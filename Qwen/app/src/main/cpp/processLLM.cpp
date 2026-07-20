// Split-ONNX Qwen3 runtime: model loading, tokenization, streaming, prefill/decode, and JNI glue.
#include "llm_runtime_state.h"
#include "ort_helpers.h"
#include "vision_runtime.h"
#include <jni.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <condition_variable>
#include <chrono>
#include <deque>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>
#include <sys/stat.h>
#include <malloc.h>
#include <cstdio>

// The text greedy decode graph is the canonical handle for the shared ORT API, allocators, and state
// layout that every merged graph uses. Per-token compute selects its graph by modality and strategy.
static ModelRuntime& mMain          = getModel(LLM_TextDecodeGreedy);
static ModelRuntime& mKVSlice       = getModel(LLM_KV_Slice);
static ModelRuntime& mKVSplit2      = getModel(LLM_KV_Split2);
static ModelRuntime& mKVConcat      = getModel(LLM_KV_Concat);
static ModelRuntime& mRopeShift     = getModel(LLM_RopeShift);

// Vision (image/video) sessions. Optionally loaded; null for a text-only export.
static ModelRuntime& mVision            = getModel(LLM_Vision);
static ModelRuntime& mImagePreprocess   = getModel(LLM_Image_Preprocess);
static ModelRuntime& mVideoPreprocess   = getModel(LLM_Video_Preprocess);

// Merged graphs share weights, but every live OrtSession still owns an optimized execution plan.
// Load selected pairs lazily and retain producer sessions while cross-turn OrtValues may use their allocators.
static std::mutex g_runtime_config_mutex;
static std::mutex g_model_session_mutex;
static std::mutex g_benchmark_correctness_mutex;
static std::string g_last_benchmark_correctness = "{}";
static jobject g_model_asset_manager = nullptr;
static std::string g_model_asset_directory;
static int g_model_ep_type = EP_CPU;
static bool g_model_low_memory = true;
static bool g_model_sessions_initialized = false;
static bool g_vision_sessions_initialized = false;
static std::atomic<uint64_t> g_vision_cancel_generation{0};
static std::array<MergedStrategy, 3> g_loaded_merged_strategies = {
    MSTRAT_GREEDY, MSTRAT_GREEDY, MSTRAT_GREEDY
};

// Config-desired strategies, cached alongside g_loaded_merged_strategies and refreshed only when
// Configure_LLM changes the decode config (both guarded by g_runtime_config_mutex). Lets the vision-init
// path reuse the resolved array instead of recomputing mergedStrategiesForConfig on every call.
static std::array<MergedStrategy, 3> g_config_merged_strategies = {
    MSTRAT_GREEDY, MSTRAT_GREEDY, MSTRAT_GREEDY
};

static std::array<MergedStrategy, 3> mergedStrategiesForConfig(
        int configuredDecodeMode, bool configuredPenalty) {
    MergedStrategy textStrategy = configuredPenalty ? MSTRAT_PENALTY_GREEDY : MSTRAT_GREEDY;
    if (configuredDecodeMode == DECODE_MODE_SAMPLING) {
        textStrategy = MSTRAT_SAMPLING;
    }

    return {textStrategy, textStrategy, textStrategy};
}

static int metadataVisionCaps() {
    int caps = 0;
    if (g_multimodal && chat_vision_start_id >= 0 && chat_vision_end_id >= 0) {
        if (chat_image_token_id >= 0) {
            caps |= 0x1;
        }
        if (chat_video_token_id >= 0) {
            caps |= 0x2;
        }
    }
    return caps;
}

static bool ensureMergedStrategySessions(
        JNIEnv* env, const std::array<MergedStrategy, 3>& strategies, int visionCaps);
static bool initializeVisionSessionsLocked(
    JNIEnv* env, const std::array<MergedStrategy, 3>& strategies, int visionCaps);

// Camera cancellation must release stale pre-encoded vision state before these helpers are defined.
static void releasePendingVisionHiddenLocked();
static void releasePendingVisionHidden();
static void setRuntimeTerminate(bool terminate);

// Token streaming.
static thread_local int g_flow_turn_id = -1;
static thread_local int g_flow_flush_index = 0;

static std::string formatTokenIdsTail(const std::vector<int>& ids, size_t limit) {
    const size_t start = ids.size() > limit ? ids.size() - limit : 0;
    std::string result = "[";
    for (size_t index = start; index < ids.size(); ++index) {
        if (index != start) {
            result.push_back(',');
        }
        result.append(std::to_string(ids[index]));
    }
    result.push_back(']');
    return result;
}

inline void append_output_words(std::string& out, int id) {
    static thread_local std::string scratch;
    const std::string& words = tokenizer->decode_id(id, scratch);
    if (words.length() == 6 && words[0] == '<' && words[5] == '>' && words[1] == '0' && words[2] == 'x') {
        const unsigned char* hex = hex_value_lut();
        char byte = static_cast<char>((hex[static_cast<unsigned char>(words[3])] << 4) |
                          hex[static_cast<unsigned char>(words[4])]);
        append_utf8_complete(out, std::string(1, byte));
    } else {
        append_utf8_complete(out, words);
    }
}

inline void flush_stream(JNIEnv* env) {
    if (stream_buf.empty()) {
        return;
    }
    const size_t byteCount = stream_buf.size();
    bool callbackInvoked = false;
    bool callbackException = false;
    if (g_main_cls != nullptr && g_on_token != nullptr) {
        jstring js = newJStringFromUtf8(env, stream_buf);
        if (js != nullptr) {
            callbackInvoked = true;
            env->CallStaticVoidMethod(g_main_cls, g_on_token, js);
            env->DeleteLocalRef(js);
            if (env->ExceptionCheck()) {
                callbackException = true;
                env->ExceptionClear();
            }
        }
    }
    LOGI("FLOW stream_flush turn=%d seq=%d bytes=%zu callback=%d exception=%d",
         g_flow_turn_id, ++g_flow_flush_index, byteCount,
         callbackInvoked ? 1 : 0, callbackException ? 1 : 0);
    stream_buf.clear();
    tokens_since_flush = 0;
    last_flush_ms = now_ms();
}

// KV shape helpers.
inline static int detectKVSeqAxis(const std::vector<int64_t>& modelDims) {
    for (size_t a = 1; a < modelDims.size(); ++a) {
        if (modelDims[a] <= 0) {
            return static_cast<int>(a);
        }
    }
    return -1;
}

inline static std::vector<int64_t> emptyKVDims(const std::vector<int64_t>& modelDims,
                                               int sequenceAxis) {
    std::vector<int64_t> dims = modelDims;
    for (int64_t& dim : dims) {
        if (dim < 1) {
            dim = 1;
        }
    }
    if (sequenceAxis >= 0 && static_cast<size_t>(sequenceAxis) < dims.size()) {
        dims[sequenceAxis] = 0;
    }
    
    return dims;
}

inline static bool validateMainIO(const ModelRuntime& m) {
    // The merged primary leads its I/O with the full state block; the KV tensors precede the linear
    // states. Validate the block is present and each full-attention KV input has a dynamic sequence axis.
    if (m.inputNames.size() < static_cast<size_t>(num_main_states) ||
        m.outputNames.size() < static_cast<size_t>(num_main_states)) {
        LOGE("ORT: merged graph state block too small: inputs=%zu outputs=%zu (need >= %d)",
             m.inputNames.size(), m.outputNames.size(), num_main_states);
        return false;
    }
    if (m.inputDims.empty() || m.inputDims[0].size() <= 4) {
        LOGE("ORT: key KV input 0 has invalid rank (need >=5 with seq axis 4)");
        return false;
    }
    for (int i = 0; i < num_keys_values; i++) {
        if (detectKVSeqAxis(m.inputDims[i]) < 0) {
            LOGE("ORT: KV input %d has no detectable dynamic sequence axis (rank %zu)", i, m.inputDims[i].size());
            return false;
        }
    }
    return true;
}

// Model loading.
static long processRssKb() {
    std::FILE* status = std::fopen("/proc/self/status", "r");
    if (status == nullptr) {
        return -1;
    }
    char line[256];
    long rssKb = -1;
    while (std::fgets(line, sizeof(line), status) != nullptr) {
        if (std::sscanf(line, "VmRSS: %ld kB", &rssKb) == 1) {
            break;
        }
    }
    std::fclose(status);
    return rssKb;
}

static bool loadModel(ModelId id, JNIEnv* env, jobject assetManager, int epType, bool lowMemoryMode) {
    ModelRuntime& m = getModel(id);
    m.id = id;
    // Merged strategy graphs share ONE injected weight blob (attachSharedInitializers), so ORT reuses the
    // single mapped copy instead of deserializing ~450MB per graph. Non-merged graphs load normally.
    const bool attachShared = modelIsMergedGraph(id);
    const long rssBeforeKb = processRssKb();
    const auto loadStarted = std::chrono::steady_clock::now();
    // LLM_Vision runs during prefill (full cores); the per-token KV/RoPE control graphs run during decode
    // and belong on the small decode pool (Top-N big cores) with the merged decode graphs.
    const bool loaded = ortLoadModelSession(m, gModelFileNames[id].c_str(), kModelExternalFileNames[id],
                                            env, assetManager, epType, lowMemoryMode, attachShared,
                                            g_model_asset_directory.c_str(), id == LLM_Vision,
                                            modelIsKvControlGraph(id));
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - loadStarted).count();
    const long rssAfterKb = processRssKb();
    if (!loaded) {
        releaseModelRuntime(m);
    }
    LOGI("Model load: id=%d file=%s ok=%d elapsed_ms=%lld rss_kb=%ld delta_kb=%ld shared=%d",
         static_cast<int>(id), gModelFileNames[id].c_str(), loaded ? 1 : 0,
         static_cast<long long>(elapsedMs), rssAfterKb,
         rssBeforeKb >= 0 && rssAfterKb >= 0 ? rssAfterKb - rssBeforeKb : -1,
         attachShared ? 1 : 0);
    return loaded;
}

// Runtime config is read from exporter-stamped ONNX metadata. Missing required keys fail load.
static void buildChatTemplates() {
    chat_user_prefix_ids              = { chat_im_start_id, chat_user_id, chat_newline_id };
    chat_assistant_prefix_ids         = { chat_im_end_id, chat_newline_id, chat_im_start_id, chat_assistant_id, chat_newline_id };
    // Thinking ON pre-fills only the opening "<think>\n"; the model then generates the reasoning body
    // and closes it with </think> (Qwen3.5 thinking template).
    chat_think_prefix_ids             = { chat_think_start_id, chat_newline_id };
    chat_previous_assistant_close_ids = { chat_im_end_id, chat_newline_id };
    chat_system_prefix_ids            = { chat_im_start_id, chat_system_id, chat_newline_id };
    chat_system_suffix_ids            = { chat_im_end_id, chat_newline_id };
}

static bool applyRuntimeModelConfig(const OrtApi* api, OrtSession* session) {
    OrtModelMetadata* md = nullptr;
    if (!ok_status(api, api->SessionGetModelMetadata(session, &md), "SessionGetModelMetadata") || md == nullptr) {
        return false;
    }
    OrtAllocator* alloc = nullptr;
    if (!ok_status(api, api->GetAllocatorWithDefaultOptions(&alloc), "GetAllocatorWithDefaultOptions")) {
        api->ReleaseModelMetadata(md);
        return false;
    }

    bool ok = true;
    auto reqInt = [&](const char* key) -> int {
        const std::string s = lookupModelMetadata(api, md, alloc, key);
        if (s.empty()) {
            LOGE("model metadata: required key '%s' is missing (re-export with Export_Qwen.py)", key);
            ok = false;
            return 0;
        }
        return std::atoi(s.c_str());
    };
    auto optInt = [&](const char* key, int fallback) -> int {
        const std::string s = lookupModelMetadata(api, md, alloc, key);
        return s.empty() ? fallback : std::atoi(s.c_str());
    };
    auto optFloat = [&](const char* key, float fallback) -> float {
        const std::string s = lookupModelMetadata(api, md, alloc, key);
        return s.empty() ? fallback : static_cast<float>(std::atof(s.c_str()));
    };
    auto validFileName = [](const std::string& value) {
        return !value.empty() && value != "." && value != ".." &&
               value.find('/') == std::string::npos && value.find('\\') == std::string::npos;
    };
    auto metadataFileName = [&](const char* key) {
        std::string value = lookupModelMetadata(api, md, alloc, key);
        if (!value.empty() && !validFileName(value)) {
            LOGE("model metadata: file-name key '%s' is not a basename: %s", key, value.c_str());
            ok = false;
            value.clear();
        }
        return value;
    };

    for (int i = 0; i < kModelCount; ++i) {
        gModelFileNames[i] = kModelFileNames[i];
    }
    gSharedInitializersFileName     = kSharedInitializersFileName;
    gSharedInitializersDataFileName = kSharedInitializersDataFileName;

    struct ModelFileMetadataKey {
        ModelId id;
        const char* key;
    };
    static constexpr ModelFileMetadataKey modelFileKeys[] = {
        {LLM_TextPrefillGreedy,         "model_file_name_text_prefill_greedy"},
        {LLM_TextPrefillPenaltyGreedy,  "model_file_name_text_prefill_penalty_greedy"},
        {LLM_TextPrefillSampling,       "model_file_name_text_prefill_sampling"},
        {LLM_TextDecodeGreedy,          "model_file_name_text_decode_greedy"},
        {LLM_TextDecodePenaltyGreedy,   "model_file_name_text_decode_penalty_greedy"},
        {LLM_TextDecodeSampling,        "model_file_name_text_decode_sampling"},
        {LLM_ImagePrefillGreedy,        "model_file_name_image_prefill_greedy"},
        {LLM_ImagePrefillPenaltyGreedy, "model_file_name_image_prefill_penalty_greedy"},
        {LLM_ImagePrefillSampling,      "model_file_name_image_prefill_sampling"},
        {LLM_ImageDecodeGreedy,         "model_file_name_image_decode_greedy"},
        {LLM_ImageDecodePenaltyGreedy,  "model_file_name_image_decode_penalty_greedy"},
        {LLM_ImageDecodeSampling,       "model_file_name_image_decode_sampling"},
        {LLM_VideoPrefillGreedy,        "model_file_name_video_prefill_greedy"},
        {LLM_VideoPrefillPenaltyGreedy, "model_file_name_video_prefill_penalty_greedy"},
        {LLM_VideoPrefillSampling,      "model_file_name_video_prefill_sampling"},
        {LLM_VideoDecodeGreedy,         "model_file_name_video_decode_greedy"},
        {LLM_VideoDecodePenaltyGreedy,  "model_file_name_video_decode_penalty_greedy"},
        {LLM_VideoDecodeSampling,       "model_file_name_video_decode_sampling"},
        {LLM_Vision,                    "model_file_name_vision"},
        {LLM_Image_Preprocess,          "model_file_name_image_preprocess"},
        {LLM_Video_Preprocess,          "model_file_name_video_preprocess"},
        {LLM_KV_Slice,                  "model_file_name_kv_slice"},
        {LLM_KV_Split2,                 "model_file_name_kv_split2"},
        {LLM_KV_Concat,                 "model_file_name_kv_concat"},
        {LLM_RopeShift,                 "model_file_name_rope_shift"},
    };
    for (const ModelFileMetadataKey& entry : modelFileKeys) {
        const std::string value = metadataFileName(entry.key);
        if (!value.empty()) {
            gModelFileNames[entry.id] = value;
        } else if (entry.id < LLM_FIRST_OPTIONAL_VISION_MODEL) {
            LOGE("model metadata: required file-name key '%s' is missing", entry.key);
            ok = false;
        }
    }
    const std::string sharedModelName = metadataFileName("model_file_name_shared_initializers");
    const std::string sharedDataName = metadataFileName("model_file_name_shared_initializers_data");
    if (sharedModelName.empty() || sharedDataName.empty()) {
        LOGE("model metadata: shared initializer file-name keys are required");
        ok = false;
    } else {
        gSharedInitializersFileName = sharedModelName;
        gSharedInitializersDataFileName = sharedDataName;
    }

    const int meta_layers        = reqInt("num_layers");
    const int meta_full_layers   = reqInt("num_full_attention_layers");
    const int meta_linear_layers = reqInt("num_linear_attention_layers");
    const int meta_kv_tensors    = reqInt("kv_num_tensors");
    const int meta_kv_blocks     = reqInt("kv_blocks_per_layer");
    const int meta_max_seq_len = reqInt("max_seq_len");
    const int meta_endoftext   = reqInt("chat_endoftext_id");
    const int meta_im_start    = reqInt("chat_im_start_id");
    const int meta_im_end      = reqInt("chat_im_end_id");
    const int meta_system      = reqInt("chat_system_id");
    const int meta_user        = reqInt("chat_user_id");
    const int meta_assistant   = reqInt("chat_assistant_id");
    const int meta_newline     = reqInt("chat_newline_id");
    const int meta_dbl_newline = reqInt("chat_double_newline_id");
    const int meta_think_start = reqInt("chat_think_start_id");
    const int meta_think_end   = reqInt("chat_think_end_id");
        const bool meta_supports_thinking = optInt(
            "chat_supports_thinking", meta_linear_layers > 0 ? 1 : 0) != 0;

    const std::string meta_stop_ids = lookupModelMetadata(api, md, alloc, "stop_token_ids");

    // Multimodal reservation (optional): only present when a vision model stamped these ids.
    const int meta_image_token  = optInt("image_token_id", -1);
    const int meta_video_token  = optInt("video_token_id", -1);
    const int meta_vision_start = optInt("vision_start_token_id", -1);
    const int meta_vision_end   = optInt("vision_end_token_id", -1);

    // Vision patch / sampling geometry (optional; defaults keep a text-only build inert).
    const int   meta_spatial_merge = optInt("spatial_merge_size", 2);
    const int   meta_patch_size    = optInt("patch_size", 16);
    const int   meta_temporal_ps   = optInt("temporal_patch_size", 2);
    const float meta_video_fps     = optFloat("video_fps", 2.0f);
    const int   meta_video_min_fr  = optInt("video_min_frames", 4);
    const int   meta_video_max_fr  = optInt("video_max_frames", 768);

    // Optional vision input geometry ("H,W"); absent for text-only models (keep the default size).
    const std::string meta_input_image_size = lookupModelMetadata(api, md, alloc, "input_image_size");
    const std::string meta_input_video_size = lookupModelMetadata(api, md, alloc, "input_video_size");

    api->ReleaseModelMetadata(md);
    if (!ok) {
        return false;
    }

    if (meta_layers <= 0 || meta_layers > max_num_layers) {
        LOGE("model metadata: num_layers=%d out of range (1..%d)", meta_layers, max_num_layers);
        return false;
    }
    if (meta_full_layers <= 0 || meta_linear_layers < 0 ||
        meta_full_layers + meta_linear_layers != meta_layers) {
        LOGE("model metadata: attention-layer split invalid (layers=%d full=%d linear=%d)",
             meta_layers, meta_full_layers, meta_linear_layers);
        return false;
    }
    if (meta_kv_blocks != 2 && meta_kv_blocks != 4 && meta_kv_blocks != 6) {
        LOGE("model metadata: kv_blocks_per_layer=%d (expected 2/4/6)", meta_kv_blocks);
        return false;
    }
    if (meta_kv_tensors != meta_full_layers * meta_kv_blocks || meta_kv_tensors > max_keys_values) {
        LOGE("model metadata: kv_num_tensors=%d inconsistent (full_layers=%d x blocks=%d; ceiling=%d)",
             meta_kv_tensors, meta_full_layers, meta_kv_blocks, max_keys_values);
        return false;
    }
    if (meta_max_seq_len < MIN_MEMORY_TOKENS + kDecodeRecycleHeadroom) {
        LOGE("model metadata: max_seq_len=%d too small", meta_max_seq_len);
        return false;
    }

    num_layers                  = meta_layers;
    num_full_attention_layers   = meta_full_layers;
    num_linear_attention_layers = meta_linear_layers;
    g_kv_blocks                 = meta_kv_blocks;
    num_keys_values             = meta_kv_tensors;
    // num_linear_states / num_main_states / g_linear_blocks are derived from the Main graph in
    // configureKVLayout, which is the authoritative source for the exact state-tensor count.

    max_seq_len = meta_max_seq_len;

    DEFAULT_MEMORY_TOKENS      = (max_seq_len * 3) / 8;
    DEFAULT_PREFILL_TOKENS     = DEFAULT_MEMORY_TOKENS / 8;
    DEFAULT_MAX_DECODE_TOKENS  = max_seq_len / 2 - 1;
    MAX_MEMORY_TOKENS_RUNTIME  = max_seq_len - kDecodeRecycleHeadroom;
    MAX_PREFILL_TOKENS_RUNTIME = MAX_MEMORY_TOKENS_RUNTIME;
    MAX_DECODE_TOKENS_RUNTIME  = max_seq_len;
    g_memory_tokens  = DEFAULT_MEMORY_TOKENS;
    g_prefill_tokens = DEFAULT_PREFILL_TOKENS;
    g_decode_tokens  = DEFAULT_MAX_DECODE_TOKENS;

    chat_endoftext_id      = meta_endoftext;
    chat_im_start_id       = meta_im_start;
    chat_im_end_id         = meta_im_end;
    chat_system_id         = meta_system;
    chat_user_id           = meta_user;
    chat_assistant_id      = meta_assistant;
    chat_newline_id        = meta_newline;
    chat_double_newline_id = meta_dbl_newline;
    chat_think_start_id    = meta_think_start;
    chat_think_end_id      = meta_think_end;
    g_supports_thinking    = meta_supports_thinking;
    chat_image_token_id    = meta_image_token;
    chat_video_token_id    = meta_video_token;
    chat_vision_start_id   = meta_vision_start;
    chat_vision_end_id     = meta_vision_end;
    g_multimodal           = (meta_image_token >= 0 || meta_video_token >= 0);

    g_stop_token_ids.clear();
    auto appendStopToken = [](int id) {
        if (id >= 0 && std::find(g_stop_token_ids.begin(), g_stop_token_ids.end(), id) ==
                               g_stop_token_ids.end()) {
            g_stop_token_ids.push_back(id);
        }
    };
    size_t stopStart = 0;
    while (stopStart < meta_stop_ids.size()) {
        const size_t comma = meta_stop_ids.find(',', stopStart);
        const std::string item = meta_stop_ids.substr(
                stopStart, comma == std::string::npos ? std::string::npos : comma - stopStart);
        char* end = nullptr;
        const long value = std::strtol(item.c_str(), &end, 10);
        if (end != item.c_str() && *end == '\0' && value >= 0 &&
            value <= std::numeric_limits<int>::max()) {
            appendStopToken(static_cast<int>(value));
        } else {
            LOGE("model metadata: ignoring invalid stop token id '%s'", item.c_str());
        }
        if (comma == std::string::npos) {
            break;
        }
        stopStart = comma + 1;
    }
    // The exporter/model generation config can name only <|endoftext|>, while the chat template closes
    // assistant turns with <|im_end|>. A generated <|im_start|> begins another role block and must also
    // terminate before its raw chat-template prefix reaches the transcript.
    appendStopToken(meta_endoftext);
    appendStopToken(meta_im_end);
    appendStopToken(meta_im_start);

    g_spatial_merge_size   = meta_spatial_merge > 0 ? meta_spatial_merge : 2;
    g_patch_size           = meta_patch_size > 0 ? meta_patch_size : 16;
    g_temporal_patch_size  = meta_temporal_ps > 0 ? meta_temporal_ps : 2;
    g_video_fps            = meta_video_fps > 0.0f ? meta_video_fps : 2.0f;
    g_video_min_frames     = meta_video_min_fr > 0 ? meta_video_min_fr : 4;
    g_video_max_frames     = meta_video_max_fr > 0 ? meta_video_max_fr : 768;

    // Camera auto-fit: parse "H,W" and mirror it into the camera globals so the capture pipeline
    // matches the exported vision input. Fall back to the default when the key is absent or malformed.
    auto parseHW = [](const std::string& s, int& h, int& w) {
        h = 0; w = 0;
        const size_t comma = s.find(',');
        if (!s.empty() && comma != std::string::npos) {
            h = std::atoi(s.substr(0, comma).c_str());
            w = std::atoi(s.substr(comma + 1).c_str());
        }
    };
    int meta_img_h = 0, meta_img_w = 0;
    parseHW(meta_input_image_size, meta_img_h, meta_img_w);
    if (meta_img_h > 0 && meta_img_w > 0) {
        input_image_height = meta_img_h;
        input_image_width  = meta_img_w;
    } else {
        input_image_height = DEFAULT_INPUT_IMAGE_SIZE;
        input_image_width  = DEFAULT_INPUT_IMAGE_SIZE;
    }
    int meta_vid_h = 0, meta_vid_w = 0;
    parseHW(meta_input_video_size, meta_vid_h, meta_vid_w);
    if (meta_vid_h > 0 && meta_vid_w > 0) {
        input_video_height = meta_vid_h;
        input_video_width  = meta_vid_w;
    } else {
        input_video_height = input_image_height;   // fall back to the image size when unspecified
        input_video_width  = input_image_width;
    }

    buildChatTemplates();

     LOGI("model metadata applied: layers=%d kv_blocks=%d kv_tensors=%d max_seq_len=%d "
            "thinking=%d (mem=%d prefill=%d decode=%d)",
            num_layers, g_kv_blocks, num_keys_values, max_seq_len,
            g_supports_thinking ? 1 : 0,
         DEFAULT_MEMORY_TOKENS, DEFAULT_PREFILL_TOKENS, DEFAULT_MAX_DECODE_TOKENS);
    if (g_multimodal) {
        LOGI("model metadata applied: vision input_image_size=%dx%d (HxW), image_token=%d video_token=%d",
             input_image_height, input_image_width, chat_image_token_id, chat_video_token_id);
    }
    return true;
}

// Preload model geometry and file-name metadata before creating the real sessions.
static bool preloadRuntimeMetadata(JNIEnv* env, jobject assetManager, bool lowMemoryMode,
                                   const std::string& assetDirectory) {
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv* ortEnv = getSharedEnv(api);
    if (ortEnv == nullptr) {
        LOGE("preloadRuntimeMetadata: shared ORT env unavailable");
        return false;
    }

    const char* fileName = kMetadataModelFileName;

    OrtSessionOptions* opts = nullptr;
    if (!ok_status(api, api->CreateSessionOptions(&opts), "CreateSessionOptions(meta)")) {
        return false;
    }
    if ((gSharedEnvUsesGlobalPools &&
         !ok_status(api, api->DisablePerSessionThreads(opts), "DisablePerSessionThreads(meta)")) ||
        !ok_status(api, api->SetSessionGraphOptimizationLevel(opts, ORT_DISABLE_ALL),
                   "SetSessionGraphOptimizationLevel(meta)")) {
        api->ReleaseSessionOptions(opts);
        return false;
    }

    OrtSession* session = nullptr;
    OrtStatus* status = nullptr;
    std::vector<char> fileBuffer;
    bool haveBuffer = false;
    if (!lowMemoryMode && assetManager != nullptr) {
        AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
        if (mgr != nullptr) {
            const std::string metadataAssetPath = assetPath(assetDirectory.c_str(), fileName);
            AAsset* asset = AAssetManager_open(mgr, metadataAssetPath.c_str(), AASSET_MODE_BUFFER);
            haveBuffer = readAssetFully(asset, fileBuffer, fileName);
            if (asset != nullptr) {
                AAsset_close(asset);
            }
        }
    }
    if (haveBuffer) {
        status = api->CreateSessionFromArray(ortEnv, fileBuffer.data(), fileBuffer.size(), opts, &session);
    } else {
        const std::string cachePath   = cache_path + fileName;
        const std::string storagePath = storage_path + fileName;
        const std::string path = cacheFileExists(cachePath) ? cachePath : storagePath;
        status = api->CreateSession(ortEnv, path.c_str(), opts, &session);
    }
    api->ReleaseSessionOptions(opts);
    if (!ok_status(api, status, "CreateSession(meta)") || session == nullptr) {
        LOGE("preloadRuntimeMetadata: could not open %s to read model metadata", fileName);
        return false;
    }

    const bool applied = applyRuntimeModelConfig(api, session);
    api->ReleaseSession(session);
    return applied;
}

// The merged graphs lead ALL I/O with the state block (in_* inputs / out_* outputs): the
// full-attention KV tensors then the hybrid linear-attention conv/recurrent states. Everything after is
// the token input, phase counters, per-strategy scalars, and the token/kv_seq/save_id outputs.
static bool configureKVLayout() {
    ModelRuntime& m = mMain;   // merged primary (text greedy decode) defines the shared state layout
    if (m.api == nullptr || m.session == nullptr) {
        LOGE("configureKVLayout: merged primary graph not loaded");
        return false;
    }
    if (num_keys_values <= 0 || num_layers <= 0 || g_kv_blocks <= 0) {
        LOGE("configureKVLayout: model metadata not applied (layers=%d blocks=%d kv=%d)",
             num_layers, g_kv_blocks, num_keys_values);
        return false;
    }
    // Count the contiguous leading state tensors (in_* inputs and out_* outputs).
    int statesByInputs = 0;
    for (const char* name : m.inputNames) {
        if (name != nullptr && std::strncmp(name, "in_", 3) == 0) { ++statesByInputs; }
        else { break; }
    }
    int statesByOutputs = 0;
    for (const char* name : m.outputNames) {
        if (name != nullptr && std::strncmp(name, "out_", 4) == 0) { ++statesByOutputs; }
        else { break; }
    }
    if (statesByInputs != statesByOutputs || statesByInputs < num_keys_values) {
        LOGE("configureKVLayout: merged state block inconsistent (in=%d out=%d, full-attn KV=%d)",
             statesByInputs, statesByOutputs, num_keys_values);
        return false;
    }
    num_main_states   = statesByInputs;
    num_linear_states = num_main_states - num_keys_values;
    if (num_linear_states < 0 || num_main_states > max_main_states) {
        LOGE("configureKVLayout: linear-state count invalid (main=%d full=%d linear=%d, ceiling=%d)",
             num_main_states, num_keys_values, num_linear_states, max_main_states);
        return false;
    }
    if (num_linear_attention_layers > 0) {
        g_linear_blocks = num_linear_states / num_linear_attention_layers;
        if (g_linear_blocks <= 0 || g_linear_blocks * num_linear_attention_layers != num_linear_states) {
            LOGE("configureKVLayout: linear states=%d not divisible by linear layers=%d",
                 num_linear_states, num_linear_attention_layers);
            return false;
        }
    } else if (num_linear_states != 0) {
        LOGE("configureKVLayout: metadata declares 0 linear layers but the graph carries %d extra states",
             num_linear_states);
        return false;
    } else {
        g_linear_blocks = 0;
    }

    kvSliceStartIdx = num_keys_values;   // KV-management graphs act on the full-attention block only
    kvSliceEndIdx   = num_keys_values + 1;

    const char* modeName = (g_kv_blocks == 2) ? "F16/F32 (key,value)"
                         : (g_kv_blocks == 4) ? "symmetric quant (key,value,k_scale,v_scale)"
                                              : "asymmetric quant (key,value,k_scale,k_bias,v_scale,v_bias)";
    LOGI("configureKVLayout: %d layers (%d full + %d linear) x %d KV blocks -> %d KV + %d linear = %d "
         "merged states [%s]", num_layers, num_full_attention_layers, num_linear_attention_layers,
         g_kv_blocks, num_keys_values, num_linear_states, num_main_states, modeName);
    return true;
}

// Empty KV-cache initialization.
static float gEmptyKVData = 0.0f;

// Fixed-shape zero backing for hybrid linear-attention init states (persists for the tensors' lifetime).
static std::array<std::vector<uint8_t>, max_linear_states> gLinearInitBacking;

// Build fresh-conversation linear-attention states: full-shape zero conv/recurrent summaries.
// No-op for pure-transformer models (num_linear_states == 0).
static bool initLinearStates() {
    ModelRuntime& m = mMain;   // merged primary: state block dims/types define the linear init shapes
    if (m.api == nullptr || m.session == nullptr) {
        return false;
    }
    const OrtApi* api = m.api;
    for (int i = 0; i < num_linear_states; ++i) {
        const int idx = num_keys_values + i;   // linear states follow the full-attention KV prefix
        if (linear_state_init[i] != nullptr) {
            api->ReleaseValue(linear_state_init[i]);
            linear_state_init[i] = nullptr;
        }
        std::vector<int64_t> dims = m.inputDims[idx];
        int64_t elems = 1;
        for (int64_t& d : dims) {
            if (d <= 0) { d = 1; }   // dynamic batch axis -> 1 at conversation start
            elems *= d;
        }
        const size_t elementSize = tensorElementSize(m.inputTypes[idx]);
        if (elementSize == 0) {
            LOGE("initLinearStates: unsupported input type %d at state %d",
                 static_cast<int>(m.inputTypes[idx]), idx);
            return false;
        }
        const size_t bytes = static_cast<size_t>(elems) * elementSize;
        gLinearInitBacking[i].assign(bytes, 0);
        if (!createTensorWithData(api, m.memoryInfo, gLinearInitBacking[i].data(), bytes,
                                  dims, m.inputTypes[idx], &linear_state_init[i],
                                  "CreateTensorWithDataAsOrtValue(linear_init)")) {
            for (OrtValue*& s : linear_state_init) {
                if (s != nullptr) { api->ReleaseValue(s); s = nullptr; }
            }
            return false;
        }
    }
    return true;
}

static bool initKVCache() {
    ModelRuntime& m = mMain;   // merged primary: state block dims/types define the empty-KV init shapes
    if (m.api == nullptr || m.session == nullptr) {
        LOGE("ORT: initKVCache called before the merged primary graph loaded");
        return false;
    }
    const OrtApi* api = m.api;
    for (int i = 0; i < num_keys_values; i++) {
        if (input_tensors_kv_init[i] != nullptr) {
            api->ReleaseValue(input_tensors_kv_init[i]);
            input_tensors_kv_init[i] = nullptr;
        }
        const int seqAxis = detectKVSeqAxis(m.inputDims[i]);
        std::vector<int64_t> kvDims = emptyKVDims(m.inputDims[i], seqAxis);
        if (!createTensorWithData(api, m.memoryInfo, reinterpret_cast<void*>(&gEmptyKVData), 0,
                                  kvDims, m.inputTypes[i], &input_tensors_kv_init[i],
                                  "CreateTensorWithDataAsOrtValue(empty_kv)")) {
            for (OrtValue*& kv : input_tensors_kv_init) {
                if (kv != nullptr) { api->ReleaseValue(kv); kv = nullptr; }
            }
            return false;
        }
    }
    return initLinearStates();
}

// ── Vision (image/video) capability + static input buffers ─────────────────────────────────────
// Derive vision-token geometry from the loaded graph shapes (the authoritative source, like the KV
// layout) and pre-allocate the static, pre-bound pixel_values tensors. Silently disables a modality
// whose graphs did not load or whose ids/geometry are missing. No-op for a text-only export.
static bool sessionReady(const ModelRuntime& m) { return m.api != nullptr && m.session != nullptr; }

// Positive static dim from a loaded graph, else fallback (handles dynamic/symbolic axes).
static int64_t staticDimOr(const ModelRuntime& m, bool input, int tensorIdx, int axis, int64_t fallback) {
    const std::vector<std::vector<int64_t>>& dl = input ? m.inputDims : m.outputDims;
    if (tensorIdx < 0 || static_cast<size_t>(tensorIdx) >= dl.size()) return fallback;
    const std::vector<int64_t>& d = dl[tensorIdx];
    if (axis < 0 || static_cast<size_t>(axis) >= d.size() || d[axis] <= 0) return fallback;
    return d[axis];
}

static int64_t rawDimOr(const ModelRuntime& m, bool input, int tensorIdx, int axis, int64_t fallback) {
    const std::vector<std::vector<int64_t>>& dl = input ? m.inputDims : m.outputDims;
    if (tensorIdx < 0 || static_cast<size_t>(tensorIdx) >= dl.size()) return fallback;
    const std::vector<int64_t>& d = dl[tensorIdx];
    if (axis < 0 || static_cast<size_t>(axis) >= d.size()) return fallback;
    return d[axis];
}

static int deriveVideoFrameSeqLen(int64_t h, int64_t w) {
    const int patch = std::max(1, g_patch_size);
    const int merge = std::max(1, g_spatial_merge_size);
    const int gridH = std::max<int64_t>(1, h / patch / merge);
    const int gridW = std::max<int64_t>(1, w / patch / merge);
    return gridH * gridW;
}

static void updateVideoTokenGeometryForFrames(int frameCount) {
    const int frames = std::max(1, frameCount);
    g_video_grid_t = std::max(1, frames / std::max(1, g_temporal_patch_size));
    if (g_video_frame_seqlen <= 0) {
        g_video_frame_seqlen = deriveVideoFrameSeqLen(input_video_height, input_video_width);
    }
    g_video_embed_size = std::max(1, g_video_grid_t * g_video_frame_seqlen);
}

static bool ensureDynamicVideoCapacity(int frameCount) {
    if (frameCount <= g_video_dynamic_capacity_frames) {
        return true;
    }
    if (g_video_frame_storage_bytes == 0 || g_video_num_frames <= 0) {
        return false;
    }
        const int initialCapacity = std::max(1, g_video_min_frames);
        int newCapacity = std::max(
            frameCount,
            g_video_dynamic_capacity_frames > 0
                ? g_video_dynamic_capacity_frames * 2
                : initialCapacity);
    newCapacity = std::min(newCapacity, g_video_num_frames);
    if (newCapacity < frameCount) {
        return false;
    }
    g_video_pixel_backing.resize(static_cast<size_t>(newCapacity) * g_video_frame_storage_bytes);
    g_video_dynamic_capacity_frames = newCapacity;
    return true;
}

static void releaseDynamicVideoInputLocked() {
    if (g_video_fixed_length_input) {
        return;
    }
    if (g_video_pixel_values != nullptr && mVideoPreprocess.api != nullptr) {
        mVideoPreprocess.api->ReleaseValue(g_video_pixel_values);
        g_video_pixel_values = nullptr;
    }
    std::vector<uint8_t>().swap(g_video_pixel_backing);
    g_video_dynamic_capacity_frames = 0;
    g_video_recorded_frames = 0;
    g_video_ring_count.store(0, std::memory_order_relaxed);
    g_video_ring_write.store(0, std::memory_order_relaxed);
}

static void setVisionEncodeTerminate(bool terminate) {
    ModelRuntime* visionModels[] = { &mImagePreprocess, &mVideoPreprocess, &mVision };
    for (ModelRuntime* m : visionModels) {
        if (m->api == nullptr || m->runOptions == nullptr) {
            continue;
        }
        if (terminate) {
            logOrtStatus(m->api, m->api->RunOptionsSetTerminate(m->runOptions), "RunOptionsSetTerminate(vision)");
        } else {
            logOrtStatus(m->api, m->api->RunOptionsUnsetTerminate(m->runOptions), "RunOptionsUnsetTerminate(vision)");
        }
    }
}

static bool writePixelValues(uint8_t* destination, size_t destinationBytes,
                             ONNXTensorElementDataType type, const uint8_t* source,
                             size_t count) {
    const size_t elementSize = tensorElementSize(type);
    if (destination == nullptr || source == nullptr || elementSize == 0 ||
        count > destinationBytes / elementSize) {
        return false;
    }
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
        std::memcpy(destination, source, count);
        return true;
    }
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        float* output = reinterpret_cast<float*>(destination);
        for (size_t i = 0; i < count; ++i) { output[i] = source[i]; }
        return true;
    }
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        uint16_t* output = reinterpret_cast<uint16_t*>(destination);
        for (size_t i = 0; i < count; ++i) { output[i] = floatToHalf(source[i]); }
        return true;
    }
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
        uint16_t* output = reinterpret_cast<uint16_t*>(destination);
        for (size_t i = 0; i < count; ++i) {
            const float value = source[i];
            uint32_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            output[i] = static_cast<uint16_t>(bits >> 16);
        }
        return true;
    }
    LOGE("vision preprocess input has unsupported element type %d", static_cast<int>(type));
    return false;
}

// Allocate one static pixel tensor over owned storage using the preprocess graph's declared input type.
static bool allocPixelValues(ModelRuntime& m, const std::vector<int64_t>& dims,
                             std::vector<uint8_t>& backing, OrtValue** out, uint8_t fill,
                             const char* op) {
    if (*out != nullptr) {
        m.api->ReleaseValue(*out);
        *out = nullptr;
    }
    if (m.inputTypes.empty()) {
        return false;
    }
    const ONNXTensorElementDataType type = m.inputTypes[0];
    const size_t elementSize = tensorElementSize(type);
    if (elementSize == 0) {
        LOGE("%s: unsupported input type %d", op, static_cast<int>(type));
        return false;
    }
    size_t elems = 1;
    for (int64_t d : dims) {
        if (d <= 0 || elems > std::numeric_limits<size_t>::max() / static_cast<size_t>(d)) {
            return false;
        }
        elems *= static_cast<size_t>(d);
    }
    if (elems > std::numeric_limits<size_t>::max() / elementSize) {
        return false;
    }
    backing.assign(elems * elementSize, 0);
    if (fill != 0) {
        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
            std::fill(backing.begin(), backing.end(), fill);
        } else {
            uint32_t floatBits = 0;
            const float floatValue = fill;
            std::memcpy(&floatBits, &floatValue, sizeof(floatBits));
            uint32_t encoded = 0;
            size_t encodedBytes = elementSize;
            if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                encoded = floatBits;
            } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
                encoded = floatToHalf(floatValue);
            } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
                encoded = floatBits >> 16;
            } else {
                LOGE("%s: unsupported fill input type %d", op, static_cast<int>(type));
                backing.clear();
                return false;
            }
            for (size_t index = 0; index < elems; ++index) {
                std::memcpy(backing.data() + index * encodedBytes, &encoded, encodedBytes);
            }
        }
    }
    return createTensorWithData(m.api, m.memoryInfo, backing.data(), backing.size(), dims, type, out, op);
}

static bool initVisionBuffers() {
    bool ok = true;
    if (g_vision_image_ready && sessionReady(mImagePreprocess)) {
        // pixel_values [N,1,3,H,W].
        const int64_t N = std::max<int64_t>(1, g_vision_batch_size);
        const int64_t H = staticDimOr(mImagePreprocess, true, 0, 3, input_image_height);
        const int64_t W = staticDimOr(mImagePreprocess, true, 0, 4, input_image_width);
        g_image_slot_bytes = static_cast<size_t>(3 * H * W);
        g_image_slot_storage_bytes = g_image_slot_bytes *
                                     tensorElementSize(mImagePreprocess.inputTypes[0]);
        if (!allocPixelValues(mImagePreprocess, {N, 1, 3, H, W}, g_image_pixel_backing,
                              &g_image_pixel_values, 128, "CreateTensor(image pixel_values)")) {
            LOGE("initVisionBuffers: image pixel_values allocation failed; disabling image");
            g_vision_image_ready = false;
            ok = false;
        }
    }
    if (g_vision_video_ready && sessionReady(mVideoPreprocess)) {
        // video_frames [F,3,H,W].
        const int64_t H = staticDimOr(mVideoPreprocess, true, 0, 2, input_video_height);
        const int64_t W = staticDimOr(mVideoPreprocess, true, 0, 3, input_video_width);
        g_video_frame_bytes = static_cast<size_t>(3 * H * W);
        g_video_frame_storage_bytes = g_video_frame_bytes *
                                      tensorElementSize(mVideoPreprocess.inputTypes[0]);
        if (g_video_fixed_length_input) {
            const int64_t F = std::max<int64_t>(1, g_video_num_frames);
            if (!allocPixelValues(mVideoPreprocess, {F, 3, H, W}, g_video_pixel_backing,
                                  &g_video_pixel_values, 0, "CreateTensor(video pixel_values)")) {
                LOGE("initVisionBuffers: video pixel_values allocation failed; disabling video");
                g_vision_video_ready = false;
                ok = false;
            }
        } else {
            if (g_video_pixel_values != nullptr) {
                mVideoPreprocess.api->ReleaseValue(g_video_pixel_values);
                g_video_pixel_values = nullptr;
            }
            g_video_pixel_backing.clear();
            g_video_dynamic_capacity_frames = 0;
        }
    }
    return ok;
}

static void releaseVisionBuffers() {
    if (g_image_pixel_values != nullptr && mImagePreprocess.api != nullptr) {
        mImagePreprocess.api->ReleaseValue(g_image_pixel_values);
        g_image_pixel_values = nullptr;
    }
    if (g_video_pixel_values != nullptr && mVideoPreprocess.api != nullptr) {
        mVideoPreprocess.api->ReleaseValue(g_video_pixel_values);
        g_video_pixel_values = nullptr;
    }
    std::vector<uint8_t>().swap(g_image_pixel_backing);
    std::vector<uint8_t>().swap(g_video_pixel_backing);
    g_image_slot_bytes = 0;
    g_image_slot_storage_bytes = 0;
    g_video_frame_bytes = 0;
    g_video_frame_storage_bytes = 0;
    g_video_dynamic_capacity_frames = 0;
    g_video_recorded_frames = 0;
    g_video_ring_count.store(0, std::memory_order_relaxed);
    g_video_ring_write.store(0, std::memory_order_relaxed);
}

static bool configureVisionLayout() {
    g_vision_image_ready = false;
    g_vision_video_ready = false;
    if (!g_multimodal) {
        return true;   // text-only export: nothing to configure
    }

    // Merged runtime: vision readiness = ViT encode + preprocess + the merged image/video prefill+decode
    // graphs (which fold concat/rotary/main internally). The concat token count is read from the merged
    // prefill graph's vision-hidden input shape (the authoritative source, like the KV layout).
    ModelRuntime& imgPrefill = getModel(mergedPrefillModel(1, g_loaded_merged_strategies[1]));
    ModelRuntime& imgDecode  = getModel(mergedDecodeModel(1, g_loaded_merged_strategies[1]));
    ModelRuntime& vidPrefill = getModel(mergedPrefillModel(2, g_loaded_merged_strategies[2]));
    ModelRuntime& vidDecode  = getModel(mergedDecodeModel(2, g_loaded_merged_strategies[2]));
    auto modalityGraphsReady = [](int modality) {
        const MergedStrategy strategy = g_loaded_merged_strategies[modality];
        return sessionReady(getModel(mergedPrefillModel(modality, strategy))) &&
               sessionReady(getModel(mergedDecodeModel(modality, strategy)));
    };
    auto visionTypesMatch = [](const ModelRuntime& preprocess) {
        if (preprocess.outputTypes.size() != mVision.inputTypes.size()) {
            return false;
        }
        for (size_t index = 0; index < preprocess.outputTypes.size(); ++index) {
            if (preprocess.outputTypes[index] != mVision.inputTypes[index]) {
                return false;
            }
        }
        return true;
    };
    const bool imageGraphs = sessionReady(mVision) && sessionReady(mImagePreprocess) &&
                             sessionReady(imgPrefill) && sessionReady(imgDecode) &&
                             modalityGraphsReady(1) && visionTypesMatch(mImagePreprocess);
    const bool videoGraphs = sessionReady(mVision) && sessionReady(mVideoPreprocess) &&
                             sessionReady(vidPrefill) && sessionReady(vidDecode) &&
                             modalityGraphsReady(2) && visionTypesMatch(mVideoPreprocess);
    auto visionHiddenTokens = [](ModelRuntime& g, const char* name) -> int {
        const int idx = findInputIdx(g, name);
        return (idx >= 0 && g.inputDims[idx].size() > 1) ? static_cast<int>(g.inputDims[idx][1]) : 0;
    };

    if (imageGraphs) {
        input_image_height = static_cast<int>(staticDimOr(
            mImagePreprocess, true, 0, 3, input_image_height));
        input_image_width = static_cast<int>(staticDimOr(
            mImagePreprocess, true, 0, 4, input_image_width));
        g_vision_batch_size = static_cast<int>(staticDimOr(mImagePreprocess, true, 0, 0, 1));
        g_image_embed_size  = visionHiddenTokens(imgPrefill, "concat_image_vision_hidden_states");
        g_vision_image_ready = g_vision_batch_size > 0 && g_image_embed_size > 0 &&
                               chat_image_token_id >= 0 && chat_vision_start_id >= 0 &&
                               chat_vision_end_id >= 0;
    }
    if (videoGraphs) {
        const int64_t rawFrames = rawDimOr(mVideoPreprocess, true, 0, 0, 0);
        g_video_fixed_length_input = rawFrames > 0;
        g_video_num_frames = g_video_fixed_length_input
                ? static_cast<int>(rawFrames)
                : std::max(1, g_video_max_frames);
        const int64_t videoH = staticDimOr(mVideoPreprocess, true, 0, 2, input_video_height);
        const int64_t videoW = staticDimOr(mVideoPreprocess, true, 0, 3, input_video_width);
        input_video_height = static_cast<int>(videoH);
        input_video_width = static_cast<int>(videoW);
        g_video_frame_seqlen = deriveVideoFrameSeqLen(videoH, videoW);
        if (g_video_fixed_length_input) {
            g_video_grid_t = (g_temporal_patch_size > 0) ? g_video_num_frames / g_temporal_patch_size : 0;
            const int staticEmbed = visionHiddenTokens(vidPrefill, "concat_video_vision_hidden_states");
            if (staticEmbed > 0) {
                g_video_embed_size = staticEmbed;
                if (g_video_grid_t > 0) {
                    g_video_frame_seqlen = std::max(1, g_video_embed_size / g_video_grid_t);
                }
            } else {
                g_video_embed_size = std::max(1, g_video_grid_t * g_video_frame_seqlen);
            }
        } else {
            updateVideoTokenGeometryForFrames(std::max(1, g_video_min_frames));
        }
        g_vision_video_ready = g_video_num_frames > 0 && g_video_frame_seqlen > 0 &&
                               chat_video_token_id >= 0 && chat_vision_start_id >= 0 &&
                               chat_vision_end_id >= 0;
    }

    if (!g_vision_image_ready && !g_vision_video_ready) {
        LOGI("configureVisionLayout: metadata declares vision but no complete graph set loaded; "
             "vision disabled (text unaffected)");
        return true;
    }

    const bool buffersReady = initVisionBuffers();
    LOGI("configureVisionLayout: image_ready=%d (N=%d, input=%dx%d, embed=%d) video_ready=%d "
         "(frames=%d, input=%dx%d, fixed=%d, grid_t=%d, frame_seqlen=%d, embed=%d)",
         g_vision_image_ready ? 1 : 0, g_vision_batch_size,
         input_image_width, input_image_height, g_image_embed_size,
         g_vision_video_ready ? 1 : 0, g_video_num_frames,
         input_video_width, input_video_height, g_video_fixed_length_input ? 1 : 0,
         g_video_grid_t, g_video_frame_seqlen, g_video_embed_size);
    return buffersReady && (g_vision_image_ready || g_vision_video_ready);
}

// Peak bytes reserved in each CPU arena. CPU decode uses a dedicated OrtEnv/thread pool, so its arena
// must be pre-grown and invalidated independently from the shared prefill/utility arena.
static int64_t g_reserved_prefill_arena_bytes = 0;
static int64_t g_reserved_decode_arena_bytes = 0;

// Pre-grow each CPU arena so cache-heavy graphs do not grow it mid-turn. Reserve the largest loaded
// cache I/O working set plus one persistent batch-1 saved KV on the decode side. The per-Env high-water
// marks make this idempotent. seqCap is the worst-case sequence length.
static void reserveSharedArena(int64_t seqCap) {
    if (!kPrewarmSharedArena || kPrewarmArenaPercent <= 0) {
        return;
    }
    const int64_t capTokens = std::max<int64_t>(1, seqCap);

    // Bytes of key/value cache I/O. Keys (and key scale/bias) store sequence on the last axis; values
    // store it on the penultimate axis. Output shape inference often leaves otherwise-static head dims
    // symbolic, so use the positionally aligned input shape as their fallback authority.
    auto cacheIoBytes = [&](const std::vector<const char*>& names,
                            const std::vector<std::vector<int64_t>>& dimsList,
                            const std::vector<ONNXTensorElementDataType>& typeList,
                            const std::vector<std::vector<int64_t>>& fallbackDims,
                            int64_t batchForDynamic) -> int64_t {
        int64_t total = 0;
        for (size_t t = 0; t < names.size() && t < dimsList.size() && t < typeList.size(); ++t) {
            const char* name = names[t];
            const bool isKey = name != nullptr && std::strstr(name, "key_") != nullptr;
            const bool isValue = name != nullptr && std::strstr(name, "value_") != nullptr;
            const std::vector<int64_t>& dims = dimsList[t];
            if ((!isKey && !isValue) || dims.size() < 5) {
                continue;
            }
            const int seqAxis = static_cast<int>(dims.size()) - (isValue ? 2 : 1);
            const size_t fallbackIndex = num_keys_values > 0
                ? t % static_cast<size_t>(num_keys_values)
                : t;
            const std::vector<int64_t>* fallback = fallbackIndex < fallbackDims.size()
                ? &fallbackDims[fallbackIndex]
                : nullptr;
            int64_t elems = 1;
            for (int a = 0; a < static_cast<int>(dims.size()); ++a) {
                int64_t d = dims[a];
                if (a == seqAxis) {
                    d = capTokens;
                } else if (a == 0) {
                    d = (d <= 0) ? batchForDynamic : d;
                } else if (d <= 0 && fallback != nullptr &&
                           static_cast<size_t>(a) < fallback->size() && (*fallback)[a] > 0) {
                    d = (*fallback)[a];
                } else if (d <= 0) {
                    d = 1;
                }
                elems *= std::max<int64_t>(1, d);
            }
            total += elems * static_cast<int64_t>(tensorElementSize(typeList[t]));
        }
        return total;
    };

    auto reserveEnv = [&](ModelRuntime& anchor, bool arenaRegistered, bool includePersistentState,
                          int64_t& reservedBytes, const char* label) {
        if (!arenaRegistered || anchor.api == nullptr || anchor.env == nullptr ||
            anchor.session == nullptr || anchor.memoryInfo == nullptr) {
            return;
        }
        int64_t peakBytes = 0;
        for (const ModelRuntime& runtime : gModelRuntimes) {
            if (runtime.session == nullptr || runtime.env != anchor.env) {
                continue;
            }
            const int64_t work = cacheIoBytes(
                    runtime.inputNames, runtime.inputDims, runtime.inputTypes,
                    runtime.inputDims, 1) +
                    cacheIoBytes(runtime.outputNames, runtime.outputDims, runtime.outputTypes,
                         runtime.inputDims, 1);
            peakBytes = std::max(peakBytes, work);
        }
        if (includePersistentState) {
            peakBytes += cacheIoBytes(anchor.inputNames, anchor.inputDims, anchor.inputTypes,
                                      anchor.inputDims, 1);
        }
        const int64_t target = peakBytes * kPrewarmArenaPercent / 100;
        const int64_t delta = target - reservedBytes;
        if (delta <= 0) {
            return;
        }
        OrtAllocator* arena = nullptr;
        if (!logOrtStatus(anchor.api,
                          anchor.api->CreateAllocator(anchor.session, anchor.memoryInfo, &arena),
                          "CreateAllocator(arena prewarm)") || arena == nullptr) {
            return;
        }
        void* block = nullptr;
        if (logOrtStatus(anchor.api,
                         anchor.api->AllocatorAlloc(arena, static_cast<size_t>(delta), &block),
                         "AllocatorAlloc(arena prewarm)") && block != nullptr) {
            logOrtStatus(anchor.api, anchor.api->AllocatorFree(arena, block),
                         "AllocatorFree(arena prewarm)");
            reservedBytes = target;
              LOGI("%s arena pre-warmed: +%lld MB (total %lld MB; seq<=%lld %d%%)",
                 label, static_cast<long long>(delta >> 20),
                 static_cast<long long>(target >> 20), static_cast<long long>(capTokens),
                  kPrewarmArenaPercent);
        }
        anchor.api->ReleaseAllocator(arena);
    };

    ModelRuntime& prefill = getModel(LLM_TextPrefillGreedy);
    reserveEnv(prefill, gSharedEnvArenaRegistered, false,
               g_reserved_prefill_arena_bytes, "Prefill");
    if (mMain.env == prefill.env) {
        reserveEnv(mMain, gSharedEnvArenaRegistered, true,
                   g_reserved_prefill_arena_bytes, "Shared");
    } else {
        reserveEnv(mMain, gDecodeEnvArenaRegistered, true,
                   g_reserved_decode_arena_bytes, "Decode");
    }
}

// JNI: exported vision input geometry, so the camera pipeline (a separate CameraService/glRender
// translation unit) can auto-fit the captured frame to the model. Returns {width, height}; both
// fall back to DEFAULT_INPUT_IMAGE_SIZE until a vision model's metadata has been applied.
extern "C" JNIEXPORT jintArray JNICALL
Java_com_example_myapplication_CameraService_nativeGetInputImageSize(JNIEnv* env, jclass /*clazz*/) {
    const jint dims[2] = {input_image_width, input_image_height};
    jintArray out = env->NewIntArray(2);
    if (out != nullptr) {
        env->SetIntArrayRegion(out, 0, 2, dims);
    }
    return out;
}

// JNI: exported VIDEO input geometry {width, height}. Falls back to the image size when the export
// carries no separate input_video_size (or before any vision model is loaded).
extern "C" JNIEXPORT jintArray JNICALL
Java_com_example_myapplication_CameraService_nativeGetInputVideoSize(JNIEnv* env, jclass /*clazz*/) {
    const jint dims[2] = {input_video_width, input_video_height};
    jintArray out = env->NewIntArray(2);
    if (out != nullptr) {
        env->SetIntArrayRegion(out, 0, 2, dims);
    }
    return out;
}

// JNI: vision capability bitmask (bit0 = image, bit1 = video). 0 for a text-only model, which the UI
// uses to grey out the camera button (and hide the video toggle when image-only).
extern "C" JNIEXPORT jint JNICALL
Java_com_example_myapplication_CameraService_nativeGetVisionCaps(JNIEnv* env, jclass /*clazz*/) {
    (void)env;
    return static_cast<jint>(metadataVisionCaps());
}

// JNI: discard any pending vision input. Called when a capture is accepted natively (pixel_values pushed)
// but the UI then declines to dispatch a turn (e.g. a generation is already running), so the next text
// turn is not mistaken for a stale vision turn.
extern "C" JNIEXPORT jlong JNICALL
Java_com_example_myapplication_CameraService_nativeRequestCancelPendingVision(
        JNIEnv* env, jclass /*clazz*/) {
    (void)env;
    const uint64_t generation =
            g_vision_cancel_generation.fetch_add(1, std::memory_order_acq_rel) + 1;
    g_vision_cancel_requested.store(true, std::memory_order_release);
    g_pending_vision.store(VISION_NONE, std::memory_order_release);
    setVisionEncodeTerminate(true);
    return static_cast<jlong>(generation);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_CameraService_nativeFinishCancelPendingVision(
        JNIEnv* env, jclass /*clazz*/, jlong requestedGeneration) {
    (void)env;
    releasePendingVisionHidden();
    {
        std::lock_guard<std::mutex> lock(g_vision_pixels_mutex);
        releaseDynamicVideoInputLocked();
    }
    if (g_vision_cancel_generation.load(std::memory_order_acquire) ==
            static_cast<uint64_t>(requestedGeneration)) {
        g_vision_cancel_requested.store(false, std::memory_order_release);
        if (!g_cancel.load(std::memory_order_acquire) &&
            !g_clean_commit_cancel.load(std::memory_order_acquire)) {
            setVisionEncodeTerminate(false);
        }
    }
}

// JNI: number of ring frames the video buffer holds (VIDEO_NUM_FRAMES) so the camera can size its
// sampling ring; 0 when video is unavailable.
extern "C" JNIEXPORT jint JNICALL
Java_com_example_myapplication_CameraService_nativeGetVideoNumFrames(JNIEnv* env, jclass /*clazz*/) {
    (void)env;
    return g_vision_video_ready ? g_video_num_frames : 0;
}

// JNI: whether video preprocess uses a fixed frame axis. Fixed => ring + black padding; dynamic => append.
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_CameraService_nativeIsVideoFixedLength(JNIEnv* env, jclass /*clazz*/) {
    (void)env;
    return g_video_fixed_length_input ? JNI_TRUE : JNI_FALSE;
}

// JNI: target video sampling FPS (metadata video_fps), decoupled from the 30fps preview.
extern "C" JNIEXPORT jfloat JNICALL
Java_com_example_myapplication_CameraService_nativeGetVideoFps(JNIEnv* env, jclass /*clazz*/) {
    (void)env;
    return g_video_fps;
}

// Copy mapped GPU CHW once into owned pixel storage, then publish the pending image atomically.
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_CameraService_nativePushImageCHW(
        JNIEnv* env, jclass /*clazz*/, jobject chwBuf, jint width, jint height) {
    if (!g_vision_image_ready || chwBuf == nullptr) {
        return JNI_FALSE;
    }
    if (width != input_image_width || height != input_image_height) {
        LOGE("nativePushImageCHW: geometry mismatch (got=%dx%d expected=%dx%d)",
             width, height, input_image_width, input_image_height);
        return JNI_FALSE;
    }
    void* addr = env->GetDirectBufferAddress(chwBuf);
    const jlong cap = env->GetDirectBufferCapacity(chwBuf);
    if (addr == nullptr || cap <= 0) {
        return JNI_FALSE;
    }
    std::lock_guard<std::mutex> lock(g_vision_pixels_mutex);
    if (g_image_slot_bytes == 0 || g_image_slot_storage_bytes == 0 ||
        g_image_pixel_backing.size() < g_image_slot_storage_bytes ||
        static_cast<size_t>(cap) < g_image_slot_bytes) {
        LOGE("nativePushImageCHW: size mismatch (cap=%lld raw=%zu storage=%zu)",
             static_cast<long long>(cap), g_image_slot_bytes, g_image_slot_storage_bytes);
        return JNI_FALSE;
    }
    if (!writePixelValues(g_image_pixel_backing.data(), g_image_slot_storage_bytes,
                          mImagePreprocess.inputTypes[0], static_cast<const uint8_t*>(addr),
                          g_image_slot_bytes)) {
        return JNI_FALSE;
    }
    // New pixels: bump the capture seq so any earlier speculative pre-encode is treated as stale, then
    // mark an image inference pending. Prewarm_Vision (kicked off by the UI) encodes these fresh pixels.
    g_vision_capture_seq.fetch_add(1, std::memory_order_acq_rel);
    g_pending_vision.store(VISION_IMAGE, std::memory_order_release);
    return JNI_TRUE;
}

// JNI: start/stop continuous video sampling. Turning ON resets the ring; the desired flag survives a
// processing pause so Run_LLM can auto-resume sampling after it consumes a window.
extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_CameraService_nativeSetVideoStream(JNIEnv* env, jclass /*clazz*/, jboolean on) {
    (void)env;
    if (on == JNI_TRUE) {
        std::lock_guard<std::mutex> lock(g_vision_pixels_mutex);
        g_video_ring_count.store(0, std::memory_order_relaxed);
        g_video_ring_write.store(0, std::memory_order_relaxed);
        g_video_recorded_frames = 0;
        if (!g_video_fixed_length_input) {
            if (g_video_pixel_values != nullptr && mVideoPreprocess.api != nullptr) {
                mVideoPreprocess.api->ReleaseValue(g_video_pixel_values);
                g_video_pixel_values = nullptr;
            }
        }
        g_video_stream_desired.store(true, std::memory_order_release);
        g_video_stream_active.store(g_vision_video_ready, std::memory_order_release);
    } else {
        g_video_stream_desired.store(false, std::memory_order_release);
        g_video_stream_active.store(false, std::memory_order_release);
    }
}

// JNI: push one sampled video frame's CHW planar uint8 ([3,H,W]) directly into the next ring slot of
// the static video pixel_values buffer (zero extra alloc). Ignored unless sampling is active.
extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_CameraService_nativePushVideoFrameCHW(
        JNIEnv* env, jclass /*clazz*/, jobject chwBuf, jint width, jint height) {
    if (!g_vision_video_ready || !g_video_stream_active.load(std::memory_order_acquire) || chwBuf == nullptr) {
        return;
    }
    if (width != input_video_width || height != input_video_height) {
        LOGE("nativePushVideoFrameCHW: geometry mismatch (got=%dx%d expected=%dx%d)",
             width, height, input_video_width, input_video_height);
        return;
    }
    void* addr = env->GetDirectBufferAddress(chwBuf);
    const jlong cap = env->GetDirectBufferCapacity(chwBuf);
    const int F = g_video_num_frames;
    if (addr == nullptr || cap <= 0 || F <= 0 || g_video_frame_bytes == 0 ||
        static_cast<size_t>(cap) < g_video_frame_bytes) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_vision_pixels_mutex);
    if (g_video_fixed_length_input) {
        if (g_video_pixel_backing.size() < static_cast<size_t>(F) * g_video_frame_storage_bytes) {
            return;
        }
        const int slot = g_video_ring_write.load(std::memory_order_relaxed) % F;
        if (!writePixelValues(
                g_video_pixel_backing.data() + static_cast<size_t>(slot) * g_video_frame_storage_bytes,
                g_video_frame_storage_bytes, mVideoPreprocess.inputTypes[0],
                static_cast<const uint8_t*>(addr), g_video_frame_bytes)) {
            return;
        }
        g_video_ring_write.store((slot + 1) % F, std::memory_order_relaxed);
        const int c = g_video_ring_count.load(std::memory_order_relaxed);
        if (c < F) {
            g_video_ring_count.store(c + 1, std::memory_order_relaxed);
        }
        g_video_recorded_frames = std::min(F, c + 1);
    } else if (g_video_recorded_frames < F && ensureDynamicVideoCapacity(g_video_recorded_frames + 1)) {
        const size_t offset = static_cast<size_t>(g_video_recorded_frames) * g_video_frame_storage_bytes;
        if (!writePixelValues(g_video_pixel_backing.data() + offset, g_video_frame_storage_bytes,
                              mVideoPreprocess.inputTypes[0], static_cast<const uint8_t*>(addr),
                              g_video_frame_bytes)) {
            return;
        }
        g_video_recorded_frames += 1;
        g_video_ring_count.store(g_video_recorded_frames, std::memory_order_relaxed);
    }
}

// Reorder the ring into oldest->newest temporal order on submit only. Partial fixed-F clips repeat
// their last sampled frame, matching the exporter/reference sampler; full unwrapped clips need no copy.
static void reorderVideoRingLocked() {
    const int F = g_video_num_frames;
    const int count = g_video_ring_count.load(std::memory_order_relaxed);
    if (F <= 0 || count <= 0 || g_video_frame_storage_bytes == 0 ||
        g_video_pixel_backing.size() < static_cast<size_t>(F) * g_video_frame_storage_bytes) {
        return;
    }
    const int have   = std::min(count, F);
    if (have < F) {
        const uint8_t* lastFrame = g_video_pixel_backing.data() +
                static_cast<size_t>(have - 1) * g_video_frame_storage_bytes;
        for (int frame = have; frame < F; ++frame) {
            std::memcpy(g_video_pixel_backing.data() +
                            static_cast<size_t>(frame) * g_video_frame_storage_bytes,
                        lastFrame, g_video_frame_storage_bytes);
        }
        g_video_recorded_frames = have;
        return;
    }
    const int write  = g_video_ring_write.load(std::memory_order_relaxed);
    if (write == 0) {
        g_video_recorded_frames = have;
        return;
    }
    const int rotateFrames = write % F;   // wrapped -> oldest is the write cursor; rotate it to slot 0
    auto gcdInt = [](int left, int right) {
        while (right != 0) {
            const int remainder = left % right;
            left = right;
            right = remainder;
        }
        return left;
    };
    std::vector<uint8_t> scratchFrame(g_video_frame_storage_bytes);
    uint8_t* base = g_video_pixel_backing.data();
    const int cycles = gcdInt(F, rotateFrames);
    for (int cycleStart = 0; cycleStart < cycles; ++cycleStart) {
        int currentFrame = cycleStart;
        std::memcpy(scratchFrame.data(),
                    base + static_cast<size_t>(currentFrame) * g_video_frame_storage_bytes,
                    g_video_frame_storage_bytes);
        while (true) {
            int nextFrame = currentFrame + rotateFrames;
            if (nextFrame >= F) {
                nextFrame -= F;
            }
            if (nextFrame == cycleStart) {
                break;
            }
            std::memcpy(base + static_cast<size_t>(currentFrame) * g_video_frame_storage_bytes,
                        base + static_cast<size_t>(nextFrame) * g_video_frame_storage_bytes,
                        g_video_frame_storage_bytes);
            currentFrame = nextFrame;
        }
        std::memcpy(base + static_cast<size_t>(currentFrame) * g_video_frame_storage_bytes,
                    scratchFrame.data(), g_video_frame_storage_bytes);
    }
    g_video_recorded_frames = have;
}

// JNI: freeze the current video window (pause sampling immediately), reorder it oldest->newest, and
// mark a video inference pending for the next Run_LLM.
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_CameraService_nativeSubmitVideoQuery(JNIEnv* env, jclass /*clazz*/) {
    (void)env;
    if (!g_vision_video_ready) {
        return JNI_FALSE;
    }
    g_video_stream_active.store(false, std::memory_order_release);   // pause sampling immediately
    std::lock_guard<std::mutex> lock(g_vision_pixels_mutex);
    int captured = g_video_fixed_length_input
            ? g_video_ring_count.load(std::memory_order_relaxed)
            : g_video_recorded_frames;
    if (captured <= 0) {
        return JNI_FALSE;   // nothing captured yet
    }
    if (g_video_fixed_length_input) {
        reorderVideoRingLocked();
        updateVideoTokenGeometryForFrames(g_video_num_frames);
    } else {
        const int temporalPatch = std::max(1, g_temporal_patch_size);
        int normalizedFrames = std::max(captured, std::max(1, g_video_min_frames));
        normalizedFrames = ((normalizedFrames + temporalPatch - 1) / temporalPatch) * temporalPatch;
        normalizedFrames = std::min(normalizedFrames, g_video_num_frames);
        if (normalizedFrames < captured || !ensureDynamicVideoCapacity(normalizedFrames)) {
            return JNI_FALSE;
        }
        const uint8_t* lastFrame = g_video_pixel_backing.data() +
                static_cast<size_t>(captured - 1) * g_video_frame_storage_bytes;
        for (int frame = captured; frame < normalizedFrames; ++frame) {
            std::memcpy(g_video_pixel_backing.data() +
                            static_cast<size_t>(frame) * g_video_frame_storage_bytes,
                        lastFrame, g_video_frame_storage_bytes);
        }
        captured = normalizedFrames;
        g_video_recorded_frames = captured;
        if (g_video_pixel_values != nullptr && mVideoPreprocess.api != nullptr) {
            mVideoPreprocess.api->ReleaseValue(g_video_pixel_values);
            g_video_pixel_values = nullptr;
        }
        const int64_t H = staticDimOr(mVideoPreprocess, true, 0, 2, input_video_height);
        const int64_t W = staticDimOr(mVideoPreprocess, true, 0, 3, input_video_width);
        const std::vector<int64_t> dims = {captured, 3, H, W};
        const size_t bytes = static_cast<size_t>(captured) * g_video_frame_storage_bytes;
        if (!createTensorWithData(mVideoPreprocess.api, mVideoPreprocess.memoryInfo,
                      g_video_pixel_backing.data(), bytes, dims,
                      mVideoPreprocess.inputTypes[0], &g_video_pixel_values,
                                  "CreateTensor(dynamic video pixel_values)")) {
            return JNI_FALSE;
        }
        updateVideoTokenGeometryForFrames(captured);
    }
    // Frozen window is a fresh capture: bump the seq (invalidate any stale pre-encode) then mark pending.
    g_vision_capture_seq.fetch_add(1, std::memory_order_acq_rel);
    g_pending_vision.store(VISION_VIDEO, std::memory_order_release);
    return JNI_TRUE;
}

// JNI: model load.
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1A(JNIEnv* env, jclass clazz,
                                                            jobject asset_manager,
                                                            jstring asset_directory,
                                                            jstring cache_directory,
                                                            jstring storage_directory,
                                                            jint ep_type,
                                                            jboolean low_memory_mode) {
    // ep_type: 0 = CPU (default), 1 = XNNPACK, 2 = QNN
    if (g_main_cls == nullptr) {
        if (clazz == nullptr) {
            LOGE("Load_Models_A: could not resolve MainActivity class");
            return JNI_FALSE;
        }
        JavaVM* vm = nullptr;
        if (env->GetJavaVM(&vm) != JNI_OK || vm == nullptr) {
            LOGE("Load_Models_A: could not resolve JavaVM");
            return JNI_FALSE;
        }
        jclass global_cls = static_cast<jclass>(env->NewGlobalRef(clazz));
        if (global_cls == nullptr || env->ExceptionCheck()) {
            if (env->ExceptionCheck()) env->ExceptionClear();
            LOGE("Load_Models_A: could not retain MainActivity class");
            return JNI_FALSE;
        }
        auto requireStaticMethod = [&](const char* name, const char* signature) {
            jmethodID method = env->GetStaticMethodID(global_cls, name, signature);
            if (method == nullptr || env->ExceptionCheck()) {
                if (env->ExceptionCheck()) env->ExceptionClear();
                LOGE("Load_Models_A: required callback missing: %s %s", name, signature);
                return static_cast<jmethodID>(nullptr);
            }
            return method;
        };
        g_on_token = requireStaticMethod("onTokenStream", "(Ljava/lang/String;)V");
        g_on_perf = requireStaticMethod("onPerfStats", "(FFIIIIZ)V");
        g_on_post_processing = requireStaticMethod("onPostProcessingState", "(Z)V");
        if (g_on_token == nullptr || g_on_perf == nullptr || g_on_post_processing == nullptr) {
            env->DeleteGlobalRef(global_cls);
            g_on_token = nullptr;
            g_on_perf = nullptr;
            g_on_post_processing = nullptr;
            return JNI_FALSE;
        }
        g_main_cls = global_cls;
        g_jvm = vm;
    }
    if (g_jvm == nullptr || g_on_token == nullptr || g_on_perf == nullptr ||
        g_on_post_processing == nullptr) {
        LOGE("Load_Models_A: JNI callbacks are not initialized");
        return JNI_FALSE;
    }
    stream_buf.reserve(256);

    const bool lowMem = (low_memory_mode == JNI_TRUE);
    std::string requestedAssetDirectory;
    if (asset_directory != nullptr &&
        !jStringToUtf8(env, asset_directory, requestedAssetDirectory)) {
        return JNI_FALSE;
    }
    std::string requestedCacheDirectory;
    std::string requestedStorageDirectory;
    if (cache_directory == nullptr || storage_directory == nullptr ||
        !jStringToUtf8(env, cache_directory, requestedCacheDirectory) ||
        !jStringToUtf8(env, storage_directory, requestedStorageDirectory) ||
        requestedCacheDirectory.empty() || requestedStorageDirectory.empty()) {
        LOGE("Load_Models_A: app cache/storage directories are required");
        return JNI_FALSE;
    }
    auto normalizeDirectory = [](std::string directory) {
        if (directory.back() != '/') {
            directory.push_back('/');
        }
        return directory;
    };
    cache_path = normalizeDirectory(std::move(requestedCacheDirectory));
    storage_path = normalizeDirectory(std::move(requestedStorageDirectory));
    vocab_path = cache_path + "vocab_Qwen3.5-0.8B.txt";
    std::lock_guard<std::mutex> modelLock(g_model_session_mutex);
    if (g_model_sessions_initialized) {
        LOGI("Load_Models_A: model sessions are already initialized");
        return JNI_TRUE;
    }

    if (!preloadRuntimeMetadata(env, asset_manager, lowMem, requestedAssetDirectory)) {
        LOGE("Load_Models_A: could not read model metadata; aborting load");
        return JNI_FALSE;
    }

    // Map the required shared merged-weight blob ONCE before loading any merged graph, so each merged
    // session attaches the single mapped copy (AddInitializer) instead of deserializing it per session.
    const OrtApi* sharedApi = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!loadSharedInitializers(sharedApi, env, asset_manager, lowMem,
                                gSharedInitializersFileName.c_str(),
                                gSharedInitializersDataFileName.c_str(),
                                requestedAssetDirectory.c_str())) {
        LOGE("Load_Models_A: shared merged initializers are required but could not be loaded");
        return JNI_FALSE;
    }

    if (g_model_asset_manager != nullptr) {
        env->DeleteGlobalRef(g_model_asset_manager);
        g_model_asset_manager = nullptr;
    }
    if (asset_manager != nullptr) {
        g_model_asset_manager = env->NewGlobalRef(asset_manager);
        if (g_model_asset_manager == nullptr) {
            LOGE("Load_Models_A: could not retain the asset manager for lazy strategy loading");
            return JNI_FALSE;
        }
    }
    g_model_ep_type = static_cast<int>(ep_type);
    g_model_low_memory = lowMem;
    g_model_asset_directory = std::move(requestedAssetDirectory);

    const int metadataCaps = metadataVisionCaps();
    const std::array<MergedStrategy, 3> startupStrategies = {
        MSTRAT_GREEDY, MSTRAT_GREEDY, MSTRAT_GREEDY
    };
    if (!ensureMergedStrategySessions(env, startupStrategies, 0)) {
        LOGE("Load_Models_A: required text greedy sessions failed to load");
        return JNI_FALSE;
    }

    constexpr int firstStandaloneModel = LLM_KV_Slice;
    for (int id = firstStandaloneModel; id < kModelCount; ++id) {
        if (modelIsBundleDescriptor(static_cast<ModelId>(id))) {
            continue;
        }
        const bool transformerOnlyHelper = id == LLM_KV_Slice || id == LLM_KV_Split2 ||
                id == LLM_KV_Concat || id == LLM_RopeShift;
        if (num_linear_attention_layers > 0 && transformerOnlyHelper) {
            LOGI("Load_Models_A: deferring unused helper %d (%s)",
                 id, gModelFileNames[id].c_str());
            continue;
        }
        if (!loadModel(static_cast<ModelId>(id), env, asset_manager, ep_type, lowMem)) {
            const bool requiredPrefillHelper = num_linear_attention_layers == 0 &&
                    (id == LLM_KV_Split2 || id == LLM_KV_Concat);
                if (requiredPrefillHelper) {
                LOGE("Load_Models_A: required graph %d (%s) failed to load",
                     id, gModelFileNames[id].c_str());
                return JNI_FALSE;
            }
            LOGI("Load_Models_A: optional model %d (%s) unavailable; that strategy/feature is disabled",
                 id, gModelFileNames[id].c_str());
            continue;
        }
    }

    ModelRuntime& main = mMain;   // merged primary (text greedy decode): canonical KV/linear state layout
    if (!configureKVLayout()) {
        return JNI_FALSE;
    }
    if (!validateMainIO(main)) {
        return JNI_FALSE;
    }

    if (!initKVCache()) {
        return JNI_FALSE;
    }

    reserveSharedArena(g_memory_tokens);
    mallopt(M_PURGE, 0);

    g_model_sessions_initialized = true;
    int loadedSessions = 0;
    for (const ModelRuntime& runtime : gModelRuntimes) {
        loadedSessions += runtime.session != nullptr ? 1 : 0;
    }

          LOGI("Load_Models_A: %d/%d sessions resident (EP=%d, lowMem=%d, strategy=%d, "
            "vision_caps=%d vision_ready=%d)",
            loadedSessions, kSessionModelCount, ep_type, lowMem ? 1 : 0,
         static_cast<int>(g_loaded_merged_strategies[0]),
            metadataCaps, g_vision_sessions_initialized ? 1 : 0);
    return JNI_TRUE;
}

// Multi-turn KV helpers.
static inline void release_saved_kv() {
    if (mMain.api == nullptr) {
        return;
    }
    for (OrtValue*& kv : saved_kv) {
        if (kv != nullptr) {
            mMain.api->ReleaseValue(kv);
            kv = nullptr;
        }
    }
}

// Hybrid linear-attention passthrough state helpers (no-ops for pure-transformer models).
static inline void release_saved_linear() {
    if (mMain.api == nullptr) {
        return;
    }
    for (OrtValue*& s : saved_linear_states) {
        if (s != nullptr) {
            mMain.api->ReleaseValue(s);
            s = nullptr;
        }
    }
    saved_linear_valid = false;
}

// Take ownership of a freshly-produced linear-state vector (from prefill/append) as the persisted state.
static inline void store_saved_linear(std::vector<OrtValue*>& produced) {
    release_saved_linear();
    for (int i = 0; i < num_linear_states && i < static_cast<int>(produced.size()); ++i) {
        saved_linear_states[i] = produced[i];
        produced[i] = nullptr;
    }
    produced.clear();
    saved_linear_valid = (num_linear_states > 0);
}

static inline void publishMemoryUsage() {
    g_memory_used_tokens.store(std::max<int64_t>(0, saved_kv_len), std::memory_order_release);
}

static inline void clearModelBindingRefs(ModelRuntime& m) {
    clearRuntimeBindings(m);
}

static inline void clearAllModelBindingRefs() {
    for (ModelRuntime& m : gModelRuntimes) {
        clearModelBindingRefs(m);
    }
}

static inline void shrinkAllocatorIfSupported(const OrtApi* api, OrtAllocator* allocator, const char* op) {
    if (api == nullptr || allocator == nullptr || allocator->Shrink == nullptr) {
        return;
    }
    logOrtStatus(api, allocator->Shrink(allocator), op);
}

static void shrinkModelAllocators(ModelRuntime& m) {
    if (m.api == nullptr || m.session == nullptr) {
        return;
    }
    OrtAllocator* sessionAllocator = nullptr;
    if (m.memoryInfo != nullptr &&
        logOrtStatus(m.api, m.api->CreateAllocator(m.session, m.memoryInfo, &sessionAllocator),
                     "CreateAllocator(runtime trim)")) {
        shrinkAllocatorIfSupported(m.api, sessionAllocator, "AllocatorShrink(runtime trim)");
        m.api->ReleaseAllocator(sessionAllocator);
    }
    if (m.ioAllocator != nullptr && m.ioAllocator != m.allocator) {
        shrinkAllocatorIfSupported(m.api, m.ioAllocator, "AllocatorShrink(io trim)");
    }
    if (m.env == gSharedEnv) {
        g_reserved_prefill_arena_bytes = 0;
    }
    if (m.env == gDecodeEnv) {
        g_reserved_decode_arena_bytes = 0;
    }
}

static void shrinkPrimaryArenas() {
    ModelRuntime& prefill = getModel(LLM_TextPrefillGreedy);
    shrinkModelAllocators(prefill);
    if (mMain.env != prefill.env) {
        shrinkModelAllocators(mMain);
    }
}

static void shrinkAllModelAllocators() {
    std::vector<OrtEnv*> shrunkEnvs;
    std::vector<OrtAllocator*> shrunkIoAllocators;
    for (ModelRuntime& m : gModelRuntimes) {
        if (m.api == nullptr || m.session == nullptr) {
            continue;
        }
        if (std::find(shrunkEnvs.begin(), shrunkEnvs.end(), m.env) == shrunkEnvs.end()) {
            OrtAllocator* sessionAllocator = nullptr;
            if (m.memoryInfo != nullptr &&
                logOrtStatus(m.api, m.api->CreateAllocator(m.session, m.memoryInfo, &sessionAllocator),
                             "CreateAllocator(runtime trim)")) {
                shrinkAllocatorIfSupported(m.api, sessionAllocator, "AllocatorShrink(runtime trim)");
                m.api->ReleaseAllocator(sessionAllocator);
            }
            shrunkEnvs.push_back(m.env);
            if (m.env == gSharedEnv) {
                g_reserved_prefill_arena_bytes = 0;
            }
            if (m.env == gDecodeEnv) {
                g_reserved_decode_arena_bytes = 0;
            }
        }
        if (m.ioAllocator != nullptr && m.ioAllocator != m.allocator &&
            std::find(shrunkIoAllocators.begin(), shrunkIoAllocators.end(), m.ioAllocator) ==
                    shrunkIoAllocators.end()) {
            shrinkAllocatorIfSupported(m.api, m.ioAllocator, "AllocatorShrink(io trim)");
            shrunkIoAllocators.push_back(m.ioAllocator);
        }
    }
}

static bool ensureMergedStrategySessions(
    JNIEnv* env, const std::array<MergedStrategy, 3>& strategies, int visionCaps) {
    constexpr int mergedEnd =
            LLM_FIRST_MERGED_MODEL + 3 * LLM_MERGED_MODELS_PER_MODALITY;
    auto loadOne = [&](ModelId id) {
        if (mergedModelReady(id)) {
            return true;
        }
        return loadModel(id, env, g_model_asset_manager, g_model_ep_type, g_model_low_memory);
    };

    // These two graphs define the canonical state layout and are also used for clean-cache rebuilds.
    if (!loadOne(LLM_TextPrefillGreedy) || !loadOne(LLM_TextDecodeGreedy)) {
        return false;
    }

    for (int modality = 0; modality < 3; ++modality) {
        if (modality > 0 && (visionCaps & (1 << (modality - 1))) == 0) {
            continue;
        }
        const ModelId prefill = mergedPrefillModel(modality, strategies[modality]);
        const ModelId decode = mergedDecodeModel(modality, strategies[modality]);
        if (loadOne(prefill) && loadOne(decode)) {
            continue;
        }
        LOGE("merged strategy load failed: modality=%d strategy=%d", modality,
             static_cast<int>(strategies[modality]));
        return false;
    }
    g_loaded_merged_strategies = strategies;

    int mergedResident = 0;
    for (int id = LLM_FIRST_MERGED_MODEL; id < mergedEnd; ++id) {
        mergedResident += getModel(static_cast<ModelId>(id)).session != nullptr ? 1 : 0;
    }
        LOGI("merged strategy sessions ready: resident=%d/18 strategy=%d/%d/%d",
         mergedResident,
         static_cast<int>(strategies[0]), static_cast<int>(strategies[1]),
            static_cast<int>(strategies[2]));
    return true;
}

static void releaseVisionRuntimeSessions() {
    releaseVisionBuffers();
    if (!saved_kv_has_vision) {
        constexpr int imageVideoMergedBegin = LLM_ImagePrefillGreedy;
        constexpr int imageVideoMergedEnd =
                LLM_FIRST_MERGED_MODEL + 3 * LLM_MERGED_MODELS_PER_MODALITY;
        for (int id = imageVideoMergedBegin; id < imageVideoMergedEnd; ++id) {
            releaseModelRuntime(getModel(static_cast<ModelId>(id)));
        }
    } else {
        LOGI("Vision trim: retaining merged producer sessions for live visual KV");
    }
    releaseModelRuntime(mVision);
    releaseModelRuntime(mImagePreprocess);
    releaseModelRuntime(mVideoPreprocess);
    g_vision_image_ready = false;
    g_vision_video_ready = false;
    g_vision_sessions_initialized = false;
}

// Called with g_model_session_mutex held. Vision/preprocess stay resident until an explicit pressure trim;
// strategy changes incrementally add missing merged graphs and retain every graph already loaded.
static bool initializeVisionSessionsLocked(
        JNIEnv* env, const std::array<MergedStrategy, 3>& strategies, int visionCaps) {
    const std::array<MergedStrategy, 3>& safeStrategies = saved_kv_has_vision
            ? g_loaded_merged_strategies : strategies;
    if (!ensureMergedStrategySessions(env, safeStrategies, visionCaps)) {
        return false;
    }
    if (visionCaps == 0) {
        return true;
    }

    auto loadVisionModel = [&](ModelId id) {
        return getModel(id).session != nullptr ||
               loadModel(id, env, g_model_asset_manager, g_model_ep_type, g_model_low_memory);
    };
    if (g_vision_sessions_initialized) {
        return true;
    }

    bool loaded = loadVisionModel(LLM_Vision);
    if ((visionCaps & 0x1) != 0) {
        loaded = loadVisionModel(LLM_Image_Preprocess) && loaded;
    }
    if ((visionCaps & 0x2) != 0) {
        loaded = loadVisionModel(LLM_Video_Preprocess) && loaded;
    }

    loaded = loaded && configureVisionLayout();
    int readyCaps = 0;
    if (g_vision_image_ready) readyCaps |= 0x1;
    if (g_vision_video_ready) readyCaps |= 0x2;
    loaded = loaded && readyCaps == visionCaps;
    if (!loaded) {
        releaseVisionRuntimeSessions();
        shrinkPrimaryArenas();
        LOGE("Vision initialization failed (requested_caps=%d ready_caps=%d)", visionCaps, readyCaps);
        return false;
    }

    g_vision_sessions_initialized = true;
    LOGI("Vision sessions ready at startup (caps=%d strategy=%d/%d)", visionCaps,
         static_cast<int>(strategies[1]), static_cast<int>(strategies[2]));
    return true;
}

// Bounded logical history. The vector keeps a lazy dead prefix so normal pruning is O(1); it compacts
// only after that prefix dominates storage. Logical offsets let rollback checkpoints survive compaction.
static size_t retainedHistorySize() {
    return g_history_begin <= g_history_ids.size()
            ? g_history_ids.size() - g_history_begin : 0;
}

static int64_t logicalHistoryEnd() {
    return g_history_base + static_cast<int64_t>(retainedHistorySize());
}

static const int* retainedHistoryData() {
    return retainedHistorySize() > 0 ? g_history_ids.data() + g_history_begin : nullptr;
}

static void pruneHistoryStorage() {
    const size_t capacity = static_cast<size_t>(std::max(1, max_seq_len));
    const size_t retained = retainedHistorySize();
    if (retained > capacity) {
        const size_t drop = retained - capacity;
        g_history_begin += drop;
        g_history_base += static_cast<int64_t>(drop);
    }
    g_turn_checkpoints.erase(
            std::remove_if(g_turn_checkpoints.begin(), g_turn_checkpoints.end(),
                           [](const TurnCheckpoint& checkpoint) {
                               return checkpoint.historyLen <= g_history_base;
                           }),
            g_turn_checkpoints.end());
    if (g_history_begin > 0 &&
        (g_history_begin >= g_history_ids.size() / 2 || g_history_begin > capacity)) {
        g_history_ids.erase(g_history_ids.begin(),
                            g_history_ids.begin() + static_cast<std::ptrdiff_t>(g_history_begin));
        g_history_begin = 0;
    }
}

static void appendHistoryIds(const std::vector<int>& ids) {
    g_history_ids.insert(g_history_ids.end(), ids.begin(), ids.end());
    pruneHistoryStorage();
}

// Per-turn rollback checkpoints.
static inline void record_turn_checkpoint(int32_t turnId, int64_t historyLen, int64_t activeLen,
                                          uint8_t modality, int64_t mropePos, bool hadVision) {
    for (TurnCheckpoint& cp : g_turn_checkpoints) {
        if (cp.turnId == turnId) {
            cp.historyLen = historyLen;
            cp.activeLen  = activeLen;
            cp.modality   = modality;
            cp.mropePos   = mropePos;
            cp.hadVision  = hadVision;
            return;
        }
    }
    g_turn_checkpoints.push_back({turnId, historyLen, activeLen, modality, mropePos, hadVision});
    constexpr size_t maxTurnCheckpoints = 256;
    if (g_turn_checkpoints.size() > maxTurnCheckpoints) {
        g_turn_checkpoints.erase(
                g_turn_checkpoints.begin(),
                g_turn_checkpoints.begin() + static_cast<std::ptrdiff_t>(
                        g_turn_checkpoints.size() - maxTurnCheckpoints));
    }
}

// Shared bind/run/fetch core for KV graph helpers.
static bool runBoundGraph(ModelRuntime& m, const std::vector<OrtValue*>& inputs, int numOutputs,
                          std::vector<OrtValue*>& out) {
    resetBinding(m);
    bool ok = true;
    for (size_t i = 0; i < inputs.size(); ++i) {
        ok = bindIn(m, static_cast<int>(i), inputs[i]) && ok;
    }
    for (int i = 0; i < numOutputs; ++i) {
        ok = bindOutDevice(m, i) && ok;
    }
    ok = ok && runBinding(m) && fetchOutputsInto(m, out);
    if (!ok || out.size() < static_cast<size_t>(numOutputs)) {
        releaseValues(m.api, out);
        return false;
    }
    return true;
}

static bool runRangeSlice(ModelRuntime& m, std::vector<OrtValue*>& in, int64_t start, int64_t end,
                          std::vector<OrtValue*>& out) {
    out.clear();
    if (m.session == nullptr || in.size() < static_cast<size_t>(num_keys_values) || !ensureBinding(m)) {
        return false;
    }
    const OrtApi* api = m.api;
    OrtBuffer startT = makeBuffer(m, {1}, m.inputTypes[kvSliceStartIdx]);
    OrtBuffer endT   = makeBuffer(m, {1}, m.inputTypes[kvSliceEndIdx]);
    bool ok = startT.value != nullptr && endT.value != nullptr;
    if (ok) {
        ok = writeScalarInt64(startT, m.inputTypes[kvSliceStartIdx], start) &&
             writeScalarInt64(endT, m.inputTypes[kvSliceEndIdx], end);
    }
    if (ok) {
        std::vector<OrtValue*> inputs;
        inputs.reserve(static_cast<size_t>(num_keys_values + 2));
        inputs.insert(inputs.end(), in.begin(), in.begin() + num_keys_values);      // KV tensors at slots 0..n-1
        inputs.push_back(startT.value);                                            // slot kvSliceStartIdx
        inputs.push_back(endT.value);                                              // slot kvSliceEndIdx
        ok = runBoundGraph(m, inputs, num_keys_values, out);
    }
    releaseBuffer(api, startT);
    releaseBuffer(api, endT);
    return ok;
}

static inline bool runKVSlice(std::vector<OrtValue*>& in, int64_t start, int64_t end, std::vector<OrtValue*>& out) {
    return runRangeSlice(mKVSlice, in, start, end, out);
}

static bool runKVSplit2(std::vector<OrtValue*>& in, int64_t splitAt,
                        std::vector<OrtValue*>& prefix, std::vector<OrtValue*>& window) {
    prefix.clear();
    window.clear();
    ModelRuntime& m = mKVSplit2;
    const int expectedOutputs = 2 * num_keys_values;
    if (m.session == nullptr || in.size() < static_cast<size_t>(num_keys_values) ||
        m.inputNames.size() < static_cast<size_t>(num_keys_values + 1) ||
        m.outputNames.size() < static_cast<size_t>(expectedOutputs) || !ensureBinding(m)) {
        return false;
    }
    const int splitIdx = num_keys_values;
    const OrtApi* api = m.api;
    OrtBuffer splitBuffer = makeBuffer(m, {1}, m.inputTypes[splitIdx]);
    bool ok = writeScalarInt64(splitBuffer, m.inputTypes[splitIdx], splitAt);
    std::vector<OrtValue*> outputs;
    if (ok) {
        std::vector<OrtValue*> inputs;
        inputs.reserve(static_cast<size_t>(num_keys_values + 1));
        inputs.insert(inputs.end(), in.begin(), in.begin() + num_keys_values);
        inputs.push_back(splitBuffer.value);
        ok = runBoundGraph(m, inputs, expectedOutputs, outputs);
    }
    releaseBuffer(api, splitBuffer);
    if (!ok || outputs.size() < static_cast<size_t>(expectedOutputs)) {
        releaseValues(api, outputs);
        return false;
    }
    prefix.assign(outputs.begin(), outputs.begin() + num_keys_values);
    window.assign(outputs.begin() + num_keys_values, outputs.begin() + expectedOutputs);
    for (size_t index = static_cast<size_t>(expectedOutputs); index < outputs.size(); ++index) {
        releaseOne(api, outputs[index]);
    }
    return true;
}

static bool runKVConcatGraph(ModelRuntime& m, std::vector<OrtValue*>& prefix,
                             std::vector<OrtValue*>& suffix, std::vector<OrtValue*>& out) {
    out.clear();
    if (m.session == nullptr || prefix.size() < static_cast<size_t>(num_keys_values) ||
        suffix.size() < static_cast<size_t>(num_keys_values) ||
        m.inputNames.size() < static_cast<size_t>(2 * num_keys_values) ||
        m.outputNames.size() < static_cast<size_t>(num_keys_values) ||
        m.inputTypes.size() < static_cast<size_t>(2 * num_keys_values) ||
        m.outputTypes.size() < static_cast<size_t>(num_keys_values) || !ensureBinding(m)) {
        return false;
    }
    for (int index = 0; index < num_keys_values; ++index) {
        const ONNXTensorElementDataType canonicalType = mMain.inputTypes[index];
        if (m.inputTypes[index] != canonicalType ||
            m.inputTypes[num_keys_values + index] != canonicalType ||
            m.outputTypes[index] != canonicalType) {
            return false;
        }
    }
    std::vector<OrtValue*> inputs;
    inputs.reserve(static_cast<size_t>(2 * num_keys_values));
    inputs.insert(inputs.end(), prefix.begin(), prefix.begin() + num_keys_values);
    inputs.insert(inputs.end(), suffix.begin(), suffix.begin() + num_keys_values);
    return runBoundGraph(m, inputs, num_keys_values, out);
}

static bool runKVConcat(std::vector<OrtValue*>& prefix, std::vector<OrtValue*>& suffix,
                        std::vector<OrtValue*>& out) {
    return runKVConcatGraph(mKVConcat, prefix, suffix, out);
}

static void appendKVSegment(int64_t tokenLen, int64_t mropeSpan, uint8_t modality) {
    if (tokenLen <= 0) {
        return;
    }
    const int64_t span = std::max<int64_t>(0, mropeSpan);
    if (!g_kv_segments.empty() && g_kv_segments.back().modality == modality) {
        g_kv_segments.back().token_len += tokenLen;
        g_kv_segments.back().mrope_span += span;
        return;
    }
    g_kv_segments.push_back({tokenLen, span, modality});
}

static int64_t kvSegmentTokenTotal() {
    int64_t total = 0;
    for (const KVSegment& seg : g_kv_segments) {
        total += std::max<int64_t>(0, seg.token_len);
    }
    return total;
}

static bool kvSegmentsContainVision() {
    for (const KVSegment& seg : g_kv_segments) {
        if (seg.modality != VISION_NONE && seg.token_len > 0) {
            return true;
        }
    }
    return false;
}

static void resetKVSegmentsToText(int64_t tokenLen) {
    std::vector<KVSegment>().swap(g_kv_segments);
    appendKVSegment(tokenLen, tokenLen, VISION_NONE);
}

static void dropFrontKVSegments(int64_t dropLen) {
    int64_t remaining = std::max<int64_t>(0, dropLen);
    while (remaining > 0 && !g_kv_segments.empty()) {
        KVSegment& front = g_kv_segments.front();
        if (remaining >= front.token_len) {
            remaining -= front.token_len;
            g_kv_segments.erase(g_kv_segments.begin());
            continue;
        }
        front.token_len -= remaining;
        front.mrope_span = std::max<int64_t>(0, front.mrope_span - remaining);
        remaining = 0;
    }
}

static void trimKVSegmentsToTotal(int64_t tokenLen) {
    int64_t total = kvSegmentTokenTotal();
    while (total > tokenLen && !g_kv_segments.empty()) {
        KVSegment& back = g_kv_segments.back();
        const int64_t excess = total - tokenLen;
        if (excess >= back.token_len) {
            total -= back.token_len;
            g_kv_segments.pop_back();
            continue;
        }
        back.token_len -= excess;
        back.mrope_span = std::max<int64_t>(0, back.mrope_span - excess);
        total = tokenLen;
    }
}

// Not fail-open: if this shrink fails, the caller must reset before prefill.
static bool shrinkSavedKVForRotaryBudget(int64_t targetLen) {
    if (targetLen >= saved_kv_len) {
        return true;
    }
    if (targetLen <= 0 || mKVSlice.session == nullptr) {
        LOGE("KV rotary-budget shrink unavailable (target=%lld, slice_model=%d)",
             static_cast<long long>(targetLen), mKVSlice.session ? 1 : 0);
        return false;
    }

    std::vector<OrtValue*> current;
    current.reserve(num_keys_values);
    for (int i = 0; i < num_keys_values; ++i) {
        if (saved_kv[i] == nullptr) {
            LOGE("KV rotary-budget shrink failed: saved_kv[%d] is null", i);
            return false;
        }
        current.push_back(saved_kv[i]);
    }

    const int64_t dropLen = saved_kv_len - targetLen;
    std::vector<OrtValue*> sliced;
    if (!runKVSlice(current, dropLen, saved_kv_len, sliced)) {
        LOGE("KV rotary-budget shrink failed; resetting history for this turn");
        return false;
    }

    release_saved_kv();
    for (int i = 0; i < num_keys_values; ++i) {
        saved_kv[i] = sliced[i];
    }
    sliced.clear();
    dropFrontKVSegments(dropLen);
    saved_kv_base += dropLen;
    saved_kv_len = targetLen;
    publishMemoryUsage();
    return true;
}

// Re-rotate key-side tensors down by `shift`; values carry no rotation.
static bool runRopeShift(std::vector<OrtValue*>& in, int64_t shift, std::vector<OrtValue*>& out) {
    out.clear();
    ModelRuntime& m = mRopeShift;
    const size_t keyCount = in.size();
    const size_t expectedInputs = keyCount + 1;
    if (m.session == nullptr || keyCount == 0 || m.inputNames.size() != expectedInputs ||
        m.inputTypes.size() != expectedInputs || m.outputNames.size() != keyCount || !ensureBinding(m)) {
        return false;
    }
    const int shiftIdx = static_cast<int>(keyCount);
    const OrtApi* api = m.api;
    OrtBuffer shiftT = makeBuffer(m, {1}, m.inputTypes[shiftIdx]);
    bool ok = shiftT.value != nullptr;
    if (ok) {
        ok = writeScalarInt64(shiftT, m.inputTypes[shiftIdx], shift);
    }
    if (ok) {
        std::vector<OrtValue*> inputs;
        inputs.reserve(expectedInputs);
        inputs.insert(inputs.end(), in.begin(), in.end());
        inputs.push_back(shiftT.value);
        ok = runBoundGraph(m, inputs, static_cast<int>(keyCount), out);
    }
    releaseBuffer(api, shiftT);
    return ok;
}

// Key-side slots are laid out over full-attention layers only:
// keys [0,L), k_scale [2L,3L), k_bias [3L,4L).
static bool ropeShiftKeys(OrtValue** kv, int count, int64_t shift) {
    if (mRopeShift.session == nullptr || shift <= 0) {
        return false;
    }
    const int L = num_full_attention_layers;
    if (L <= 0 || count < L * g_kv_blocks) {
        return false;
    }
    std::vector<OrtValue*> keyInputs;
    keyInputs.reserve(static_cast<size_t>(3 * L));
    auto appendSlot = [&](int slot) -> bool {
        if (slot >= count || kv[slot] == nullptr) {
            return false;
        }
        keyInputs.push_back(kv[slot]);
        return true;
    };
    for (int i = 0; i < L; ++i) {
        if (!appendSlot(i)) return false;
    }
    if (g_kv_blocks >= 4) {
        for (int i = 0; i < L; ++i) {
            if (!appendSlot(2 * L + i)) return false;
        }
    }
    if (g_kv_blocks >= 6) {
        for (int i = 0; i < L; ++i) {
            if (!appendSlot(3 * L + i)) return false;
        }
    }

    std::vector<OrtValue*> shifted;
    if (!runRopeShift(keyInputs, shift, shifted) || shifted.size() < keyInputs.size()) {
        releaseValues(mMain.api, shifted);
        return false;
    }

    size_t outIndex = 0;
    auto installSlot = [&](int slot) {
        releaseOne(mMain.api, kv[slot]);
        kv[slot] = shifted[outIndex];
        shifted[outIndex] = nullptr;
        outIndex++;
    };
    for (int i = 0; i < L; ++i) {
        installSlot(i);
    }
    if (g_kv_blocks >= 4) {
        for (int i = 0; i < L; ++i) {
            installSlot(2 * L + i);
        }
    }
    if (g_kv_blocks >= 6) {
        for (int i = 0; i < L; ++i) {
            installSlot(3 * L + i);
        }
    }
    return true;
}

// Renumber persisted KV from [saved_kv_base, +len) back to [0, len).
static bool renumberSavedKV() {
    if (mRopeShift.session == nullptr || saved_kv_base <= 0 || saved_kv_len <= 0) {
        return false;
    }
    if (!ropeShiftKeys(saved_kv.data(), num_keys_values, saved_kv_base)) {
        return false;
    }
    saved_kv_base = 0;
    return true;
}

// Hard RoPE-table recycle for long streaming turns.
static bool slideDecodeWindow(std::vector<OrtValue*>& kv, int64_t absNext, int64_t keep,
                              OrtValue* kvSeqLenTensor, ONNXTensorElementDataType kvSeqLenType,
                              int64_t& newAbsNext) {
    const OrtApi* api = mMain.api;
    if (mKVSlice.session == nullptr || mRopeShift.session == nullptr || kvSeqLenTensor == nullptr ||
        kv.size() < static_cast<size_t>(num_keys_values)) {
        return false;
    }
    const int64_t physLen = tensorDim(api, kv[0], 4);
    if (physLen <= 0) {
        return false;
    }
    const int64_t keepLen = std::min<int64_t>(keep, physLen);
    const int64_t drop    = physLen - keepLen;
    const int64_t shift   = absNext - keepLen;

    if (drop > 0) {
        std::vector<OrtValue*> kept;
        if (!runKVSlice(kv, drop, physLen, kept)) {
            return false;
        }
        releaseValues(api, kv);
        kv.swap(kept);
    }
    if (shift > 0 && !ropeShiftKeys(kv.data(), num_keys_values, shift)) {
        return false;
    }
    void* raw = nullptr;
    if (!logOrtStatus(api, api->GetTensorMutableData(kvSeqLenTensor, &raw),
                      "GetTensorMutableData(kv_seq_len)") || raw == nullptr) {
        return false;
    }
    if (kvSeqLenType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        *reinterpret_cast<int32_t*>(raw) = static_cast<int32_t>(keepLen);
    } else if (kvSeqLenType == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        *reinterpret_cast<int64_t*>(raw) = keepLen;
    } else {
        LOGE("slideDecodeWindow: unsupported kv_seq_len type %d", static_cast<int>(kvSeqLenType));
        return false;
    }
    newAbsNext = keepLen;
    return true;
}

// Token decode helpers.
inline static bool is_stop_token(int id) {
    return std::find(g_stop_token_ids.begin(), g_stop_token_ids.end(), id) != g_stop_token_ids.end();
}

// The Locked variant requires g_vision_encode_mutex; the wrapper waits out any active prewarm.
static void releasePendingVisionHiddenLocked() {
    if (!g_pending_vision_outputs.empty() && mMain.api != nullptr) {
        releaseValues(mMain.api, g_pending_vision_outputs);
    }
    g_pending_vision_outputs.clear();
    g_vision_hidden_ready    = false;
    g_vision_hidden_modality = VISION_NONE;
}
static void releasePendingVisionHidden() {
    std::lock_guard<std::mutex> lock(g_vision_encode_mutex);
    releasePendingVisionHiddenLocked();
}

inline static void clear_history(const char* reason = "unspecified",
                                 bool preservePendingVisionHidden = false) {
    LOGI("clear_history: reason=%s saved_len=%lld base=%lld mrope=%lld hasVision=%d history_ids=%zu",
         reason ? reason : "unspecified",
         static_cast<long long>(saved_kv_len),
         static_cast<long long>(saved_kv_base),
         static_cast<long long>(saved_mrope_pos),
         saved_kv_has_vision ? 1 : 0,
         retainedHistorySize());
    clearAllModelBindingRefs();
    release_saved_kv();
    release_saved_linear();
    ids_len       = 0;
    saved_kv_len  = 0;
    saved_kv_base = 0;
    saved_mrope_pos = 0;               // vision: mRoPE position tracker resets with the conversation
    saved_kv_has_vision = false;       // vision: no un-rebuildable vision segment remains
    if (!preservePendingVisionHidden) {
        g_pending_vision.store(VISION_NONE, std::memory_order_release);
        releasePendingVisionHidden();  // ordinary reset invalidates any unconsumed speculative encode
    }
    std::vector<KVSegment>().swap(g_kv_segments);
    std::vector<int>().swap(g_history_ids);
    g_history_begin = 0;
    g_history_base = 0;
    std::vector<TurnCheckpoint>().swap(g_turn_checkpoints);
    reset_output_words_decoder();
    stream_buf.clear();
    tokens_since_flush = 0;
    publishMemoryUsage();
}

// Synchronization.
static std::mutex g_system_prompt_mutex;
static std::mutex g_tokenizer_mutex;
static std::mutex g_clean_commit_mutex;
static std::condition_variable g_clean_commit_cv;
static bool g_clean_commit_running = false;
// Whether the in-flight clean commit surfaces the "整理记忆 Consolidating" UI. Only Organize-Memory ON
// commits are user-visible; OFF-mode internal rebuilds and red-zone trims run
// silently so the box never appears while the toggle is off. Bookkeeping (g_clean_commit_running) still
// tracks every commit so waitForCleanCommit() serialization is unaffected either way.
static bool g_clean_commit_status_visible = false;

class SerialMaintenanceWorker {
public:
    ~SerialMaintenanceWorker() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        condition_.notify_one();
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    void submit(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!thread_.joinable()) {
                thread_ = std::thread([this] { run(); });
            }
            tasks_.push_back(std::move(task));
        }
        condition_.notify_one();
    }

private:
    void run() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
                if (stopping_ && tasks_.empty()) {
                    return;
                }
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            task();
        }
    }

    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<std::function<void()>> tasks_;
    std::thread thread_;
    bool stopping_ = false;
};

static SerialMaintenanceWorker g_maintenance_worker;
static std::vector<int> g_manual_stop_notice_ids;
static std::vector<int> g_vision_image_note_ids;
static std::vector<int> g_vision_video_note_ids;
static std::vector<std::vector<int>> g_video_timestamp_token_cache;
static int g_video_timestamp_cache_grid_t = -1;
static int g_video_timestamp_cache_temporal_patch = -1;
static float g_video_timestamp_cache_fps = -1.0f;

// Memory profile and hysteresis.
struct MemoryProfile {
    int memoryTokens;
    int prefillTokens;
    int prefillDecoupleMinNewTokens;
    int decodeTokens;
    int decodeKeepWindow;
};

static int clampInt(int value, int lo, int hi) {
    return std::max(lo, std::min(value, hi));
}

static int memoryRedUsedThreshold(int memoryTokens) {
    const int capacity = std::max(1, memoryTokens);
    const int redPercent = g_memory_red_zone_percent.load(std::memory_order_relaxed);
    return clampInt((capacity * redPercent) / 100, 1, capacity);
}

static int memoryGreenTargetUsed(int memoryTokens) {
    const int capacity = std::max(1, memoryTokens);
    const int greenPercent = g_memory_green_target_percent.load(std::memory_order_relaxed);
    return clampInt((capacity * greenPercent) / 100, 1, capacity);
}

static bool memoryUsedInRedZone(int memoryTokens, int64_t usedTokens) {
    return usedTokens > static_cast<int64_t>(memoryRedUsedThreshold(memoryTokens));
}

static int memoryRetainTargetAfterRed(int memoryTokens, int64_t usedTokens) {
    return memoryUsedInRedZone(memoryTokens, usedTokens)
            ? memoryGreenTargetUsed(memoryTokens)
            : std::max(1, memoryTokens);
}

static MemoryProfile makeMemoryProfile(int memoryTokens, int prefillTokens, int decodeTokens) {
    MemoryProfile p{};
    p.memoryTokens = clampInt(memoryTokens, MIN_MEMORY_TOKENS, MAX_MEMORY_TOKENS_RUNTIME);
    p.prefillTokens = clampInt(prefillTokens, MIN_PREFILL_TOKENS,
                               std::min(MAX_PREFILL_TOKENS_RUNTIME, p.memoryTokens));
    p.prefillDecoupleMinNewTokens = std::max(1, p.prefillTokens / 4);
    p.decodeTokens = clampInt(decodeTokens, MIN_DECODE_TOKENS, MAX_DECODE_TOKENS_RUNTIME);
    p.decodeKeepWindow = std::min(p.memoryTokens, max_seq_len - kDecodeRecycleHeadroom);
    return p;
}

static MemoryProfile snapshotMemoryProfile() {
    std::lock_guard<std::mutex> lock(g_runtime_config_mutex);
    // Runtime pressure (thermal/memory) must NOT shorten the user's decode limit: clamping decodeTokens
    // here truncated replies mid-sentence across every decode strategy whenever pressure was briefly
    // elevated (a transient onTrimMemory / thermal SEVERE tick capped a full reply to 64/256 tokens).
    // Pressure is handled where it belongs -- Trim_Runtime (vision release + arena shrink) reclaims
    // memory and the OS CPU governor throttles heat -- so the reply is always generated in full.
    return makeMemoryProfile(g_memory_tokens, g_prefill_tokens, g_decode_tokens);
}

static void applyMemoryProfileLocked(const MemoryProfile& p) {
    g_memory_tokens = p.memoryTokens;
    g_prefill_tokens = p.prefillTokens;
    g_decode_tokens = p.decodeTokens;
}

// Clean-commit lifecycle.
static void pushPostProcessingState(bool active) {
    if (g_jvm == nullptr || g_main_cls == nullptr || g_on_post_processing == nullptr) {
        return;
    }
    JNIEnv* env = nullptr;
    bool attached = false;
    jint state = g_jvm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6);
    if (state == JNI_EDETACHED) {
        if (g_jvm->AttachCurrentThread(&env, nullptr) != JNI_OK) {
            return;
        }
        attached = true;
    } else if (state != JNI_OK || env == nullptr) {
        return;
    }
    env->CallStaticVoidMethod(g_main_cls, g_on_post_processing, active ? JNI_TRUE : JNI_FALSE);
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
    }
    if (attached) {
        g_jvm->DetachCurrentThread();
    }
}

static void waitForCleanCommit() {
    std::unique_lock<std::mutex> lock(g_clean_commit_mutex);
    g_clean_commit_cv.wait(lock, [] { return !g_clean_commit_running; });
}

static void beginCleanCommit(bool showStatus) {
    {
        std::unique_lock<std::mutex> lock(g_clean_commit_mutex);
        g_clean_commit_cv.wait(lock, [] { return !g_clean_commit_running; });
        g_clean_commit_running = true;
        g_clean_commit_status_visible = showStatus;
    }
    if (showStatus) {
        pushPostProcessingState(true);
    }
}

static void finishCleanCommit() {
    clearAllModelBindingRefs();
    bool wasVisible;
    {
        std::lock_guard<std::mutex> lock(g_clean_commit_mutex);
        g_clean_commit_running = false;
        wasVisible = g_clean_commit_status_visible;
        g_clean_commit_status_visible = false;
    }
    g_clean_commit_cv.notify_all();
    if (wasVisible) {
        pushPostProcessingState(false);
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Ensure_1Vision_1Models(JNIEnv* env, jclass clazz) {
    (void)clazz;
    if (metadataVisionCaps() == 0 || !g_model_sessions_initialized) {
        return JNI_FALSE;
    }
    if (g_vision_sessions_initialized) {
        return JNI_TRUE;
    }
    if (g_busy.exchange(true, std::memory_order_acq_rel)) {
        return JNI_FALSE;
    }
    struct BusyGuard {
        ~BusyGuard() { g_busy.store(false, std::memory_order_release); }
    } busyGuard;
    waitForCleanCommit();
    std::array<MergedStrategy, 3> strategies;
    {
        std::lock_guard<std::mutex> configLock(g_runtime_config_mutex);
        strategies = g_config_merged_strategies;
    }
    std::lock_guard<std::mutex> visionLock(g_vision_encode_mutex);
    std::lock_guard<std::mutex> modelLock(g_model_session_mutex);
    return initializeVisionSessionsLocked(env, strategies, metadataVisionCaps())
            ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Vision_1Models_1Ready(JNIEnv* env, jclass clazz) {
    (void)env;
    (void)clazz;
    std::lock_guard<std::mutex> modelLock(g_model_session_mutex);
    return g_vision_sessions_initialized ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Configure_1Runtime_1Pressure(
        JNIEnv* env, jclass clazz, jint pressure_level) {
    (void)env;
    (void)clazz;
    const int level = clampInt(static_cast<int>(pressure_level), 0, 2);
    g_runtime_pressure_level.store(level, std::memory_order_release);
    LOGI("Runtime pressure level: %d", level);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Trim_1Runtime(
        JNIEnv* env, jclass clazz, jboolean release_vision) {
    (void)env;
    (void)clazz;
    if (!g_model_sessions_initialized) {
        return JNI_TRUE;
    }
    if (g_busy.exchange(true, std::memory_order_acq_rel)) {
        return JNI_FALSE;
    }
    struct BusyGuard {
        ~BusyGuard() { g_busy.store(false, std::memory_order_release); }
    } busyGuard;
    waitForCleanCommit();
    std::lock_guard<std::mutex> visionLock(g_vision_encode_mutex);
    std::lock_guard<std::mutex> modelLock(g_model_session_mutex);
    clearAllModelBindingRefs();
    if (release_vision == JNI_TRUE) {
        g_pending_vision.store(VISION_NONE, std::memory_order_release);
        releasePendingVisionHiddenLocked();
        std::lock_guard<std::mutex> pixelsLock(g_vision_pixels_mutex);
        releaseVisionRuntimeSessions();
    }
    shrinkAllModelAllocators();
    g_reserved_prefill_arena_bytes = 0;
    g_reserved_decode_arena_bytes = 0;
    mallopt(M_PURGE, 0);
    return JNI_TRUE;
}

// Slot system prompt.
inline static bool is_blank_system_prompt(const std::string& text) {
    return std::all_of(text.begin(), text.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });
}

inline static void rebuild_system_prompt_ids_locked() {
    g_system_prompt_ids.clear();
    g_system_block_len = 0;
    if (tokenizer == nullptr || g_system_prompt_text.empty()) {
        return;
    }
    {
        std::lock_guard<std::mutex> tokenizerLock(g_tokenizer_mutex);
        g_system_prompt_ids = tokenizer->encode(g_system_prompt_text);
    }
    g_system_block_len = static_cast<int64_t>(chat_system_prefix_ids.size()) +
                         static_cast<int64_t>(g_system_prompt_ids.size()) +
                         static_cast<int64_t>(chat_system_suffix_ids.size());
}

// Shared one-shot prefill returning KV/linear state only, via the merged text greedy prefill graph. Used
// to reconstruct cross-turn KV from token ids; the graph's token/kv_seq outputs are produced but ignored.
static bool runPrefillToKV(const int* idsData, int64_t tokenCount,
                           OrtValue* const* kvInputs, OrtValue* const* linearInputs,
                           int64_t historyLen, int64_t cacheLen,
                           std::vector<OrtValue*>& outKV, std::vector<OrtValue*>& outLinear,
                           const char* inputOpName) {
    outKV.clear();
    outLinear.clear();
    const OrtApi* api = mMain.api;
    ModelRuntime& g = getModel(mergedPrefillModel(0, MSTRAT_GREEDY));   // text greedy prefill
    if (api == nullptr || g.session == nullptr || idsData == nullptr || tokenCount <= 0 || kvInputs == nullptr ||
        (num_linear_states > 0 && linearInputs == nullptr) || !ensureBinding(g)) {
        return false;
    }
    const int tokenIdx      = findInputIdxAny(g, {"embed_input_ids", "input_ids"});
    const int idsLenIdx     = findInputIdx(g, "prefill_ids_len");
    const int historyLenIdx = findInputIdx(g, "prefill_history_len");
    const int cacheLenIdx   = findInputIdx(g, "prefill_cache_len");   // absent on the vision prefill graphs
    if (tokenIdx < 0 || idsLenIdx < 0 || historyLenIdx < 0) {
        LOGE("runPrefillToKV: merged prefill graph missing token/phase inputs");
        return false;
    }
    bool ok = false;
    OrtBuffer idsBuffer, idsLenBuffer, historyLenBuffer, cacheLenBuffer;
    do {
        idsBuffer = makeBuffer(g, {1, tokenCount}, g.inputTypes[tokenIdx]);
        if (!writeIntegerValues(idsBuffer, g.inputTypes[tokenIdx], idsData,
                                static_cast<size_t>(tokenCount))) {
            LOGE("%s: unsupported token input type %d", inputOpName,
                 static_cast<int>(g.inputTypes[tokenIdx]));
            break;
        }
        idsLenBuffer = makeBuffer(g, {1}, g.inputTypes[idsLenIdx]);
        historyLenBuffer = makeBuffer(g, {1}, g.inputTypes[historyLenIdx]);
        if (!writeScalarInt64(idsLenBuffer, g.inputTypes[idsLenIdx], tokenCount) ||
            !writeScalarInt64(historyLenBuffer, g.inputTypes[historyLenIdx], historyLen)) {
            break;
        }
        if (cacheLenIdx >= 0) {
            cacheLenBuffer = makeBuffer(g, {1}, g.inputTypes[cacheLenIdx]);
            if (!writeScalarInt64(cacheLenBuffer, g.inputTypes[cacheLenIdx], cacheLen)) {
                break;
            }
        }
        resetBinding(g);
        bool bound = true;
        for (int i = 0; i < num_keys_values; ++i)   { bound = bindIn(g, i, kvInputs[i]) && bound; }
        for (int i = 0; i < num_linear_states; ++i) { bound = bindIn(g, num_keys_values + i, linearInputs[i]) && bound; }
        bound = bindIn(g, tokenIdx, idsBuffer.value) && bound;
        bound = bindIn(g, idsLenIdx, idsLenBuffer.value) && bound;
        bound = bindIn(g, historyLenIdx, historyLenBuffer.value) && bound;
        if (cacheLenIdx >= 0) { bound = bindIn(g, cacheLenIdx, cacheLenBuffer.value) && bound; }
        // Bind every output to device: the state block leads (KV then linear); the trailing token/kv_seq
        // outputs are produced but discarded (this helper only reconstructs KV state from ids).
        for (int i = 0; i < static_cast<int>(g.outputNames.size()); ++i) { bound = bindOutDevice(g, i) && bound; }
        std::vector<OrtValue*> allOut;
        const bool runOk = bound && runBinding(g) && fetchOutputsInto(g, allOut) &&
                           allOut.size() >= static_cast<size_t>(num_main_states);
        if (!runOk) {
            releaseValues(api, allOut);
            break;
        }
        outKV.assign(allOut.begin(), allOut.begin() + num_keys_values);
        outLinear.assign(allOut.begin() + num_keys_values, allOut.begin() + num_main_states);
        for (size_t i = static_cast<size_t>(num_main_states); i < allOut.size(); ++i) {
            releaseOne(api, allOut[i]);   // token / kv_seq tail: ignored
        }
        ok = true;
    } while (false);
    releaseBuffer(api, idsBuffer);
    releaseBuffer(api, idsLenBuffer);
    releaseBuffer(api, historyLenBuffer);
    releaseBuffer(api, cacheLenBuffer);
    if (!ok) {
        releaseValues(api, outKV);
        releaseValues(api, outLinear);
    }
    return ok;
}

// Rebuild from ids; never excise KV in-place because survivors still attended dropped tokens.
static bool rebuildSavedKVFromHistoryRetaining(int retainTarget) {
    release_saved_kv();
    release_saved_linear();
    saved_kv_len = 0;
    saved_kv_base = 0;
    // A rebuild reconstructs the window purely from the text ids: any spliced vision segment is gone,
    // so the mRoPE position tracker collapses back to the physical token count (text-equal invariant).
    saved_kv_has_vision = false;
    std::vector<KVSegment>().swap(g_kv_segments);
    publishMemoryUsage();
    const size_t retainedCount = retainedHistorySize();
    if (retainedCount == 0) {
        saved_mrope_pos = 0;
        return true;
    }
    const int64_t keep = std::max(1, retainTarget);
    const int64_t total = static_cast<int64_t>(retainedCount);
    const int64_t startIdx = (total > keep) ? (total - keep) : 0;
    const OrtApi* api = mMain.api;
    const int64_t N = total - startIdx;   // = min(keep, total)
    if (api == nullptr || mMain.session == nullptr) {
        return false;
    }
    std::vector<OrtValue*> kept, keptLinear;
    if (!runPrefillToKV(retainedHistoryData() + startIdx, N, input_tensors_kv_init.data(),
                        linear_state_init.data(), 0, 0, kept, keptLinear, "recompute input_ids")) {
        g_history_ids.clear();
        g_history_begin = 0;
        g_history_base = 0;
        return false;
    }
    for (int i=0;i<num_keys_values;++i) saved_kv[i]=kept[i];
    store_saved_linear(keptLinear);   // rebuilt from zero over the retained window -> consistent linear state
    saved_kv_len = N;
    saved_kv_base = 0;
    saved_mrope_pos = N;   // text-only rebuild: mRoPE position == physical length
    resetKVSegmentsToText(N);
    publishMemoryUsage();
    return true;
}

static bool rebuildSavedKVFromHistory(int memoryTokens) {
    const int retainTarget = memoryRetainTargetAfterRed(memoryTokens, saved_kv_len);
    return rebuildSavedKVFromHistoryRetaining(retainTarget);
}

// Incremental clean-cache update; falls back to full rebuild on failure. Vision organization must
// rebuild from note-backed history so no spliced vision KV or dependent linear state survives.
static bool appendCleanTurnKV(const std::vector<int>& ids, int memoryTokens,
                              bool rebuildVisionMemory) {
    if (ids.empty()) {
        return true;
    }
    const OrtApi* api = mMain.api;
    const int64_t N = static_cast<int64_t>(ids.size());
    const int64_t prevLen = saved_kv_len;
    const int64_t prevBase = saved_kv_base;
        appendHistoryIds(ids);
    const int64_t projectedLen = rebuildVisionMemory
            ? static_cast<int64_t>(retainedHistorySize())
            : prevLen + N;
    const int intendedRetain = static_cast<int>(
            memoryUsedInRedZone(memoryTokens, projectedLen)
                ? memoryGreenTargetUsed(memoryTokens)
                : projectedLen);
    if (rebuildVisionMemory) {
        LOGI("Clean-cache: replacing vision KV with note-backed history "
             "(physical=%lld text_ids=%lld retain=%d)",
             static_cast<long long>(prevLen), static_cast<long long>(projectedLen), intendedRetain);
        return rebuildSavedKVFromHistoryRetaining(intendedRetain);
    }
    if (api == nullptr || prevLen == 0 || saved_kv[0] == nullptr ||
        (hasLinearState() && !saved_linear_valid) ||                      // linear state stale: rebuild
        prevBase + prevLen + N + 2 > max_seq_len ||                       // table would overflow: rebuild
        mMain.session == nullptr) {
        return rebuildSavedKVFromHistoryRetaining(intendedRetain);
    }
    std::vector<OrtValue*> kept, keptLinear;
    if (!runPrefillToKV(ids.data(), N, saved_kv.data(), saved_linear_states.data(),
                        prevBase + prevLen, prevLen, kept, keptLinear, "append input_ids")) {
        return rebuildSavedKVFromHistoryRetaining(intendedRetain);
    }
    release_saved_kv();
    for (int i=0;i<num_keys_values;++i) saved_kv[i]=kept[i];
    store_saved_linear(keptLinear);   // continues the recurrence from the persisted linear state
    saved_kv_len = prevLen + N;
    saved_kv_base = prevBase;
    if (memoryUsedInRedZone(memoryTokens, saved_kv_len)) {
        return rebuildSavedKVFromHistoryRetaining(intendedRetain);
    }
    publishMemoryUsage();
    return true;
}

constexpr int kInlineCleanCommitMaxTokens = 256;

static void runCleanCommit(const std::vector<int>& ids, bool kvStateValid, int memoryTokens,
                           bool rebuildVisionMemory) {
    if (g_clean_commit_cancel.load(std::memory_order_relaxed)) {
        return;
    }
    if (kvStateValid) {
        if (!appendCleanTurnKV(ids, memoryTokens, rebuildVisionMemory)) {
            LOGE("Clean-cache recompute failed; resetting conversation history");
            clear_history("clean_commit_recompute_failed");
        }
    } else {
        clear_history("clean_commit_invalid_kv_state");
    }
}

static void commitCleanTurn(std::vector<int>&& dialogueIds, bool kvStateValid, int memoryTokens,
                            bool rebuildVisionMemory, bool showStatus) {
    if (dialogueIds.size() <= static_cast<size_t>(kInlineCleanCommitMaxTokens)) {
        beginCleanCommit(showStatus);
        runCleanCommit(dialogueIds, kvStateValid, memoryTokens, rebuildVisionMemory);
        finishCleanCommit();
        return;
    }
    beginCleanCommit(showStatus);
    g_maintenance_worker.submit(
            [ids = std::move(dialogueIds), kvStateValid, memoryTokens,
             rebuildVisionMemory]() mutable {
        runCleanCommit(ids, kvStateValid, memoryTokens, rebuildVisionMemory);
        finishCleanCommit();
    });
}

// OFF red-zone cap rebuild; green-zone OFF saves KV directly. This path runs ONLY when the retained KV
// crossed the red-zone trigger, i.e. a forced memory organization -- so it surfaces the "整理记忆" box even
// though the Organize Memory toggle is off, matching the ON path (the box tracks whether organization runs,
// not whether the toggle is set).
static void commitOffCap(int cap) {
    beginCleanCommit(true);
    g_maintenance_worker.submit([cap]() {
        if (!g_clean_commit_cancel.load(std::memory_order_relaxed)) {
            if (rebuildSavedKVFromHistoryRetaining(cap)) {
                LOGI("OFF memory organization complete: retained=%lld target=%d",
                     static_cast<long long>(saved_kv_len), cap);
            } else {
                LOGE("OFF memory organization failed; resetting conversation history");
                clear_history("off_memory_organization_failed");
            }
        }
        finishCleanCommit();
    });
}

// OFF mode: transfer decode KV directly; red-zone trimming is a background re-prefill, not a slice.
static bool saveDecodeKVDirect(std::vector<OrtValue*>& decodeKV, std::vector<OrtValue*>& decodeLinear,
                               int64_t decodeBase,
                               const std::vector<int>& promptIds, const std::vector<int>& replyIds,
                               int memoryTokens, int* pendingCap,
                               bool visionTurn, int64_t visionMropePos,
                               int64_t promptPhysicalTokens, int64_t promptMropeSpan,
                               uint8_t promptModality) {
    *pendingCap = -1;
    const OrtApi* api = mMain.api;
    if (api == nullptr || decodeKV.size() < static_cast<size_t>(num_keys_values) || decodeKV[0] == nullptr) {
        return false;
    }
    const int64_t physLen = tensorDim(api, decodeKV[0], 4);   // keys seq axis (5-D KV layout)
    if (physLen <= 0) {
        return false;
    }
    const bool keepsVision = visionTurn || saved_kv_has_vision;
    if (keepsVision && !visionTurn && saved_kv_has_vision && g_kv_segments.empty() && saved_kv_len > 0) {
        appendKVSegment(saved_kv_len, std::max<int64_t>(0, saved_mrope_pos - saved_kv_base), VISION_IMAGE);
    }
    const int64_t previousSegmentTokens = kvSegmentTokenTotal();
    const int64_t previousMropePos = saved_kv_has_vision ? saved_mrope_pos : (saved_kv_base + saved_kv_len);
    release_saved_kv();
    for (int i = 0; i < num_keys_values; ++i) {
        saved_kv[i] = decodeKV[i];
        decodeKV[i] = nullptr;
    }
    for (size_t i = static_cast<size_t>(num_keys_values); i < decodeKV.size(); ++i) {
        releaseOne(api, decodeKV[i]);
    }
    decodeKV.clear();
    store_saved_linear(decodeLinear);   // decode linear is consistent (no in-turn slide for hybrid)
    // Whether the saved window still carries an un-rebuildable vision segment: a vision turn adds one;
    // a text turn on top of an existing vision segment keeps it. Snapshot BEFORE overwriting globals.
    saved_kv_len  = physLen;
    saved_kv_base = decodeBase;
    appendHistoryIds(promptIds);
    appendHistoryIds(replyIds);
    if (keepsVision) {
        if (visionTurn) {
            std::vector<KVSegment>().swap(g_kv_segments);
        }
        const int64_t newPhysicalTokens = std::max<int64_t>(0, physLen - previousSegmentTokens);
        const int64_t promptTokens = std::min<int64_t>(std::max<int64_t>(0, promptPhysicalTokens), newPhysicalTokens);
        const int64_t promptSpan = std::min<int64_t>(std::max<int64_t>(0, promptMropeSpan),
                                                     std::max<int64_t>(0, visionMropePos - previousMropePos));
        appendKVSegment(promptTokens, promptSpan, promptModality);
        const int64_t replyTokens = std::max<int64_t>(0, newPhysicalTokens - promptTokens);
        const int64_t replySpan = std::max<int64_t>(0, visionMropePos - previousMropePos - promptSpan);
        appendKVSegment(replyTokens, replySpan, VISION_NONE);
        trimKVSegmentsToTotal(physLen);
        // Keep live vision while below the red threshold. Once physical KV usage crosses it, rebuild
        // note-backed history to the green target; this drops unrebuildable vision tokens and recreates
        // both full-attention KV and hybrid linear state from text. mRoPE diverges only while vision lives.
        saved_kv_has_vision = kvSegmentsContainVision();
        saved_mrope_pos = (visionMropePos > 0) ? visionMropePos : physLen;
        const int64_t cap = memoryRetainTargetAfterRed(memoryTokens, saved_kv_len);
        if (saved_kv_len > cap) {
            *pendingCap = static_cast<int>(cap);
            LOGI("OFF multimodal memory trigger: physical=%lld capacity=%d target=%d",
                 static_cast<long long>(saved_kv_len), memoryTokens, *pendingCap);
        }
        publishMemoryUsage();
        return true;
    }
    resetKVSegmentsToText(physLen);
    const int64_t cap = memoryRetainTargetAfterRed(memoryTokens, saved_kv_len);
    if (saved_kv_len > cap) {
        *pendingCap = static_cast<int>(cap);
        return true;
    }
    renumberSavedKV();
    saved_mrope_pos = saved_kv_len;   // pure-text invariant: mRoPE position == physical length
    publishMemoryUsage();
    return true;
}

// ── Vision (image/video) prefill pipeline ────────────────────────────────────────────────────
static inline bool visionCancelRequested() {
    if (g_vision_cancel_requested.load(std::memory_order_acquire) ||
        g_clean_commit_cancel.load(std::memory_order_acquire)) {
        return true;
    }
    // A manual Stop sets g_cancel to end the turn cooperatively; unlike Clear/destructive cancel it must not
    // abort an in-flight vision encode. g_manual_stop is published (release) before g_cancel, so an acquire
    // load of g_cancel that observes true is guaranteed to observe the matching manual-stop value.
    if (!g_cancel.load(std::memory_order_acquire)) {
        return false;
    }
    return !g_manual_stop.load(std::memory_order_acquire);
}

// Preprocess(static pixel_values) -> 5 device outs -> Vision -> vision_hidden. This is the expensive,
// query-INDEPENDENT ViT pass, so it can be pre-computed the moment the user captures (Prewarm_Vision)
// while they type the query. Assumes the caller holds g_vision_encode_mutex (serializes every user of
// the vision encoder sessions). Returns all owned deepstack + vision hidden outputs, or empty on failure/cancel.
static std::vector<OrtValue*> runVisionEncodeOnly(uint8_t visionMode) {
    const OrtApi* api = mMain.api;
    ModelRuntime& pre     = (visionMode == VISION_IMAGE) ? mImagePreprocess : mVideoPreprocess;
    OrtValue* pixelValues = (visionMode == VISION_IMAGE) ? g_image_pixel_values : g_video_pixel_values;
    if (visionCancelRequested() || api == nullptr || pixelValues == nullptr ||
        !sessionReady(pre) || !sessionReady(mVision) ||
        !ensureBinding(pre) || !ensureBinding(mVision)) {
        return {};
    }

    const bool isVideo = visionMode == VISION_VIDEO;
    const int64_t inputCount = tensorDim(api, pixelValues, 0);
    const int64_t inputChannels = tensorDim(api, pixelValues, isVideo ? 1 : 2);
    const int64_t inputHeight = tensorDim(api, pixelValues, isVideo ? 2 : 3);
    const int64_t inputWidth = tensorDim(api, pixelValues, isVideo ? 3 : 4);
    const int expectedHeight = isVideo ? input_video_height : input_image_height;
    const int expectedWidth = isVideo ? input_video_width : input_image_width;
    const bool countMatches = isVideo
            ? inputCount > 0 && (!g_video_fixed_length_input || inputCount == g_video_num_frames)
            : inputCount == g_vision_batch_size;
    if (!countMatches || inputChannels != 3 || inputHeight != expectedHeight ||
        inputWidth != expectedWidth) {
        LOGE("Vision encode input contract mismatch: mode=%s graph=%s input=%lldx%lldx%lldx%lld "
             "expected_count=%d expected_hw=%dx%d",
             isVideo ? "video" : "image", pre.fileName.c_str(),
             static_cast<long long>(inputCount), static_cast<long long>(inputChannels),
             static_cast<long long>(inputHeight), static_cast<long long>(inputWidth),
             isVideo ? g_video_num_frames : g_vision_batch_size, expectedHeight, expectedWidth);
           return {};
    }

    // 1) Preprocess: the static pre-bound pixel_values -> 5 device outputs. Hold the pixels mutex only
    //    for the bind+run so the camera producer cannot overwrite the buffer mid-preprocess.
    std::vector<OrtValue*> preOut;
    const int preOutN = static_cast<int>(pre.outputNames.size());
    const auto preprocessStarted = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(g_vision_pixels_mutex);
        resetBinding(pre);
        if (!bindIn(pre, 0, pixelValues)) {
            return {};
        }
        for (int i = 0; i < preOutN; ++i) {
            bindOutDevice(pre, i);
        }
        if (!runBinding(pre) || !fetchOutputsInto(pre, preOut) ||
            preOut.size() < static_cast<size_t>(preOutN)) {
            releaseValues(api, preOut);
            return {};
        }
    }
    const auto preprocessFinished = std::chrono::steady_clock::now();
    if (visionCancelRequested()) {
        releaseValues(api, preOut);
        return {};
    }
    const int64_t patchTokens = preOut.empty() ? -1 : tensorDim(api, preOut[0], 0);

    // 2) Vision encoder: the 5 preprocess outputs (in graph order) -> vision_hidden.
    resetBinding(mVision);
    bool ok = true;
    const int visInN = static_cast<int>(mVision.inputNames.size());
    for (int i = 0; i < visInN && i < static_cast<int>(preOut.size()); ++i) {
        ok = bindIn(mVision, i, preOut[i]) && ok;
    }
    for (int i = 0; i < static_cast<int>(mVision.outputNames.size()); ++i) {
        ok = bindOutDevice(mVision, i) && ok;
    }
    std::vector<OrtValue*> visOut;
    const auto visionStarted = std::chrono::steady_clock::now();
    ok = ok && runBinding(mVision) && fetchOutputsInto(mVision, visOut) && !visOut.empty();
    const auto visionFinished = std::chrono::steady_clock::now();
    releaseValues(api, preOut);   // consumed by Vision
    if (!ok || visOut.empty() || visOut[0] == nullptr) {
        releaseValues(api, visOut);
        return {};
    }
    const int visionHiddenIndex = static_cast<int>(visOut.size()) - 1;
    if (visOut[static_cast<size_t>(visionHiddenIndex)] == nullptr) {
        LOGE("Vision encoder output contract has no final vision hidden tensor");
        releaseValues(api, visOut);
        return {};
    }
    OrtValue* visionHidden = visOut[static_cast<size_t>(visionHiddenIndex)];
        const int64_t hiddenTokens = tensorDim(api, visionHidden, 1);
        const int64_t expectedHiddenTokens = isVideo ? g_video_embed_size : g_image_embed_size;
        const int64_t merge = std::max<int64_t>(1, g_spatial_merge_size);
        const int64_t expectedPatchTokens = expectedHiddenTokens * merge * merge;
        if (hiddenTokens != expectedHiddenTokens || patchTokens != expectedPatchTokens) {
        LOGE("Vision encode output contract mismatch: mode=%s graph=%s patches=%lld/%lld hidden=%lld/%lld",
             isVideo ? "video" : "image", pre.fileName.c_str(),
             static_cast<long long>(patchTokens), static_cast<long long>(expectedPatchTokens),
             static_cast<long long>(hiddenTokens), static_cast<long long>(expectedHiddenTokens));
        releaseValues(api, visOut);
        return {};
        }
        const auto preprocessMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            preprocessFinished - preprocessStarted).count();
        const auto visionMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            visionFinished - visionStarted).count();
        LOGI("Vision encode route: mode=%s preprocess=%s input=%lldx%lldx%lldx%lld "
         "patches=%lld hidden=%lld preprocess_ms=%lld vision_ms=%lld",
         isVideo ? "video" : "image", pre.fileName.c_str(),
         static_cast<long long>(inputCount), static_cast<long long>(inputChannels),
         static_cast<long long>(inputHeight), static_cast<long long>(inputWidth),
         static_cast<long long>(patchTokens), static_cast<long long>(hiddenTokens),
         static_cast<long long>(preprocessMs), static_cast<long long>(visionMs));
    if (visionCancelRequested()) {
        releaseValues(api, visOut);
        return {};
    }
    if (visionMode == VISION_VIDEO && !g_video_fixed_length_input) {
        if (pre.binding != nullptr) {
            pre.api->ClearBoundInputs(pre.binding);
        }
        std::lock_guard<std::mutex> lock(g_vision_pixels_mutex);
        releaseDynamicVideoInputLocked();
    }
    return visOut;
}

// Transfer a matching cached hidden to this turn, or encode inline. The mutex serializes every vision
// session user and ensures stale cached output is released before another encode.
static std::vector<OrtValue*> getOrEncodeVisionOutputs(uint8_t visionMode) {
    const uint32_t seq = g_vision_capture_seq.load(std::memory_order_acquire);
    std::lock_guard<std::mutex> lock(g_vision_encode_mutex);
    LOGI("Vision cache handoff: request=%u seq=%u ready=%d cached_mode=%u cached_seq=%u outputs=%zu",
         static_cast<unsigned>(visionMode), seq, g_vision_hidden_ready ? 1 : 0,
         static_cast<unsigned>(g_vision_hidden_modality), g_vision_hidden_seq,
         g_pending_vision_outputs.size());
    if (g_vision_hidden_ready && g_vision_hidden_modality == visionMode &&
        g_vision_hidden_seq == seq && !g_pending_vision_outputs.empty()) {
        std::vector<OrtValue*> outputs = std::move(g_pending_vision_outputs);
        g_vision_hidden_ready    = false;
        g_vision_hidden_modality = VISION_NONE;
        if (visionMode == VISION_VIDEO && !g_video_fixed_length_input) {
            std::lock_guard<std::mutex> pixelLock(g_vision_pixels_mutex);
            releaseDynamicVideoInputLocked();
        }
        return outputs;
    }
    releasePendingVisionHiddenLocked();   // drop any stale pre-encode, then compute fresh inline
    return runVisionEncodeOnly(visionMode);
}

// Best-effort query-independent ViT pre-encode. A miss or race falls back to inline encoding.
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Prewarm_1Vision(JNIEnv* env, jclass /*clazz*/) {
    (void)env;
    const uint8_t mode = g_pending_vision.load(std::memory_order_acquire);
    if (mode == VISION_NONE) {
        return JNI_FALSE;
    }
    if ((mode == VISION_IMAGE && !g_vision_image_ready) ||
        (mode == VISION_VIDEO && !g_vision_video_ready)) {
        return JNI_FALSE;
    }
    if (g_busy.load(std::memory_order_acquire)) {
        return JNI_FALSE;   // a generation owns the vision sessions; the turn will encode inline
    }
    const uint32_t seq = g_vision_capture_seq.load(std::memory_order_acquire);
    std::lock_guard<std::mutex> lock(g_vision_encode_mutex);
    // Re-check under the lock: Run_LLM exchanges the pending flag to NONE at turn start, so if the user
    // already sent the query there is nothing left to pre-encode (the turn is encoding it inline).
    if (g_pending_vision.load(std::memory_order_acquire) != mode) {
        return JNI_FALSE;
    }
    if (g_vision_hidden_ready && g_vision_hidden_modality == mode && g_vision_hidden_seq == seq &&
        !g_pending_vision_outputs.empty()) {
        return JNI_TRUE;   // already pre-encoded for exactly this capture
    }
    releasePendingVisionHiddenLocked();   // drop any stale encode from an earlier capture
    std::vector<OrtValue*> outputs = runVisionEncodeOnly(mode);
    if (!outputs.empty()) {
        if (visionCancelRequested()) {
            releaseValues(mMain.api, outputs);
            return JNI_FALSE;
        }
        g_pending_vision_outputs = std::move(outputs);
        g_vision_hidden_modality = mode;
        g_vision_hidden_seq      = seq;
        g_vision_hidden_ready    = true;
        return JNI_TRUE;
    }
    return JNI_FALSE;
}

// Build the vision content token span inserted between the user prefix and the query, replicating the
// Python build_image_prompt / build_video_prompt exactly (special tokens are atomic so they act as BPE
// merge barriers; only free text — the video timestamps — is encoded, matching the text path's existing
// piece-wise tokenization). num_images == g_vision_batch_size for the fixed-slot image concat contract.
static std::vector<int> buildVisionContentIds(uint8_t visionMode) {
    std::vector<int> ids;
    if (visionMode == VISION_IMAGE) {
        for (int n = 0; n < std::max(1, g_vision_batch_size); ++n) {
            ids.push_back(chat_vision_start_id);
            ids.push_back(chat_vision_end_id);
        }
    } else if (visionMode == VISION_VIDEO) {
        std::vector<std::vector<int>> timestampTokenCache;
        {
            std::lock_guard<std::mutex> tokenizerLock(g_tokenizer_mutex);
            if (g_video_timestamp_cache_grid_t != g_video_grid_t ||
                g_video_timestamp_cache_temporal_patch != g_temporal_patch_size ||
                g_video_timestamp_cache_fps != g_video_fps) {
                g_video_timestamp_token_cache.clear();
                g_video_timestamp_token_cache.reserve(static_cast<size_t>(std::max(0, g_video_grid_t)));
                for (int f = 0; f < g_video_grid_t; ++f) {
                    const double ts = static_cast<double>(f) * g_temporal_patch_size /
                                      std::max(g_video_fps, 1e-6f);
                    char tsbuf[32];
                    std::snprintf(tsbuf, sizeof(tsbuf), "<%.1f seconds>", ts);
                    g_video_timestamp_token_cache.push_back(tokenizer->encode(tsbuf));
                }
                g_video_timestamp_cache_grid_t = g_video_grid_t;
                g_video_timestamp_cache_temporal_patch = g_temporal_patch_size;
                g_video_timestamp_cache_fps = g_video_fps;
            }
            timestampTokenCache = g_video_timestamp_token_cache;
        }
        for (int f = 0; f < g_video_grid_t; ++f) {
            if (f < static_cast<int>(timestampTokenCache.size())) {
                const std::vector<int>& tsIds = timestampTokenCache[static_cast<size_t>(f)];
                ids.insert(ids.end(), tsIds.begin(), tsIds.end());
            }
            ids.push_back(chat_vision_start_id);
            for (int p = 0; p < g_video_frame_seqlen; ++p) {
                ids.push_back(chat_video_token_id);
            }
            ids.push_back(chat_vision_end_id);
        }
    }
    return ids;
}

// Must match THINK_OPEN / THINK_CLOSE in ChatAdapter.java.
static constexpr char kThinkOpenSentinel[]  = "\x02\x02think\x03\x03";
static constexpr char kThinkCloseSentinel[] = "\x02\x02/think\x03\x03";

inline static void buffer_token(int id) {
    if (id == chat_think_start_id) {
        reset_output_words_decoder();
        stream_buf.append(kThinkOpenSentinel);
    } else if (id == chat_think_end_id) {
        reset_output_words_decoder();
        stream_buf.append(kThinkCloseSentinel);
    } else {
        // No tokenizer lock here: decode runs under g_busy (one generation at a time) and decode_id() is
        // a pure const read of the immutable decoder table. That table is (re)built only by Pre_Process
        // at load time, which never overlaps a decode, so the per-token lock was pure overhead.
        append_output_words(stream_buf, id);
    }
    tokens_since_flush += 1;
}

inline static void stream_token(JNIEnv* env, int id) {
    buffer_token(id);
    if (tokens_since_flush >= STREAM_BATCH || (now_ms() - last_flush_ms) >= STREAM_FLUSH_MS) {
        flush_stream(env);
    }
}

static void pushPerfStats(JNIEnv* env, float prefillRate, float decodeRate,
                          int remainingTokens, int capacityTokens,
                          int prefillTokens, int decodeTokens, bool finalSample) {
    if (g_main_cls == nullptr || g_on_perf == nullptr) {
        return;
    }
    jvalue args[7]{};
    args[0].f = prefillRate;
    args[1].f = decodeRate;
    args[2].i = remainingTokens;
    args[3].i = capacityTokens;
    args[4].i = prefillTokens;
    args[5].i = decodeTokens;
    args[6].z = finalSample ? JNI_TRUE : JNI_FALSE;
    env->CallStaticVoidMethodA(g_main_cls, g_on_perf, args);
    if (env->ExceptionCheck()) {
        env->ExceptionClear();
    }
}

static void clearBenchmarkCorrectness() {
    std::lock_guard<std::mutex> lock(g_benchmark_correctness_mutex);
    g_last_benchmark_correctness = "{}";
}

static void setBenchmarkCorrectness(const std::vector<int>& tokenIds,
                                    int generated,
                                    int64_t initialKvSeq,
                                    int64_t finalKvSeq) {
    std::string json;
    json.reserve(96 + tokenIds.size() * 8);
    json.append("{\"token_ids\":[");
    for (size_t index = 0; index < tokenIds.size(); ++index) {
        if (index != 0) {
            json.push_back(',');
        }
        json.append(std::to_string(tokenIds[index]));
    }
    json.append("],\"generated_tokens\":");
    json.append(std::to_string(generated));
    json.append(",\"initial_kv_seq\":");
    json.append(std::to_string(initialKvSeq));
    json.append(",\"final_kv_seq\":");
    json.append(std::to_string(finalKvSeq));
    json.push_back('}');

    std::lock_guard<std::mutex> lock(g_benchmark_correctness_mutex);
    g_last_benchmark_correctness = std::move(json);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_Get_1Last_1Benchmark_1Correctness(JNIEnv* env, jclass clazz) {
    (void)clazz;
    std::lock_guard<std::mutex> lock(g_benchmark_correctness_mutex);
    return env->NewStringUTF(g_last_benchmark_correctness.c_str());
}

// Decode strategy.
struct DecodeStrategy {
    bool sampling = false;
    int  topK = 1;
    float repeatPenalty = DEFAULT_REPEAT_PENALTY;
    int  penaltyRange = DEFAULT_PENALTY_RANGE;
    float temperature = DEFAULT_TEMPERATURE;
    float topP = DEFAULT_TOP_P;
};

static DecodeStrategy resolveDecodeStrategy() {
    DecodeStrategy s;
    std::lock_guard<std::mutex> lock(g_runtime_config_mutex);
    s.sampling = decodeMode == DECODE_MODE_SAMPLING;
    s.topK = clampInt(topK, 1, MAX_SAMPLING_TOP_K_RUNTIME);
    s.repeatPenalty = s.sampling
        ? std::max(1.0f, std::min(repeatPenalty, MAX_SAMPLING_PENALTY))
        : std::max(0.0f, std::min(repeatPenalty, 1.0f));
    s.penaltyRange = clampInt(penaltyRange, 1, MAX_PENALTY_RANGE_RUNTIME);
    s.temperature = std::max(MIN_TEMPERATURE_RUNTIME,
        std::min(temperature, MAX_TEMPERATURE_RUNTIME));
    s.topP = std::max(MIN_TOP_P_RUNTIME, std::min(topP, MAX_TOP_P_RUNTIME));
    return s;
}

// JNI: decode config.
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Configure_1LLM(JNIEnv* env, jclass clazz,
                                                           jint decode_mode,
                                                           jint top_k,
                                                           jfloat repeat_penalty,
                                                           jint penalty_range,
                                                           jfloat sampling_temperature,
                                                           jfloat top_p) {
    (void)clazz;
    if (decode_mode != DECODE_MODE_GREEDY && decode_mode != DECODE_MODE_SAMPLING) {
        LOGE("LLM decode config rejected invalid mode=%d", static_cast<int>(decode_mode));
        return JNI_FALSE;
    }
    const int cfgDecodeMode = static_cast<int>(decode_mode);
    const bool cfgUseSampling = cfgDecodeMode == DECODE_MODE_SAMPLING;
    const int cfgTopK = clampInt(top_k, 1, MAX_SAMPLING_TOP_K_RUNTIME);
    const float cfgRepeatPenalty = cfgUseSampling
            ? std::max(1.0f, std::min(repeat_penalty, MAX_SAMPLING_PENALTY))
            : std::max(0.0f, std::min(repeat_penalty, 1.0f));
    const int cfgPenaltyRange = clampInt(penalty_range, 1, MAX_PENALTY_RANGE_RUNTIME);
    const float cfgTemperature = std::max(MIN_TEMPERATURE_RUNTIME,
            std::min(sampling_temperature, MAX_TEMPERATURE_RUNTIME));
    const float cfgTopP = std::max(MIN_TOP_P_RUNTIME, std::min(top_p, MAX_TOP_P_RUNTIME));
    const bool cfgUsePenalty = !cfgUseSampling && cfgRepeatPenalty < DEFAULT_REPEAT_PENALTY;

    const std::array<MergedStrategy, 3> requestedStrategies =
            mergedStrategiesForConfig(cfgDecodeMode, cfgUsePenalty);
    bool needsSessionSwap = false;
    {
        std::lock_guard<std::mutex> modelLock(g_model_session_mutex);
        needsSessionSwap = g_model_sessions_initialized &&
                requestedStrategies[0] != g_loaded_merged_strategies[0];
    }
    bool ownsBusy = false;
    if (needsSessionSwap) {
        if (g_busy.exchange(true, std::memory_order_acq_rel)) {
            LOGE("LLM decode strategy change rejected while inference or maintenance is busy");
            return JNI_FALSE;
        }
        ownsBusy = true;
        waitForCleanCommit();
    }
    struct BusyGuard {
        bool armed;
        ~BusyGuard() {
            if (armed) {
                g_busy.store(false, std::memory_order_release);
            }
        }
    } busyGuard{ownsBusy};

    if (needsSessionSwap) {
        std::lock_guard<std::mutex> visionLock(g_vision_encode_mutex);
        std::lock_guard<std::mutex> modelLock(g_model_session_mutex);
        // Strategy edits must not rebuild every resident image/video graph while a large visual KV
        // window is live. Keep those pairs until the next visual turn clears that window; only the text
        // pair is required immediately for follow-up text turns.
        std::array<MergedStrategy, 3> residentStrategies = requestedStrategies;
        residentStrategies[1] = g_loaded_merged_strategies[1];
        residentStrategies[2] = g_loaded_merged_strategies[2];
        if (g_model_sessions_initialized &&
            requestedStrategies[0] != g_loaded_merged_strategies[0] &&
            !ensureMergedStrategySessions(env, residentStrategies,
                                          g_vision_sessions_initialized ? metadataVisionCaps() : 0)) {
            LOGE("LLM decode config rejected because its merged graph pair could not be loaded");
            return JNI_FALSE;
        }
    }
    {
        std::lock_guard<std::mutex> configLock(g_runtime_config_mutex);
        decodeMode = cfgDecodeMode;
        topK = cfgTopK;
        repeatPenalty = cfgRepeatPenalty;
        penaltyRange = cfgPenaltyRange;
        temperature = cfgTemperature;
        topP = cfgTopP;
        g_config_merged_strategies = requestedStrategies;
    }
    LOGI("LLM decode config mode=%d sampling=%d penalty=%d topK=%d "
         "repeatPenalty=%.3f penaltyRange=%d temperature=%.3f topP=%.3f",
         cfgDecodeMode,
         cfgUseSampling ? 1 : 0,
         cfgUsePenalty ? 1 : 0,
         cfgTopK,
         cfgRepeatPenalty,
         cfgPenaltyRange,
         cfgTemperature,
         cfgTopP);
    return JNI_TRUE;
}

// Organize-memory toggle takes effect on the next turn.
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Configure_1Organize_1Memory(JNIEnv* env, jclass clazz,
                                                                       jboolean enable) {
    (void)env;
    (void)clazz;
    const bool organize = (enable == JNI_TRUE);
    {
        std::lock_guard<std::mutex> lock(g_runtime_config_mutex);
        g_organize_memory = organize;
    }
    LOGI("LLM organize-memory config: %d", organize ? 1 : 0);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Configure_1Run_1Profiling(JNIEnv* env, jclass clazz,
                                                                      jstring profile_prefix,
                                                                      jboolean enable) {
    (void)clazz;
    std::string prefix;
    if (enable == JNI_TRUE) {
        if (profile_prefix == nullptr) {
            return JNI_FALSE;
        }
        if (!jStringToUtf8(env, profile_prefix, prefix)) {
            return JNI_FALSE;
        }
        if (prefix.empty()) {
            return JNI_FALSE;
        }
    }

    bool ok = true;
    for (ModelRuntime& runtime : gModelRuntimes) {
        if (runtime.api == nullptr || runtime.runOptions == nullptr) {
            continue;
        }
        if (enable == JNI_TRUE) {
            const std::string modelPrefix = prefix + "_model_" + std::to_string(runtime.id);
            ok = logOrtStatus(runtime.api,
                              runtime.api->RunOptionsEnableProfiling(runtime.runOptions, modelPrefix.c_str()),
                              "RunOptionsEnableProfiling") && ok;
        } else {
            ok = logOrtStatus(runtime.api,
                              runtime.api->RunOptionsDisableProfiling(runtime.runOptions),
                              "RunOptionsDisableProfiling") && ok;
        }
    }
    return ok ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Configure_1Memory(JNIEnv* env, jclass clazz,
                                                              jint memory_tokens,
                                                              jint prefill_tokens,
                                                              jint decode_tokens) {
    (void)env;
    (void)clazz;
    const MemoryProfile p = makeMemoryProfile(memory_tokens, prefill_tokens, decode_tokens);
    {
        std::lock_guard<std::mutex> lock(g_runtime_config_mutex);
        applyMemoryProfileLocked(p);
    }
    LOGI("LLM memory config memory=%d, prefill=%d, decode=%d",
         p.memoryTokens, p.prefillTokens, p.decodeTokens);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_example_myapplication_MainActivity_Get_1Memory_1Limits(JNIEnv* env, jclass clazz) {
    (void)clazz;
    const jint values[] = {
            MIN_MEMORY_TOKENS, MAX_MEMORY_TOKENS_RUNTIME, DEFAULT_MEMORY_TOKENS,
            MIN_PREFILL_TOKENS, MAX_PREFILL_TOKENS_RUNTIME, DEFAULT_PREFILL_TOKENS,
            MIN_DECODE_TOKENS, MAX_DECODE_TOKENS_RUNTIME, DEFAULT_MAX_DECODE_TOKENS
    };
    constexpr jsize valueCount = static_cast<jsize>(sizeof(values) / sizeof(values[0]));
    jintArray out = env->NewIntArray(valueCount);
    if (out != nullptr) {
        env->SetIntArrayRegion(out, 0, valueCount, values);
    }
    return out;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Supports_1Prefill_1Lookback(
        JNIEnv* env, jclass clazz) {
    (void)env;
    (void)clazz;
    const bool supported = num_linear_attention_layers == 0 &&
            mKVSplit2.session != nullptr && mKVConcat.session != nullptr;
    return supported ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Supports_1Thinking(JNIEnv* env, jclass clazz) {
    (void)env;
    (void)clazz;
    return g_supports_thinking ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Configure_1Memory_1Thresholds(JNIEnv* env, jclass clazz,
                                                                          jint red_percent,
                                                                          jint green_percent) {
    (void)env;
    (void)clazz;
    const int red = clampInt(red_percent,
                             MIN_MEMORY_BAND_PERCENT + MIN_MEMORY_BAND_GAP, MAX_MEMORY_BAND_PERCENT);
    const int green = clampInt(green_percent,
                               MIN_MEMORY_BAND_PERCENT, red - MIN_MEMORY_BAND_GAP);
    {
        std::lock_guard<std::mutex> lock(g_runtime_config_mutex);
        g_memory_red_zone_percent.store(red, std::memory_order_relaxed);
        g_memory_green_target_percent.store(green, std::memory_order_relaxed);
    }
    LOGI("LLM memory thresholds: red=%d%% green=%d%%", red, green);
    return JNI_TRUE;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_example_myapplication_MainActivity_Get_1Memory_1Thresholds(JNIEnv* env, jclass clazz) {
    (void)clazz;
    jint values[2];
    {
        std::lock_guard<std::mutex> lock(g_runtime_config_mutex);
        values[0] = g_memory_red_zone_percent.load(std::memory_order_relaxed);
        values[1] = g_memory_green_target_percent.load(std::memory_order_relaxed);
    }
    jintArray out = env->NewIntArray(2);
    if (out != nullptr) {
        env->SetIntArrayRegion(out, 0, 2, values);
    }
    return out;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_example_myapplication_MainActivity_Get_1Memory_1Stats(JNIEnv* env, jclass clazz) {
    (void)clazz;
    if (!g_busy.load(std::memory_order_acquire)) {
        waitForCleanCommit();
    }
    const MemoryProfile memory = snapshotMemoryProfile();
    const int capacity = std::max(1, memory.memoryTokens);
    const int used = static_cast<int>(std::max<int64_t>(
            0, std::min<int64_t>(g_memory_used_tokens.load(std::memory_order_acquire),
                                  static_cast<int64_t>(capacity))));
    const int remaining = std::max(0, capacity - used);
    const int remainingPercent = (remaining == capacity)
            ? 100
            : clampInt(static_cast<int>((static_cast<int64_t>(remaining) * 100) / capacity), 0, 99);
    const jint values[] = { used, remaining, remainingPercent, capacity };
    constexpr jsize valueCount = static_cast<jsize>(sizeof(values) / sizeof(values[0]));
    jintArray out = env->NewIntArray(valueCount);
    if (out != nullptr) {
        env->SetIntArrayRegion(out, 0, valueCount, values);
    }
    return out;
}

// JNI: tokenizer setup.
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Pre_1Process(JNIEnv* /*env*/, jobject /*clazz*/) {
    if (tokenizer != nullptr) {
        delete tokenizer;
        tokenizer = nullptr;
    }
    {
        std::lock_guard<std::mutex> tokenizerLock(g_tokenizer_mutex);
        // Auto-discover the vocab table (vocab_*.txt, case-insensitive) instead of forcing an exact name
        // match, so the exported vocab can keep its model-specific file name.
        const std::string resolvedVocab = resolveVocabPath();
        LOGI("Pre_Process: loading tokenizer vocab from %s", resolvedVocab.c_str());
        tokenizer = Tokenizer::createTokenizer(resolvedVocab);
    }
    if (tokenizer == nullptr) {
        LOGE("Pre_Process: tokenizer creation failed (no readable vocab_*.txt found)");
        return JNI_FALSE;
    }
    {
        std::lock_guard<std::mutex> lock(g_system_prompt_mutex);
        if (g_system_prompt_text.empty()) {
            g_system_prompt_text = DEFAULT_SYSTEM_PROMPT;
        }
        rebuild_system_prompt_ids_locked();
    }
    {
        std::lock_guard<std::mutex> tokenizerLock(g_tokenizer_mutex);
        g_manual_stop_notice_ids = tokenizer->encode(MANUAL_STOP_NOTICE);
        g_vision_image_note_ids = tokenizer->encode(VISION_IMAGE_MEMORY_NOTE);
        g_vision_video_note_ids = tokenizer->encode(VISION_VIDEO_MEMORY_NOTE);
        g_video_timestamp_token_cache.clear();
        g_video_timestamp_cache_grid_t = -1;
        g_video_timestamp_cache_temporal_patch = -1;
        g_video_timestamp_cache_fps = -1.0f;
    }
    return JNI_TRUE;
}

// JNI: slot system prompt.
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Set_1System_1Prompt(JNIEnv* env, jclass /*clazz*/,
                                                                jstring system_prompt) {
    if (tokenizer == nullptr) {
        return JNI_FALSE;   // tokenizer not ready (Pre_Process must run first)
    }
    std::string newPrompt;
    if (system_prompt == nullptr) {
        newPrompt.clear();
    } else if (!jStringToUtf8(env, system_prompt, newPrompt)) {
        return JNI_FALSE;
    }
    if (is_blank_system_prompt(newPrompt)) {
        newPrompt.clear();
    }
    size_t bodyTokens = 0;
    int64_t blockLen = 0;
    {
        std::lock_guard<std::mutex> lock(g_system_prompt_mutex);
        g_system_prompt_text = std::move(newPrompt);
        rebuild_system_prompt_ids_locked();
        bodyTokens = g_system_prompt_ids.size();
        blockLen = g_system_block_len;
    }
    LOGI("Slot system prompt set: %zu body tokens, block len %lld",
         bodyTokens, static_cast<long long>(blockLen));
    return JNI_TRUE;
}

static jstring makeRunResult(JNIEnv* env, const char* category, const char* code) {
    std::string payload(category);
    payload.push_back('|');
    payload.append(code);
    return newJStringFromUtf8(env, payload);
}

static jstring runError(JNIEnv* env, const char* code) {
    LOGE("FLOW native_error turn=%d code=%s", g_flow_turn_id, code);
    return makeRunResult(env, "LLM_ERROR", code);
}

static jstring runCancelled(JNIEnv* env) {
    LOGI("FLOW native_cancelled turn=%d", g_flow_turn_id);
    return makeRunResult(env, "LLM_STATUS", "CANCELLED");
}

// JNI: full generation.
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_Run_1LLM(JNIEnv* env, jclass /*clazz*/, jstring jquery,
                                                     jboolean clear, jboolean use_think, jint turn_id) {
    g_flow_turn_id = static_cast<int>(turn_id);
    g_flow_flush_index = 0;
    const bool requestedThink = use_think == JNI_TRUE;
    const bool effectiveThink = requestedThink && g_supports_thinking;
    if (g_busy.exchange(true, std::memory_order_acq_rel)) {
        LOGE("FLOW native_busy turn=%d", static_cast<int>(turn_id));
        return runError(env, "BUSY");
    }
    struct BusyGuard { ~BusyGuard() { g_busy.store(false, std::memory_order_release); } } busy_guard;
    struct TerminateGuard { ~TerminateGuard() { setRuntimeTerminate(false); } } terminate_guard;
    clearBenchmarkCorrectness();
    const bool pendingPrestartCancel = g_pending_prestart_cancel.exchange(false, std::memory_order_acq_rel);
    const bool pendingPrestartManualStop = g_pending_prestart_manual_stop.exchange(false, std::memory_order_acq_rel);
    g_cancel.store(false, std::memory_order_relaxed);
    g_manual_stop.store(false, std::memory_order_relaxed);
    LOGI("FLOW native_start turn=%d clear=%d think_requested=%d think_effective=%d "
         "thinking_supported=%d prestart_cancel=%d prestart_manual=%d",
         static_cast<int>(turn_id), clear == JNI_TRUE ? 1 : 0, requestedThink ? 1 : 0,
         effectiveThink ? 1 : 0, g_supports_thinking ? 1 : 0,
         pendingPrestartCancel ? 1 : 0, pendingPrestartManualStop ? 1 : 0);
    if (pendingPrestartCancel) {
        if (pendingPrestartManualStop) {
            LOGI("Run_LLM: consumed manual pre-start stop before native decode began");
        }
        return runCancelled(env);
    }
    waitForCleanCommit();
    if (g_cancel.load(std::memory_order_relaxed)) {
        return runCancelled(env);
    }

    const OrtApi* api = mMain.api;
    // Merged runtime readiness: the primary merged graph (text greedy decode) anchors the api/allocator and
    // its sibling prefill graph must be present. Rotary/embed/head are all fused into the merged graphs.
    if (api == nullptr || mMain.session == nullptr ||
        !mergedModelReady(mergedPrefillModel(0, MSTRAT_GREEDY)) || tokenizer == nullptr) {
        return runError(env, "NOT_READY");
    }
    if (jquery == nullptr) {
        return runError(env, "INVALID_INPUT");
    }

    // Turn setup.
    // Consume any pending vision input (image/video) handed over by the camera pipeline. A vision turn
    // begins a fresh visual context: force-clear history so the vision prefill runs from history_len=0,
    // exactly matching the single-turn Python reference (build_image_prompt / build_video_prompt).
    uint8_t visionMode = VISION_NONE;
    {
        const uint8_t pend = g_pending_vision.exchange(VISION_NONE, std::memory_order_acq_rel);
        if (pend == VISION_IMAGE && g_vision_image_ready) {
            visionMode = VISION_IMAGE;
        } else if (pend == VISION_VIDEO && g_vision_video_ready) {
            visionMode = VISION_VIDEO;
        }
    }
    const bool isVisionTurn = (visionMode != VISION_NONE);

    if (clear == JNI_TRUE || isVisionTurn) {
        // The pending modality was already consumed above. Keep its matching speculative ViT output
        // across this mandatory fresh-context reset so getOrEncodeVisionOutputs can take ownership.
        clear_history(isVisionTurn ? "vision_turn_fresh_context" : "java_clear_flag", isVisionTurn);
    }
    const MemoryProfile memory = snapshotMemoryProfile();
    bool organizeMemory;
    {
        std::lock_guard<std::mutex> lock(g_runtime_config_mutex);
        organizeMemory = g_organize_memory;
    }
    // Organize-memory ON intentionally replaces unrebuildable vision embeddings with the stored
    // natural-language note after generation, then rebuilds both KV and linear state from those ids.
    // OFF keeps the live spliced vision state for follow-up turns.
    if (!saved_kv_has_vision && saved_kv_len > memory.memoryTokens) {
        if (!rebuildSavedKVFromHistory(memory.memoryTokens)) {
            clear_history("memory_cap_rebuild_failed");
        }
    }
    std::string query;
    if (!jStringToUtf8(env, jquery, query)) {
        return runError(env, "INPUT_DECODE");
    }
    std::vector<int> queryIds;
    {
        std::lock_guard<std::mutex> tokenizerLock(g_tokenizer_mutex);
        const size_t queryTokenLimit = static_cast<size_t>(std::max(1, max_seq_len)) + 1;
        queryIds = tokenizer->encode(query, queryTokenLimit);
    }
    if (queryIds.size() > static_cast<size_t>(max_seq_len)) {
        return runError(env, "INPUT_TOO_LONG");
    }
    // In OFF mode the system block becomes part of the retained KV once, then is not appended again.
    // Organize-memory strips it from each clean rebuild, so ON injects it on every live turn. Vision
    // turns use the same policy instead of silently dropping the configured system instruction.
    std::vector<int> activeSystemBlock;
    int64_t activeSystemBlockLen = 0;
    {
        std::lock_guard<std::mutex> lock(g_system_prompt_mutex);
        if (g_system_block_len > 0 && !g_system_prompt_ids.empty()) {
            activeSystemBlockLen = g_system_block_len;
            activeSystemBlock.reserve(static_cast<size_t>(activeSystemBlockLen));
            activeSystemBlock.insert(activeSystemBlock.end(), chat_system_prefix_ids.begin(), chat_system_prefix_ids.end());
            activeSystemBlock.insert(activeSystemBlock.end(), g_system_prompt_ids.begin(), g_system_prompt_ids.end());
            activeSystemBlock.insert(activeSystemBlock.end(), chat_system_suffix_ids.begin(), chat_system_suffix_ids.end());
        }
    }
    const bool injectSystem = activeSystemBlockLen > 0 &&
            (organizeMemory || saved_kv_len == 0);

    // Build the live prompt once in final order. This avoids repeatedly shifting a long video prompt
    // for vision, user and system prefixes.
    const std::vector<int> visionContent = isVisionTurn
            ? buildVisionContentIds(visionMode) : std::vector<int>{};
        const size_t thinkSuffixLen = effectiveThink ? chat_think_prefix_ids.size() : 0;
    const size_t commonPromptSize = (injectSystem ? activeSystemBlock.size() : 0) +
            chat_user_prefix_ids.size() + queryIds.size() + chat_assistant_prefix_ids.size() +
            thinkSuffixLen;
    std::vector<int> get_ids;
    get_ids.reserve(commonPromptSize + visionContent.size());
    if (injectSystem) {
        get_ids.insert(get_ids.end(), activeSystemBlock.begin(), activeSystemBlock.end());
    }
    get_ids.insert(get_ids.end(), chat_user_prefix_ids.begin(), chat_user_prefix_ids.end());
    get_ids.insert(get_ids.end(), visionContent.begin(), visionContent.end());
    get_ids.insert(get_ids.end(), queryIds.begin(), queryIds.end());
    get_ids.insert(get_ids.end(), chat_assistant_prefix_ids.begin(), chat_assistant_prefix_ids.end());
    if (effectiveThink) {
        get_ids.insert(get_ids.end(), chat_think_prefix_ids.begin(), chat_think_prefix_ids.end());
    }

    // Stored history replaces non-rebuildable vision embeddings with a text note. Assemble that variant
    // independently in final order so neither prompt needs front insertion.
    std::vector<int> visionHistoryPromptIds;
    if (isVisionTurn) {
        std::vector<int> noteIds;
        {
            std::lock_guard<std::mutex> tokenizerLock(g_tokenizer_mutex);
            const std::vector<int>& cachedNote = (visionMode == VISION_IMAGE)
                    ? g_vision_image_note_ids : g_vision_video_note_ids;
            noteIds = !cachedNote.empty()
                    ? cachedNote
                    : tokenizer->encode(visionMode == VISION_IMAGE ? VISION_IMAGE_MEMORY_NOTE
                                                                   : VISION_VIDEO_MEMORY_NOTE);
        }
        visionHistoryPromptIds.reserve(commonPromptSize + noteIds.size());
        if (injectSystem) {
            visionHistoryPromptIds.insert(visionHistoryPromptIds.end(),
                                          activeSystemBlock.begin(), activeSystemBlock.end());
        }
        visionHistoryPromptIds.insert(visionHistoryPromptIds.end(),
                                      chat_user_prefix_ids.begin(), chat_user_prefix_ids.end());
        visionHistoryPromptIds.insert(visionHistoryPromptIds.end(), noteIds.begin(), noteIds.end());
        visionHistoryPromptIds.insert(visionHistoryPromptIds.end(), queryIds.begin(), queryIds.end());
        visionHistoryPromptIds.insert(visionHistoryPromptIds.end(),
                                      chat_assistant_prefix_ids.begin(), chat_assistant_prefix_ids.end());
        if (effectiveThink) {
            visionHistoryPromptIds.insert(visionHistoryPromptIds.end(),
                                          chat_think_prefix_ids.begin(),
                                          chat_think_prefix_ids.end());
        }
    }
    const DecodeStrategy strat = resolveDecodeStrategy();
    const bool useSampling = strat.sampling;
    // OFF mode direct-saves live KV; Organize Memory rebuilds from ids and substitutes a vision note.
    const bool rebuildCleanStateFromIds = organizeMemory;
    std::vector<int> dialogueBaseIds;
    if (rebuildCleanStateFromIds) {
        const int64_t dialogueSkip = (injectSystem && organizeMemory) ? activeSystemBlockLen : 0;
        const std::vector<int>& dialogueSourceIds = isVisionTurn ? visionHistoryPromptIds : get_ids;
        dialogueBaseIds.assign(dialogueSourceIds.begin() + dialogueSkip, dialogueSourceIds.end());
    }
    reserveSharedArena(memory.memoryTokens);
    const int  topK = strat.topK;
    const float repeatPenalty = strat.repeatPenalty;
    const int  penaltyRange = strat.penaltyRange;
    const float samplingTemperature = strat.temperature;
    const float samplingTopP = strat.topP;

    // mRoPE positions (history_pos) and the physical KV length diverge once a vision segment is cached,
    // so history_pos tracks the mRoPE
    // position (fed as history_len to the rotary prefill) while saved_kv_len stays the physical count.
    int64_t history_pos = saved_kv_has_vision ? saved_mrope_pos : (saved_kv_base + saved_kv_len);
    constexpr int64_t decodeBudgetMargin = 2;
    const int prev_close_len = static_cast<int>(chat_previous_assistant_close_ids.size());
    const int sep_len = (saved_kv_len > 0) ? prev_close_len : 0;
    const auto core_len = static_cast<int64_t>(get_ids.size());
    const int64_t prompt_span = static_cast<int64_t>(sep_len) + core_len;
    // Streaming pure-text decode relies on hard RoPE recycle: it slides the window mid-decode, so it
    // reserves only the fixed recycle headroom here. Hybrid linear attention cannot slide, but its decode is
    // separately capped to the remaining rotary budget at decode time (decodeBudget =
    // min(rotaryDecodeBudget, decodeTokens)); reserving the FULL configured decode up front would force
    // clearing history that otherwise fits — a small history under a max-length decode setting cleared
    // the whole conversation every turn. Cap their reserve to the room left AFTER the retained history so
    // the guard below only trims when history + prompt alone overflow the table; the decode then adapts.
    const bool recyclerDrivesDecode = !hasLinearState() &&
            mKVSlice.session != nullptr && mRopeShift.session != nullptr;
    const int64_t rotaryBudgetRoom = static_cast<int64_t>(max_seq_len) - prompt_span - decodeBudgetMargin;
    const int64_t decode_reserve = std::max<int64_t>(0,
            recyclerDrivesDecode
                ? std::min<int64_t>(static_cast<int64_t>(kDecodeRecycleHeadroom), rotaryBudgetRoom)
                : std::min<int64_t>(static_cast<int64_t>(memory.decodeTokens), rotaryBudgetRoom - history_pos));
    const int64_t turn_room = prompt_span + decode_reserve + decodeBudgetMargin;

    // Reclaim absolute RoPE positions before prefill when needed.
    if (history_pos + turn_room > max_seq_len) {
        const int64_t targetLen = static_cast<int64_t>(max_seq_len) - turn_room;
        if (hasLinearState()) {
            // Hybrid: can't slice/rope-shift the linear recurrent state; rebuild the retained window
            // from ids so the full-attention KV and the linear state stay consistent.
            if (targetLen > 0 && rebuildSavedKVFromHistoryRetaining(static_cast<int>(targetLen))) {
                history_pos = saved_kv_base + saved_kv_len;
            }
        } else {
            if (targetLen > 0 && saved_kv_len + turn_room > max_seq_len) {
                if (shrinkSavedKVForRotaryBudget(targetLen)) {
                    history_pos = saved_kv_base + saved_kv_len;
                }
            }
            if (saved_kv_len + turn_room <= max_seq_len && renumberSavedKV()) {
                history_pos = saved_kv_base + saved_kv_len;
            }
        }
        if (history_pos + turn_room > max_seq_len) {
            clear_history("rotary_budget_overflow");
            history_pos = 0;
            if (core_len + decodeBudgetMargin > max_seq_len) {
                return runError(env, "INPUT_TOO_LONG");
            }
        }
    }
    // Trim history before prefill if prompt + retained KV enters the red zone.
    if (saved_kv_len > 0 && !saved_kv_has_vision) {
        const int64_t livePromptLen = core_len + prev_close_len;
        int64_t targetHistoryLen = static_cast<int64_t>(memory.memoryTokens) - livePromptLen;
        const int64_t liveUsedAfterPrompt = saved_kv_len + livePromptLen;
        if (memoryUsedInRedZone(memory.memoryTokens, liveUsedAfterPrompt)) {
            targetHistoryLen = static_cast<int64_t>(memoryGreenTargetUsed(memory.memoryTokens)) - livePromptLen;
        }
        if (targetHistoryLen <= 0) {
            clear_history("prompt_exceeds_memory_window");
            history_pos = 0;
        } else if (saved_kv_len > targetHistoryLen) {
            if (rebuildSavedKVFromHistoryRetaining(static_cast<int>(targetHistoryLen))) {
                history_pos = saved_kv_base + saved_kv_len;
            } else {
                clear_history("red_zone_rebuild_failed");
                history_pos = 0;
            }
        }
    }
    const bool reusedHistory = (saved_kv_len > 0);
    if (reusedHistory) {
        std::vector<int> promptWithHistoryClose;
        promptWithHistoryClose.reserve(chat_previous_assistant_close_ids.size() + get_ids.size());
        promptWithHistoryClose.insert(promptWithHistoryClose.end(),
                                      chat_previous_assistant_close_ids.begin(),
                                      chat_previous_assistant_close_ids.end());
        promptWithHistoryClose.insert(promptWithHistoryClose.end(), get_ids.begin(), get_ids.end());
        get_ids.swap(promptWithHistoryClose);
    }
    ids_len = static_cast<int64_t>(get_ids.size());
            LOGI("FLOW prompt turn=%d modality=%u query_ids=%zu prompt_ids=%lld tail=%s stops=%s "
                "saved_kv=%lld saved_vision=%d sampling=%d repeat_penalty=%.3f clear=%d",
            static_cast<int>(turn_id), static_cast<unsigned>(visionMode), queryIds.size(),
            static_cast<long long>(ids_len), formatTokenIdsTail(get_ids, 24).c_str(),
            formatTokenIdsTail(g_stop_token_ids, g_stop_token_ids.size()).c_str(),
            static_cast<long long>(saved_kv_len), saved_kv_has_vision ? 1 : 0,
                strat.sampling ? 1 : 0, strat.repeatPenalty,
                clear == JNI_TRUE ? 1 : 0);
    // Snapshot the turn-start restore point: history_pos is the mRoPE position fed to this turn's prefill
    // (diverges from saved_kv_len when the retained window holds a vision segment), and saved_kv_has_vision
    // records whether that window carries un-rebuildable vision KV so a later rollback can preserve it.
    record_turn_checkpoint(turn_id,
                           logicalHistoryEnd(), saved_kv_len,
                           visionMode, history_pos, saved_kv_has_vision);

    // ================================ MERGED GENERATION ================================
    // The merged export runs ONE prefill graph (Rotary+embed+Main+head fused) producing the KV/linear
    // state block AND the first token, then a decode graph per token. Selection is by (modality, strategy).
    // This path reuses the turn-setup above (get_ids, history_pos, memory, strategy, vision) and the KV-block
    // persistence (saveDecodeKVDirect / commitCleanTurn).
    {
        const OrtApi* mApi = mMain.api;
        const int modality = static_cast<int>(visionMode);   // 0 text, 1 image, 2 video
        const bool directPenalty = !useSampling && repeatPenalty < DEFAULT_REPEAT_PENALTY;
        MergedStrategy mstrat = directPenalty ? MSTRAT_PENALTY_GREEDY : MSTRAT_GREEDY;
        if (useSampling) {
            mstrat = MSTRAT_SAMPLING;
        }
        ModelId prefillId = mergedPrefillModel(modality, mstrat);
        ModelId decodeId  = mergedDecodeModel(modality, mstrat);
        if (!mergedModelReady(prefillId) || !mergedModelReady(decodeId)) {
            std::lock_guard<std::mutex> visionLock(g_vision_encode_mutex);
            std::lock_guard<std::mutex> modelLock(g_model_session_mutex);
            std::array<MergedStrategy, 3> turnStrategies = g_loaded_merged_strategies;
            turnStrategies[modality] = mstrat;
            const int turnVisionCap = isVisionTurn ? 1 << (modality - 1) : 0;
            if (!ensureMergedStrategySessions(env, turnStrategies, turnVisionCap)) {
                LOGE("Run_LLM(merged): could not load deferred strategy "
                     "(modality=%d strategy=%d)", modality, static_cast<int>(mstrat));
                return runError(env, "STRATEGY_UNAVAILABLE");
            }
            prefillId = mergedPrefillModel(modality, mstrat);
            decodeId = mergedDecodeModel(modality, mstrat);
        }
        if (!mergedModelReady(prefillId) || !mergedModelReady(decodeId)) {
            LOGE("Run_LLM(merged): selected graph pair is unavailable (modality=%d strategy=%d)",
                 modality, static_cast<int>(mstrat));
            return runError(env, "STRATEGY_UNAVAILABLE");
        }
        ModelRuntime& pg = getModel(prefillId);
        ModelRuntime& dg = getModel(decodeId);
        if (mApi == nullptr || pg.session == nullptr || dg.session == nullptr ||
            !ensureBinding(pg) || !ensureBinding(dg) || !ensureAlternateBinding(dg) ||
            tokenizer == nullptr) {
            return runError(env, "RUNTIME_UNAVAILABLE");
        }
        if (isVisionTurn) {
            LOGI("Vision generation route: modality=%s strategy=%d prefill=%s decode=%s expected_hidden=%d",
                 visionMode == VISION_VIDEO ? "video" : "image", static_cast<int>(mstrat),
                 pg.fileName.c_str(), dg.fileName.c_str(),
                 visionMode == VISION_VIDEO ? g_video_embed_size : g_image_embed_size);
        }
        const int pTokenIn = findInputIdxAny(pg, {"embed_input_ids", "input_ids"});
        const int dTokenIn = findInputIdxAny(dg, {"embed_input_ids", "input_ids"});
        const int dKvSeqIn = findInputIdxPrefix(dg, "decode_kv_seq_len");
        const int pIdsLenIn = findInputIdx(pg, "prefill_ids_len");
        const int pHistoryLenIn = findInputIdx(pg, "prefill_history_len");
        const int pCacheLenIn = findInputIdx(pg, "prefill_cache_len");
        std::vector<int> pVisionInputs;
        int pVisionHiddenOutput = -1;
        bool visionInputContractValid = !isVisionTurn;
        if (isVisionTurn) {
            const char* concatPrefix = visionMode == VISION_IMAGE
                    ? "concat_image_" : "concat_video_";
            pVisionInputs.assign(mVision.outputNames.size(), -1);
            visionInputContractValid = !pVisionInputs.empty() &&
                    mVision.outputTypes.size() == pVisionInputs.size();
            pVisionHiddenOutput = static_cast<int>(pVisionInputs.size()) - 1;
            for (size_t i = 0; i < pVisionInputs.size(); ++i) {
                const std::string inputName = static_cast<int>(i) == pVisionHiddenOutput
                        ? std::string(concatPrefix) + "vision_hidden_states"
                        : std::string(concatPrefix) + "in_deepstack_feature_" + std::to_string(i);
                const int inputIndex = findInputIdx(pg, inputName.c_str());
                pVisionInputs[i] = inputIndex;
                visionInputContractValid = inputIndex >= num_main_states &&
                        pg.inputTypes[inputIndex] == mVision.outputTypes[i] &&
                        visionInputContractValid;
            }
            visionInputContractValid = pVisionHiddenOutput >= 0 && visionInputContractValid;
        }
        if (pTokenIn < num_main_states || dTokenIn < num_main_states || dKvSeqIn < num_main_states ||
            pIdsLenIn < num_main_states || pHistoryLenIn < num_main_states ||
            (!isVisionTurn && pCacheLenIn < num_main_states) ||
            !visionInputContractValid ||
            !isIntegerTensorType(pg.inputTypes[pTokenIn]) ||
            !isIntegerTensorType(dg.inputTypes[dTokenIn]) ||
            !isIntegerTensorType(dg.inputTypes[dKvSeqIn]) ||
            !isIntegerTensorType(pg.inputTypes[pIdsLenIn]) ||
            !isIntegerTensorType(pg.inputTypes[pHistoryLenIn]) ||
            (pCacheLenIn >= 0 && !isIntegerTensorType(pg.inputTypes[pCacheLenIn]))) {
            LOGE("Run_LLM(merged): invalid phase input contract (prefill=%s token=%d ids=%d history=%d "
             "cache=%d vision_outputs=%zu; decode=%s token=%d kv=%d)", pg.fileName.c_str(), pTokenIn,
             pIdsLenIn, pHistoryLenIn, pCacheLenIn, pVisionInputs.size(),
             dg.fileName.c_str(), dTokenIn, dKvSeqIn);
            return runError(env, "MODEL_CONTRACT");
        }
        const bool isSampling = (mstrat == MSTRAT_SAMPLING);
        const bool isPenalty = (mstrat == MSTRAT_PENALTY_GREEDY);
        const bool requiresSaveId = isPenalty || isSampling;
        // Inverse of the outer-scope persistence route: OFF mode direct-saves live state.
        const bool offDirectSave = !rebuildCleanStateFromIds;
        const bool prefillConcatReady = mKVConcat.session != nullptr;
        const bool prefillDecoupleEligible =
            !isVisionTurn && !saved_kv_has_vision && !hasLinearState() &&
            reusedHistory && saved_kv[0] != nullptr && pCacheLenIn >= num_main_states &&
            mKVSplit2.session != nullptr && prefillConcatReady;
        const bool prefillZeroWindow = prefillDecoupleEligible && memory.prefillTokens == 0;
        const bool prefillRecentWindow = prefillDecoupleEligible && memory.prefillTokens > 0 &&
            saved_kv_len > static_cast<int64_t>(memory.prefillTokens) &&
            ids_len >= static_cast<int64_t>(memory.prefillDecoupleMinNewTokens);
        const bool prefillDecoupled = prefillZeroWindow || prefillRecentWindow;
        const int64_t prefillCacheLen = prefillZeroWindow
            ? 0
            : (prefillRecentWindow ? static_cast<int64_t>(memory.prefillTokens) : saved_kv_len);

        const int penaltyValueIn = findInputIdx(dg, "penalty_penalty_value");
        const int penaltyRangeIn = findInputIdx(dg, "penalty_penalty_range");
        const int samplingTemperatureIn = findInputIdx(dg, "sampling_temperature");
        const int samplingTopKIn = findInputIdx(dg, "sampling_top_k");
        const int samplingTopPIn = findInputIdx(dg, "sampling_top_p");
        const int samplingPenaltyIn = findInputIdx(dg, "sampling_repetition_penalty");
        auto hasInput = [](const ModelRuntime& graph, const char* name) {
            return findInputIdx(graph, name) >= 0;
        };
        auto inputTypeMatches = [](const ModelRuntime& graph, const char* name,
                                   ONNXTensorElementDataType expected) {
            const int idx = findInputIdx(graph, name);
            return idx < 0 || graph.inputTypes[idx] == expected;
        };
        ONNXTensorElementDataType saveIdType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        auto collectSaveIdInputs = [](const ModelRuntime& graph) {
            std::vector<int> indices;
            for (const char* name : {"penalty_save_id_in", "penalty_greedy_save_id_in",
                                     "sampling_previous_ids"}) {
                const int index = findInputIdx(graph, name);
                if (index >= 0) {
                    indices.push_back(index);
                }
            }
            return indices;
        };
        const std::vector<int> prefillSaveIdInputs = collectSaveIdInputs(pg);
        const std::vector<int> decodeSaveIdInputs = collectSaveIdInputs(dg);
        auto mergeSaveIdType = [&](const ModelRuntime& graph, int idx) {
            const ONNXTensorElementDataType type = graph.inputTypes[idx];
            if (!isIntegerTensorType(type)) {
                return false;
            }
            if (saveIdType == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
                saveIdType = type;
                return true;
            }
            return saveIdType == type;
        };
        bool saveIdTypesValid = true;
        for (int index : prefillSaveIdInputs) {
            saveIdTypesValid = mergeSaveIdType(pg, index) && saveIdTypesValid;
        }
        for (int index : decodeSaveIdInputs) {
            saveIdTypesValid = mergeSaveIdType(dg, index) && saveIdTypesValid;
        }
        bool strategyContractValid = true;
        switch (mstrat) {
            case MSTRAT_GREEDY:
                break;
            case MSTRAT_PENALTY_GREEDY:
                strategyContractValid = hasInput(pg, "penalty_greedy_save_id_in") &&
                        hasInput(dg, "penalty_save_id_in") && hasInput(dg, "penalty_greedy_save_id_in");
                break;
            case MSTRAT_SAMPLING:
                strategyContractValid = hasInput(pg, "sampling_previous_ids") &&
                        hasInput(dg, "sampling_previous_ids");
                break;
        }
        if (!strategyContractValid ||
                        !saveIdTypesValid ||
                        (requiresSaveId && saveIdType == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) ||
            (isPenalty &&
             (penaltyValueIn < 0 || penaltyRangeIn < 0 ||
                            !isFloatingTensorType(dg.inputTypes[penaltyValueIn]) ||
                            !isIntegerTensorType(dg.inputTypes[penaltyRangeIn]) ||
                            !inputTypeMatches(pg, "penalty_penalty_value", dg.inputTypes[penaltyValueIn]) ||
                            !inputTypeMatches(pg, "penalty_penalty_range", dg.inputTypes[penaltyRangeIn]))) ||
            (isSampling &&
             (samplingTemperatureIn < 0 || samplingTopKIn < 0 || samplingTopPIn < 0 ||
                            samplingPenaltyIn < 0 ||
                            !isFloatingTensorType(dg.inputTypes[samplingTemperatureIn]) ||
                            !isIntegerTensorType(dg.inputTypes[samplingTopKIn]) ||
                            !isFloatingTensorType(dg.inputTypes[samplingTopPIn]) ||
                            !isFloatingTensorType(dg.inputTypes[samplingPenaltyIn]) ||
                            !inputTypeMatches(pg, "sampling_temperature", dg.inputTypes[samplingTemperatureIn]) ||
                            !inputTypeMatches(pg, "sampling_top_k", dg.inputTypes[samplingTopKIn]) ||
                            !inputTypeMatches(pg, "sampling_top_p", dg.inputTypes[samplingTopPIn]) ||
                            !inputTypeMatches(pg, "sampling_repetition_penalty",
                                              dg.inputTypes[samplingPenaltyIn])))) {
            LOGE("Run_LLM(merged): invalid strategy input contract for %s / %s",
                 pg.fileName.c_str(), dg.fileName.c_str());
              return runError(env, "MODEL_CONTRACT");
        }

        auto readTokenScalar = [&](OrtValue* value, ONNXTensorElementDataType type) -> int {
            int64_t token = -1;
            if (!readScalarInteger(mApi, value, type, token) ||
                token < std::numeric_limits<int>::min() || token > std::numeric_limits<int>::max()) {
                LOGE("Run_LLM(merged): invalid token scalar for type %d", static_cast<int>(type));
                return -1;
            }
            return static_cast<int>(token);
        };

        // ---- vision encode (image/video): raw ViT hidden; the merged prefill graph does the concat ----
        std::vector<OrtValue*> visionOutputs;
        OrtValue* visionHidden = nullptr;
        int64_t   visionGrowth = 0;
        if (isVisionTurn) {
            visionOutputs = getOrEncodeVisionOutputs(visionMode);
            if (visionOutputs.size() != pVisionInputs.size()) {
                LOGE("Run_LLM(merged): vision output count mismatch (actual=%zu expected=%zu); "
                     "aborting vision turn", visionOutputs.size(), pVisionInputs.size());
                releaseValues(mApi, visionOutputs);
                return visionCancelRequested()
                        ? runCancelled(env)
                        : runError(env, "VISION_ENCODE");
            }
            visionHidden = visionOutputs[static_cast<size_t>(pVisionHiddenOutput)];
            if (visionMode == VISION_IMAGE) {
                visionGrowth = std::max<int64_t>(0, tensorDim(mApi, visionHidden, 1));   // image concat grows seq
            }
            if (visionMode == VISION_VIDEO && g_video_stream_desired.load(std::memory_order_acquire)) {
                std::lock_guard<std::mutex> lock(g_vision_pixels_mutex);
                g_video_ring_count.store(0, std::memory_order_relaxed);
                g_video_ring_write.store(0, std::memory_order_relaxed);
                g_video_stream_active.store(true, std::memory_order_release);
            }
        }
        const int64_t prefillTokenCount = ids_len + visionGrowth;

        // ---- reusable metadata-typed input buffers ----
        OrtBuffer idsBuf, idsLenBuf, historyLenBuf, cacheLenBuf;
        OrtBuffer penaltyValBuf, penaltyRangeBuf;
        OrtBuffer samplingTemperatureBuf, samplingTopKBuf, samplingTopPBuf, samplingPenaltyBuf;
        OrtBuffer emptySaveIdBuf;
        std::vector<OrtValue*> prefillPrefixKV;
        std::vector<OrtValue*> prefillWindowKV;
        auto releasePrefillSlices = [&]() {
            releaseValues(mApi, prefillPrefixKV);
            releaseValues(mApi, prefillWindowKV);
        };
        bool inputBuffersOk = true;
        if (isPenalty) {
            const ONNXTensorElementDataType valueType = dg.inputTypes[penaltyValueIn];
            const ONNXTensorElementDataType rangeType = dg.inputTypes[penaltyRangeIn];
            penaltyValBuf   = makeBuffer(dg, {1}, valueType);
            penaltyRangeBuf = makeBuffer(dg, {1}, rangeType);
            inputBuffersOk = writeScalarFloat(penaltyValBuf, valueType, repeatPenalty) &&
                             writeScalarInt64(penaltyRangeBuf, rangeType, penaltyRange) &&
                             inputBuffersOk;
        }
        if (isSampling) {
            const ONNXTensorElementDataType temperatureType = dg.inputTypes[samplingTemperatureIn];
            const ONNXTensorElementDataType topKType = dg.inputTypes[samplingTopKIn];
            const ONNXTensorElementDataType topPType = dg.inputTypes[samplingTopPIn];
            const ONNXTensorElementDataType penaltyType = dg.inputTypes[samplingPenaltyIn];
            samplingTemperatureBuf = makeBuffer(dg, {1}, temperatureType);
            samplingTopKBuf = makeBuffer(dg, {1}, topKType);
            samplingTopPBuf = makeBuffer(dg, {1}, topPType);
            samplingPenaltyBuf = makeBuffer(dg, {1}, penaltyType);
            inputBuffersOk = writeScalarFloat(samplingTemperatureBuf, temperatureType,
                                              samplingTemperature) &&
                             writeScalarInt64(samplingTopKBuf, topKType, topK) &&
                             writeScalarFloat(samplingTopPBuf, topPType, samplingTopP) &&
                             writeScalarFloat(samplingPenaltyBuf, penaltyType, repeatPenalty) &&
                             inputBuffersOk;
        }
        if (requiresSaveId) {
            emptySaveIdBuf = makeBuffer(pg, {1, 0}, saveIdType);
            inputBuffersOk = emptySaveIdBuf.value != nullptr && inputBuffersOk;
        }

        // Bind the empty save-id / phase / strategy scalars into a merged graph's inputs (by name; absent
        // names are skipped). saveId is the growing accumulated-id tensor (empty on prefill).
        auto bindStrategyInputs = [&](ModelRuntime& g, OrtValue* saveId) {
            bool bound = true;
            const std::vector<int>& saveInputs = &g == &pg
                    ? prefillSaveIdInputs : decodeSaveIdInputs;
            for (int index : saveInputs) {
                bound = saveId != nullptr && bindIn(g, index, saveId) && bound;
            }
            if (isPenalty) {
                const int valueIdx = findInputIdx(g, "penalty_penalty_value");
                const int rangeIdx = findInputIdx(g, "penalty_penalty_range");
                if (valueIdx >= 0) { bound = bindIn(g, valueIdx, penaltyValBuf.value) && bound; }
                if (rangeIdx >= 0) { bound = bindIn(g, rangeIdx, penaltyRangeBuf.value) && bound; }
            }
            if (isSampling) {
                const int temperatureIdx = findInputIdx(g, "sampling_temperature");
                const int topKIdx = findInputIdx(g, "sampling_top_k");
                const int topPIdx = findInputIdx(g, "sampling_top_p");
                const int penaltyIdx = findInputIdx(g, "sampling_repetition_penalty");
                if (temperatureIdx >= 0) {
                    bound = bindIn(g, temperatureIdx, samplingTemperatureBuf.value) && bound;
                }
                if (topKIdx >= 0) { bound = bindIn(g, topKIdx, samplingTopKBuf.value) && bound; }
                if (topPIdx >= 0) { bound = bindIn(g, topPIdx, samplingTopPBuf.value) && bound; }
                if (penaltyIdx >= 0) {
                    bound = bindIn(g, penaltyIdx, samplingPenaltyBuf.value) && bound;
                }
            }
            return bound;
        };

        auto bindDynamicStrategyInput = [&](OrtValue* saveId) {
            bool bound = true;
            for (int index : decodeSaveIdInputs) {
                bound = saveId != nullptr && bindIn(dg, index, saveId) && bound;
            }
            return bound;
        };

        idsBuf = makeBuffer(pg, {1, ids_len}, pg.inputTypes[pTokenIn]);
        idsLenBuf = makeBuffer(pg, {1}, pg.inputTypes[pIdsLenIn]);
        historyLenBuf = makeBuffer(pg, {1}, pg.inputTypes[pHistoryLenIn]);
        inputBuffersOk = writeIntegerValues(idsBuf, pg.inputTypes[pTokenIn], get_ids.data(),
                                            static_cast<size_t>(ids_len)) &&
                         writeScalarInt64(idsLenBuf, pg.inputTypes[pIdsLenIn], prefillTokenCount) &&
                         writeScalarInt64(historyLenBuf, pg.inputTypes[pHistoryLenIn], history_pos) &&
                         inputBuffersOk;
        if (pCacheLenIn >= 0) {
            cacheLenBuf = makeBuffer(pg, {1}, pg.inputTypes[pCacheLenIn]);
            inputBuffersOk = writeScalarInt64(cacheLenBuf, pg.inputTypes[pCacheLenIn],
                                              reusedHistory ? prefillCacheLen : 0) && inputBuffersOk;
        }
        auto releaseInputBuffers = [&]() {
            releaseBuffer(mApi, idsBuf);
            releaseBuffer(mApi, idsLenBuf);
            releaseBuffer(mApi, historyLenBuf);
            releaseBuffer(mApi, cacheLenBuf);
            releaseBuffer(mApi, penaltyValBuf);
            releaseBuffer(mApi, penaltyRangeBuf);
            releaseBuffer(mApi, samplingTemperatureBuf);
            releaseBuffer(mApi, samplingTopKBuf);
            releaseBuffer(mApi, samplingTopPBuf);
            releaseBuffer(mApi, samplingPenaltyBuf);
            releaseBuffer(mApi, emptySaveIdBuf);
        };
        if (!inputBuffersOk) {
            releaseValues(mApi, visionOutputs);
            releaseInputBuffers();
            return runError(env, "ALLOCATION");
        }

        if (prefillDecoupled) {
            std::vector<OrtValue*> savedInputs(saved_kv.begin(), saved_kv.begin() + num_keys_values);
            const int64_t splitAt = saved_kv_len - prefillCacheLen;
            if (!runKVSplit2(savedInputs, splitAt, prefillPrefixKV, prefillWindowKV)) {
                LOGE("Run_LLM(merged): prefill KV split failed (history=%lld cache=%lld)",
                     static_cast<long long>(saved_kv_len), static_cast<long long>(prefillCacheLen));
                releaseValues(mApi, visionOutputs);
                releasePrefillSlices();
                releaseInputBuffers();
                clearAllModelBindingRefs();
                return runError(env, "PREFILL_STATE");
            }
            LOGI("Run_LLM(merged): prefill window decoupled (prefix=%lld cache=%lld prompt=%lld)",
                 static_cast<long long>(splitAt), static_cast<long long>(prefillCacheLen),
                 static_cast<long long>(ids_len));
        }

        // ---- prefill I/O binding ----
        resetBinding(pg);
        bool prefillBound = true;
        for (int i = 0; i < num_keys_values; ++i) {
            OrtValue* kvIn = prefillDecoupled
                    ? prefillWindowKV[i]
                    : ((reusedHistory && saved_kv[i] != nullptr) ? saved_kv[i] : input_tensors_kv_init[i]);
            prefillBound = bindIn(pg, i, kvIn) && prefillBound;
        }
        for (int i = 0; i < num_linear_states; ++i) {
            OrtValue* linIn = (reusedHistory && saved_linear_valid) ? saved_linear_states[i] : linear_state_init[i];
            prefillBound = bindIn(pg, num_keys_values + i, linIn) && prefillBound;
        }
        prefillBound = bindIn(pg, pTokenIn, idsBuf.value) && prefillBound;
        prefillBound = bindIn(pg, pIdsLenIn, idsLenBuf.value) && prefillBound;
        prefillBound = bindIn(pg, pHistoryLenIn, historyLenBuf.value) && prefillBound;
        if (pCacheLenIn >= 0) {
            prefillBound = bindIn(pg, pCacheLenIn, cacheLenBuf.value) && prefillBound;
        }
        for (size_t i = 0; i < visionOutputs.size(); ++i) {
            prefillBound = bindIn(pg, pVisionInputs[i], visionOutputs[i]) && prefillBound;
        }
        prefillBound = bindStrategyInputs(pg, emptySaveIdBuf.value) && prefillBound;
        for (int i = 0; i < static_cast<int>(pg.outputNames.size()); ++i) {
            prefillBound = bindOutDevice(pg, i) && prefillBound;
        }

        LOGI("FLOW prefill_begin turn=%d graph=%s ids=%lld expanded=%lld history=%lld cache=%lld "
             "vision_outputs=%zu bound=%d cancel=%d",
             static_cast<int>(turn_id), pg.fileName.c_str(), static_cast<long long>(ids_len),
             static_cast<long long>(prefillTokenCount), static_cast<long long>(history_pos),
             static_cast<long long>(reusedHistory ? prefillCacheLen : 0), visionOutputs.size(),
             prefillBound ? 1 : 0, g_cancel.load(std::memory_order_acquire) ? 1 : 0);
        const auto prefill_start_clock = std::chrono::steady_clock::now();
        const int64_t prefill_start_ms = now_ms();
        std::vector<OrtValue*> prefillOut;
        if (!prefillBound || !runBinding(pg) || !fetchOutputsInto(pg, prefillOut) ||
            prefillOut.size() < static_cast<size_t>(num_main_states)) {
            releaseValues(mApi, prefillOut);
            releaseValues(mApi, visionOutputs);
            releasePrefillSlices();
            releaseInputBuffers();
            clearAllModelBindingRefs();
                return g_cancel.load(std::memory_order_relaxed)
                    ? runCancelled(env)
                    : runError(env, "PREFILL_RUN");
        }
        releaseValues(mApi, visionOutputs);   // consumed by the merged prefill (concat happened inside)
        const float prefill_s = std::chrono::duration<float>(
                std::chrono::steady_clock::now() - prefill_start_clock).count();
        const float prefill_rate = (prefill_s > 0.0f)
            ? (static_cast<float>(prefillTokenCount) / prefill_s) : 0.0f;

        // Resolve prefill output positions by name.
        const int pMaxIdx  = findOutputIdxAny(pg, {"greedy_max_logits_idx", "penalty_greedy_max_logits_idx",
                               "sampling_sampled_id"});
        const int pNextTok = pMaxIdx;
        const int pKvSeq   = findOutputIdx(pg, "prefill_kv_seq_len");
        const int pSaveId  = findOutputIdxAny(pg, {"penalty_greedy_save_id_out",
                       "sampling_save_id_out"});
        const int dMaxIdx  = findOutputIdxAny(dg, {"greedy_max_logits_idx", "penalty_greedy_max_logits_idx",
                               "sampling_sampled_id"});
        const int dNextTok = dMaxIdx;
        const int dKvSeq   = findOutputIdxAny(dg, {"decode_kv_seq_len_next", "decode_kv_seq_len"});
        const int dSaveId  = findOutputIdxAny(dg, {"penalty_greedy_save_id_out",
                       "sampling_save_id_out"});

        auto hasPrefillOutput = [&](int idx) {
            return idx >= num_main_states && idx < static_cast<int>(prefillOut.size());
        };
        auto hasDecodeOutput = [&](int idx) {
            return idx >= num_main_states && idx < static_cast<int>(dg.outputNames.size());
        };
        bool stateTypesValid = true;
        for (int i = 0; i < num_main_states; ++i) {
            const ONNXTensorElementDataType canonicalType = mMain.inputTypes[i];
            stateTypesValid = pg.inputTypes[i] == canonicalType &&
                              pg.outputTypes[i] == canonicalType &&
                              dg.inputTypes[i] == canonicalType &&
                              dg.outputTypes[i] == canonicalType &&
                              stateTypesValid;
        }
        const bool prefillTypesValid = hasPrefillOutput(pMaxIdx) && hasPrefillOutput(pNextTok) &&
            hasPrefillOutput(pKvSeq) &&
            isIntegerTensorType(pg.outputTypes[pMaxIdx]) &&
            pg.outputTypes[pNextTok] == dg.inputTypes[dTokenIn] &&
            pg.outputTypes[pKvSeq] == dg.inputTypes[dKvSeqIn] &&
            (!requiresSaveId || (hasPrefillOutput(pSaveId) &&
             pg.outputTypes[pSaveId] == saveIdType));
        const bool decodeTypesValid = hasDecodeOutput(dMaxIdx) && hasDecodeOutput(dNextTok) &&
            hasDecodeOutput(dKvSeq) &&
            isIntegerTensorType(dg.outputTypes[dMaxIdx]) &&
            dg.outputTypes[dNextTok] == dg.inputTypes[dTokenIn] &&
            dg.outputTypes[dKvSeq] == dg.inputTypes[dKvSeqIn] &&
            (!requiresSaveId || (hasDecodeOutput(dSaveId) &&
             dg.outputTypes[dSaveId] == saveIdType));
        if (!stateTypesValid || !prefillTypesValid || !decodeTypesValid ||
            !hasPrefillOutput(pMaxIdx) || !hasPrefillOutput(pNextTok) ||
            !hasPrefillOutput(pKvSeq) || (requiresSaveId && !hasPrefillOutput(pSaveId)) ||
            !hasDecodeOutput(dMaxIdx) || !hasDecodeOutput(dNextTok) ||
              !hasDecodeOutput(dKvSeq) || (requiresSaveId && !hasDecodeOutput(dSaveId))) {
            LOGE("Run_LLM(merged): invalid output contract (prefill=%s max=%d next=%d kv=%d save=%d "
                  "outputs=%zu; decode=%s max=%d next=%d kv=%d save=%d outputs=%zu; state_count=%d)",
                  pg.fileName.c_str(), pMaxIdx, pNextTok, pKvSeq, pSaveId,
                  prefillOut.size(), dg.fileName.c_str(), dMaxIdx, dNextTok, dKvSeq, dSaveId,
                 dg.outputNames.size(), num_main_states);
            clearAllModelBindingRefs();
            releaseValues(mApi, prefillOut);
            releasePrefillSlices();
            releaseInputBuffers();
            return runError(env, "MODEL_CONTRACT");
        }

        // Take ownership of the state block + control tensors; release the rest of the tail.
        std::vector<OrtValue*> stateVec(prefillOut.begin(), prefillOut.begin() + num_main_states);
        for (int i = 0; i < num_main_states; ++i) { prefillOut[i] = nullptr; }
        if (prefillDecoupled) {
            std::vector<OrtValue*> suffixKV(stateVec.begin(), stateVec.begin() + num_keys_values);
            std::vector<OrtValue*> fullKV;
                const bool concatOk = runKVConcat(prefillPrefixKV, suffixKV, fullKV);
            if (!concatOk ||
                fullKV.size() < static_cast<size_t>(num_keys_values)) {
                LOGE("Run_LLM(merged): prefill KV concat failed");
                releaseValues(mApi, fullKV);
                releaseValues(mApi, stateVec);
                releaseValues(mApi, prefillOut);
                releasePrefillSlices();
                releaseInputBuffers();
                clearAllModelBindingRefs();
                return runError(env, "PREFILL_STATE");
            }
            for (int i = 0; i < num_keys_values; ++i) {
                releaseOne(mApi, stateVec[i]);
                stateVec[i] = fullKV[i];
                fullKV[i] = nullptr;
            }
            releaseValues(mApi, fullKV);
            releasePrefillSlices();
        }
        auto takeOut = [&](int idx) -> OrtValue* {
            if (idx < 0 || idx >= static_cast<int>(prefillOut.size())) { return nullptr; }
            OrtValue* v = prefillOut[idx]; prefillOut[idx] = nullptr; return v;
        };
        int selectedToken = readTokenScalar(prefillOut[pMaxIdx], pg.outputTypes[pMaxIdx]);
        OrtValue* nextToken = takeOut(pNextTok);
        OrtValue* kvSeqTok  = takeOut(pKvSeq);
        OrtValue* saveId    = takeOut(pSaveId);
        int64_t initialKvSeq = -1;
        if (selectedToken < 0 ||
            !readScalarInteger(mApi, kvSeqTok, pg.outputTypes[pKvSeq], initialKvSeq) ||
            initialKvSeq <= 0) {
            releaseValues(mApi, stateVec);
            releaseOne(mApi, nextToken);
            releaseOne(mApi, kvSeqTok);
            releaseOne(mApi, saveId);
            releaseValues(mApi, prefillOut);
            releasePrefillSlices();
            releaseInputBuffers();
            clearAllModelBindingRefs();
            return runError(env, "PREFILL_OUTPUT");
        }
        releaseValues(mApi, prefillOut);   // any remaining tail outputs

        const int memoryCapacity = std::max(1, memory.memoryTokens);
        int64_t currentPhysicalLen = stateVec.empty() ? -1 : tensorDim(mApi, stateVec[0], 4);
        if (currentPhysicalLen < 0) {
            currentPhysicalLen = saved_kv_len + prefillTokenCount;
        }
        auto memoryRemainingForState = [&]() {
            const int used = static_cast<int>(std::min<int64_t>(
                    std::max<int64_t>(0, currentPhysicalLen), memoryCapacity));
            return memoryCapacity - used;
        };
        pushPerfStats(env, prefill_rate, -1.0f, memoryRemainingForState(),
                  memoryCapacity, static_cast<int>(prefillTokenCount), 0, false);

        // ---- generation bookkeeping ----
        std::vector<int> replyVec;         // answer-only (think excluded), used by ON commit
        std::vector<int> fullReplyVec;     // OFF: every decoded token (mirrors saved decode KV)
        int generated = 0, numDecode = 0;
        // Thinking ON pre-fills "<think>\n" into the prompt (never decoded), so start already inside the
        // think block: the reasoning body is excluded from the retained answer, exactly as a generated <think>.
        int replyThinkDepth = effectiveThink ? 1 : 0;
        bool kvStateValid = true;
        bool pendingTokenInState = false;  // selected/streamed, but not yet consumed by a decode graph
        int64_t absNext = initialKvSeq;
        int64_t decodeBase = reusedHistory ? saved_kv_base : 0;
        const int64_t decodeMargin = 2;
        int decodeBudget = memory.decodeTokens;
        const int rotaryRoom = static_cast<int>(std::max<int64_t>(
                0, static_cast<int64_t>(max_seq_len) - initialKvSeq - decodeMargin));
        const bool canRecycleDecode = !hasLinearState() && !isVisionTurn &&
            mKVSlice.session != nullptr && mRopeShift.session != nullptr;
        if (!canRecycleDecode) { decodeBudget = std::min(decodeBudget, rotaryRoom); }
        LOGI("FLOW prefill_end turn=%d token=%d stop=%d kv_seq=%lld prefill_ms=%lld "
             "decode_budget=%d rotary_room=%d cancel=%d output=%s",
             static_cast<int>(turn_id), selectedToken, is_stop_token(selectedToken) ? 1 : 0,
             static_cast<long long>(initialKvSeq),
             static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - prefill_start_clock).count()),
             decodeBudget, rotaryRoom, g_cancel.load(std::memory_order_acquire) ? 1 : 0,
             pMaxIdx >= 0 ? pg.outputNames[pMaxIdx] : "<missing>");
        if (rebuildCleanStateFromIds) {
            replyVec.reserve(static_cast<size_t>(std::max(0, decodeBudget)));
        }
        if (!organizeMemory) {
            fullReplyVec.reserve(static_cast<size_t>(std::max(0, decodeBudget)));
        }

        auto appendReply = [&](int id) {
            if (id == chat_think_start_id) { replyThinkDepth += 1; return; }
            if (id == chat_think_end_id) { if (replyThinkDepth > 0) replyThinkDepth -= 1; return; }
            if (replyThinkDepth == 0 && rebuildCleanStateFromIds) { replyVec.push_back(id); }
        };

        reset_output_words_decoder();
        stream_buf.clear();
        if (effectiveThink) {
            // The <think> token lives in the pre-filled prompt, so the model never streams it. Emit the open
            // sentinel here so the UI opens the reasoning panel; the model still closes it with </think>.
            stream_buf.append(kThinkOpenSentinel);
        }
        tokens_since_flush = 0;
        last_flush_ms = prefill_start_ms;
        // Commit the prefill's first token.
        if (selectedToken >= 0 && !is_stop_token(selectedToken) && generated < decodeBudget) {
            generated += 1; numDecode += 1;
            appendReply(selectedToken);
            if (!organizeMemory) { fullReplyVec.push_back(selectedToken); }
            stream_token(env, selectedToken);
            pendingTokenInState = true;
        }

        const auto decode_start_clock = std::chrono::steady_clock::now();
        int64_t last_decode_perf_ms = now_ms();
        auto pushLiveDecodeStats = [&](int decodedTokens) {
            const float live_decode_s = std::chrono::duration<float>(
                std::chrono::steady_clock::now() - decode_start_clock).count();
            const float live_decode_rate = (live_decode_s > 0.0f)
                ? (static_cast<float>(decodedTokens) / live_decode_s) : 0.0f;
            pushPerfStats(env, -1.0f, live_decode_rate, memoryRemainingForState(),
                          memoryCapacity, 0, decodedTokens, false);
        };

        OrtIoBinding* decodeBindings[2] = {dg.binding, dg.alternateBinding};
        bool decodeBindingsReady = true;
        for (OrtIoBinding* binding : decodeBindings) {
            dg.binding = binding;
            resetBinding(dg);
            decodeBindingsReady = bindStrategyInputs(dg, emptySaveIdBuf.value) &&
                                  decodeBindingsReady;
            for (int index = 0; index < static_cast<int>(dg.outputNames.size()); ++index) {
                decodeBindingsReady = bindOutDevice(dg, index) && decodeBindingsReady;
            }
        }
        dg.binding = decodeBindings[0];
        int decodeBindingStep = 0;
        int linearRebindsLeft[2] = {2, 2};
        int controlRebindsLeft[2] = {2, 2};
        std::vector<OrtValue*> stepOut;
        stepOut.reserve(dg.outputNames.size());
        if (!decodeBindingsReady) {
            LOGE("Run_LLM(merged): could not initialize decode ping-pong bindings");
            kvStateValid = false;
        }
        LOGI("FLOW decode_begin turn=%d first_token=%d stop=%d budget=%d bindings=%d cancel=%d",
             static_cast<int>(turn_id), selectedToken, is_stop_token(selectedToken) ? 1 : 0,
             decodeBudget, decodeBindingsReady ? 1 : 0,
             g_cancel.load(std::memory_order_acquire) ? 1 : 0);

        // ---- decode loop: dual bindings retain static outputs/scalars and peer fixed-shape inputs ----
        while (numDecode < decodeBudget && selectedToken >= 0 && !is_stop_token(selectedToken) &&
               !g_cancel.load(std::memory_order_acquire) && kvStateValid) {
            const int bindingIndex = decodeBindingStep & 1;
            dg.binding = decodeBindings[bindingIndex];
            bool decodeBound = true;
            for (int i = 0; i < num_keys_values; ++i) {
                decodeBound = bindIn(dg, i, stateVec[i]) && decodeBound;
            }
            if (linearRebindsLeft[bindingIndex] > 0) {
                for (int i = num_keys_values; i < num_main_states; ++i) {
                    decodeBound = bindIn(dg, i, stateVec[i]) && decodeBound;
                }
            }
            if (controlRebindsLeft[bindingIndex] > 0) {
                decodeBound = bindIn(dg, dTokenIn, nextToken) && decodeBound;
                decodeBound = bindIn(dg, dKvSeqIn, kvSeqTok) && decodeBound;
            }
                decodeBound = bindDynamicStrategyInput(
                    saveId != nullptr ? saveId : emptySaveIdBuf.value) && decodeBound;
            for (int i = 0; i < num_keys_values; ++i) {
                decodeBound = bindOutDevice(dg, i) && decodeBound;
            }
            if (requiresSaveId) {
                decodeBound = bindOutDevice(dg, dSaveId) && decodeBound;
            }

            if (!decodeBound || !runBinding(dg) || !fetchOutputsInto(dg, stepOut) ||
                stepOut.size() != dg.outputNames.size()) {
                releaseValues(mApi, stepOut);
                kvStateValid = false;
                break;
            }
            if (linearRebindsLeft[bindingIndex] > 0) {
                linearRebindsLeft[bindingIndex] -= 1;
            }
            if (controlRebindsLeft[bindingIndex] > 0) {
                controlRebindsLeft[bindingIndex] -= 1;
            }
            decodeBindingStep += 1;
            selectedToken = readTokenScalar(stepOut[dMaxIdx], dg.outputTypes[dMaxIdx]);
            // Swap in the new state + control tensors (release the previous ones).
            releaseValues(mApi, stateVec);
            stateVec.assign(stepOut.begin(), stepOut.begin() + num_main_states);
            for (int i = 0; i < num_main_states; ++i) { stepOut[i] = nullptr; }
            auto takeStep = [&](int idx) -> OrtValue* {
                if (idx < 0 || idx >= static_cast<int>(stepOut.size())) { return nullptr; }
                OrtValue* v = stepOut[idx]; stepOut[idx] = nullptr; return v;
            };
            OrtValue* newNext = takeStep(dNextTok);
            OrtValue* newKvSeq = takeStep(dKvSeq);
            OrtValue* newSaveId = takeStep(dSaveId);
            releaseValues(mApi, stepOut);
            releaseOne(mApi, nextToken); nextToken = newNext;
            releaseOne(mApi, kvSeqTok);  kvSeqTok  = newKvSeq;
            if (newSaveId != nullptr) { releaseOne(mApi, saveId); saveId = newSaveId; }

            // This successful run consumed the previously selected token. The returned counter is the
            // authoritative next mRoPE position.
            pendingTokenInState = false;
            int64_t decodedAbsNext = -1;
            if (!readScalarInteger(mApi, kvSeqTok, dg.outputTypes[dKvSeq], decodedAbsNext) ||
                decodedAbsNext <= 0) {
                kvStateValid = false;
                break;
            }
            absNext = decodedAbsNext;
            currentPhysicalLen += 1;

            if (is_stop_token(selectedToken)) { break; }
            generated += 1;
            appendReply(selectedToken);
            if (!organizeMemory) { fullReplyVec.push_back(selectedToken); }
            stream_token(env, selectedToken);
            const int64_t perf_now_ms = now_ms();
            if (tokens_since_flush == 0 &&
                perf_now_ms - last_decode_perf_ms >= PERF_STATS_INTERVAL_MS) {
                pushLiveDecodeStats(generated);
                last_decode_perf_ms = perf_now_ms;
            }
            pendingTokenInState = true;

            if (canRecycleDecode && absNext + decodeMargin >= static_cast<int64_t>(max_seq_len)) {
                const int64_t keepForAppend = memoryGreenTargetUsed(memory.decodeKeepWindow);
                const int64_t preSlideLen = currentPhysicalLen;
                int64_t slidNext = absNext;
                if (slideDecodeWindow(stateVec, absNext, keepForAppend, kvSeqTok,
                                      dg.outputTypes[dKvSeq], slidNext)) {
                    absNext = slidNext;
                    decodeBase = 0;
                    currentPhysicalLen = std::min<int64_t>(keepForAppend, preSlideLen);
                } else {
                    LOGE("Run_LLM(merged): in-turn KV recycle failed at abs=%lld len=%lld keep=%lld",
                         static_cast<long long>(absNext), static_cast<long long>(preSlideLen),
                         static_cast<long long>(keepForAppend));
                    if (offDirectSave) {
                        kvStateValid = false;
                    }
                    break;
                }
            }
            numDecode += 1;
        }

            const char* decodeExitReason = numDecode >= decodeBudget ? "budget"
                : selectedToken < 0 ? "invalid_token"
                : is_stop_token(selectedToken) ? "stop"
                : g_cancel.load(std::memory_order_acquire) ? "cancel"
                : !kvStateValid ? "state_invalid"
                : "other";
            LOGI("FLOW decode_end turn=%d reason=%s generated=%d steps=%d token=%d stop=%d "
                 "cancel=%d manual=%d state_valid=%d pending_stream_bytes=%zu",
                 static_cast<int>(turn_id), decodeExitReason, generated, numDecode, selectedToken,
                 is_stop_token(selectedToken) ? 1 : 0,
                 g_cancel.load(std::memory_order_acquire) ? 1 : 0,
                 g_manual_stop.load(std::memory_order_acquire) ? 1 : 0,
                 kvStateValid ? 1 : 0, stream_buf.size());

        auto appendExplicitTokensToMergedState = [&](const std::vector<int>& ids) -> bool {
            for (int id : ids) {
                if (canRecycleDecode && absNext + decodeMargin >= static_cast<int64_t>(max_seq_len)) {
                    const int64_t keepForAppend = memoryGreenTargetUsed(memory.decodeKeepWindow);
                    int64_t slidNext = absNext;
                    if (!slideDecodeWindow(stateVec, absNext, keepForAppend, kvSeqTok,
                                           dg.outputTypes[dKvSeq], slidNext)) {
                        return false;
                    }
                    absNext = slidNext;
                    decodeBase = 0;
                    currentPhysicalLen = std::min<int64_t>(keepForAppend, currentPhysicalLen);
                } else if (!canRecycleDecode && absNext + decodeMargin >= static_cast<int64_t>(max_seq_len)) {
                    return false;
                }

                OrtBuffer explicitToken = makeBuffer(dg, {1, 1}, dg.inputTypes[dTokenIn]);
                if (!writeIntegerValues(explicitToken, dg.inputTypes[dTokenIn], &id, 1)) {
                    releaseBuffer(mApi, explicitToken);
                    return false;
                }

                resetBinding(dg);
                bool bound = true;
                for (int i = 0; i < num_main_states; ++i) {
                    bound = bindIn(dg, i, stateVec[i]) && bound;
                }
                bound = bindIn(dg, dTokenIn, explicitToken.value) && bound;
                bound = bindIn(dg, dKvSeqIn, kvSeqTok) && bound;
                bound = bindStrategyInputs(dg, saveId != nullptr ? saveId : emptySaveIdBuf.value) && bound;
                for (int i = 0; i < static_cast<int>(dg.outputNames.size()); ++i) {
                    bound = bindOutDevice(dg, i) && bound;
                }

                std::vector<OrtValue*> appendOut;
                const bool ran = bound && runBinding(dg) && fetchOutputsInto(dg, appendOut) &&
                        appendOut.size() == dg.outputNames.size();
                releaseBuffer(mApi, explicitToken);
                if (!ran) {
                    releaseValues(mApi, appendOut);
                    return false;
                }

                releaseValues(mApi, stateVec);
                stateVec.assign(appendOut.begin(), appendOut.begin() + num_main_states);
                for (int i = 0; i < num_main_states; ++i) {
                    appendOut[i] = nullptr;
                }
                auto takeAppend = [&](int idx) -> OrtValue* {
                    OrtValue* value = appendOut[idx];
                    appendOut[idx] = nullptr;
                    return value;
                };
                OrtValue* appendedKvSeq = takeAppend(dKvSeq);
                OrtValue* appendedSaveId = requiresSaveId ? takeAppend(dSaveId) : nullptr;
                releaseValues(mApi, appendOut);
                releaseOne(mApi, kvSeqTok);
                kvSeqTok = appendedKvSeq;
                if (appendedSaveId != nullptr) {
                    releaseOne(mApi, saveId);
                    saveId = appendedSaveId;
                }
                int64_t appendedAbsNext = -1;
                if (!readScalarInteger(mApi, kvSeqTok, dg.outputTypes[dKvSeq], appendedAbsNext) ||
                    appendedAbsNext <= 0) {
                    return false;
                }
                absNext = appendedAbsNext;
                currentPhysicalLen += 1;
            }
            return true;
        };

        // Detach the final state now so OFF mode can append the last selected token (which has been
        // streamed but not consumed when decode stops at its budget/cancel boundary) and a manual-stop
        // notice before persistence. Organize-memory mode rebuilds from IDs and does not consume this live state.
        std::vector<int> manualStopIds;
        if (g_manual_stop.load(std::memory_order_acquire)) {
            {
                std::lock_guard<std::mutex> tokenizerLock(g_tokenizer_mutex);
                manualStopIds = !g_manual_stop_notice_ids.empty()
                        ? g_manual_stop_notice_ids
                        : tokenizer->encode(MANUAL_STOP_NOTICE);
            }
            if (rebuildCleanStateFromIds) {
                replyVec.insert(replyVec.end(), manualStopIds.begin(), manualStopIds.end());
            }
            stream_buf.append(MANUAL_STOP_NOTICE);
        }

        if (offDirectSave && kvStateValid) {
            std::vector<int> stateAppendIds;
            if (pendingTokenInState) {
                stateAppendIds.push_back(selectedToken);
            }
            stateAppendIds.insert(stateAppendIds.end(), manualStopIds.begin(), manualStopIds.end());
            if (!stateAppendIds.empty()) {
                if (appendExplicitTokensToMergedState(stateAppendIds)) {
                    pendingTokenInState = false;
                    if (!manualStopIds.empty()) {
                        fullReplyVec.insert(fullReplyVec.end(), manualStopIds.begin(), manualStopIds.end());
                    }
                } else {
                    LOGE("Run_LLM(merged): failed to append pending/manual-stop tokens to direct-save KV");
                    kvStateValid = false;
                }
            }
        }

        std::vector<OrtValue*> finalKV(stateVec.begin(), stateVec.begin() + num_keys_values);
        std::vector<OrtValue*> finalLinear(stateVec.begin() + num_keys_values, stateVec.begin() + num_main_states);
        for (int i = 0; i < num_main_states; ++i) { stateVec[i] = nullptr; }

        const auto end_clock = std::chrono::steady_clock::now();
        flush_stream(env);
        reset_output_words_decoder();

        // ---- perf ----
        const float decode_s = std::chrono::duration<float>(end_clock - decode_start_clock).count();
        const float decode_rate  = (decode_s > 0.0f) ? (static_cast<float>(generated) / decode_s) : 0.0f;
        setBenchmarkCorrectness(fullReplyVec, generated, initialKvSeq, absNext);
        const int finalRemaining = memoryRemainingForState();
        char stats[96];
        std::snprintf(stats, sizeof(stats), "%.2f|%.2f|%d|%d",
                  prefill_rate, decode_rate, finalRemaining, memoryCapacity);
        pushPerfStats(env, prefill_rate, decode_rate, finalRemaining, memoryCapacity,
                  static_cast<int>(prefillTokenCount), generated, true);
           LOGI("FLOW native_complete turn=%d generated=%d prefill_rate=%.2f decode_rate=%.2f "
               "remaining=%d state_valid=%d",
               static_cast<int>(turn_id), generated, prefill_rate, decode_rate,
               finalRemaining, kvStateValid ? 1 : 0);

        auto cleanupMerged = [&]() {
            releaseValues(mApi, stateVec);
            releaseValues(mApi, finalKV);
            releaseValues(mApi, finalLinear);
            releaseOne(mApi, nextToken);
            releaseOne(mApi, kvSeqTok);
            releaseOne(mApi, saveId);
            releaseInputBuffers();
            clearAllModelBindingRefs();
        };

        if (offDirectSave) {
            int pendingCap = -1;
            const std::vector<int>& historyPromptIds = isVisionTurn ? visionHistoryPromptIds : get_ids;
            const int64_t promptMropeSpan = std::max<int64_t>(0, initialKvSeq - history_pos);
            const bool saved = kvStateValid &&
                    saveDecodeKVDirect(finalKV, finalLinear, decodeBase, historyPromptIds, fullReplyVec,
                                       memory.memoryTokens, &pendingCap, isVisionTurn, absNext,
                                       prefillTokenCount, promptMropeSpan,
                                       isVisionTurn ? visionMode : VISION_NONE);
            if (!saved) { clear_history("merged_direct_kv_save_failed"); }
            cleanupMerged();
            if (saved && pendingCap >= 0) { commitOffCap(pendingCap); }
            return env->NewStringUTF(stats);
        }

        // Organize-memory ON rebuilds clean KV and linear state from ids. Vision prompts
        // use dialogueBaseIds' natural-language note instead of unrebuildable vision placeholders.
        // The "整理记忆" box tracks whether an organization actually runs.
        std::vector<int> dialogueIds;
        if (kvStateValid) {
            dialogueIds.reserve(static_cast<size_t>(prev_close_len) + dialogueBaseIds.size() + replyVec.size());
            if (reusedHistory) {
                dialogueIds.insert(dialogueIds.end(), chat_previous_assistant_close_ids.begin(),
                                   chat_previous_assistant_close_ids.end());
            }
            dialogueIds.insert(dialogueIds.end(), dialogueBaseIds.begin(), dialogueBaseIds.end());
            dialogueIds.insert(dialogueIds.end(), replyVec.begin(), replyVec.end());
        }
        cleanupMerged();
        const bool rebuildVisionMemory = organizeMemory &&
            (isVisionTurn || saved_kv_has_vision);
        // Mirror appendCleanTurnKV's red-zone decision so the box shows when a toggle-off turn is forced
        // to rebuild to the green target. rebuildVisionMemory only occurs with Organize Memory ON, which
        // already shows the box, so the pure-text projection (prevLen + new dialogue ids) suffices here.
        const bool forcedRedZoneRebuild = kvStateValid && !rebuildVisionMemory &&
            memoryUsedInRedZone(memory.memoryTokens,
                                saved_kv_len + static_cast<int64_t>(dialogueIds.size()));
        commitCleanTurn(std::move(dialogueIds), kvStateValid, memory.memoryTokens,
                rebuildVisionMemory, organizeMemory || forcedRedZoneRebuild);
        return env->NewStringUTF(stats);
    }
}
// JNI: cooperative cancel.
extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_Stop_1LLM(JNIEnv* /*env*/, jclass /*clazz*/, jboolean manual) {
    const bool isManual = manual == JNI_TRUE;
    const bool busy = g_busy.load(std::memory_order_acquire);
    LOGI("FLOW stop_request turn=%d manual=%d busy=%d", g_flow_turn_id,
         isManual ? 1 : 0, busy ? 1 : 0);
    if (!busy) {
        // Publish the manual-stop classification before the cancel flag (release) so a pre-start consumer
        // that observes the cancel exchange also observes the matching manual-stop value.
        g_pending_prestart_manual_stop.store(isManual, std::memory_order_release);
        g_pending_prestart_cancel.store(true, std::memory_order_release);
    }
    // Manual Stop is COOPERATIVE: publish g_manual_stop (release) BEFORE g_cancel (release) so any reader
    // that observes g_cancel with an acquire load also observes the manual-stop classification. The decode
    // loop and vision pipeline poll g_cancel and halt at the next token boundary, leaving the KV state valid
    // so the partial reply + manual-stop notice persist.
    if (isManual) {
        g_manual_stop.store(true, std::memory_order_release);
    }
    g_cancel.store(true, std::memory_order_release);
    // ORT-level termination aborts the in-flight decode step, which invalidates the KV state (kvStateValid
    // becomes false) and forces the commit path to clear native history. Reserve it for destructive
    // cancel/Clear; a manual Stop must not trigger it.
    if (!isManual) {
        setRuntimeTerminate(true);
    }
}

// Abort any in-flight ORT run during Clear (generation, vision prewarm, clean commit).
static void setRuntimeTerminate(bool terminate) {
    for (ModelRuntime& m : gModelRuntimes) {
        if (m.api == nullptr || m.runOptions == nullptr) {
            continue;
        }
        if (terminate) {
            logOrtStatus(m.api, m.api->RunOptionsSetTerminate(m.runOptions), "RunOptionsSetTerminate");
        } else {
            logOrtStatus(m.api, m.api->RunOptionsUnsetTerminate(m.runOptions), "RunOptionsUnsetTerminate");
        }
    }
}

// JNI: panic reset.
extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_Clear_1Cache(JNIEnv* env, jclass clazz) {
    (void)env;
    (void)clazz;
    if (g_busy.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    struct BusyGuard { ~BusyGuard() { g_busy.store(false, std::memory_order_release); } } busy_guard;
    g_clean_commit_cancel.store(true, std::memory_order_relaxed);
    g_cancel.store(true, std::memory_order_relaxed);
    g_pending_prestart_cancel.store(false, std::memory_order_release);
    g_pending_prestart_manual_stop.store(false, std::memory_order_release);
    g_pending_vision.store(VISION_NONE, std::memory_order_release);
    setRuntimeTerminate(true);
    waitForCleanCommit();
    clear_history("clear_cache_jni");
    {
        std::lock_guard<std::mutex> lock(g_vision_pixels_mutex);
        releaseDynamicVideoInputLocked();
    }
    shrinkAllModelAllocators();
    g_reserved_prefill_arena_bytes = 0;
    g_reserved_decode_arena_bytes = 0;   // arenas were shrunk; the next turn re-warms them
    setRuntimeTerminate(false);
    g_cancel.store(false, std::memory_order_relaxed);
    g_clean_commit_cancel.store(false, std::memory_order_relaxed);
}

// JNI: rollback before edit-and-resend.
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Rollback_1LLM(JNIEnv* env, jclass clazz, jint turn_id) {
    (void)env;
    (void)clazz;
    if (g_busy.exchange(true, std::memory_order_acq_rel)) {
        return JNI_FALSE;
    }
    struct BusyGuard { ~BusyGuard() { g_busy.store(false, std::memory_order_release); } } busy_guard;
    waitForCleanCommit();
    const MemoryProfile memory = snapshotMemoryProfile();

    int64_t historyLen = -1;
    int64_t activeLen = 0;
    int64_t mropePos = 0;
    bool    hadVision = false;
    size_t cut = g_turn_checkpoints.size();
    for (size_t i = 0; i < g_turn_checkpoints.size(); ++i) {
        if (g_turn_checkpoints[i].turnId == turn_id) {
            historyLen = g_turn_checkpoints[i].historyLen;
            activeLen  = g_turn_checkpoints[i].activeLen;
            mropePos   = g_turn_checkpoints[i].mropePos;
            hadVision  = g_turn_checkpoints[i].hadVision;
            cut = i;
            break;
        }
    }
    if (historyLen < 0) {
        clear_history("rollback_checkpoint_missing");
        return JNI_FALSE;
    }
    g_turn_checkpoints.resize(cut);

    const int64_t oldFull = logicalHistoryEnd();
    historyLen = std::max<int64_t>(g_history_base, std::min<int64_t>(historyLen, oldFull));
    // activeLen counts PHYSICAL KV tokens; a retained vision segment makes it exceed the text-id history,
    // so clamp it to the physical window (saved_kv_len), never to historyLen (which would drop vision KV).
    activeLen  = std::max<int64_t>(0, std::min<int64_t>(activeLen, saved_kv_len));
    if (historyLen <= g_history_base) {
        const bool rolledToConversationStart = g_history_base == 0 && historyLen == 0;
        clear_history("rollback_to_empty_history");
        return rolledToConversationStart ? JNI_TRUE : JNI_FALSE;
    }
    if (oldFull > historyLen) {
        const int64_t retainedLength = historyLen - g_history_base;
        g_history_ids.resize(g_history_begin + static_cast<size_t>(retainedLength));
    }
    // Fast path when the active window front has not moved since turn start. Slicing keeps the front
    // [0, activeLen), so a vision segment (always at the front of the window) is preserved intact: a
    // sufficient-memory edit/resend in a vision chat no longer forces a full reset. Only the full-
    // attention KV can be sliced, so hybrid (linear-state) models fall through to the ids rebuild.
    const int64_t curOffset    = oldFull - saved_kv_len;
    const int64_t targetOffset = historyLen - activeLen;
    if (curOffset == targetOffset && mKVSlice.session != nullptr && saved_kv[0] != nullptr) {
        if (activeLen >= saved_kv_len) {
            return JNI_TRUE;   // no tokens dropped: full-attention KV, linear state and vision KV stay valid
        }
        std::vector<OrtValue*> cur(saved_kv.begin(), saved_kv.begin() + num_keys_values), sliced;
        if (!hasLinearState() && runKVSlice(cur, 0, activeLen, sliced)) {
            release_saved_kv();
            for (int i = 0; i < num_keys_values; ++i) saved_kv[i] = sliced[i];
            saved_kv_len = activeLen;
            // Restore the turn-start mRoPE bookkeeping. A retained vision segment keeps mRoPE diverged
            // from the physical length (use the recorded position); pure text stays 1:1 (base + len).
            saved_kv_has_vision = hadVision;
            saved_mrope_pos = hadVision ? mropePos : (saved_kv_base + activeLen);
            publishMemoryUsage();
            return JNI_TRUE;
        }
    }
    // Fallback: rebuild the retained window from text ids. A vision segment cannot be reproduced this way
    // (it collapses to its natural-language memory note, with the assistant's own description still in
    // history), but the conversation stays coherent instead of being cleared.
    const int rebuildTarget = static_cast<int>(std::min<int64_t>(
            std::max<int64_t>(1, activeLen), static_cast<int64_t>(memory.memoryTokens)));
    if (!rebuildSavedKVFromHistoryRetaining(rebuildTarget)) {
        clear_history("rollback_rebuild_failed");
        return JNI_FALSE;
    }
    return JNI_TRUE;
}
