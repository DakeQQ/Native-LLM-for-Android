// Split-ONNX Qwen3 runtime: model loading, tokenization, streaming, prefill/decode, and JNI glue.
#include "llm_runtime_state.h"
#include "ort_helpers.h"
#include <jni.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <sys/stat.h>
#include <cstdio>

// Model sessions used across the pipeline.
static ModelRuntime& mEmbed         = getModel(LLM_Embed);
static ModelRuntime& mMain          = getModel(LLM_Main);
static ModelRuntime& mRotaryPrefill = getModel(LLM_RotaryPrefill);
static ModelRuntime& mRotaryDecode  = getModel(LLM_RotaryDecode);
static ModelRuntime& mGreedy        = getModel(LLM_Greedy);
static ModelRuntime& mFirstBeam     = getModel(LLM_FirstBeam);
static ModelRuntime& mSecondBeam    = getModel(LLM_SecondBeam);
static ModelRuntime& mPenalty       = getModel(LLM_Penalty);
static ModelRuntime& mArgmax        = getModel(LLM_Argmax);
static ModelRuntime& mKVSlice       = getModel(LLM_KV_Slice);
static ModelRuntime& mKVSplit2      = getModel(LLM_KV_Split2);
static ModelRuntime& mKVConcat      = getModel(LLM_KV_Concat);
static ModelRuntime& mRopeShift     = getModel(LLM_RopeShift);

// Token streaming.
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
    if (g_main_cls != nullptr && g_on_token != nullptr) {
        jstring js = env->NewStringUTF(stream_buf.c_str());
        if (js != nullptr) {
            env->CallStaticVoidMethod(g_main_cls, g_on_token, js);
            env->DeleteLocalRef(js);
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
            }
        }
    }
    stream_buf.clear();
    tokens_since_flush = 0;
    last_flush_ms = now_ms();
}

// KV shape helpers.
inline static int detectKVSeqAxis(const std::vector<int64_t>& modelDims) {
    for (int a = 1; a < modelDims.size(); ++a) {
        if (modelDims[a] <= 0) {
            return a;
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
    if (m.inputNames.size() <= static_cast<size_t>(mainMaskIdx) ||
        m.outputNames.size() <= static_cast<size_t>(mainLogitsOutIdx)) {
        LOGE("ORT: unexpected Main I/O count: inputs=%zu outputs=%zu", m.inputNames.size(), m.outputNames.size());
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
bool loadModel(ModelId id, JNIEnv* env, jobject assetManager, int epType, bool lowMemoryMode) {
    ModelRuntime& m = getModel(id);
    m.id = id;
    return ortLoadModelSession(m, kModelFileNames[id], kModelExternalFileNames[id],
                               env, assetManager, epType, lowMemoryMode);
}

// Runtime config is read from exporter-stamped ONNX metadata. Missing required keys fail load.
static void buildChatTemplates() {
    chat_user_prefix_ids              = { chat_im_start_id, chat_user_id, chat_newline_id };
    chat_assistant_prefix_ids         = { chat_im_end_id, chat_newline_id, chat_im_start_id, chat_assistant_id, chat_newline_id };
    chat_empty_think_block_ids        = { chat_think_start_id, chat_double_newline_id, chat_think_end_id, chat_double_newline_id };
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

    const int meta_layers        = reqInt("num_layers");
    // Hybrid split; defaults keep pre-hybrid (pure full-attention) exports working unchanged.
    const int meta_full_layers   = optInt("num_full_attention_layers", meta_layers);
    const int meta_linear_layers = optInt("num_linear_attention_layers", 0);
    const int meta_kv_tensors    = reqInt("kv_num_tensors");
    const int meta_kv_blocks     = reqInt("kv_blocks_per_layer");
    const int meta_max_seq_len = reqInt("max_seq_len");
    const int meta_fp16        = reqInt("activations_fp16");
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

    // Multimodal reservation (optional): only present when a vision model stamped these ids.
    const int meta_image_token  = optInt("image_token_id", -1);
    const int meta_video_token  = optInt("video_token_id", -1);
    const int meta_vision_start = optInt("vision_start_token_id", -1);
    const int meta_vision_end   = optInt("vision_end_token_id", -1);

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

    ORT_FP16    = (meta_fp16 != 0);
    max_seq_len = meta_max_seq_len;

    DEFAULT_MEMORY_TOKENS      = max_seq_len / 4;
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
    chat_image_token_id    = meta_image_token;
    chat_video_token_id    = meta_video_token;
    chat_vision_start_id   = meta_vision_start;
    chat_vision_end_id     = meta_vision_end;
    g_multimodal           = (meta_image_token >= 0 || meta_video_token >= 0);
    end_id_0 = chat_endoftext_id;
    end_id_1 = chat_im_end_id;
    buildChatTemplates();

    LOGI("model metadata applied: layers=%d kv_blocks=%d kv_tensors=%d max_seq_len=%d fp16=%d "
         "(mem=%d prefill=%d decode=%d)",
         num_layers, g_kv_blocks, num_keys_values, max_seq_len, ORT_FP16 ? 1 : 0,
         DEFAULT_MEMORY_TOKENS, DEFAULT_PREFILL_TOKENS, DEFAULT_MAX_DECODE_TOKENS);
    return true;
}

// Preload metadata before real sessions so ORT_FP16 is correct during session creation.
static bool preloadRuntimeMetadata(JNIEnv* env, jobject assetManager, bool lowMemoryMode) {
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
    if (gSharedEnvUsesGlobalPools) {
        api->DisablePerSessionThreads(opts);
    }
    api->SetSessionGraphOptimizationLevel(opts, ORT_DISABLE_ALL);   // metadata only: skip all optimization

    OrtSession* session = nullptr;
    OrtStatus* status = nullptr;
    std::vector<char> fileBuffer;
    bool haveBuffer = false;
    if (!lowMemoryMode && assetManager != nullptr) {
        AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
        if (mgr != nullptr) {
            AAsset* asset = AAssetManager_open(mgr, fileName, AASSET_MODE_BUFFER);
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

// KV tensors must be first. Inputs = KV + hidden/cos/sin/mask; outputs = KV + logits.
bool configureKVLayout() {
    ModelRuntime& m = getModel(LLM_Main);
    if (m.api == nullptr || m.session == nullptr) {
        LOGE("configureKVLayout: LLM_Main not loaded");
        return false;
    }
    if (num_keys_values <= 0 || num_layers <= 0 || g_kv_blocks <= 0) {
        LOGE("configureKVLayout: model metadata not applied (layers=%d blocks=%d kv=%d)",
             num_layers, g_kv_blocks, num_keys_values);
        return false;
    }
    const int inCount   = static_cast<int>(m.inputNames.size());
    const int outCount  = static_cast<int>(m.outputNames.size());
    // Main state block = [windowed full-attn KV] + [passthrough linear states]; then hidden/cos/sin/mask
    // on the input side and logits on the output side.
    const int statesByInputs  = inCount - 4;    // minus hidden_states, rotary_cos, rotary_sin, attention_mask
    const int statesByOutputs = outCount - 1;   // minus logits
    if (statesByInputs != statesByOutputs || statesByInputs < num_keys_values) {
        LOGE("configureKVLayout: LLM_Main I/O (inputs=%d outputs=%d -> states=%d/%d) inconsistent with "
             "full-attention KV count=%d", inCount, outCount, statesByInputs, statesByOutputs, num_keys_values);
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
        LOGE("configureKVLayout: metadata declares 0 linear layers but Main carries %d extra states",
             num_linear_states);
        return false;
    } else {
        g_linear_blocks = 0;
    }

    mainHiddenIdx    = num_main_states;
    mainCosIdx       = num_main_states + 1;
    mainSinIdx       = num_main_states + 2;
    mainMaskIdx      = num_main_states + 3;
    mainLogitsOutIdx = num_main_states;

    beamSaveIdInIdx       = num_main_states + 1;
    firstBeamSizeIdx      = num_main_states + 2;
    secondBeamPrevProbIdx = num_main_states + 2;
    secondBeamSizeIdx     = num_main_states + 3;
    secondBeamTopKIdx     = num_main_states + 4;
    beamSaveIdOutIdx      = num_main_states;
    beamScoreOutIdx       = num_main_states + 1;
    beamIdsOutIdx         = num_main_states + 2;
    beamMaxOutIdx         = num_main_states + 3;

    kvSliceStartIdx = num_keys_values;   // KV-management graphs act on the full-attention block only
    kvSliceEndIdx   = num_keys_values + 1;

    const char* modeName = (g_kv_blocks == 2) ? "F16/F32 (key,value)"
                         : (g_kv_blocks == 4) ? "symmetric quant (key,value,k_scale,v_scale)"
                                              : "asymmetric quant (key,value,k_scale,k_bias,v_scale,v_bias)";
    LOGI("configureKVLayout: %d layers (%d full + %d linear) x %d KV blocks -> %d KV + %d linear = %d "
         "Main states [%s]", num_layers, num_full_attention_layers, num_linear_attention_layers,
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
    ModelRuntime& m = getModel(LLM_Main);
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
        const size_t bytes = static_cast<size_t>(elems) * tensorElementSize(m.inputTypes[idx]);
        gLinearInitBacking[i].assign(bytes, 0);
        if (!createTensorWithData(api, m.ioMemoryInfo, gLinearInitBacking[i].data(), bytes,
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

bool initKVCache() {
    ModelRuntime& m = getModel(LLM_Main);
    if (m.api == nullptr || m.session == nullptr) {
        LOGE("ORT: initKVCache called before LLM_Main loaded");
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
        if (!createTensorWithData(api, m.ioMemoryInfo, reinterpret_cast<void*>(&gEmptyKVData), 0,
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

// Peak shared-arena bytes already reserved (high-water mark); reset when the arena is shrunk (Clear_Cache).
static int64_t g_reserved_arena_bytes = 0;

// Pre-grow the shared CPU arena so cache-heavy graphs do not grow it mid-turn. Reserve the
// largest loaded cache I/O working set plus one persistent batch-1 saved KV; g_reserved_arena_bytes
// makes this idempotent. seqCap is the worst-case sequence length; beamBatch is the decode batch.
static void reserveSharedArena(int64_t seqCap, int64_t beamBatch) {
    if (!kPrewarmSharedArena || kPrewarmArenaPercent <= 0 || !gSharedEnvArenaRegistered) {
        return;
    }
    ModelRuntime& main = getModel(LLM_Main);
    if (main.api == nullptr || main.session == nullptr || main.memoryInfo == nullptr) {
        return;
    }
    const OrtApi* api = main.api;
    const int64_t capTokens = std::max<int64_t>(1, seqCap);
    const int64_t beamB     = std::max<int64_t>(1, beamBatch);

    // Bytes of the cache-shaped I/O (rank>=5 KV tensors) of one graph: last axis = sequence -> capTokens;
    // a dynamic batch axis (0) -> batchForDynamic; any other dynamic axis -> 1. Scalars/logits/cos-sin/mask
    // are rank<5 and skipped.
    auto cacheIoBytes = [&](const std::vector<std::vector<int64_t>>& dimsList,
                            const std::vector<ONNXTensorElementDataType>& typeList,
                            int64_t batchForDynamic) -> int64_t {
        int64_t total = 0;
        for (size_t t = 0; t < dimsList.size() && t < typeList.size(); ++t) {
            const std::vector<int64_t>& dims = dimsList[t];
            if (dims.size() < 5) {
                continue;
            }
            const int lastAxis = static_cast<int>(dims.size()) - 1;
            int64_t elems = 1;
            for (int a = 0; a < static_cast<int>(dims.size()); ++a) {
                int64_t d = dims[a];
                if (a == lastAxis) {
                    d = capTokens;
                } else if (d <= 0) {
                    d = (a == 0) ? batchForDynamic : 1;
                }
                elems *= std::max<int64_t>(1, d);
            }
            total += elems * static_cast<int64_t>(tensorElementSize(typeList[t]));
        }
        return total;
    };

    int64_t peakBytes = 0;
    for (const ModelRuntime& mr : gModelRuntimes) {
        if (mr.session == nullptr) {
            continue;
        }
        const int64_t batchB = (mr.id == LLM_Main || mr.id == LLM_FirstBeam || mr.id == LLM_SecondBeam)
                               ? beamB : 1;
        const int64_t work = cacheIoBytes(mr.inputDims, mr.inputTypes, batchB) +
                             cacheIoBytes(mr.outputDims, mr.outputTypes, batchB);
        peakBytes = std::max(peakBytes, work);
    }
    peakBytes += cacheIoBytes(main.inputDims, main.inputTypes, 1);   // + persistent batch-1 saved_kv

    const int64_t target = peakBytes / 100 * kPrewarmArenaPercent;
    const int64_t delta  = target - g_reserved_arena_bytes;
    if (delta <= 0) {
        return;   // arena already holds enough
    }
    OrtAllocator* arena = nullptr;
    if (!logOrtStatus(api, api->CreateAllocator(main.session, main.memoryInfo, &arena),
                      "CreateAllocator(arena prewarm)") || arena == nullptr) {
        return;
    }
    void* block = nullptr;
    if (logOrtStatus(api, api->AllocatorAlloc(arena, static_cast<size_t>(delta), &block),
                     "AllocatorAlloc(arena prewarm)") && block != nullptr) {
        logOrtStatus(api, api->AllocatorFree(arena, block),   // back to the arena free-list; pages stay resident
                     "AllocatorFree(arena prewarm)");
        g_reserved_arena_bytes = target;
        LOGI("Shared arena pre-warmed: +%lld MB (total %lld MB; all graphs, seq<=%lld, beam<=%lld, %d%%)",
             static_cast<long long>(delta >> 20), static_cast<long long>(target >> 20),
             static_cast<long long>(capTokens), static_cast<long long>(beamB), kPrewarmArenaPercent);
    }
    api->ReleaseAllocator(arena);
}

// JNI: model load.
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1A(JNIEnv* env, jobject clazz,
                                                            jobject asset_manager,
                                                            jint ep_type,
                                                            jboolean low_memory_mode) {
    // ep_type: 0 = CPU (default), 1 = XNNPACK, 2 = QNN
    if (g_main_cls == nullptr) {
        jclass local_cls = env->GetObjectClass(clazz);
        if (local_cls != nullptr) {
            env->GetJavaVM(&g_jvm);
            g_main_cls = static_cast<jclass>(env->NewGlobalRef(local_cls));
            env->DeleteLocalRef(local_cls);
            g_on_token = env->GetStaticMethodID(g_main_cls, "onTokenStream", "(Ljava/lang/String;)V");
            g_on_perf  = env->GetStaticMethodID(g_main_cls, "onPerfStats", "(Ljava/lang/String;)V");
            g_on_post_processing = env->GetStaticMethodID(g_main_cls, "onPostProcessingState", "(Z)V");
        }
    }
    stream_buf.reserve(256);

    const bool lowMem = (low_memory_mode == JNI_TRUE);

    if (!preloadRuntimeMetadata(env, asset_manager, lowMem)) {
        LOGE("Load_Models_A: could not read model metadata; aborting load");
        return JNI_FALSE;
    }

    for (int id = 0; id < kModelCount; ++id) {
        if (!loadModel(static_cast<ModelId>(id), env, asset_manager, ep_type, lowMem)) {
            if (id == LLM_KV_Slice || id == LLM_KV_Split2 ||
                id == LLM_KV_Concat || id == LLM_RopeShift) {
                LOGE("Load_Models_A: optional model %d (%s) unavailable; that long-chat feature is disabled",
                     id, kModelFileNames[id]);
                continue;
            }
            LOGE("Load_Models_A: failed to load model %d (%s)", id, kModelFileNames[id]);
            return JNI_FALSE;
        }
    }

    ModelRuntime& main = getModel(LLM_Main);
    if (!configureKVLayout()) {
        return JNI_FALSE;
    }
    if (!validateMainIO(main)) {
        return JNI_FALSE;
    }

    if (!initKVCache()) {
        return JNI_FALSE;
    }

    reserveSharedArena(g_memory_tokens, useBeamSearch ? beamSize : 1);

    LOGI("Load_Models_A: all %d models loaded (EP=%d, lowMem=%d)", kModelCount, ep_type, lowMem ? 1 : 0);
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
    if (m.api == nullptr || m.binding == nullptr) {
        return;
    }
    m.api->ClearBoundInputs(m.binding);
    m.api->ClearBoundOutputs(m.binding);
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
}

static void shrinkAllModelAllocators() {
    for (ModelRuntime& m : gModelRuntimes) {
        shrinkModelAllocators(m);
    }
}

// Per-turn rollback checkpoints.
static inline void invalidate_checkpoints() {
    g_turn_checkpoints.clear();
}
static inline void record_turn_checkpoint(int32_t turnId, int64_t historyLen, int64_t activeLen) {
    for (TurnCheckpoint& cp : g_turn_checkpoints) {
        if (cp.turnId == turnId) {
            cp.historyLen = historyLen;
            cp.activeLen  = activeLen;
            return;
        }
    }
    g_turn_checkpoints.push_back({turnId, historyLen, activeLen});
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
        writeScalarInt64(startT, m.inputTypes[kvSliceStartIdx], start);
        writeScalarInt64(endT,   m.inputTypes[kvSliceEndIdx],   end);
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

// Split KV into prefix/window in one graph run.
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
    OrtBuffer splitT = makeBuffer(m, {1}, m.inputTypes[splitIdx]);
    bool ok = splitT.value != nullptr;
    std::vector<OrtValue*> combined;
    if (ok) {
        writeScalarInt64(splitT, m.inputTypes[splitIdx], splitAt);
        std::vector<OrtValue*> inputs;
        inputs.reserve(static_cast<size_t>(num_keys_values + 1));
        inputs.insert(inputs.end(), in.begin(), in.begin() + num_keys_values);
        inputs.push_back(splitT.value);
        ok = runBoundGraph(m, inputs, expectedOutputs, combined);
    }
    releaseBuffer(api, splitT);
    if (!ok || combined.size() < static_cast<size_t>(expectedOutputs)) {
        releaseValues(api, combined);
        return false;
    }
    prefix.assign(combined.begin(), combined.begin() + num_keys_values);
    window.assign(combined.begin() + num_keys_values, combined.begin() + expectedOutputs);
    for (size_t i = static_cast<size_t>(expectedOutputs); i < combined.size(); ++i) {
        releaseOne(api, combined[i]);
    }
    combined.clear();
    return true;
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
    saved_kv_base += dropLen;
    saved_kv_len = targetLen;
    publishMemoryUsage();
    return true;
}

// KV_Concat restores prefix + window after prefill-window decoupling.
static bool runKVConcat(std::vector<OrtValue*>& prefix, std::vector<OrtValue*>& suffix,
                        std::vector<OrtValue*>& out) {
    out.clear();
    ModelRuntime& m = mKVConcat;
    if (m.session == nullptr || prefix.size() < static_cast<size_t>(num_keys_values) ||
        suffix.size() < static_cast<size_t>(num_keys_values) ||
        m.inputNames.size() < static_cast<size_t>(2 * num_keys_values) ||
        m.outputNames.size() < static_cast<size_t>(num_keys_values) || !ensureBinding(m)) {
        return false;
    }
    std::vector<OrtValue*> inputs;
    inputs.reserve(static_cast<size_t>(2 * num_keys_values));
    inputs.insert(inputs.end(), prefix.begin(), prefix.begin() + num_keys_values);
    inputs.insert(inputs.end(), suffix.begin(), suffix.begin() + num_keys_values);
    return runBoundGraph(m, inputs, num_keys_values, out);
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
        writeScalarInt64(shiftT, m.inputTypes[shiftIdx], shift);
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
                              OrtValue* kvSeqLenTensor, int64_t& newAbsNext) {
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
    if (mRotaryDecode.inputTypes[0] == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        *reinterpret_cast<int32_t*>(raw) = static_cast<int32_t>(keepLen);
    } else {
        *reinterpret_cast<int64_t*>(raw) = keepLen;
    }
    newAbsNext = keepLen;
    return true;
}

// Token decode helpers.
inline static bool is_stop_token(int id) {
    return (id == end_id_0) || (id == end_id_1);
}

inline static void clear_history() {
    clearAllModelBindingRefs();
    release_saved_kv();
    release_saved_linear();
    ids_len       = 0;
    saved_kv_len  = 0;
    saved_kv_base = 0;
    std::vector<int>().swap(g_history_ids);
    std::vector<TurnCheckpoint>().swap(g_turn_checkpoints);
    reset_output_words_decoder();
    stream_buf.clear();
    tokens_since_flush = 0;
    publishMemoryUsage();
}

// Synchronization.
static std::mutex g_system_prompt_mutex;
static std::mutex g_tokenizer_mutex;
static std::mutex g_runtime_config_mutex;
static std::mutex g_clean_commit_mutex;
static std::condition_variable g_clean_commit_cv;
static bool g_clean_commit_running = false;

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

static void beginCleanCommit() {
    {
        std::unique_lock<std::mutex> lock(g_clean_commit_mutex);
        g_clean_commit_cv.wait(lock, [] { return !g_clean_commit_running; });
        g_clean_commit_running = true;
    }
    pushPostProcessingState(true);
}

static void finishCleanCommit() {
    clearAllModelBindingRefs();
    {
        std::lock_guard<std::mutex> lock(g_clean_commit_mutex);
        g_clean_commit_running = false;
    }
    g_clean_commit_cv.notify_all();
    pushPostProcessingState(false);
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

inline static void rebuild_system_prompt_ids() {
    std::lock_guard<std::mutex> lock(g_system_prompt_mutex);
    rebuild_system_prompt_ids_locked();
}

// Shared one-shot prefill returning KV outputs only.
static bool runPrefillToKV(const int* idsData, int64_t tokenCount,
                           OrtValue* const* kvInputs, OrtValue* const* linearInputs,
                           int64_t historyLen, int64_t cacheLen,
                           std::vector<OrtValue*>& outKV, std::vector<OrtValue*>& outLinear,
                           const char* inputOpName) {
    outKV.clear();
    outLinear.clear();
    const OrtApi* api = mMain.api;
    if (api == nullptr || idsData == nullptr || tokenCount <= 0 || kvInputs == nullptr ||
        (num_linear_states > 0 && linearInputs == nullptr) ||
        !ensureBinding(mEmbed) || !ensureBinding(mRotaryPrefill) || !ensureBinding(mMain)) {
        return false;
    }

    bool ok = false;
    OrtBuffer logitsBuf{};
    do {
        int64_t idsShape[2] = {1, tokenCount};
        OrtValue* idsT = nullptr;
        if (!logOrtStatus(api, api->CreateTensorWithDataAsOrtValue(
                mEmbed.memoryInfo, const_cast<int*>(idsData),
                static_cast<size_t>(tokenCount) << input_ids_elem_shift,
                idsShape, 2, mEmbed.inputTypes[0], &idsT), inputOpName)) {
            break;
        }

        std::vector<OrtValue*> embOut;
        embOut.reserve(1);
        resetBinding(mEmbed);
        bindIn(mEmbed, 0, idsT);
        bindOutDevice(mEmbed, 0);
        const bool embedOk = runBinding(mEmbed) && fetchOutputsInto(mEmbed, embOut);
        releaseOne(api, idsT);

        OrtValue* hidden = nullptr;
        if (!embedOk || !takeFirstOutput(api, embOut, hidden)) {
            releaseValues(api, embOut);
            break;
        }

        int64_t vN = tokenCount, vHist = historyLen, vCache = cacheLen, scalarShape[1] = {1};
        OrtValue *t0 = nullptr, *t1 = nullptr, *t2 = nullptr;
        const bool scalarOk =
                logOrtStatus(api, api->CreateTensorWithDataAsOrtValue(
                        mRotaryPrefill.memoryInfo, &vN, sizeof(vN), scalarShape, 1,
                        mRotaryPrefill.inputTypes[0], &t0), "prefill ids_len") &&
                logOrtStatus(api, api->CreateTensorWithDataAsOrtValue(
                        mRotaryPrefill.memoryInfo, &vHist, sizeof(vHist), scalarShape, 1,
                        mRotaryPrefill.inputTypes[1], &t1), "prefill history_len") &&
                logOrtStatus(api, api->CreateTensorWithDataAsOrtValue(
                        mRotaryPrefill.memoryInfo, &vCache, sizeof(vCache), scalarShape, 1,
                        mRotaryPrefill.inputTypes[2], &t2), "prefill cache_len");
        if (!scalarOk) {
            releaseOne(api, hidden);
            releaseOne(api, t0);
            releaseOne(api, t1);
            releaseOne(api, t2);
            break;
        }

        std::vector<OrtValue*> roOut;
        roOut.reserve(4);
        resetBinding(mRotaryPrefill);
        bindIn(mRotaryPrefill, 0, t0);
        bindIn(mRotaryPrefill, 1, t1);
        bindIn(mRotaryPrefill, 2, t2);
        for (int i = 0; i < 4; ++i) {
            bindOutDevice(mRotaryPrefill, i);
        }
        const bool rotaryOk = runBinding(mRotaryPrefill) &&
                              fetchOutputsInto(mRotaryPrefill, roOut) && roOut.size() >= 4;
        releaseOne(api, t0);
        releaseOne(api, t1);
        releaseOne(api, t2);
        if (!rotaryOk) {
            releaseOne(api, hidden);
            releaseValues(api, roOut);
            break;
        }

        const int vocab = static_cast<int>(mMain.outputDims[mainLogitsOutIdx].back());
        logitsBuf = makeBuffer(mMain, {1, static_cast<int64_t>(vocab)}, mMain.outputTypes[mainLogitsOutIdx]);
        if (logitsBuf.value == nullptr) {
            releaseOne(api, hidden);
            releaseValues(api, roOut);
            break;
        }

        resetBinding(mMain);
        bool bound = true;
        for (int i = 0; i < num_keys_values; ++i) {
            bound = bindIn(mMain, i, kvInputs[i]) && bound;
        }
        for (int i = 0; i < num_linear_states; ++i) {
            bound = bindIn(mMain, num_keys_values + i, linearInputs[i]) && bound;
        }
        bound = bindIn(mMain, mainHiddenIdx, hidden) && bound;
        bound = bindIn(mMain, mainCosIdx, roOut[0]) && bound;
        bound = bindIn(mMain, mainSinIdx, roOut[1]) && bound;
        bound = bindIn(mMain, mainMaskIdx, roOut[2]) && bound;
        for (int i = 0; i < num_keys_values; ++i) {
            bound = bindOutDevice(mMain, i) && bound;
        }
        for (int i = 0; i < num_linear_states; ++i) {
            bound = bindOutDevice(mMain, num_keys_values + i) && bound;
        }
        bound = bindOut(mMain, mainLogitsOutIdx, logitsBuf.value) && bound;
        const bool mainOk = bound && runBinding(mMain) && fetchOutputsInto(mMain, outKV) &&
                            outKV.size() >= static_cast<size_t>(num_main_states);
        releaseOne(api, hidden);
        releaseValues(api, roOut);
        if (!mainOk) {
            break;
        }
        // Fetched order: [full KV] [linear states] [logits]. Peel linear off, release the logits handle.
        outLinear.assign(outKV.begin() + num_keys_values, outKV.begin() + num_main_states);
        for (size_t i = static_cast<size_t>(num_main_states); i < outKV.size(); ++i) {
            releaseOne(api, outKV[i]);
        }
        outKV.resize(num_keys_values);   // linear pointers are now solely owned by outLinear
        ok = true;
    } while (false);
    releaseBuffer(api, logitsBuf);
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
    publishMemoryUsage();
    if (g_history_ids.empty()) {
        return true;
    }
    const int64_t keep = std::max(1, retainTarget);
    const int64_t total = static_cast<int64_t>(g_history_ids.size());
    const int64_t startIdx = (total > keep) ? (total - keep) : 0;
    const OrtApi* api = mMain.api;
    const int64_t N = total - startIdx;   // = min(keep, total)
    if (api == nullptr || !ensureBinding(mEmbed) || !ensureBinding(mRotaryPrefill) || !ensureBinding(mMain)) {
        return false;
    }
    std::vector<OrtValue*> kept, keptLinear;
    if (!runPrefillToKV(g_history_ids.data() + startIdx, N, input_tensors_kv_init.data(),
                        linear_state_init.data(), 0, 0, kept, keptLinear, "recompute input_ids")) {
        g_history_ids.clear();
        return false;
    }
    for (int i=0;i<num_keys_values;++i) saved_kv[i]=kept[i];
    store_saved_linear(keptLinear);   // rebuilt from zero over the retained window -> consistent linear state
    saved_kv_len = N;
    saved_kv_base = 0;
    publishMemoryUsage();
    return true;
}

static bool rebuildSavedKVFromHistory(int memoryTokens) {
    const int retainTarget = memoryRetainTargetAfterRed(memoryTokens, saved_kv_len);
    return rebuildSavedKVFromHistoryRetaining(retainTarget);
}

// Incremental clean-cache update; falls back to full rebuild on failure.
static bool appendCleanTurnKV(const std::vector<int>& ids, int memoryTokens) {
    if (ids.empty()) {
        return true;
    }
    const OrtApi* api = mMain.api;
    const int64_t N = static_cast<int64_t>(ids.size());
    const int64_t prevLen = saved_kv_len;
    const int64_t prevBase = saved_kv_base;
    const int intendedRetain = static_cast<int>(
            memoryUsedInRedZone(memoryTokens, prevLen + N)
                ? memoryGreenTargetUsed(memoryTokens)
                : prevLen + N);
    g_history_ids.insert(g_history_ids.end(), ids.begin(), ids.end());
    if (api == nullptr || prevLen == 0 || saved_kv[0] == nullptr ||
        (hasLinearState() && !saved_linear_valid) ||                      // linear state stale: rebuild
        prevBase + prevLen + N + 2 > max_seq_len ||                       // table would overflow: rebuild
        !ensureBinding(mEmbed) || !ensureBinding(mRotaryPrefill) || !ensureBinding(mMain)) {
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

static void runCleanCommit(const std::vector<int>& ids, bool kvStateValid, int memoryTokens) {
    if (g_clean_commit_cancel.load(std::memory_order_relaxed)) {
        return;
    }
    if (kvStateValid) {
        if (!appendCleanTurnKV(ids, memoryTokens)) {
            LOGE("Clean-cache recompute failed; resetting conversation history");
            clear_history();
        }
    } else {
        clear_history();
    }
}

static void commitCleanTurn(std::vector<int>&& dialogueIds, bool kvStateValid, int memoryTokens) {
    if (dialogueIds.size() <= static_cast<size_t>(kInlineCleanCommitMaxTokens)) {
        beginCleanCommit();
        runCleanCommit(dialogueIds, kvStateValid, memoryTokens);
        finishCleanCommit();
        return;
    }
    beginCleanCommit();
    std::thread([ids = std::move(dialogueIds), kvStateValid, memoryTokens]() mutable {
        runCleanCommit(ids, kvStateValid, memoryTokens);
        finishCleanCommit();
    }).detach();
}

// OFF red-zone cap rebuild; green-zone OFF saves KV directly.
static void commitOffCap(int cap) {
    beginCleanCommit();
    std::thread([cap]() {
        if (!g_clean_commit_cancel.load(std::memory_order_relaxed)) {
            rebuildSavedKVFromHistoryRetaining(cap);
        }
        finishCleanCommit();
    }).detach();
}

// OFF mode: transfer decode KV directly; red-zone trimming is a background re-prefill, not a slice.
static bool saveDecodeKVDirect(std::vector<OrtValue*>& decodeKV, std::vector<OrtValue*>& decodeLinear,
                               int64_t decodeBase,
                               const std::vector<int>& promptIds, const std::vector<int>& replyIds,
                               int memoryTokens, int* pendingCap) {
    *pendingCap = -1;
    const OrtApi* api = mMain.api;
    if (api == nullptr || decodeKV.size() < static_cast<size_t>(num_keys_values) || decodeKV[0] == nullptr) {
        return false;
    }
    const int64_t physLen = tensorDim(api, decodeKV[0], 4);   // keys seq axis (5-D KV layout)
    if (physLen <= 0) {
        return false;
    }
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
    saved_kv_len  = physLen;
    saved_kv_base = decodeBase;
    g_history_ids.insert(g_history_ids.end(), promptIds.begin(), promptIds.end());
    g_history_ids.insert(g_history_ids.end(), replyIds.begin(), replyIds.end());
    const int64_t cap = memoryRetainTargetAfterRed(memoryTokens, saved_kv_len);
    if (saved_kv_len > cap) {
        *pendingCap = static_cast<int>(cap);
        return true;
    }
    renumberSavedKV();
    publishMemoryUsage();
    return true;
}

// Fold manual-stop notice into OFF-mode decode KV.
static bool appendIdsToDecodeKV(std::vector<OrtValue*>& kv, std::vector<OrtValue*>& linear,
                                const std::vector<int>& ids, int64_t absStart, int64_t cacheLen) {
    const OrtApi* api = mMain.api;
    const auto n = static_cast<int64_t>(ids.size());
    if (api == nullptr || n <= 0 || kv.size() < static_cast<size_t>(num_keys_values) || kv[0] == nullptr ||
        !ensureBinding(mEmbed) || !ensureBinding(mRotaryPrefill) || !ensureBinding(mMain)) {
        return false;
    }
    std::vector<OrtValue*> appended, appendedLinear;
    if (!runPrefillToKV(ids.data(), n, kv.data(), linear.data(), absStart, cacheLen,
                        appended, appendedLinear, "notice input_ids")) {
        return false;
    }
    releaseValues(api, kv);
    kv.swap(appended);
    releaseValues(api, linear);
    linear.swap(appendedLinear);
    return true;
}

// ── Multimodal reservation ───────────────────────────────────────────────────────────────────
// Hook where a future vision build splices image/video embeddings into the text hidden states before the
// prefill Main pass. No-op today: g_multimodal is only set when a vision model stamped image/video token
// ids, and the text-only LLM_MODEL_TABLE never loads such a model. See user_settings.h for the activation
// plan (add the vision graphs to the table as optional loads, then fill this in).
static inline void maybeInjectVisionEmbeddings(OrtValue* /*prefillHidden*/, const std::vector<int>& /*ids*/) {
    if (!g_multimodal) {
        return;
    }
    // TODO(vision): LLM_Image/Video_Preprocess -> LLM_Vision -> LLM_Concat_* then overwrite the
    // <image_pad>/<video_pad> rows of prefillHidden with the projected vision embeddings.
}

// Must match THINK_OPEN / THINK_CLOSE in ChatAdapter.java.
static constexpr char kThinkOpenSentinel[]  = "\x02\x02think\x03\x03";
static constexpr char kThinkCloseSentinel[] = "\x02\x02/think\x03\x03";

inline static void stream_token(JNIEnv* env, int id) {
    if (id == chat_think_start_id) {
        reset_output_words_decoder();
        stream_buf.append(kThinkOpenSentinel);
    } else if (id == chat_think_end_id) {
        reset_output_words_decoder();
        stream_buf.append(kThinkCloseSentinel);
    } else {
        std::lock_guard<std::mutex> tokenizerLock(g_tokenizer_mutex);
        append_output_words(stream_buf, id);
    }
    tokens_since_flush += 1;
    if (tokens_since_flush >= STREAM_BATCH || (now_ms() - last_flush_ms) >= STREAM_FLUSH_MS) {
        flush_stream(env);
    }
}

static void pushPerfStats(JNIEnv* env, const char* payload) {
    if (payload == nullptr || g_main_cls == nullptr || g_on_perf == nullptr) {
        return;
    }
    jstring perfStr = env->NewStringUTF(payload);
    if (perfStr != nullptr) {
        env->CallStaticVoidMethod(g_main_cls, g_on_perf, perfStr);
        env->DeleteLocalRef(perfStr);
        if (env->ExceptionCheck()) {
            env->ExceptionClear();
        }
    }
}

static void pushPerfRates(JNIEnv* env, bool includePrefill, float prefillRate,
                          bool includeDecode, float decodeRate,
                          int memoryRemaining = -1, int memoryCapacity = -1,
                          int prefillTokens = -1, int decodeTokens = -1) {
    char stats[128];
    const bool includeMemory = memoryRemaining >= 0 && memoryCapacity > 0;
    const bool includeTokens = prefillTokens >= 0 || decodeTokens >= 0;
    char prefillTokenField[16] = "";
    char decodeTokenField[16] = "";
    if (prefillTokens >= 0) {
        std::snprintf(prefillTokenField, sizeof(prefillTokenField), "%d", prefillTokens);
    }
    if (decodeTokens >= 0) {
        std::snprintf(decodeTokenField, sizeof(decodeTokenField), "%d", decodeTokens);
    }
    if (includePrefill && includeDecode) {
        if (includeMemory && includeTokens) {
            std::snprintf(stats, sizeof(stats), "%.2f|%.2f|%d|%d|%s|%s",
                          prefillRate, decodeRate, memoryRemaining, memoryCapacity,
                          prefillTokenField, decodeTokenField);
        } else if (includeMemory) {
            std::snprintf(stats, sizeof(stats), "%.2f|%.2f|%d|%d",
                          prefillRate, decodeRate, memoryRemaining, memoryCapacity);
        } else if (includeTokens) {
            std::snprintf(stats, sizeof(stats), "%.2f|%.2f|||%s|%s",
                          prefillRate, decodeRate, prefillTokenField, decodeTokenField);
        } else {
            std::snprintf(stats, sizeof(stats), "%.2f|%.2f", prefillRate, decodeRate);
        }
    } else if (includePrefill) {
        if (includeMemory && includeTokens) {
            std::snprintf(stats, sizeof(stats), "%.2f||%d|%d|%s|",
                          prefillRate, memoryRemaining, memoryCapacity,
                          prefillTokenField);
        } else if (includeMemory) {
            std::snprintf(stats, sizeof(stats), "%.2f||%d|%d",
                          prefillRate, memoryRemaining, memoryCapacity);
        } else if (includeTokens) {
            std::snprintf(stats, sizeof(stats), "%.2f||||%s|",
                          prefillRate, prefillTokenField);
        } else {
            std::snprintf(stats, sizeof(stats), "%.2f|", prefillRate);
        }
    } else if (includeDecode) {
        if (includeMemory && includeTokens) {
            std::snprintf(stats, sizeof(stats), "|%.2f|%d|%d||%s",
                          decodeRate, memoryRemaining, memoryCapacity,
                          decodeTokenField);
        } else if (includeMemory) {
            std::snprintf(stats, sizeof(stats), "|%.2f|%d|%d",
                          decodeRate, memoryRemaining, memoryCapacity);
        } else if (includeTokens) {
            std::snprintf(stats, sizeof(stats), "|%.2f||||%s",
                          decodeRate, decodeTokenField);
        } else {
            std::snprintf(stats, sizeof(stats), "|%.2f", decodeRate);
        }
    } else {
        return;
    }
    pushPerfStats(env, stats);
}

// Decode strategy.
struct DecodeStrategy {
    bool beam = false;
    bool penalty = false;
    bool argmax = true;
    bool greedy = false;
    int  decodeBatch = 1;
    int  effectiveTopK = 1;
    int  beamSize = 1;
    float repeatPenalty = DEFAULT_REPEAT_PENALTY;
};

static DecodeStrategy resolveDecodeStrategy() {
    DecodeStrategy s;
    bool cfgUseBeam = false;
    int cfgTopK = 1;
    int cfgBeamSize = 1;
    float cfgRepeatPenalty = DEFAULT_REPEAT_PENALTY;
    {
        std::lock_guard<std::mutex> lock(g_runtime_config_mutex);
        cfgUseBeam = useBeamSearch;
        cfgTopK = clampInt(topK, 0, MAX_TOP_K_RUNTIME);
        cfgBeamSize = clampInt(beamSize, 0, MAX_BEAM_SIZE_RUNTIME);
        cfgRepeatPenalty = std::max(0.0f, std::min(repeatPenalty, 1.0f));
        if (cfgUseBeam && cfgTopK < cfgBeamSize) {
            cfgTopK = cfgBeamSize;
        }
    }
    const bool cfgUsePenalty = cfgRepeatPenalty < DEFAULT_REPEAT_PENALTY;

    s.beam = cfgUseBeam && cfgTopK >= 2 && cfgBeamSize >= 2 && mFirstBeam.session && mSecondBeam.session;
    s.penalty = cfgUsePenalty && mPenalty.session;
    s.argmax = !s.beam && !s.penalty && mArgmax.session;
    s.greedy = !s.beam && !s.argmax;
    s.decodeBatch = s.beam ? cfgBeamSize : 1;
    s.effectiveTopK = std::max(cfgTopK, cfgBeamSize);
    s.beamSize = cfgBeamSize;
    s.repeatPenalty = cfgRepeatPenalty;
    return s;
}

// JNI: decode config.
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Configure_1LLM(JNIEnv* env, jclass clazz,
                                                           jboolean use_beam_search,
                                                           jint top_k,
                                                           jint beam_size,
                                                           jfloat repeat_penalty) {
    bool cfgUseBeam;
    bool cfgUsePenalty;
    int cfgTopK;
    int cfgBeamSize;
    float cfgRepeatPenalty;
    {
        std::lock_guard<std::mutex> lock(g_runtime_config_mutex);
        useBeamSearch = (use_beam_search == JNI_TRUE);
        topK = clampInt(static_cast<int>(top_k), 0, MAX_TOP_K_RUNTIME);
        beamSize = clampInt(static_cast<int>(beam_size), 0, MAX_BEAM_SIZE_RUNTIME);
        repeatPenalty = std::max(0.0f, std::min(static_cast<float>(repeat_penalty), 1.0f));
        usePenalty = repeatPenalty < DEFAULT_REPEAT_PENALTY;
        if (useBeamSearch && topK < beamSize) {
            topK = beamSize;
        }
        cfgUseBeam = useBeamSearch;
        cfgUsePenalty = usePenalty;
        cfgTopK = topK;
        cfgBeamSize = beamSize;
        cfgRepeatPenalty = repeatPenalty;
    }
    LOGI("LLM decode config beam=%d, penalty=%d, topK=%d, beamSize=%d, repeatPenalty=%.3f",
         cfgUseBeam ? 1 : 0,
         cfgUsePenalty ? 1 : 0,
         cfgTopK,
         cfgBeamSize,
         cfgRepeatPenalty);
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
Java_com_example_myapplication_MainActivity_Configure_1Memory(JNIEnv* env, jclass clazz,
                                                              jint memory_tokens,
                                                              jint prefill_tokens,
                                                              jint decode_tokens) {
    (void)env;
    (void)clazz;
    const MemoryProfile p = makeMemoryProfile(static_cast<int>(memory_tokens),
                                              static_cast<int>(prefill_tokens),
                                              static_cast<int>(decode_tokens));
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
Java_com_example_myapplication_MainActivity_Configure_1Memory_1Thresholds(JNIEnv* env, jclass clazz,
                                                                          jint red_percent,
                                                                          jint green_percent) {
    (void)env;
    (void)clazz;
    const int red = clampInt(static_cast<int>(red_percent),
                             MIN_MEMORY_BAND_PERCENT + MIN_MEMORY_BAND_GAP, MAX_MEMORY_BAND_PERCENT);
    const int green = clampInt(static_cast<int>(green_percent),
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
Java_com_example_myapplication_MainActivity_Pre_1Process(JNIEnv* env, jobject clazz) {
    if (tokenizer != nullptr) {
        delete tokenizer;
        tokenizer = nullptr;
    }
    {
        std::lock_guard<std::mutex> tokenizerLock(g_tokenizer_mutex);
        tokenizer = Tokenizer::createTokenizer(vocab_path);
    }
    if (tokenizer == nullptr) {
        return JNI_FALSE;
    }
    {
        std::lock_guard<std::mutex> lock(g_system_prompt_mutex);
        if (g_system_prompt_text.empty()) {
            g_system_prompt_text = DEFAULT_SYSTEM_PROMPT;
        }
        rebuild_system_prompt_ids_locked();
    }
    return JNI_TRUE;
}

// JNI: slot system prompt.
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Set_1System_1Prompt(JNIEnv* env, jclass clazz,
                                                                jstring system_prompt) {
    if (tokenizer == nullptr) {
        return JNI_FALSE;   // tokenizer not ready (Pre_Process must run first)
    }
    std::string newPrompt;
    if (system_prompt == nullptr) {
        newPrompt.clear();
    } else {
        const char* text = env->GetStringUTFChars(system_prompt, nullptr);
        if (text == nullptr) {
            return JNI_FALSE;
        }
        newPrompt.assign(text);
        env->ReleaseStringUTFChars(system_prompt, text);
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

// JNI: full generation.
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_Run_1LLM(JNIEnv* env, jclass clazz, jstring jquery,
                                                     jboolean clear, jboolean use_think, jint turn_id) {
    if (g_busy.exchange(true, std::memory_order_acq_rel)) {
        return env->NewStringUTF("");
    }
    struct BusyGuard { ~BusyGuard() { g_busy.store(false, std::memory_order_release); } } busy_guard;
    const bool pendingPrestartCancel = g_pending_prestart_cancel.exchange(false, std::memory_order_acq_rel);
    g_pending_prestart_manual_stop.store(false, std::memory_order_release);
    g_cancel.store(false, std::memory_order_relaxed);
    g_manual_stop.store(false, std::memory_order_relaxed);
    if (pendingPrestartCancel) {
        return env->NewStringUTF("");
    }
    waitForCleanCommit();
    if (g_cancel.load(std::memory_order_relaxed)) {
        return env->NewStringUTF("");
    }

    const OrtApi* api = mMain.api;
    if (api == nullptr || mMain.session == nullptr || mEmbed.session == nullptr ||
        mRotaryPrefill.session == nullptr || mRotaryDecode.session == nullptr || tokenizer == nullptr) {
        return env->NewStringUTF("");
    }
    if (jquery == nullptr) {
        return env->NewStringUTF("");
    }

    // Turn setup.
    if (clear == JNI_TRUE) {
        clear_history();
    }
    // Retry once with full-history prefill if a shortened prefill collapses immediately.
    bool forceFullPrefill = false;
    int  genAttempt = 0;
    const MemoryProfile memory = snapshotMemoryProfile();
    bool organizeMemory;
    {
        std::lock_guard<std::mutex> lock(g_runtime_config_mutex);
        organizeMemory = g_organize_memory;
    }
    if (saved_kv_len > memory.memoryTokens) {
        if (!rebuildSavedKVFromHistory(memory.memoryTokens)) {
            clear_history();
        }
    }
    const char* query = env->GetStringUTFChars(jquery, nullptr);
    if (query == nullptr) {
        return env->NewStringUTF("");
    }
    std::vector<int> get_ids;
    {
        std::lock_guard<std::mutex> tokenizerLock(g_tokenizer_mutex);
        get_ids = tokenizer->encode(query);
    }
    env->ReleaseStringUTFChars(jquery, query);
    // Chat template; previous assistant close is prepended only when history exists.
    get_ids.insert(get_ids.begin(), chat_user_prefix_ids.begin(), chat_user_prefix_ids.end());
    get_ids.insert(get_ids.end(), chat_assistant_prefix_ids.begin(), chat_assistant_prefix_ids.end());
    if (use_think != JNI_TRUE) {
        get_ids.insert(get_ids.end(), chat_empty_think_block_ids.begin(), chat_empty_think_block_ids.end());
    }
    // System block is injected live; organize-memory decides whether it persists cross-turn.
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
    const bool injectSystem = activeSystemBlockLen > 0;
    if (injectSystem) {
        get_ids.insert(get_ids.begin(), activeSystemBlock.begin(), activeSystemBlock.end());
    }
    const int64_t dialogueSkip = (injectSystem && organizeMemory) ? activeSystemBlockLen : 0;
    std::vector<int> dialogueBaseIds(get_ids.begin() + dialogueSkip, get_ids.end());

    // Decode strategy.
    const DecodeStrategy strat = resolveDecodeStrategy();
    const bool useBeam = strat.beam;
    bool usePen = strat.penalty;
    bool useArgmax = strat.argmax;
    bool useGreedy = strat.greedy;
    const int  decodeBatch = strat.decodeBatch;
    // Top up the shared-arena reservation for this turn's actual decode batch (beam expands the cache).
    reserveSharedArena(memory.memoryTokens, decodeBatch);
    const int  beamSize = strat.beamSize;
    const float repeatPenalty = strat.repeatPenalty;
    bool escapePenaltyActive = false;
    const bool canUseEscapePenalty = !useBeam && mGreedy.session != nullptr && mPenalty.session != nullptr;

    // Retained KV is batch-1; beam expands only after prefill.
    int64_t history_pos = saved_kv_base + saved_kv_len;
    constexpr int64_t decodeBudgetMargin = 2;
    const int prev_close_len = static_cast<int>(chat_previous_assistant_close_ids.size());
    const int sep_len = (saved_kv_len > 0) ? prev_close_len : 0;
    const auto core_len = static_cast<int64_t>(get_ids.size());
    const int64_t prompt_span = static_cast<int64_t>(sep_len) + core_len;
    // Streaming relies on hard RoPE recycle; beam reserves the whole decode budget up front. Hybrid
    // (linear-attention) models can't slide/rope-shift their recurrent state, so they also reserve.
    const bool recyclerDrivesDecode = !useBeam && !hasLinearState() &&
            mKVSlice.session != nullptr && mRopeShift.session != nullptr;
    const int64_t decode_reserve = std::max<int64_t>(0,
            std::min<int64_t>(recyclerDrivesDecode ? static_cast<int64_t>(kDecodeRecycleHeadroom)
                                                   : static_cast<int64_t>(memory.decodeTokens),
                              static_cast<int64_t>(max_seq_len) - prompt_span - decodeBudgetMargin));
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
            clear_history();
            history_pos = 0;
            if (core_len + decodeBudgetMargin > max_seq_len) {
                return env->NewStringUTF("Over_Inputs");
            }
        }
    }
    // Trim history before prefill if prompt + retained KV enters the red zone.
    if (saved_kv_len > 0) {
        const int64_t livePromptLen = core_len + prev_close_len;
        int64_t targetHistoryLen = static_cast<int64_t>(memory.memoryTokens) - livePromptLen;
        const int64_t liveUsedAfterPrompt = saved_kv_len + livePromptLen;
        if (memoryUsedInRedZone(memory.memoryTokens, liveUsedAfterPrompt)) {
            targetHistoryLen = static_cast<int64_t>(memoryGreenTargetUsed(memory.memoryTokens)) - livePromptLen;
        }
        if (targetHistoryLen <= 0) {
            clear_history();
            history_pos = 0;
        } else if (saved_kv_len > targetHistoryLen) {
            if (rebuildSavedKVFromHistoryRetaining(static_cast<int>(targetHistoryLen))) {
                history_pos = saved_kv_base + saved_kv_len;
            } else {
                clear_history();
                history_pos = 0;
            }
        }
    }
    const bool reusedHistory = (saved_kv_len > 0);
    if (reusedHistory) {
        get_ids.insert(get_ids.begin(), chat_previous_assistant_close_ids.begin(),
                   chat_previous_assistant_close_ids.end());
    }
    ids_len = static_cast<int64_t>(get_ids.size());
    // ids_len is global; keep a local for final throughput after possible cleanup/reset.
    const int64_t prefillTokenCount = ids_len;
    record_turn_checkpoint(static_cast<int32_t>(turn_id),
                           static_cast<int64_t>(g_history_ids.size()), saved_kv_len);

    if (useGreedy && !mGreedy.session) {
        return env->NewStringUTF("");
    }
    if (!ensureBinding(mEmbed) || !ensureBinding(mMain) || !ensureBinding(mRotaryPrefill) ||
        !ensureBinding(mRotaryDecode) ||
        (useArgmax && !ensureBinding(mArgmax)) ||
        ((useGreedy || canUseEscapePenalty) && !ensureBinding(mGreedy)) ||
        ((usePen || canUseEscapePenalty) && !ensureBinding(mPenalty)) ||
        (useBeam && (!ensureBinding(mFirstBeam) || !ensureBinding(mSecondBeam)))) {
        return env->NewStringUTF("");
    }

    // Prefill-window decoupling: shorten prefill attention, then concat full KV back for decode.
retry_generation:
    // Prefill-window decoupling splits/concats the full-attention KV; it is disabled for hybrid models
    // because the linear recurrent state cannot be windowed to match a shortened full-attention window.
    const bool wantDecouple = !forceFullPrefill && !hasLinearState() && reusedHistory && saved_kv[0] != nullptr &&
            memory.prefillTokens > 0 &&
            saved_kv_len > static_cast<int64_t>(memory.prefillTokens) &&
            ids_len >= static_cast<int64_t>(memory.prefillDecoupleMinNewTokens) &&
            mKVSplit2.session != nullptr && mKVConcat.session != nullptr;
    const bool prefillZeroWindow = !forceFullPrefill && !hasLinearState() && reusedHistory && saved_kv[0] != nullptr &&
            memory.prefillTokens == 0 && mKVConcat.session != nullptr;
    const int64_t prefillCacheLen = prefillZeroWindow ? 0
            : (wantDecouple ? static_cast<int64_t>(memory.prefillTokens) : saved_kv_len);
    const int64_t decoupleDropLen = wantDecouple ? (saved_kv_len - static_cast<int64_t>(memory.prefillTokens)) : 0;

    std::vector<OrtValue*> oneTime;
    oneTime.reserve(8);
    auto keep = [&oneTime](OrtValue* v) { if (v) oneTime.push_back(v); };
    std::vector<OrtValue*> setupOut;
    setupOut.reserve(8);

    // Embed prompt.
    std::vector<int64_t> idsShape = {1, ids_len};
    OrtValue* idsTensor = nullptr;
    if (!logOrtStatus(api, api->CreateTensorWithDataAsOrtValue(
            mEmbed.memoryInfo, reinterpret_cast<void*>(get_ids.data()),
            static_cast<size_t>(ids_len) << input_ids_elem_shift,
            idsShape.data(), idsShape.size(), mEmbed.inputTypes[0], &idsTensor),
            "CreateTensorWithDataAsOrtValue(input_ids)")) {
        return env->NewStringUTF("");
    }
    resetBinding(mEmbed);
    bindIn(mEmbed, 0, idsTensor);
    bindOutDevice(mEmbed, 0);
    if (!runBinding(mEmbed) || !fetchOutputsInto(mEmbed, setupOut)) {
        releaseOne(api, idsTensor);
        return env->NewStringUTF("");
    }
    releaseOne(api, idsTensor);
    OrtValue* prefillHidden = nullptr;
    if (!takeFirstOutput(api, setupOut, prefillHidden)) {
        return env->NewStringUTF("");
    }
    keep(prefillHidden);
    maybeInjectVisionEmbeddings(prefillHidden, get_ids);   // reserved no-op until a vision model is wired in
    const int hiddenSize = static_cast<int>(mEmbed.outputDims[0].back());
    const ONNXTensorElementDataType hiddenType = mEmbed.outputTypes[0];

    // Rotary prefill emits cos/sin, append-aware mask, and kv_seq_len.
    int64_t idsLenVal = ids_len;
    int64_t historyLenVal = history_pos;
    int64_t cacheLenVal = prefillCacheLen;   // window width when prefill-window decoupling is active
    OrtValue* idsLenTensor = nullptr;
    OrtValue* historyLenTensor = nullptr;
    OrtValue* cacheLenTensor = nullptr;
    int64_t scalarShape[1] = {1};
    if (!logOrtStatus(api, api->CreateTensorWithDataAsOrtValue(
            mRotaryPrefill.memoryInfo, &idsLenVal, sizeof(int64_t), scalarShape, 1,
            mRotaryPrefill.inputTypes[0], &idsLenTensor), "CreateTensorWithDataAsOrtValue(ids_len)") ||
        !logOrtStatus(api, api->CreateTensorWithDataAsOrtValue(
            mRotaryPrefill.memoryInfo, &historyLenVal, sizeof(int64_t), scalarShape, 1,
            mRotaryPrefill.inputTypes[1], &historyLenTensor), "CreateTensorWithDataAsOrtValue(history_len)")) {
        releaseOne(api, idsLenTensor);
        releaseValues(api, oneTime);
        return env->NewStringUTF("");
    }
    keep(idsLenTensor);
    keep(historyLenTensor);
    if (!logOrtStatus(api, api->CreateTensorWithDataAsOrtValue(
            mRotaryPrefill.memoryInfo, &cacheLenVal, sizeof(int64_t), scalarShape, 1,
            mRotaryPrefill.inputTypes[2], &cacheLenTensor), "CreateTensorWithDataAsOrtValue(cache_len)")) {
        releaseValues(api, oneTime);
        return env->NewStringUTF("");
    }
    keep(cacheLenTensor);
    resetBinding(mRotaryPrefill);
    bindIn(mRotaryPrefill, 0, idsLenTensor);
    bindIn(mRotaryPrefill, 1, historyLenTensor);
    bindIn(mRotaryPrefill, 2, cacheLenTensor);
    bindOutDevice(mRotaryPrefill, 0);
    bindOutDevice(mRotaryPrefill, 1);
    bindOutDevice(mRotaryPrefill, 2);
    bindOutDevice(mRotaryPrefill, 3);
    if (!runBinding(mRotaryPrefill) || !fetchOutputsInto(mRotaryPrefill, setupOut) || setupOut.size() < 4) {
        releaseValues(api, setupOut);
        releaseValues(api, oneTime);
        return env->NewStringUTF("");
    }
    OrtValue* prefillCos     = setupOut[0];
    OrtValue* prefillSin     = setupOut[1];
    OrtValue* kvSeqLenTensor = setupOut[3];   // incremented in-place by each Rotary-Decode step
    keep(prefillCos); keep(prefillSin); keep(kvSeqLenTensor);
    OrtValue* prefillMaskFromRotary = setupOut[2];
    keep(prefillMaskFromRotary);
    for (size_t k = 4; k < setupOut.size(); ++k) releaseOne(api, setupOut[k]);
    setupOut.clear();

    // Reusable IOBinding buffers.
    const int vocabSize = static_cast<int>(mMain.outputDims[mainLogitsOutIdx].back());
    const ONNXTensorElementDataType logitsType = mMain.outputTypes[mainLogitsOutIdx];

    OrtBuffer prefillLogitsBuf = makeBuffer(mMain, {1, (int64_t)vocabSize}, logitsType);
    OrtBuffer decodeLogitsBuf  = makeBuffer(mMain, {(int64_t)decodeBatch, (int64_t)vocabSize}, logitsType);
    OrtBuffer hiddenBuf        = makeBuffer(mMain, {(int64_t)decodeBatch, 1, (int64_t)hiddenSize}, hiddenType);
    OrtBuffer cosBuf           = makeBuffer(mMain, mRotaryDecode.outputDims[0], mRotaryDecode.outputTypes[0]);
    OrtBuffer sinBuf           = makeBuffer(mMain, mRotaryDecode.outputDims[1], mRotaryDecode.outputTypes[1]);
    OrtBuffer maskBuf          = makeBuffer(mMain, {1, 1, 1, 1, 1}, mMain.inputTypes[mainMaskIdx]);
    OrtValue* prefillMaskValue = prefillMaskFromRotary;
    OrtBuffer maxIdBuf         = makeBuffer(mMain, {1, 1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    OrtBuffer saveIdEmptyBuf   = makeBuffer(mMain, {(int64_t)decodeBatch, 0}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);

    OrtBuffer beamIdsBuf, beamScoreBuf, beamSizeBuf, topKBuf;
    if (useBeam) {
        beamIdsBuf   = makeBuffer(mMain, {(int64_t)beamSize, 1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
        beamScoreBuf = makeBuffer(mMain, {(int64_t)beamSize, 1}, logitsType);
        beamSizeBuf  = makeBuffer(mMain, {1}, mFirstBeam.inputTypes[firstBeamSizeIdx]);
        topKBuf      = makeBuffer(mMain, {1}, mSecondBeam.inputTypes[secondBeamTopKIdx]);
        *reinterpret_cast<int64_t*>(beamSizeBuf.data()) = beamSize;
        *reinterpret_cast<int64_t*>(topKBuf.data())     = strat.effectiveTopK;
    }

    OrtBuffer penaltyValBuf, penaltyRangeBuf;
    if (usePen || canUseEscapePenalty) {
        if (mPenalty.inputNames.size() > 2) {
            penaltyValBuf = makeBuffer(mMain, {1}, mPenalty.inputTypes[2]);
            writeScalarFloat(penaltyValBuf, mPenalty.inputTypes[2], repeatPenalty);
        }
        if (mPenalty.inputNames.size() > 3) {
            penaltyRangeBuf = makeBuffer(mMain, {1}, mPenalty.inputTypes[3]);
            *reinterpret_cast<int64_t*>(penaltyRangeBuf.data()) = PENALTY_RANGE;
        }
    }

    // Bind-once per-step models.
    resetBinding(mEmbed);
    bindIn(mEmbed, 0, useBeam ? beamIdsBuf.value : maxIdBuf.value);
    bindOut(mEmbed, 0, hiddenBuf.value);

    resetBinding(mRotaryDecode);
    bindIn(mRotaryDecode, 0, kvSeqLenTensor);
    bindOut(mRotaryDecode, 0, cosBuf.value);
    bindOut(mRotaryDecode, 1, sinBuf.value);
    bindOut(mRotaryDecode, 2, kvSeqLenTensor);   // in-place increment

    // Decode state.
    std::vector<OrtValue*> mainKV;       // current Main KV outputs = next Main KV inputs (non-beam)
    std::vector<OrtValue*> beamKV;       // current beam-reordered KV = next Main KV inputs (beam)
    std::vector<OrtValue*> nextMainKV;
    std::vector<OrtValue*> nextBeamKV;
    std::vector<OrtValue*> beamOut;
    std::vector<OrtValue*> greedyOut;
    std::vector<OrtValue*> windowKV;     // prefill-decouple: most-recent prefillTokens fed to the prefill
    std::vector<OrtValue*> prefixKV;     // prefill-decouple: older history concatenated back after prefill
    std::vector<OrtValue*> mainLinear;   // hybrid linear-attention passthrough states (empty for pure transformers)
    std::vector<OrtValue*> beamLinear;
    std::vector<OrtValue*> nextMainLinear;
    std::vector<OrtValue*> nextBeamLinear;
    mainKV.reserve(num_keys_values);
    beamKV.reserve(num_keys_values);
    nextMainKV.reserve(num_main_states + 1);   // KV + linear outputs + logits OrtValue handle
    nextBeamKV.reserve(num_keys_values);
    beamOut.reserve(num_main_states + 4);      // reordered KV + linear + save_ids + score/ids/max handles
    greedyOut.reserve(2);                      // max + updated save_ids handles
    mainLinear.reserve(num_linear_states);
    beamLinear.reserve(num_linear_states);
    nextMainLinear.reserve(num_linear_states);
    nextBeamLinear.reserve(num_linear_states);
    OrtValue* saveIdDev = nullptr;       // greedy/beam save_ids (device), grows each step
    std::vector<int>     recentDecodedIds;
    std::vector<int32_t> escapeSeedData;
    if (canUseEscapePenalty) {
        recentDecodedIds.reserve(static_cast<size_t>(PENALTY_RANGE));
    }

    const auto* const maxIdPtr = reinterpret_cast<const int32_t*>(maxIdBuf.data());

    auto cleanup = [&]() {
        releaseOne(api, saveIdDev);
        releaseValues(api, mainKV);
        releaseValues(api, beamKV);
        releaseValues(api, nextMainKV);
        releaseValues(api, nextBeamKV);
        releaseValues(api, beamOut);
        releaseValues(api, greedyOut);
        releaseValues(api, windowKV);
        releaseValues(api, prefixKV);
        releaseValues(api, mainLinear);
        releaseValues(api, beamLinear);
        releaseValues(api, nextMainLinear);
        releaseValues(api, nextBeamLinear);
        releaseValues(api, oneTime);
        releaseBuffer(api, prefillLogitsBuf);
        releaseBuffer(api, decodeLogitsBuf);
        releaseBuffer(api, hiddenBuf);
        releaseBuffer(api, cosBuf);
        releaseBuffer(api, sinBuf);
        releaseBuffer(api, maskBuf);
        releaseBuffer(api, maxIdBuf);
        releaseBuffer(api, saveIdEmptyBuf);
        releaseBuffer(api, beamIdsBuf);
        releaseBuffer(api, beamScoreBuf);
        releaseBuffer(api, beamSizeBuf);
        releaseBuffer(api, topKBuf);
        releaseBuffer(api, penaltyValBuf);
        releaseBuffer(api, penaltyRangeBuf);
        clearAllModelBindingRefs();
    };

    // Streaming clock.
    reset_output_words_decoder();
    stream_buf.clear();
    tokens_since_flush = 0;
    const auto prefill_start_clock = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point decode_start_clock = prefill_start_clock;
    const int64_t prefill_start_ms = now_ms();
    last_flush_ms = prefill_start_ms;
    int64_t last_perf_ms = prefill_start_ms;
    std::chrono::steady_clock::time_point last_perf_clock = prefill_start_clock;
    int last_perf_decoded = 0;
    bool prefillStatsSent = false;
    constexpr int32_t PERF_UPDATE_MS = 500;

    const int rotaryDecodeBudget = static_cast<int>(std::max<int64_t>(
            0, static_cast<int64_t>(max_seq_len) - (history_pos + ids_len) - decodeBudgetMargin));
    // Hybrid models have no in-turn RoPE recycle, so (like beam) they cap decode to the rotary budget.
    const bool decodeBudgetBounded = useBeam || hasLinearState();
    const int decodeBudget = decodeBudgetBounded ? std::min(rotaryDecodeBudget, memory.decodeTokens)
                                                 : memory.decodeTokens;
    int64_t absNext = history_pos + ids_len;
    int64_t decodeBase = saved_kv_base;   // absolute RoPE pos of the oldest live cached token (0 after a slide)
    bool isPrefill = true;
    bool mainDecodeFixedBound = false;
    bool argmaxDecodeBound = false;   // Argmax's logits input / max_id output are fixed once decode starts
    int maxId = -1;
    int numDecode = 0;
    int generated = 0;   // committed (streamed) reply tokens; drives the throughput figure
    std::vector<int> replyVec;   // answer-only reply ids (think excluded); used by the ON-mode clean recompute
    std::vector<int> fullReplyVec;   // OFF mode: EVERY decoded token (think included) to mirror the saved decode KV
    const size_t replyReserve = static_cast<size_t>(std::min(decodeBudget, 256));
    replyVec.reserve(replyReserve);
    if (!organizeMemory) {
        fullReplyVec.reserve(replyReserve);   // only OFF mode records every decoded token here
    }
    int replyThinkDepth = 0;     // generated <think> spans are display-only and excluded from clean history
    int beamMemoryReplyTokens = 0;
    bool preparedTokenPendingKV = false;
    bool kvStateValid = true;

    const int livePromptMemoryTokens = static_cast<int>(dialogueBaseIds.size()) + (reusedHistory ? prev_close_len : 0);
    auto remainingFromUsedTokens = [&](int64_t usedTokens) -> int {
        const int used = clampInt(static_cast<int>(std::min<int64_t>(std::max<int64_t>(0, usedTokens),
                                                                     memory.memoryTokens)),
                                  0, memory.memoryTokens);
        return std::max(0, memory.memoryTokens - used);
    };
    auto liveMemoryRemainingTokens = [&]() -> int {
        const int64_t replyTokens = static_cast<int64_t>(
                organizeMemory ? replyVec.size() : fullReplyVec.size());
        const int64_t liveReplyTokens = useBeam
                ? std::max<int64_t>(beamMemoryReplyTokens, replyTokens)
                : replyTokens;
        const int64_t liveUsed = saved_kv_len + static_cast<int64_t>(livePromptMemoryTokens) + liveReplyTokens;
        return remainingFromUsedTokens(liveUsed);
    };

    auto countCleanReplyTokens = [&](const std::vector<int>& ids) -> int {
        int cleanTokens = 0;
        int thinkDepth = 0;
        for (int id : ids) {
            if (is_stop_token(id)) {
                break;
            }
            if (id == chat_think_start_id) {
                thinkDepth += 1;
                continue;
            }
            if (id == chat_think_end_id) {
                if (thinkDepth > 0) {
                    thinkDepth -= 1;
                }
                continue;
            }
            if (thinkDepth == 0) {
                cleanTokens += 1;
            }
        }
        return cleanTokens;
    };

    auto countFullReplyTokens = [&](const std::vector<int>& ids) -> int {
        int tokens = 0;
        for (int id : ids) {
            if (is_stop_token(id)) {
                break;
            }
            tokens += 1;
        }
        return tokens;
    };

    auto refreshBeamMemoryReplyTokens = [&]() {
        if (useBeam && saveIdDev != nullptr) {
            const std::vector<int> best = extractFirstRowInt32(api, saveIdDev);
            beamMemoryReplyTokens = organizeMemory ? countCleanReplyTokens(best)
                                                   : countFullReplyTokens(best);
        }
    };

    auto appendReplyTokenForHistory = [&](int id) {
        if (id == chat_think_start_id) {
            replyThinkDepth += 1;
            return;
        }
        if (id == chat_think_end_id) {
            if (replyThinkDepth > 0) {
                replyThinkDepth -= 1;
            }
            return;
        }
        if (replyThinkDepth == 0) {
            replyVec.push_back(id);
        }
    };

    auto appendPreparedTokenToKV = [&]() -> bool {
        std::vector<OrtValue*>& kvTarget  = useBeam ? beamKV     : mainKV;
        std::vector<OrtValue*>& linTarget = useBeam ? beamLinear : mainLinear;
        if (kvTarget.size() < static_cast<size_t>(num_keys_values) || kvTarget[0] == nullptr) {
            LOGE("KV append flush skipped: current KV is empty");
            return false;
        }

        resetBinding(mMain);
        bool ok = true;
        for (int i = 0; i < num_keys_values; ++i) {
            ok = bindIn(mMain, i, kvTarget[i]) && ok;
        }
        for (int i = 0; i < num_linear_states; ++i) {
            ok = bindIn(mMain, num_keys_values + i, linTarget[i]) && ok;
        }
        ok = bindIn(mMain, mainHiddenIdx, hiddenBuf.value) && ok;
        ok = bindIn(mMain, mainCosIdx,    cosBuf.value)    && ok;
        ok = bindIn(mMain, mainSinIdx,    sinBuf.value)    && ok;
        ok = bindIn(mMain, mainMaskIdx,   maskBuf.value)   && ok;
        for (int i = 0; i < num_keys_values; ++i) {
            ok = bindOutDevice(mMain, i) && ok;
        }
        for (int i = 0; i < num_linear_states; ++i) {
            ok = bindOutDevice(mMain, num_keys_values + i) && ok;
        }
        ok = bindOut(mMain, mainLogitsOutIdx, decodeLogitsBuf.value) && ok;
        ok = ok && runBinding(mMain) && fetchOutputsInto(mMain, nextMainKV);
        if (!ok || nextMainKV.size() < static_cast<size_t>(num_main_states)) {
            releaseValues(api, nextMainKV);
            LOGE("KV append flush failed; dropping saved history to avoid stale context");
            return false;
        }
        nextMainLinear.assign(nextMainKV.begin() + num_keys_values, nextMainKV.begin() + num_main_states);
        for (size_t i = static_cast<size_t>(num_main_states); i < nextMainKV.size(); ++i) {
            releaseOne(api, nextMainKV[i]);
        }
        nextMainKV.resize(num_keys_values);
        releaseValues(api, kvTarget);
        kvTarget.swap(nextMainKV);
        releaseValues(api, linTarget);
        linTarget.swap(nextMainLinear);
        return true;
    };

    // Split retained KV into prefix/window for prefill decoupling.
    if (wantDecouple) {
        std::vector<OrtValue*> savedVec(saved_kv.begin(), saved_kv.begin() + num_keys_values);
        std::vector<OrtValue*> prefixSlice, windowSlice;
        if (runKVSplit2(savedVec, decoupleDropLen, prefixSlice, windowSlice)) {
            prefixKV.swap(prefixSlice);
            windowKV.swap(windowSlice);
            // OFF no longer needs saved_kv after split; releasing it cuts peak KV memory.
            if (!organizeMemory && !useBeam) {
                release_saved_kv();
            }
        } else {
            releaseValues(api, prefixSlice);
            releaseValues(api, windowSlice);
            LOGE("Prefill-window slice failed; aborting turn (history preserved)");
            cleanup();
            return env->NewStringUTF("");
        }
    }

    // Decode loop: Main -> [Penalty] -> Beam/Greedy/Argmax -> Embed -> RotaryDecode.
    while (LIKELY(numDecode < decodeBudget)) {
        if (isPrefill) {
            resetBinding(mMain);
            for (int i = 0; i < num_keys_values; ++i) {
                OrtValue* kvIn = !windowKV.empty() ? windowKV[i]
                               : ((reusedHistory && !prefillZeroWindow) ? saved_kv[i]
                                                                        : input_tensors_kv_init[i]);
                bindIn(mMain, i, kvIn);
            }
            for (int i = 0; i < num_linear_states; ++i) {
                OrtValue* linIn = (reusedHistory && saved_linear_valid) ? saved_linear_states[i]
                                                                        : linear_state_init[i];
                bindIn(mMain, num_keys_values + i, linIn);
            }
            bindIn(mMain, mainHiddenIdx, prefillHidden);
            bindIn(mMain, mainCosIdx,    prefillCos);
            bindIn(mMain, mainSinIdx,    prefillSin);
            bindIn(mMain, mainMaskIdx,   prefillMaskValue);
            for (int i = 0; i < num_keys_values; ++i) bindOutDevice(mMain, i);
            for (int i = 0; i < num_linear_states; ++i) bindOutDevice(mMain, num_keys_values + i);
            bindOut(mMain, mainLogitsOutIdx, prefillLogitsBuf.value);
        } else {
            const std::vector<OrtValue*>& kvIn  = useBeam ? beamKV     : mainKV;
            const std::vector<OrtValue*>& linIn = useBeam ? beamLinear : mainLinear;
            for (int i = 0; i < num_keys_values; ++i) {
                bindIn(mMain, i, kvIn[i]);
            }
            for (int i = 0; i < num_linear_states; ++i) {
                bindIn(mMain, num_keys_values + i, linIn[i]);
            }
            if (UNLIKELY(!mainDecodeFixedBound)) {
                bindIn(mMain, mainHiddenIdx, hiddenBuf.value);
                bindIn(mMain, mainCosIdx,    cosBuf.value);
                bindIn(mMain, mainSinIdx,    sinBuf.value);
                bindIn(mMain, mainMaskIdx,   maskBuf.value);   // (1,1,1,1,1) zeros: decode attends to all
                bindOut(mMain, mainLogitsOutIdx, decodeLogitsBuf.value);
                mainDecodeFixedBound = true;
            }
            for (int i = 0; i < num_keys_values; ++i) bindOutDevice(mMain, i);
            for (int i = 0; i < num_linear_states; ++i) bindOutDevice(mMain, num_keys_values + i);
        }
        if (!runBinding(mMain) || !fetchOutputsInto(mMain, nextMainKV) ||
            nextMainKV.size() < static_cast<size_t>(num_main_states)) {
            if (!isPrefill && preparedTokenPendingKV) {
                kvStateValid = false;
            }
            releaseValues(api, nextMainKV);
            break;
        }
        // Fetched order: [full KV] [linear states] [logits]. Peel linear off, release the logits handle.
        nextMainLinear.assign(nextMainKV.begin() + num_keys_values, nextMainKV.begin() + num_main_states);
        for (size_t i = static_cast<size_t>(num_main_states); i < nextMainKV.size(); ++i) {
            releaseOne(api, nextMainKV[i]);     // logits read from the buffer
        }
        nextMainKV.resize(num_keys_values);
        if (!isPrefill) {
            if (useBeam) { releaseValues(api, beamKV); releaseValues(api, beamLinear); }   // consumed by this Main run
            else         { releaseValues(api, mainKV); releaseValues(api, mainLinear); }
        }
        mainKV.swap(nextMainKV);
        mainLinear.swap(nextMainLinear);
        if (!isPrefill) {
            preparedTokenPendingKV = false;
        }
        if (isPrefill) {
            // Restore full KV after decoupled prefill.
            if (!windowKV.empty()) {
                releaseValues(api, windowKV);
            }
            if (!prefixKV.empty()) {
                std::vector<OrtValue*> fullKV;
                if (runKVConcat(prefixKV, mainKV, fullKV)) {
                    releaseValues(api, mainKV);
                    mainKV.swap(fullKV);
                    releaseValues(api, prefixKV);
                } else {
                    releaseValues(api, prefixKV);
                    decodeBase = saved_kv_base + decoupleDropLen;
                    LOGE("KV_Concat failed after prefill; dropping the older prefix for this turn");
                }
            } else if (prefillZeroWindow) {
                std::vector<OrtValue*> savedRef(saved_kv.begin(), saved_kv.begin() + num_keys_values);
                std::vector<OrtValue*> fullKV;
                if (runKVConcat(savedRef, mainKV, fullKV)) {
                    releaseValues(api, mainKV);
                    mainKV.swap(fullKV);
                } else {
                    decodeBase = history_pos;
                    LOGE("KV_Concat failed for zero prefill window; decode attends the new prompt only this turn");
                }
            }
            // ON/beam need saved_kv for clean commit; OFF saves mainKV directly.
            if (!organizeMemory && !useBeam) {
                release_saved_kv();
                release_saved_linear();   // OFF direct-saves the decode linear state at turn end
            }
            decode_start_clock = std::chrono::steady_clock::now();
            const int64_t nowMs = now_ms();
            last_flush_ms = nowMs;
            last_perf_ms = nowMs;
            last_perf_clock = decode_start_clock;   // decode throughput measured from the first token
            const float prefill_s = std::chrono::duration<float>(decode_start_clock - prefill_start_clock).count();
            const float prefill_rate = (prefill_s > 0.0f)
                    ? (static_cast<float>(prefillTokenCount) / prefill_s) : 0.0f;
            pushPerfRates(env, true, prefill_rate, false, 0.0f,
                          liveMemoryRemainingTokens(), memory.memoryTokens,
                          static_cast<int>(prefillTokenCount), -1);
            prefillStatsSent = true;
        }

        OrtValue* logits = isPrefill ? prefillLogitsBuf.value : decodeLogitsBuf.value;

        // Token selection.
        if (usePen && numDecode >= PENALTY_RANGE && saveIdDev) {
            resetBinding(mPenalty);
            bindIn(mPenalty, 0, logits);
            bindIn(mPenalty, 1, saveIdDev);
            if (penaltyValBuf.value)   bindIn(mPenalty, 2, penaltyValBuf.value);
            if (penaltyRangeBuf.value) bindIn(mPenalty, 3, penaltyRangeBuf.value);
            bindOut(mPenalty, 0, logits);
            if (!runBinding(mPenalty)) {
                break;
            }
        }

        OrtValue* saveIdInput = saveIdDev ? saveIdDev : saveIdEmptyBuf.value;

        if (useBeam) {
            ModelRuntime& beam = isPrefill ? mFirstBeam : mSecondBeam;
            resetBinding(beam);
            for (int i = 0; i < num_keys_values; ++i) bindIn(beam, i, mainKV[i]);
            for (int i = 0; i < num_linear_states; ++i) bindIn(beam, num_keys_values + i, mainLinear[i]);
            bindIn(beam, mainLogitsOutIdx, logits);
            bindIn(beam, beamSaveIdInIdx, saveIdInput);
            if (isPrefill) {
                bindIn(beam, firstBeamSizeIdx, beamSizeBuf.value);
            } else {
                bindIn(beam, secondBeamPrevProbIdx, beamScoreBuf.value);
                bindIn(beam, secondBeamSizeIdx,     beamSizeBuf.value);
                bindIn(beam, secondBeamTopKIdx,     topKBuf.value);
            }
            for (int i = 0; i < num_keys_values; ++i) bindOutDevice(beam, i);   // reordered KV
            for (int i = 0; i < num_linear_states; ++i) bindOutDevice(beam, num_keys_values + i); // reordered linear
            bindOutDevice(beam, beamSaveIdOutIdx);                              // save_ids
            bindOut(beam, beamScoreOutIdx, beamScoreBuf.value);                 // score (in-place)
            bindOut(beam, beamIdsOutIdx,   beamIdsBuf.value);                   // beam_ids
            bindOut(beam, beamMaxOutIdx,   maxIdBuf.value);                     // max_id
            if (!runBinding(beam) || !fetchOutputsInto(beam, beamOut) ||
                beamOut.size() < static_cast<size_t>(num_main_states + 1)) {
                releaseValues(api, beamOut);
                break;
            }
            releaseValues(api, mainKV);       // Main KV consumed by the beam step
            releaseValues(api, mainLinear);   // Main linear consumed by the beam step

            maxId = *maxIdPtr;
            // Keep reordered beams even on stop so finalize can stream row-0 save_ids.
            nextBeamKV.assign(beamOut.begin(), beamOut.begin() + num_keys_values);
            nextBeamLinear.assign(beamOut.begin() + num_keys_values, beamOut.begin() + num_main_states);
            for (int i = 0; i < num_main_states; ++i) beamOut[i] = nullptr;
            OrtValue* newSaveId = beamOut[num_main_states];
            beamOut[num_main_states] = nullptr;
            for (size_t k = num_main_states + 1; k < beamOut.size(); ++k) {
                releaseOne(api, beamOut[k]);   // score/ids/max read from their buffers
            }
            beamOut.clear();
            beamKV.swap(nextBeamKV);
            beamLinear.swap(nextBeamLinear);
            releaseOne(api, saveIdDev);
            saveIdDev = newSaveId;

        } else if (useArgmax) {
            // IOBinding-once: Argmax reads the fixed decode logits buffer and writes the fixed max_id
            // buffer, so bind only until the decode logits buffer is live (prefill uses a different
            // buffer), then just re-run each step. Mirrors mainDecodeFixedBound above.
            if (UNLIKELY(!argmaxDecodeBound)) {
                resetBinding(mArgmax);
                bindIn(mArgmax, 0, logits);
                bindOut(mArgmax, 0, maxIdBuf.value);
                if (!isPrefill) {
                    argmaxDecodeBound = true;
                }
            }
            if (!runBinding(mArgmax)) {
                break;
            }
            maxId = *maxIdPtr;

        } else {   // greedy (+ penalty)
            resetBinding(mGreedy);
            bindIn(mGreedy, 0, logits);
            bindIn(mGreedy, 1, saveIdInput);
            bindOut(mGreedy, 0, maxIdBuf.value);
            bindOutDevice(mGreedy, 1);   // updated save_ids -> device
            if (!runBinding(mGreedy) || !fetchOutputsInto(mGreedy, greedyOut) || greedyOut.size() < 2) {
                releaseValues(api, greedyOut);
                break;
            }
            releaseOne(api, greedyOut[0]);   // max read from the buffer
            OrtValue* newSaveId = greedyOut[1];
            greedyOut[1] = nullptr;
            releaseValues(api, greedyOut);
            releaseOne(api, saveIdDev);
            saveIdDev = newSaveId;
            maxId = *maxIdPtr;
        }

        if (UNLIKELY(is_stop_token(maxId))) {
            break;
        }
        if (UNLIKELY(g_cancel.load(std::memory_order_relaxed))) {
            break;
        }

        if (!useBeam) {
            generated += 1;
            appendReplyTokenForHistory(maxId);
            if (!organizeMemory) {
                fullReplyVec.push_back(maxId);   // OFF mode persists the decode KV as-is -> record every token
            }
            stream_token(env, maxId);
            if (canUseEscapePenalty && useArgmax) {
                recentDecodedIds.push_back(maxId);
                if (recentDecodedIds.size() > static_cast<size_t>(PENALTY_RANGE)) {
                    recentDecodedIds.erase(recentDecodedIds.begin());
                }
            }
        }

        // Only hard RoPE-table pressure slides in-flight KV; red-zone cleanup waits until turn end.
        // Hybrid models can't slide their linear recurrent state, so they never in-turn recycle (the
        // decode budget is already capped to the rotary table above).
        if (!useBeam && !hasLinearState()) {
            const bool rotaryBudgetLow = absNext + decodeBudgetMargin >= static_cast<int64_t>(max_seq_len);
            if (rotaryBudgetLow) {
                const int64_t keepForAppend = memoryGreenTargetUsed(memory.decodeKeepWindow);
                const int64_t preSlideLen = tensorDim(api, mainKV[0], 4);
                int64_t slidNext = absNext;
                if (slideDecodeWindow(mainKV, absNext, keepForAppend, kvSeqLenTensor, slidNext)) {
                    absNext = slidNext;
                    decodeBase = 0;   // the slide renumbered the kept window back to absolute position 0
                    if (canUseEscapePenalty && !escapePenaltyActive) {
                        const float boostedPenalty = std::min(repeatPenalty, ESCAPE_REPEAT_PENALTY);
                        if (penaltyValBuf.value) {
                            writeScalarFloat(penaltyValBuf, mPenalty.inputTypes[2], boostedPenalty);
                        }
                        usePen = true;
                        useArgmax = false;
                        useGreedy = true;
                        escapePenaltyActive = true;
                        if (saveIdDev == nullptr && !recentDecodedIds.empty()) {
                            escapeSeedData.assign(recentDecodedIds.begin(), recentDecodedIds.end());
                            const int64_t seedShape[2] = {1, static_cast<int64_t>(escapeSeedData.size())};
                            OrtValue* seedT = nullptr;
                            if (logOrtStatus(api, api->CreateTensorWithDataAsOrtValue(
                                    mMain.memoryInfo, escapeSeedData.data(),
                                    escapeSeedData.size() * sizeof(int32_t), seedShape, 2,
                                    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &seedT),
                                    "escape-penalty save_id seed")) {
                                saveIdDev = seedT;
                            }
                        }
                        LOGI("Hard RoPE slide: temporary repeat penalty %.3f (user %.3f)",
                             boostedPenalty, repeatPenalty);
                    }
                } else {
                    LOGE("In-turn KV recycle failed at abs pos %lld (len=%lld, keep=%lld); stopping decode early",
                         static_cast<long long>(absNext),
                         static_cast<long long>(preSlideLen),
                         static_cast<long long>(keepForAppend));
                    // Token streamed above is not embedded yet; keep OFF direct-save ids in lockstep.
                    if (!organizeMemory && !useBeam && !fullReplyVec.empty()) {
                        fullReplyVec.pop_back();
                    }
                    break;
                }
            }
        }

        if (!runBinding(mEmbed) || !runBinding(mRotaryDecode)) {
            kvStateValid = false;
            break;
        }
        preparedTokenPendingKV = true;

        if (isPrefill) {
            isPrefill = false;
        }
        absNext += 1;
        numDecode += 1;
        const int liveDecoded = useBeam ? numDecode : generated;
        const int64_t nowMs = now_ms();
        if (liveDecoded > 0 && nowMs - last_perf_ms >= PERF_UPDATE_MS) {
            refreshBeamMemoryReplyTokens();
            const auto nowClock = std::chrono::steady_clock::now();
            const float interval_s = std::chrono::duration<float>(nowClock - last_perf_clock).count();
            const int intervalTokens = liveDecoded - last_perf_decoded;
            if (interval_s > 0.0f && intervalTokens > 0) {
                pushPerfRates(env, false, 0.0f, true, static_cast<float>(intervalTokens) / interval_s,
                              liveMemoryRemainingTokens(), memory.memoryTokens,
                              -1, liveDecoded);
                last_perf_ms = nowMs;
                last_perf_clock = nowClock;
                last_perf_decoded = liveDecoded;
            }
        }
    }

    if (preparedTokenPendingKV) {
        // Only OFF direct-save consumes in-flight mainKV; ON/beam rebuild from ids.
        if (!organizeMemory && !useBeam) {
            kvStateValid = appendPreparedTokenToKV();
        }
        preparedTokenPendingKV = false;
    }

    // Retry only shortened-prefill empty replies; full-prefill stops are accepted.
    const bool usedShortenedPrefill = wantDecouple || prefillZeroWindow;
    if (genAttempt == 0 && usedShortenedPrefill && numDecode <= MIN_DECODE_TOKENS &&
        !g_cancel.load(std::memory_order_relaxed)) {
        genAttempt = 1;
        forceFullPrefill = true;
        cleanup();   // release this attempt's KV / device tensors / buffers (saved_kv globals untouched)
        bool restored = true;
        if (saved_kv[0] == nullptr && saved_kv_len > 0) {
            restored = rebuildSavedKVFromHistoryRetaining(static_cast<int>(saved_kv_len));
        }
        if (restored) {
            history_pos = saved_kv_base + saved_kv_len;
            reset_output_words_decoder();
            stream_buf.clear();
            tokens_since_flush = 0;
            goto retry_generation;
        }
        clear_history();
        return env->NewStringUTF("");
    }

    // Finalize.
    if (useBeam && saveIdDev) {
        std::vector<int> best = extractFirstRowInt32(api, saveIdDev);
        for (int id : best) {
            if (is_stop_token(id)) {
                break;
            }
            if (generated >= memory.decodeTokens) {
                break;
            }
            generated += 1;
            appendReplyTokenForHistory(id);
            if (!organizeMemory) {
                fullReplyVec.push_back(id);
            }
            stream_token(env, id);
        }
        beamMemoryReplyTokens = static_cast<int>(organizeMemory ? replyVec.size() : fullReplyVec.size());
    }
    if (g_manual_stop.load(std::memory_order_relaxed)) {
        std::vector<int> noticeIds;
        {
            std::lock_guard<std::mutex> tokenizerLock(g_tokenizer_mutex);
            noticeIds = tokenizer->encode(MANUAL_STOP_NOTICE);
        }
        replyVec.insert(replyVec.end(), noticeIds.begin(), noticeIds.end());
        // OFF direct-save must fold the notice into live KV; ON/beam recompute from ids.
        if (!organizeMemory && !useBeam && kvStateValid && !mainKV.empty() && mainKV[0] != nullptr) {
            const int64_t noticeCacheLen = tensorDim(api, mainKV[0], 4);
            if (noticeCacheLen > 0 &&
                appendIdsToDecodeKV(mainKV, mainLinear, noticeIds, decodeBase + noticeCacheLen, noticeCacheLen)) {
                fullReplyVec.insert(fullReplyVec.end(), noticeIds.begin(), noticeIds.end());
            }
        }
        if (!organizeMemory && useBeam) {
            fullReplyVec.insert(fullReplyVec.end(), noticeIds.begin(), noticeIds.end());
            beamMemoryReplyTokens = static_cast<int>(fullReplyVec.size());
        }
        stream_buf.append(MANUAL_STOP_NOTICE);
    }
    flush_stream(env);
    reset_output_words_decoder();

    const auto end_clock = std::chrono::steady_clock::now();

    const bool decodeStarted = decode_start_clock > prefill_start_clock;
    const float prefill_s = std::chrono::duration<float>(
        (decodeStarted ? decode_start_clock : end_clock) - prefill_start_clock).count();
    const float prefill_rate = (prefill_s > 0.0f) ? (static_cast<float>(prefillTokenCount) / prefill_s) : 0.0f;
    const auto base_clock = decodeStarted ? decode_start_clock : prefill_start_clock;
    const float decode_s = std::chrono::duration<float>(end_clock - base_clock).count();
    const float decode_rate = (decode_s > 0.0f) ? (static_cast<float>(generated) / decode_s) : 0.0f;
    (void)numDecode;
    char stats[96];
    const int finalMemoryRemaining = liveMemoryRemainingTokens();
    std::snprintf(stats, sizeof(stats), "%.2f|%.2f|%d|%d",
                  prefill_rate, decode_rate, finalMemoryRemaining, memory.memoryTokens);
    if (!prefillStatsSent && prefill_rate > 0.0f) {
        pushPerfRates(env, true, prefill_rate, false, 0.0f,
                      finalMemoryRemaining, memory.memoryTokens,
                      static_cast<int>(prefillTokenCount), -1);
    }
    char statsFull[128];
    std::snprintf(statsFull, sizeof(statsFull), "%.2f|%.2f|%d|%d|%d|%d",
                  prefill_rate, decode_rate, finalMemoryRemaining, memory.memoryTokens,
                  static_cast<int>(prefillTokenCount), generated);
    pushPerfStats(env, statsFull);

    // Persist turn: OFF transfers decode KV directly; ON/beam rebuild clean KV from ids.
    if (!organizeMemory && !useBeam) {
        int pendingCap = -1;
        const bool saved = kvStateValid && saveDecodeKVDirect(mainKV, mainLinear, decodeBase, get_ids,
                                                              fullReplyVec, memory.memoryTokens, &pendingCap);
        if (!saved) {
            clear_history();
        }
        cleanup();
        if (saved && pendingCap >= 0) {
            commitOffCap(pendingCap);
        }
        return env->NewStringUTF(stats);
    }

    std::vector<int> dialogueIds;
    if (kvStateValid) {
        const std::vector<int>& persistedReplyIds = (!organizeMemory && useBeam) ? fullReplyVec : replyVec;
        dialogueIds.reserve(static_cast<size_t>(prev_close_len) + dialogueBaseIds.size() + persistedReplyIds.size());
        if (reusedHistory) {
            dialogueIds.insert(dialogueIds.end(), chat_previous_assistant_close_ids.begin(),
                               chat_previous_assistant_close_ids.end());
        }
        dialogueIds.insert(dialogueIds.end(), dialogueBaseIds.begin(), dialogueBaseIds.end());
        dialogueIds.insert(dialogueIds.end(), persistedReplyIds.begin(), persistedReplyIds.end());
    }
    cleanup();
    commitCleanTurn(std::move(dialogueIds), kvStateValid, memory.memoryTokens);

    return env->NewStringUTF(stats);
}

// JNI: cooperative cancel.
extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_Stop_1LLM(JNIEnv* env, jclass clazz, jboolean manual) {
    if (!g_busy.load(std::memory_order_acquire)) {
        g_pending_prestart_cancel.store(true, std::memory_order_release);
        g_pending_prestart_manual_stop.store(manual == JNI_TRUE, std::memory_order_release);
    }
    if (manual == JNI_TRUE) {
        g_manual_stop.store(true, std::memory_order_relaxed);
    }
    g_cancel.store(true, std::memory_order_relaxed);
}

// Abort detached clean commits during Clear.
static void setCleanCommitTerminate(bool terminate) {
    ModelRuntime* commitModels[] = { &mEmbed, &mRotaryPrefill, &mMain };
    for (ModelRuntime* m : commitModels) {
        if (m->api == nullptr || m->runOptions == nullptr) {
            continue;
        }
        if (terminate) {
            logOrtStatus(m->api, m->api->RunOptionsSetTerminate(m->runOptions), "RunOptionsSetTerminate");
        } else {
            logOrtStatus(m->api, m->api->RunOptionsUnsetTerminate(m->runOptions), "RunOptionsUnsetTerminate");
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
    g_pending_prestart_cancel.store(false, std::memory_order_release);
    g_pending_prestart_manual_stop.store(false, std::memory_order_release);
    setCleanCommitTerminate(true);
    waitForCleanCommit();
    setCleanCommitTerminate(false);
    clear_history();
    shrinkAllModelAllocators();
    g_reserved_arena_bytes = 0;   // arena was shrunk; the next turn re-warms it
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
    size_t cut = g_turn_checkpoints.size();
    for (size_t i = 0; i < g_turn_checkpoints.size(); ++i) {
        if (g_turn_checkpoints[i].turnId == static_cast<int32_t>(turn_id)) {
            historyLen = g_turn_checkpoints[i].historyLen;
            activeLen  = g_turn_checkpoints[i].activeLen;
            cut = i;
            break;
        }
    }
    if (historyLen < 0) {
        clear_history();
        return JNI_FALSE;
    }
    g_turn_checkpoints.resize(cut);

    const int64_t oldFull = static_cast<int64_t>(g_history_ids.size());
    historyLen = std::max<int64_t>(0, std::min<int64_t>(historyLen, oldFull));
    activeLen  = std::max<int64_t>(0, std::min<int64_t>(activeLen, historyLen));
    if (historyLen <= 0) {
        clear_history();
        return JNI_TRUE;
    }
    if (oldFull > historyLen) {
        g_history_ids.resize(static_cast<size_t>(historyLen));
    }
    // Fast path when the active window front has not moved since turn start.
    const int64_t curOffset    = oldFull - saved_kv_len;
    const int64_t targetOffset = historyLen - activeLen;
    if (curOffset == targetOffset && mKVSlice.session != nullptr && saved_kv[0] != nullptr) {
        if (activeLen >= saved_kv_len) {
            return JNI_TRUE;   // no tokens dropped: full-attention KV and linear state both stay valid
        }
        // Slicing drops tokens; the hybrid linear recurrent state can't follow, so hybrid models
        // fall through to a full rebuild-from-ids instead.
        std::vector<OrtValue*> cur(saved_kv.begin(), saved_kv.begin() + num_keys_values), sliced;
        if (!hasLinearState() && runKVSlice(cur, 0, activeLen, sliced)) {
            release_saved_kv();
            for (int i = 0; i < num_keys_values; ++i) saved_kv[i] = sliced[i];
            saved_kv_len = activeLen;
            publishMemoryUsage();
            return JNI_TRUE;
        }
    }
    const int rebuildTarget = static_cast<int>(std::min<int64_t>(
            std::max<int64_t>(1, activeLen), static_cast<int64_t>(memory.memoryTokens)));
    if (!rebuildSavedKVFromHistoryRetaining(rebuildTarget)) {
        clear_history();
        return JNI_FALSE;
    }
    return JNI_TRUE;
}
