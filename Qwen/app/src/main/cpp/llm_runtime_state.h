#pragma once

#include "ort_helpers.h"
#include "tokenizer.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

using MNN::Transformer::Tokenizer;

// Model registry. Order must match LLM_MODEL_TABLE.
enum ModelId {
#define X(id, onnx, data) id,
    LLM_MODEL_TABLE(X)
#undef X
    kModelCount
};

inline constexpr std::array<const char*, kModelCount> kModelFileNames = {
#define X(id, onnx, data) onnx,
    LLM_MODEL_TABLE(X)
#undef X
};

inline std::array<std::string, kModelCount> gModelFileNames = [] {
    std::array<std::string, kModelCount> names;
    for (int i = 0; i < kModelCount; ++i) {
        names[i] = kModelFileNames[i];
    }
    return names;
}();

inline constexpr std::array<const char*, kModelCount> kModelExternalFileNames = {
#define X(id, onnx, data) data,
    LLM_MODEL_TABLE(X)
#undef X
};

// Shared merged-initializer descriptor filenames: compile-time defaults straight from the model table,
// plus mutable copies the runtime overwrites from stamped ONNX metadata (mirrors gModelFileNames).
inline constexpr const char* kSharedInitializersFileName =
        kModelFileNames[LLM_SharedInitializers];
inline constexpr const char* kSharedInitializersDataFileName =
        kModelExternalFileNames[LLM_SharedInitializers];
inline std::string gSharedInitializersFileName = kSharedInitializersFileName;
inline std::string gSharedInitializersDataFileName = kSharedInitializersDataFileName;
// Metadata descriptor filename (read once to load the exporter-stamped runtime config).
inline constexpr const char* kMetadataModelFileName = kModelFileNames[LLM_Metadata];

inline constexpr std::string_view kSharedInitializersModelSuffix = "_SharedInitializers.onnx";
inline constexpr std::string_view kMetadataModelSuffix = "_Metadata.onnx";

constexpr bool modelFileNameEndsWith(std::string_view fileName, std::string_view suffix) {
    return fileName.size() >= suffix.size() &&
           fileName.compare(fileName.size() - suffix.size(), suffix.size(), suffix) == 0;
}

constexpr int countModelsWithSuffix(std::string_view suffix) {
    int count = 0;
    for (const char* fileName : kModelFileNames) {
        count += modelFileNameEndsWith(fileName, suffix) ? 1 : 0;
    }
    return count;
}

static_assert(modelFileNameEndsWith(kModelFileNames[LLM_SharedInitializers],
                                    kSharedInitializersModelSuffix),
              "LLM shared-initializer descriptor must match *_SharedInitializers.onnx");
static_assert(modelFileNameEndsWith(kModelFileNames[LLM_Metadata], kMetadataModelSuffix),
              "LLM metadata descriptor must match *_Metadata.onnx");

inline constexpr int kSessionModelCount =
        kModelCount - countModelsWithSuffix(kSharedInitializersModelSuffix) -
        countModelsWithSuffix(kMetadataModelSuffix);

constexpr bool modelIsBundleDescriptor(ModelId id) {
    return modelFileNameEndsWith(kModelFileNames[id], kSharedInitializersModelSuffix) ||
           modelFileNameEndsWith(kModelFileNames[id], kMetadataModelSuffix);
}

inline std::array<ModelRuntime, kModelCount> gModelRuntimes;
inline ModelRuntime& getModel(ModelId id) { return gModelRuntimes[id]; }

// Decode strategy identity for merged-graph selection. Order is independent of the graph layout; the
// prefill/decode helpers translate a strategy into the right slot inside a modality's 6-graph block.
enum MergedStrategy {
    MSTRAT_GREEDY         = 0,
    MSTRAT_PENALTY_GREEDY = 1,
    MSTRAT_SAMPLING       = 2,
};
static_assert(MSTRAT_GREEDY == 0 && MSTRAT_PENALTY_GREEDY == 1 && MSTRAT_SAMPLING == 2,
              "Merged strategies must match their graph-slot offsets");

// modality: 0 = text, 1 = image, 2 = video (matches VisionModality's numeric value). The 18 available graphs
// are laid out modality-major with a fixed 6-slot block: prefill {greedy,penaltyGreedy,sampling} then
// decode {greedy,penaltyGreedy,sampling}.
inline ModelId mergedPrefillModel(int modality, MergedStrategy strat) {
    return static_cast<ModelId>(LLM_FIRST_MERGED_MODEL +
            modality * LLM_MERGED_MODELS_PER_MODALITY + static_cast<int>(strat));
}

inline ModelId mergedDecodeModel(int modality, MergedStrategy strat) {
    return static_cast<ModelId>(LLM_FIRST_MERGED_MODEL +
            modality * LLM_MERGED_MODELS_PER_MODALITY + 3 + static_cast<int>(strat));
}

// True for the 18 available merged strategy graphs, only a bounded selected subset of which is resident.
constexpr bool modelIsMergedGraph(int id) {
    return id >= LLM_FIRST_MERGED_MODEL &&
           id < LLM_FIRST_MERGED_MODEL + 3 * LLM_MERGED_MODELS_PER_MODALITY;
}

// KV-cache management + RoPE-shift utility graphs. They run once per decoded token alongside the merged
// decode graph (append/slice/split/concat KV and shift the rotary window), so they are tiny
// and latency-bound. Like the merged decode graphs they belong on the small Top-N big-core decode thread
// pool, NOT the wide prefill pool that drives the heavy prefill/vision GEMMs. Enumerated explicitly so the
// classification does not depend on the model-table ordering.
constexpr bool modelIsKvControlGraph(int id) {
    switch (id) {
        case LLM_KV_Slice:
        case LLM_KV_Split2:
        case LLM_KV_Concat:
        case LLM_RopeShift:
            return true;
        default:
            return false;
    }
}

// A merged strategy graph that actually loaded (present + session ready).
inline bool mergedModelReady(ModelId id) {
    return getModel(id).session != nullptr;
}

// Runtime geometry is filled from ONNX metadata during model load.
inline int num_layers                  = kDefaultNumLayers;
inline int num_full_attention_layers   = kDefaultNumLayers;      // layers with a windowed KV cache
inline int num_linear_attention_layers = 0;                      // hybrid (Qwen3.5) linear-attention layers
inline int num_keys_values             = kDefaultNumLayers * 2;  // windowed full-attention KV tensors
inline int num_linear_states           = 0;                      // passthrough conv/recurrent state tensors
inline int num_main_states             = kDefaultNumLayers * 2;  // = num_keys_values + num_linear_states
inline int g_kv_blocks                 = 2;                      // KV tensors per full-attention layer (2/4/6)
inline int g_linear_blocks             = 0;                      // state tensors per linear layer (conv+recurrent)

// True for hybrid (linear + full attention) models; false for pure transformers (Qwen3).
inline bool hasLinearState() { return num_linear_states > 0; }

// KV_Slice / Split2 / Concat / RopeShift act on the full-attention KV block only.
inline int kvSliceStartIdx       = num_keys_values;
inline int kvSliceEndIdx         = num_keys_values + 1;

// Mutable runtime options, snapshotted by Run_LLM where needed.
inline int g_memory_tokens  = DEFAULT_MEMORY_TOKENS;
inline int g_prefill_tokens = DEFAULT_PREFILL_TOKENS;
inline int g_decode_tokens  = DEFAULT_MAX_DECODE_TOKENS;
inline std::atomic<int> g_memory_green_target_percent{DEFAULT_MEMORY_GREEN_TARGET_PERCENT};
inline std::atomic<int> g_memory_red_zone_percent{DEFAULT_MEMORY_RED_ZONE_PERCENT};

inline int   decodeMode    = DECODE_MODE_GREEDY;
inline int   topK          = DEFAULT_TOP_K;
inline float repeatPenalty = DEFAULT_REPEAT_PENALTY;
inline int   penaltyRange  = DEFAULT_PENALTY_RANGE;
inline float temperature   = DEFAULT_TEMPERATURE;
inline float topP          = DEFAULT_TOP_P;
inline bool  g_organize_memory = DEFAULT_ORGANIZE_MEMORY;
inline bool  g_supports_thinking = false;

// Tokenization and prompt state.
inline Tokenizer* tokenizer = nullptr;
inline std::string      g_system_prompt_text;
inline std::vector<int> g_system_prompt_ids;
inline int64_t          g_system_block_len = 0;
inline int64_t          ids_len = 0;
inline std::vector<int> g_stop_token_ids;

// Cross-turn KV and clean dialogue history.
inline std::array<OrtValue*, max_keys_values> input_tensors_kv_init = {};
inline std::array<OrtValue*, max_keys_values> saved_kv = {};
inline int64_t saved_kv_len  = 0;
inline int64_t saved_kv_base = 0;

// Hybrid (Qwen3.5) linear-attention passthrough states. Empty for pure-transformer models. These are
// fixed-shape conv/recurrent summaries threaded output->input each step; they are never windowed, so any
// token-drop on saved_kv routes through a full re-prefill to keep the linear state consistent.
inline std::array<OrtValue*, max_linear_states> linear_state_init   = {};   // fixed-shape zero init
inline std::array<OrtValue*, max_linear_states> saved_linear_states = {};   // cross-turn persisted final states
inline bool saved_linear_valid = false;   // saved_linear_states corresponds to saved_kv's current window
inline std::atomic<int64_t> g_memory_used_tokens{0};
// 0 normal, 1 severe thermal pressure, 2 critical thermal/memory pressure. This is transient and
// never overwrites the user's persisted decode strategy or memory profile.
inline std::atomic<int> g_runtime_pressure_level{0};
inline std::vector<int> g_history_ids;
inline size_t g_history_begin = 0;
inline int64_t g_history_base = 0;

struct TurnCheckpoint {
    int32_t turnId;
    int64_t historyLen;
    int64_t activeLen;
    uint8_t modality  = 0;    // VisionModality of this turn (0 text, 1 image, 2 video)
    int64_t mropePos  = 0;    // mRoPE position at the START of this turn (fed as history_len to the prefill);
                              // diverges from activeLen once the retained window carries a vision segment
    bool    hadVision = false;// retained window held an un-rebuildable vision KV segment at this turn's start
};
inline std::vector<TurnCheckpoint> g_turn_checkpoints;

// Java callbacks and streaming buffer.
inline JavaVM*  g_jvm = nullptr;
inline jclass    g_main_cls = nullptr;
inline jmethodID g_on_token = nullptr;
inline jmethodID g_on_perf = nullptr;
inline jmethodID g_on_post_processing = nullptr;
inline std::string stream_buf;
inline int         tokens_since_flush = 0;
inline int64_t     last_flush_ms      = 0;

// Generation cancellation. Clear-cache cancellation is separate from manual Stop.
inline std::atomic<bool> g_cancel{false};
inline std::atomic<bool> g_busy{false};
inline std::atomic<bool> g_manual_stop{false};
inline std::atomic<bool> g_pending_prestart_cancel{false};
inline std::atomic<bool> g_pending_prestart_manual_stop{false};
inline std::atomic<bool> g_clean_commit_cancel{false};

extern "C" {
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Pre_1Process(JNIEnv* env, jobject clazz);

JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1A(JNIEnv* env, jclass clazz,
                                                            jobject asset_manager,
                                                            jstring asset_directory,
                                                            jstring cache_directory,
                                                            jstring storage_directory,
                                                            jint ep_type,
                                                            jboolean low_memory_mode);

JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Ensure_1Vision_1Models(JNIEnv* env, jclass clazz);

JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Vision_1Models_1Ready(JNIEnv* env, jclass clazz);

JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Configure_1Runtime_1Pressure(
    JNIEnv* env, jclass clazz, jint pressure_level);

JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Trim_1Runtime(
    JNIEnv* env, jclass clazz, jboolean release_vision);

JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_Run_1LLM(JNIEnv* env, jclass clazz, jstring jquery, jboolean clear, jboolean use_think, jint turn_id);

JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Rollback_1LLM(JNIEnv* env, jclass clazz, jint turn_id);

JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Configure_1LLM(JNIEnv* env, jclass clazz,
                                                           jint decode_mode,
                                                           jint top_k,
                                                           jfloat repeat_penalty,
                                                           jint penalty_range,
                                                           jfloat temperature,
                                                           jfloat top_p);

JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Configure_1Memory(JNIEnv* env, jclass clazz,
                                                              jint memory_tokens,
                                                              jint prefill_tokens,
                                                              jint decode_tokens);

JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Configure_1Organize_1Memory(JNIEnv* env, jclass clazz,
                                                                       jboolean enable);

JNIEXPORT jintArray JNICALL
Java_com_example_myapplication_MainActivity_Get_1Memory_1Limits(JNIEnv* env, jclass clazz);

JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Configure_1Memory_1Thresholds(JNIEnv* env, jclass clazz,
                                                                          jint red_percent,
                                                                          jint green_percent);

JNIEXPORT jintArray JNICALL
Java_com_example_myapplication_MainActivity_Get_1Memory_1Thresholds(JNIEnv* env, jclass clazz);

JNIEXPORT jintArray JNICALL
Java_com_example_myapplication_MainActivity_Get_1Memory_1Stats(JNIEnv* env, jclass clazz);

JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Set_1System_1Prompt(JNIEnv* env, jclass clazz, jstring system_prompt);

JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_Stop_1LLM(JNIEnv* env, jclass clazz, jboolean manual);

JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_Clear_1Cache(JNIEnv* env, jclass clazz);

// Speculative vision pre-encode: runs the query-independent ViT pass for the pending capture ahead of
// the send so the vision turn can reuse it. Best-effort; safe to call on a background thread.
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Prewarm_1Vision(JNIEnv* env, jclass clazz);
}
