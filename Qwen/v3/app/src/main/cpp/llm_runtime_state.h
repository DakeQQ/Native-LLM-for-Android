#pragma once

#include "ort_helpers.h"
#include "tokenizer.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

using MNN::Transformer::Tokenizer;

// Model registry. Order must match LLM_MODEL_TABLE.
enum ModelId {
#define X(id, onnx, data) id,
    LLM_MODEL_TABLE(X)
#undef X
    kModelCount
};

inline const std::array<const char*, kModelCount> kModelFileNames = {
#define X(id, onnx, data) onnx,
    LLM_MODEL_TABLE(X)
#undef X
};

inline const std::array<const char*, kModelCount> kModelExternalFileNames = {
#define X(id, onnx, data) data,
    LLM_MODEL_TABLE(X)
#undef X
};

inline std::array<ModelRuntime, kModelCount> gModelRuntimes;
inline ModelRuntime& getModel(ModelId id) { return gModelRuntimes[id]; }

// Runtime geometry is filled from ONNX metadata during model load.
inline int num_layers      = kDefaultNumLayers;
inline int num_keys_values = kDefaultNumLayers * 2;
inline int g_kv_blocks     = 2;

// Main / beam / KV-slice indices are recomputed after the active KV layout is known.
inline int mainHiddenIdx     = num_keys_values;
inline int mainCosIdx        = num_keys_values + 1;
inline int mainSinIdx        = num_keys_values + 2;
inline int mainMaskIdx       = num_keys_values + 3;
inline int mainLogitsOutIdx  = num_keys_values;

inline int beamSaveIdInIdx       = num_keys_values + 1;
inline int firstBeamSizeIdx      = num_keys_values + 2;
inline int secondBeamPrevProbIdx = num_keys_values + 2;
inline int secondBeamSizeIdx     = num_keys_values + 3;
inline int secondBeamTopKIdx     = num_keys_values + 4;
inline int beamSaveIdOutIdx      = num_keys_values;
inline int beamScoreOutIdx       = num_keys_values + 1;
inline int beamIdsOutIdx         = num_keys_values + 2;
inline int beamMaxOutIdx         = num_keys_values + 3;

inline int kvSliceStartIdx       = num_keys_values;
inline int kvSliceEndIdx         = num_keys_values + 1;

// Mutable runtime options, snapshotted by Run_LLM where needed.
inline int g_memory_tokens  = DEFAULT_MEMORY_TOKENS;
inline int g_prefill_tokens = DEFAULT_PREFILL_TOKENS;
inline int g_decode_tokens  = DEFAULT_MAX_DECODE_TOKENS;
inline std::atomic<int> g_memory_green_target_percent{DEFAULT_MEMORY_GREEN_TARGET_PERCENT};
inline std::atomic<int> g_memory_red_zone_percent{DEFAULT_MEMORY_RED_ZONE_PERCENT};

inline bool  useBeamSearch = DEFAULT_USE_BEAM_SEARCH;
inline bool  usePenalty    = false;
inline int   topK          = DEFAULT_TOP_K;
inline int   beamSize      = DEFAULT_BEAM_SIZE;
inline float repeatPenalty = DEFAULT_REPEAT_PENALTY;
inline bool  g_organize_memory = DEFAULT_ORGANIZE_MEMORY;

// Tokenization and prompt state.
inline Tokenizer* tokenizer = nullptr;
inline std::string      g_system_prompt_text;
inline std::vector<int> g_system_prompt_ids;
inline int64_t          g_system_block_len = 0;
inline int64_t          ids_len = 0;
constexpr unsigned input_ids_elem_shift = 2;

// Cross-turn KV and clean dialogue history.
inline std::array<OrtValue*, max_keys_values> input_tensors_kv_init = {};
inline std::array<OrtValue*, max_keys_values> saved_kv = {};
inline int64_t saved_kv_len  = 0;
inline int64_t saved_kv_base = 0;
inline std::atomic<int64_t> g_memory_used_tokens{0};
inline std::vector<int> g_history_ids;

struct TurnCheckpoint {
    int32_t turnId;
    int64_t historyLen;
    int64_t activeLen;
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

bool loadModel(ModelId id, JNIEnv* env, jobject assetManager, int epType, bool lowMemoryMode);
bool configureKVLayout();
bool initKVCache();

extern "C" {
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Pre_1Process(JNIEnv* env, jobject clazz);

JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1A(JNIEnv* env, jobject clazz,
                                                            jobject asset_manager,
                                                            jint ep_type,
                                                            jboolean low_memory_mode);

JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_Run_1LLM(JNIEnv* env, jclass clazz, jstring jquery, jboolean clear, jboolean use_think, jint turn_id);

JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Rollback_1LLM(JNIEnv* env, jclass clazz, jint turn_id);

JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Configure_1LLM(JNIEnv* env, jclass clazz,
                                                           jboolean use_beam_search,
                                                           jint top_k,
                                                           jint beam_size,
                                                           jfloat repeat_penalty);

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
}
