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

// Main / beam indices are recomputed after the active state layout is known. The Main state block is
// [windowed full-attn KV : 0 .. num_keys_values) | [passthrough linear : num_keys_values .. num_main_states);
// hidden/cos/sin/mask/logits and the beam scalars all sit AFTER the whole state block (= num_main_states).
inline int mainHiddenIdx     = num_main_states;
inline int mainCosIdx        = num_main_states + 1;
inline int mainSinIdx        = num_main_states + 2;
inline int mainMaskIdx       = num_main_states + 3;
inline int mainLogitsOutIdx  = num_main_states;

inline int beamSaveIdInIdx       = num_main_states + 1;
inline int firstBeamSizeIdx      = num_main_states + 2;
inline int secondBeamPrevProbIdx = num_main_states + 2;
inline int secondBeamSizeIdx     = num_main_states + 3;
inline int secondBeamTopKIdx     = num_main_states + 4;
inline int beamSaveIdOutIdx      = num_main_states;
inline int beamScoreOutIdx       = num_main_states + 1;
inline int beamIdsOutIdx         = num_main_states + 2;
inline int beamMaxOutIdx         = num_main_states + 3;

// KV_Slice / Split2 / Concat / RopeShift act on the full-attention KV block only.
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

// Hybrid (Qwen3.5) linear-attention passthrough states. Empty for pure-transformer models. These are
// fixed-shape conv/recurrent summaries threaded output->input each step; they are never windowed, so any
// token-drop on saved_kv routes through a full re-prefill to keep the linear state consistent.
inline std::array<OrtValue*, max_linear_states> linear_state_init   = {};   // fixed-shape zero init
inline std::array<OrtValue*, max_linear_states> saved_linear_states = {};   // cross-turn persisted final states
inline bool saved_linear_valid = false;   // saved_linear_states corresponds to saved_kv's current window
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
