#pragma once
#include <jni.h>
#include <iostream>
#include <fstream>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_float16.h"
#include "tokenizer.hpp"

// Android logcat helpers. Mirrors the reference project's LOGI/LOGE so the ONNX Runtime status checks
// (logOrtStatus in processLLM.cpp) surface real diagnostics instead of silently swallowing failures.
#define LOG_TAG "Hunyuan_MT"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// The MNN tokenizer lives in namespace MNN::Transformer (tokenizer.hpp). Hoist the type name into the
// global scope so the LLM code can refer to it unqualified as `Tokenizer`, matching the reference.
using MNN::Transformer::Tokenizer;

// ----- Model enumeration -----
// Hunyuan-MT ships as ONE combined graph (token embedding + KV-cache decoder + argmax fused into a
// single .onnx), so the pipeline has a single model. The enum still follows the reference's
// ModelId/kModelCount pattern so the per-model containers and loader generalise if more graphs are
// added later (e.g. a separate draft model for speculative decoding).
enum ModelId {
    LLM_Decoder = 0,   // input_ids + KV cache + history/ids_len + attention_mask -> KV cache + max_logit_id
    kModelCount
};

// ----- Central model file-name config -----
// Loaded from cachePath/storagePath (external storage / asset-staged cache) or from an asset buffer.
inline const std::array<const char*, kModelCount> kModelFileNames = {
    "Hunyuan_MT.onnx",   // LLM_Decoder
};

// ----- External-data (two-part model) file-name config -----
// "" for self-contained single-file models. This is a NAME HINT, not a hard switch: loadModel()
// auto-detects whether the *.data sibling is actually present and falls back to single-file mode when
// it is absent (a configured-but-missing *.data is NOT an error). When the file does exist its name
// MUST match the external-data reference recorded inside the corresponding .onnx, because loading
// resolves the *.data sibling by that exact name from the .onnx's directory.
inline const std::array<const char*, kModelCount> kModelExternalFileNames = {
    "a1d48530-692c-11f1-b321-bc091bee2d5c.data",   // LLM_Decoder (Hunyuan_MT-1.8B demo weights)
};

// ----- Execution provider selection -----
// Chosen at runtime via the Load_Models_A JNI arg (EP_TYPE), not baked per-model, because the app lets
// the user switch backends without a rebuild.
enum ExecutionProviderType {
    EP_CPU     = 0,   // Default CPU execution provider
    EP_XNNPACK = 1,   // XNNPACK (optimized for mobile CPU)
    EP_QNN     = 2    // QNN (Qualcomm NPU/HTP)
};

// ----- Per-model runtime container -----
// Encapsulates every ORT handle + cached I/O metadata for one session (reference ModelRuntime style),
// replacing the loose session_model_A / input_names_A / ... globals of the original single-file port.
struct ModelRuntime {
    ModelId id = LLM_Decoder;
    std::string fileName;
    std::string externalFileName;
    const OrtApi* api = nullptr;
    OrtEnv* env = nullptr;
    OrtSession* session = nullptr;
    OrtRunOptions* runOptions = nullptr;
    OrtIoBinding* binding = nullptr;
    OrtAllocator* allocator = nullptr;
    const OrtMemoryInfo* memoryInfo = nullptr;
    bool ioPrepared = false;
    // I/O-binding residency for the dynamic-length KV cache outputs (BindOutputToDevice). Defaults to
    // the CPU memoryInfo/allocator above so the CPU and XNNPACK paths stay byte-for-byte unchanged;
    // when the QNN HTP backend exposes its CPU<->device shared-memory allocator (EP option
    // enable_htp_shared_memory_allocator=1) these are repointed at it so the kv_out -> kv_in recursion
    // stays resident on the accelerator with no per-step device<->CPU copy. ioAllocator is
    // session-scoped (created via CreateAllocator) and therefore never released explicitly.
    const OrtMemoryInfo* ioMemoryInfo = nullptr;
    OrtAllocator* ioAllocator = nullptr;
    std::vector<const char*> inputNames;
    std::vector<std::vector<int64_t>> inputDims;
    std::vector<ONNXTensorElementDataType> inputTypes;
    std::vector<OrtValue*> inputTensors;
    std::vector<const char*> outputNames;
    std::vector<std::vector<int64_t>> outputDims;
    std::vector<ONNXTensorElementDataType> outputTypes;
};

// ----- Global model table -----
inline std::array<ModelRuntime, kModelCount> gModelRuntimes;
inline ModelRuntime& getModel(ModelId id) { return gModelRuntimes[id]; }

// ----- Model configuration -----
constexpr int num_layers           = 32;                          // Transformer layers (config.json). Hunyuan_MT-1.8B = 32
constexpr int max_seq_len          = 4096;                        // Must match the exported model.
constexpr int num_keys_values      = num_layers + num_layers;     // keys + values = 64
constexpr int single_chat_limit    = max_seq_len / 2 - 1;         // > (max_seq_len/2 - 1) may clear previous context
constexpr int next_chat_buffer     = max_seq_len - single_chat_limit;

// Input/output index map for the combined Hunyuan graph. KV pairs occupy [0, num_keys_values): the
// first num_layers are keys (dynamic seq axis at dim 4), the next num_layers are values (dynamic seq
// axis at dim 3). The four runtime scalars/ids follow, and the single fixed output is the argmax id.
constexpr int inputIdsIdx          = num_keys_values;       // input_ids
constexpr int historyLenIdx        = num_keys_values + 1;   // history_len
constexpr int idsLenIdx            = num_keys_values + 2;   // ids_len
constexpr int attentionMaskIdx     = num_keys_values + 3;   // attention_mask
constexpr int maxLogitOutIdx       = num_keys_values;       // max_logit_id output

// ----- Runtime configuration -----
constexpr bool ORT_FP16 = true;   // FP16 model: requires ARM64-v8.2a+ on CPU; disables the FP16->FP32 optimizers.

// ----- Token IDs -----
constexpr int end_id_0 = 120020;  // Hunyuan_MT2-1.8B
constexpr int end_id_1 = 127960;  // Hunyuan_MT2-7B
constexpr int end_id_2 = 127967;  // Hunyuan_MT2-7B

// ----- Storage paths -----
inline const std::string storage_path = "/storage/emulated/0/Android/data/com.example.myapplication/";
inline const std::string cache_path   = "/data/user/0/com.example.myapplication/cache/";
inline const std::string vocab_path   = "/data/user/0/com.example.myapplication/cache/vocab_Hunyuan_MT.txt";  // moved from assets to cache in the Java process.

// ----- Streaming batch knobs -----
// Flush the streaming batch every STREAM_BATCH decoded tokens OR every STREAM_FLUSH_MS ms, whichever
// comes first. Collapses N per-token JNI call-outs + UI rebinds into a few while keeping streaming
// visually smooth (~30 fps). Tune STREAM_BATCH up to cut call-outs further.
constexpr int     STREAM_BATCH     = 6;
constexpr int64_t STREAM_FLUSH_MS  = 33;

// ============================================================================
// LLM global state
// Shared between loadModels.cpp (creates the session, fixed buffers and KV-init tensors) and
// processLLM.cpp (runs the autoregressive decode). All `inline` because configs.h is now included by
// two translation units.
// ============================================================================

// Tokenizer (created in Pre_Process, consumed by Run_LLM).
inline Tokenizer* tokenizer = nullptr;

// Decode bookkeeping (multi-turn input_ids buffer maintenance + per-reply counters).
inline int     token_id        = 0;
inline int     response_count  = 0;
inline int     save_index      = 0;
inline int64_t history_len     = 0;
inline int64_t ids_len         = 0;
inline int8_t  attention_mask  = 1;

// Fixed (1,1) shared decode buffers, bound to the IOBinding once and reused every step:
//   max_idx_buf    : receives the model's argmax token output (backed by argmax_id).
//   decode_ids_buf : feeds that token back as the next-step input_ids (backed by decode_input_id).
// Only the backing value changes between runs, so neither the output nor the input is ever rebound.
inline OrtValue* max_idx_buf     = nullptr;
inline OrtValue* decode_ids_buf  = nullptr;
inline int32_t   argmax_id       = 0;
inline int32_t   decode_input_id = 0;

// Empty (zero-length) prefill KV inputs (distinct from the runtime KV bound from the previous step's
// outputs). Created once in Load_Models_A and auto-bound for the prefill forward pass.
inline std::array<OrtValue*, num_keys_values> input_tensors_kv_init = {};

// IOBinding carry-over for the dynamic-length KV cache only: the previous decode step's on-device KV
// outputs that are currently bound as the next step's KV inputs, released once the following step has
// rebound past them. The fixed (1,1) token buffers never need this.
inline OrtValue** carryover_outputs = nullptr;
inline size_t     carryover_count   = 0;

// Multi-turn scratch buffers.
inline std::vector<int>    input_ids(max_seq_len, 0);
inline std::vector<int>    accumulate_num_ids(20, 0);          // size must outlast max_seq_len worth of turns
inline std::vector<int>    num_ids_per_chat(20, 0);            // same size as accumulate_num_ids
inline std::vector<int>    save_max_logit_position(max_seq_len, 0);
inline std::vector<uint16_t> past_key_values_init(1, 0);

// input_ids is exported as int32, so a (1, n) tensor occupies n * sizeof(int32_t) == n * 4 == n << 2
// bytes. This is computed inline at bind time (see processLLM.cpp), replacing the former
// max_seq_len-entry lookup table (saves the table's memory and its one-time precompute loop).
constexpr unsigned input_ids_elem_shift = 2;   // log2(sizeof(int32_t))

// C++ -> Java token-streaming callback, cached ONCE in Load_Models_A (process-lifetime singleton, so
// FindClass/GetStaticMethodID never run per token). g_main_cls is a global ref to MainActivity;
// g_on_token is the static void onTokenStream(String) method id, invoked via CallStaticVoidMethod.
inline jclass    g_main_cls = nullptr;
inline jmethodID g_on_token = nullptr;

// Native streaming batch buffer. One generation's decoded words accumulate here and are flushed to the
// UI in batches; the buffer is reused for the whole process (reserve() once in Load_Models_A) so the
// decode loop never returns a std::string per token nor churns the heap.
inline std::string stream_buf;
inline int         tokens_since_flush = 0;
inline int64_t     last_flush_ms      = 0;

// Generation control. g_cancel: set by Stop_LLM(), polled each decode iteration to abort promptly.
// g_busy: non-reentrancy guard for the (global, non-thread-safe) inference state; the Java side joins
// the previous LLMThread before launching a new one, so contention should never actually occur.
inline std::atomic<bool> g_cancel{false};
inline std::atomic<bool> g_busy{false};

// ----- Function declarations -----
bool loadModel(ModelId id, JNIEnv* env, jobject assetManager, int epType, bool lowMemoryMode);

// ----- JNI entry points -----
extern "C" {
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Pre_1Process(JNIEnv* env, jobject clazz);

JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1A(JNIEnv* env, jobject clazz,
                                                            jobject asset_manager,
                                                            jint ep_type,
                                                            jboolean low_memory_mode);

JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_Run_1LLM(JNIEnv* env, jclass clazz, jstring jquery);

JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_Stop_1LLM(JNIEnv* env, jclass clazz);
}
