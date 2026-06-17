#include <jni.h>
#include <iostream>
#include <fstream>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <android/asset_manager_jni.h>
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_float16.h"
#include "tokenizer.hpp"

// ONNX Runtime API Components
const OrtApi *ort_runtime_A;
OrtAllocator* allocator;
OrtMemoryInfo *memory_info;
OrtSession *session_model_A;
OrtRunOptions *run_options_A;
OrtIoBinding *io_binding_A;
// Fixed-shape (1,1) shared decode buffers, bound to the IOBinding once and reused every step:
//  - max_idx_buf   : receives the model's argmax token output (backed by argmax_id).
//  - decode_ids_buf: feeds that token back as the next-step input_ids (backed by decode_input_id).
// Only the backing value changes between runs, so neither the output nor the input is ever rebound.
OrtValue *max_idx_buf = nullptr;
OrtValue *decode_ids_buf = nullptr;
int argmax_id = 0;
int decode_input_id = 0;

// Model File Settings
const std::string file_name_A = "Hunyuan_MT.onnx";
const std::string file_name_A_external = "a1d48530-692c-11f1-b321-bc091bee2d5c.data";  // The demo model is Hunyuan_MT-1.8B

// Model Configuration
const int num_layers = 32;                                                        // Transformer layers. Refer to config.json for the value. Hunyuan_MT-1.8B = 32
const int max_seq_len = 4096;                                                     // Please set this value to match the exported model.
const int num_keys_values = num_layers + num_layers;
const int num_keys_values_plus_1 = num_keys_values + 1;
const int num_keys_values_plus_2 = num_keys_values + 2;
const int single_chat_limit = max_seq_len / 2 - 1;                                // You can adjust this value. If you set it greater than (max_seq_len / 2 - 1), the previous context may be cleared.
const int next_chat_buffer = max_seq_len - single_chat_limit;

// Runtime Configuration
const bool ORT_FP16 = true;                                                      // Set to true for FP16 ONNX Runtime settings (FP16 model). For CPUs, this requires ARM64-v8.2a or newer.

// Token IDs
int token_id = 0;
const int end_id_0 = 120020;  // For Hunyuan_MT2-1.8B
const int end_id_1 = 127960;  // For Hunyuan_MT2-7B
const int end_id_2 = 127967;  // For Hunyuan_MT2-7B

// Storage Paths
const std::string storage_path = "/storage/emulated/0/Android/data/com.example.myapplication/";
const std::string cache_path = "/data/user/0/com.example.myapplication/cache/";
const std::string voacb_path = "/data/user/0/com.example.myapplication/cache/vocab_Hunyuan_MT.txt";  // We have moved the vocab.txt from assets to the cache folder in Java process.

// Counters and Indices
bool chatting = true;
int response_count = 0;
int save_index = 0;
int64_t history_len = 0;
int64_t ids_len = 0;
size_t amount_of_output;

// Init Values
int8_t attention_mask = 1;

// Input and Output Names
std::vector<const char*> input_names_A;
std::vector<const char*> output_names_A;

// Dimensions
std::vector<std::vector<std::int64_t>> input_dims_A;
std::vector<std::vector<std::int64_t>> output_dims_A;

// Data Types
std::vector<ONNXTensorElementDataType> input_types_A;
std::vector<ONNXTensorElementDataType> output_types_A;

// Tensors
std::vector<OrtValue*> input_tensors_A;
std::vector<OrtValue*> input_tensors_kv_init_A(num_keys_values);
// IOBinding carry-over for the dynamic-length KV cache only: the previous decode step's on-device
// KV outputs that are currently bound as the next step's KV inputs, released once the following
// step has rebound past them. The fixed (1,1) token buffers above never need this.
OrtValue** carryover_outputs = nullptr;
size_t carryover_count = 0;

// Arrays and Vectors
std::vector<int> input_ids(max_seq_len, 0);
std::vector<int> accumulate_num_ids(20, 0);                    // Just make sure the size is enough before reaching max_seq_len.
std::vector<int> num_ids_per_chat(20, 0);                      // Same size with accumulate_num_ids.
std::vector<int> save_max_logit_position(max_seq_len, 0);
std::vector<int> layer_indices(num_keys_values, 1);
std::vector<size_t> input_ids_buffer_size(max_seq_len, 0);
std::vector<uint16_t> past_key_values_init(1, 0);

// Tokenizer
MNN::Transformer::Tokenizer* tokenizer;

// C++ -> Java token-streaming callback, cached ONCE in Load_Models_A (process-lifetime singleton, so
// FindClass/GetStaticMethodID never run per token). g_main_cls is a global ref to MainActivity;
// g_on_token is the static void onTokenStream(String) method id, invoked via CallStaticVoidMethod.
jclass g_main_cls = nullptr;
jmethodID g_on_token = nullptr;

// Generation control. g_cancel: set by Stop_LLM(), polled each decode iteration to abort promptly.
// g_busy: non-reentrancy guard for the (global, non-thread-safe) inference state; the Java side joins
// the previous LLMThread before launching a new one, so contention should never actually occur.
std::atomic<bool> g_cancel{false};
std::atomic<bool> g_busy{false};

// Native streaming batch buffer. One generation's decoded words accumulate here and are flushed to the
// UI in batches; the buffer is reused for the whole process (reserve() once in Load_Models_A) so the
// decode loop never returns a std::string per token nor churns the heap.
std::string stream_buf;
int tokens_since_flush = 0;
int64_t last_flush_ms = 0;

// Batching knobs: flush the batch every STREAM_BATCH decoded tokens OR every STREAM_FLUSH_MS
// milliseconds, whichever comes first. This collapses N per-token JNI call-outs + UI rebinds into a
// few, while keeping streaming visually smooth (~30 fps). Tune STREAM_BATCH up to cut call-outs more.
constexpr int STREAM_BATCH = 6;
constexpr int64_t STREAM_FLUSH_MS = 33;

// Execution Provider Types
enum ExecutionProviderType {
    EP_CPU = 0,      // Default CPU execution provider
    EP_XNNPACK = 1,  // XNNPACK execution provider (optimized for mobile CPU)
    EP_QNN = 2       // QNN execution provider (Qualcomm NPU/HTP)
};
