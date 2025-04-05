#include <jni.h>
#include <omp.h>
#include <iostream>
#include <fstream>
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

// Model File Settings
const std::string file_name_A = "Qwen.onnx";
const std::string file_name_A_external = "59da4aba-1227-11f0-aada-bc091bee2d5c";
const bool use_deepseek = true;                                                 // Enable if using DeepSeek-Distill-Qwen.

// Model Configuration
const int num_layers = 28;                                                      // Transformer layers. Refer to config.json for the value.
const int max_seq_len = 4096;                                                   // Please set this value to match the exported model.
const int num_keys_values = num_layers + num_layers;
const int last_indices = num_keys_values + 1;
const int single_chat_limit = max_seq_len / 2 - 1;                              // You can adjust this value. If you set it greater than (max_seq_len / 2 - 1), the previous context may be cleared.
const int next_chat_buffer = max_seq_len - single_chat_limit;

// Token IDs
int start_id = 151644;
int end_id_1 = 151645;
const int end_id_0 = 151643;

// Storage Paths
const std::string storage_path = "/storage/emulated/0/Android/data/com.example.myapplication/";
const std::string cache_path = "/data/user/0/com.example.myapplication/cache/"; // We have moved the vocab.txt from assets to the cache folder in Java process.

// Counters and Indices
int response_count = 0;
int save_index = 0;
int buffer_index = 0;
int history_len = 0;
int ids_len = 0;

// Init Values
float attention_mask = -65504.f;

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
std::vector<std::vector<OrtValue *>> output_tensors_A(max_seq_len);

// Arrays and Vectors
std::vector<int> input_ids(max_seq_len, 0);
std::vector<int> accumulate_num_ids(20, 0);                    // Just make sure the size is enough before reaching max_seq_len.
std::vector<int> num_ids_per_chat(20, 0);                      // Same size with accumulate_num_ids.
std::vector<int> save_max_logit_position(max_seq_len, 0);
std::vector<int> layer_indices(num_keys_values, 0);
std::vector<Ort::Float16_t> past_key_values_init(1, Ort::Float16_t(0.f));

// NLP Components
MNN::Transformer::Tokenizer* tokenizer;
