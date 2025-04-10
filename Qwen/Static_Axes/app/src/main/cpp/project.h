#include <jni.h>
#include <iostream>
#include <fstream>
#include <android/asset_manager_jni.h>
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_float16.h"
#include "tokenizer.hpp"

const OrtApi *ort_runtime_A;
OrtSession *session_model_A;
OrtRunOptions *run_options_A;
std::vector<const char *> input_names_A;
std::vector<const char *> output_names_A;
std::vector<std::vector<std::int64_t>> input_dims_A;
std::vector<std::vector<std::int64_t>> output_dims_A;
std::vector<ONNXTensorElementDataType> input_types_A;
std::vector<ONNXTensorElementDataType> output_types_A;
std::vector<OrtValue *> input_tensors_A;
std::vector<OrtValue *> output_tensors_A;
Tokenizer* tokenizer;
int response_count = 0;
int save_index = 0;
int start_id = 151644;
int end_id_1 = 151645;
int64_t history_len = 0;
int64_t ids_len = 0;
// Ort::Float16_t attention_mask = Ort::Float16_t(-65504.f);        // Enable if using all fp16 format, and project.cpp line: 73, 189, 202, 463
float attention_mask = -65504.f;                                    // Disable if using all fp16 format, and project.cpp line: 72, 188, 201, 462

const bool use_deepseek = false;                                    // Enable if using DeepSeek-Distill-Qwen.
const std::string file_name_A = "Model_Qwen_1_5B_1024.ort";
const std::string file_name_A_external = "";                        // If using external data to load the model, provide the file name; otherwise, set to "". If contains many parts, please modify the project.cpp line 384-387.
const int max_token_history = 1024;                                 // Please set this value to match the model name flag.
const int end_id_0 = 151643;
const size_t past_key_value_size = 28 * 2 * 128 * max_token_history; // Remember edit the value if using others param size model.
const int single_chat_limit = max_token_history / 2 - 1;             // You can adjust it. If you set the value greater than (max_token_history / 2 - 1), the previous context might be cleared.
const int next_chat_buffer = max_token_history - single_chat_limit;
std::vector<int32_t> input_ids(max_token_history, 0);
std::vector<int> accumulate_num_ids(10, 0);                    // Just make sure the size is enough before reaching max_token_history.
std::vector<int> num_ids_per_chat(10, 0);                      // Same size with accumulate_num_ids.
std::vector<int> save_max_logit_position(max_token_history, 0);
const std::string storage_path = "/storage/emulated/0/Android/data/com.example.myapplication/";
const std::string cache_path = "/data/user/0/com.example.myapplication/cache/"; // We have moved the vocab.txt from assets to the cache folder in Java process.
