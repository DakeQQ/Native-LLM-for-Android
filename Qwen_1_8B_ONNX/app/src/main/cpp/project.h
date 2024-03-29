
#include <jni.h>
#include <iostream>
#include <fstream>
#include <android/asset_manager_jni.h>
#include "onnxruntime_cxx_api.h"
#include "nnapi_provider_factory.h"
#include "tokenizer.hpp"

const OrtApi* ort_runtime_A;
OrtSession* session_model_A;
std::vector<const char*> input_names_A;
std::vector<const char*> output_names_A;
std::vector<std::vector<std::int64_t>> input_dims_A;
std::vector<std::vector<std::int64_t>> output_dims_A;
std::vector<ONNXTensorElementDataType> input_types_A;
std::vector<ONNXTensorElementDataType> output_types_A;
std::vector<OrtValue*> input_tensors_A;
std::vector<OrtValue*> output_tensors_A;
std::unique_ptr<Tokenizer> tokenizer;
int response_count = 0;
int save_index = 0;
int64_t kv_shape = 0;
int64_t history_len = 0;
int64_t ids_len = 0;
int64_t offset = 0;
const int max_token_history = 256;
const int vocab_size = 151936;
const int end_id_0 = 151643;
const int start_id = 151644;
const int end_id_1 = 151645;
const int past_key_value_size = 98304 * max_token_history;  // 24 * 2 * 1 * 16 * 128
const int single_chat_limit = 65; // It is recommended to set it to max_token_history/4, and use phrases like 'go ahead', 'go on', or 'and then?' to continue answering."
const int input_ids_buffer_size = max_token_history * sizeof(int32_t);
const int next_chat_buffer = max_token_history - single_chat_limit;
std::vector<int32_t> input_ids(max_token_history, 0);
std::vector<int> accumulate_num_ids(10,0);  // Pre-allocate space for recording the num_ids per chat round. Just make sure the space size is sufficient to accommodate many chat sessions before reaching the max_token_history limit.
std::vector<int> num_ids_per_chat(10,0); // Same size with accumulate_num_ids.
std::vector<int> save_max_logit_position(max_token_history,0);
float attention_mask = -999999999.f;
std::vector<float> theta(64, 0.f);
std::vector<float> idx_theta(max_token_history * theta.size(),0.f);
const int idx_theta_buffer_size = idx_theta.size() * sizeof(float);
const int past_key_values_buffer_size = past_key_value_size * sizeof(float);
const std::string file_name_A = "Qwen_1_8B.ort";
const std::string storage_path = "/storage/emulated/0/Android/data/com.example.myapplication/files/";
const std::string vocab_file = "/data/user/0/com.example.myapplication/cache/vocab.txt";  // We have moved the vocab.txt from assets to the cache folder in Java process.
const char* qnn_htp_so = "/data/user/0/com.example.myapplication/cache/libQnnHtp.so";  //  If use (std::string + "libQnnHtp.so").c_str() instead, it will open failed.
const char* qnn_cpu_so = "/data/user/0/com.example.myapplication/cache/libQnnCpu.so";  //  If use (std::string + "libQnnCpu.so").c_str() instead, it will open failed.
void* output_tensors_buffer_past_key_values;
