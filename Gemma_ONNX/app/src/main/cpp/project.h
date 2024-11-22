
#include <jni.h>
#include <iostream>
#include <fstream>
#include <android/asset_manager_jni.h>
#include "onnxruntime_cxx_api.h"
#include "tokenizer.hpp"
#include "onnxruntime_float16.h"

const OrtApi* ort_runtime_A;
OrtSession* session_model_A;
OrtRunOptions *run_options_A;
std::vector<const char*> input_names_A;
std::vector<const char*> output_names_A;
std::vector<std::vector<std::int64_t>> input_dims_A;
std::vector<std::vector<std::int64_t>> output_dims_A;
std::vector<ONNXTensorElementDataType> input_types_A;
std::vector<ONNXTensorElementDataType> output_types_A;
std::vector<OrtValue*> input_tensors_A;
std::vector<OrtValue*> output_tensors_A;
const OrtApi* ort_runtime_B;
OrtSession* session_model_B;
OrtRunOptions *run_options_B;
std::vector<const char*> input_names_B;
std::vector<const char*> output_names_B;
std::vector<std::vector<std::int64_t>> input_dims_B;
std::vector<std::vector<std::int64_t>> output_dims_B;
std::vector<ONNXTensorElementDataType> input_types_B;
std::vector<ONNXTensorElementDataType> output_types_B;
std::vector<OrtValue*> input_tensors_B;
std::vector<OrtValue*> output_tensors_B;
const OrtApi* ort_runtime_C;
OrtSession* session_model_C;
OrtRunOptions *run_options_C;
std::vector<const char*> input_names_C;
std::vector<const char*> output_names_C;
std::vector<std::vector<std::int64_t>> input_dims_C;
std::vector<std::vector<std::int64_t>> output_dims_C;
std::vector<ONNXTensorElementDataType> input_types_C;
std::vector<ONNXTensorElementDataType> output_types_C;
std::vector<OrtValue*> input_tensors_C;
std::vector<OrtValue*> output_tensors_C;
Tokenizer* tokenizer;
int response_count = 0;
int save_index = 0;
int64_t history_len = 0;
int64_t ids_len = 0;
//Ort::Float16_t attention_mask = Ort::Float16_t(-65504.f);
float attention_mask = -65504.f;
const std::string file_name_A = "Gemma_2B_1024_A.ort";
const std::string file_name_B = "Gemma_2B_1024_B.ort";
const std::string file_name_C = "Gemma_2B_1024_C.ort";
const int max_token_history = 1024;  // Please set this value to match the model name flag.
const int hidden_size = 2304;
const int start_id = 2;
const int end_id_0 = 107;
const int end_id_1 = 1;
const int past_key_value_size = 26624 * max_token_history;  // 4 * 26 * 256
const int single_chat_limit = 341; // It is recommended to set it to max_token_history/3, and use phrases like 'go ahead', 'go on', or 'and then?' to continue answering."
const int next_chat_buffer = max_token_history - single_chat_limit;
std::vector<int32_t> input_ids(max_token_history, 0);
std::vector<int> accumulate_num_ids(30, 0);  // Just make sure the size is enough before reaching max_token_history.
std::vector<int> num_ids_per_chat(30, 0); // Same size with accumulate_num_ids.
std::vector<int> save_max_logit_position(max_token_history, 0);
const std::string vocab_file = "/data/user/0/com.example.myapplication/cache/vocab_Gemma.txt";  // We have moved the vocab.txt from assets to the cache folder in Java process.
