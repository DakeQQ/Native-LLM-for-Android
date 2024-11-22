
#include <jni.h>
#include <iostream>
#include <fstream>
#include <android/asset_manager_jni.h>
#include "onnxruntime_cxx_api.h"
#include "tokenizer.hpp"
#include "onnxruntime_float16.h"

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
int64_t history_len = 0;
int64_t ids_len = 0;
// Ort::Float16_t attention_mask = Ort::Float16_t(-65504.f);
float attention_mask = -65504.f;
const std::string file_name_A = "Model_Llama_1B_1024.ort";
const std::string file_name_A_external = "NONE";   // If using external data to load the model, provide the file name; otherwise, set to "NONE". If contains many parts, please modify the project.cpp line 313-315.
const int max_token_history = 1024; // Please set this value to match the model name flag.
const int start_id = 128000;
const int end_id_0 = 128001;
const int end_id_1 = 128009;
const int past_key_value_size = 16 * 8 * 64 * max_token_history; // 16 * 4 * 64, Remember edit the value if using others param size model.
const int single_chat_limit = 341;                        // It is recommended to set it to max_token_history/3, and use phrases like 'go ahead', 'go on', or 'and then?' to continue answering."
const int next_chat_buffer = max_token_history - single_chat_limit;
std::vector<int32_t> input_ids(max_token_history, 0);
std::vector<int> accumulate_num_ids(30, 0); // Just make sure the size is enough before reaching max_token_history.
std::vector<int> num_ids_per_chat(30, 0);   // Same size with accumulate_num_ids.
std::vector<int> save_max_logit_position(max_token_history, 0);
const std::string storage_path = "/storage/emulated/0/Android/data/com.example.myapplication/";
const std::string vocab_file = "/data/user/0/com.example.myapplication/cache/vocab_Llama.txt"; // We have moved the vocab.txt from assets to the cache folder in Java process.
