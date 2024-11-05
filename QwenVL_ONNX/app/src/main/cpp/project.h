
#include <jni.h>
#include <iostream>
#include <fstream>
#include <android/asset_manager_jni.h>
#include "onnxruntime_cxx_api.h"
#include <GLES3/gl32.h>
#include "tokenizer.hpp"


const char* computeShaderSource = "#version 320 es\n"
                                  "#extension GL_OES_EGL_image_external_essl3 : require\n"
                                  "precision highp float;\n"
                                  "layout(local_size_x = 16, local_size_y = 16) in;\n"  // gpu_num_group=16, Customize it to fit your device's specifications.
                                  "layout(binding = 0) uniform samplerExternalOES yuvTex;\n"
                                  "const int camera_width = 960;\n"  //  camera_width
                                  "const int camera_height = 960;\n"  //  camera_height
                                  "layout(std430, binding = 1) buffer Output {\n"
                                  "    int result[camera_height * camera_width];\n"  // pixelCount
                                  "} outputData;\n"
                                  "const vec3 bias = vec3(0.0, -0.5, -0.5);\n"
                                  "const mat3 YUVtoRGBMatrix = mat3(255.0, 0.0, 1.402 * 255.0, "
                                  "                                 255.0, -0.344136 * 255.0, -0.714136 * 255.0, "
                                  "                                 255.0, 1.772 * 255.0, 0.0);\n"
                                  "void main() {\n"
                                  "    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);\n"
                                  "    vec3 rgb = clamp(YUVtoRGBMatrix * (texelFetch(yuvTex, texelPos, 0).rgb + bias), 0.0, 255.0);\n"
                                  // Use int8 packing the pixels, it would be 1.6 times faster than using float32 buffer.
                                  "    outputData.result[texelPos.x * camera_height + (camera_height - texelPos.y - 1)] = (int(rgb.b * 0.55) << 16) | (int(rgb.r * 0.5) << 8) | (int(rgb.g * 1.4));\n"
                                  "}";
// OpenGL Setting
GLuint pbo_A = 0;
GLuint computeProgram = 0;
GLint yuvTexLoc = 0;
const GLsizei camera_width = 960;
const GLsizei camera_height = 960;
const GLsizei pixelCount = camera_width * camera_height;
const int pixelCount_rgb = 3 * pixelCount;
const int gpu_num_group = 16;  // Customize it to fit your device's specifications.
const GLsizei rgbSize = pixelCount_rgb * sizeof(float);
const GLsizei rgbSize_i8 = pixelCount * sizeof(int);
const GLsizei workGroupCountX = camera_width / gpu_num_group;
const GLsizei workGroupCountY = camera_height / gpu_num_group;

// ONNX Runtime & LLM Setting
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
const OrtApi* ort_runtime_D;
OrtSession* session_model_D;
OrtRunOptions *run_options_D;
std::vector<const char*> input_names_D;
std::vector<const char*> output_names_D;
std::vector<std::vector<std::int64_t>> input_dims_D;
std::vector<std::vector<std::int64_t>> output_dims_D;
std::vector<ONNXTensorElementDataType> input_types_D;
std::vector<ONNXTensorElementDataType> output_types_D;
std::vector<OrtValue*> input_tensors_D;
std::vector<OrtValue*> output_tensors_D;
const OrtApi* ort_runtime_E;
OrtSession* session_model_E;
OrtRunOptions *run_options_E;
std::vector<const char*> input_names_E;
std::vector<const char*> output_names_E;
std::vector<std::vector<std::int64_t>> input_dims_E;
std::vector<std::vector<std::int64_t>> output_dims_E;
std::vector<ONNXTensorElementDataType> input_types_E;
std::vector<ONNXTensorElementDataType> output_types_E;
std::vector<OrtValue*> input_tensors_E;
std::vector<OrtValue*> output_tensors_E;
std::unique_ptr<Tokenizer> tokenizer;
int response_count = 0;
int save_index = 0;
int dummy = 0;
int ids_len_minus = 0;
int split_factor = 0;
int64_t history_len = 0;
int64_t ids_len = 0;
Ort::Float16_t attention_mask = Ort::Float16_t(-65504.f);
Ort::Float16_t pos_factor = Ort::Float16_t(0.f);
const std::string file_name_A = "QwenVL_A.ort";
const std::string file_name_B = "QwenVL_B.ort";
const std::string file_name_C = "QwenVL_C.ort";
const std::string file_name_D = "QwenVL_D.ort";
const std::string file_name_E = "QwenVL_E.ort";
const int WIDTH_FACTOR = 10;
const int HEIGHT_FACTOR = 10;
const int prompt_head_len = 5;
const int hidden_size = 1536;
const int max_token_history = 1024;                                       // Please set this value to match the exported model.
const int end_id_0 = 151643;
const int end_id_1 = 151645;
const int past_key_value_size = 7168 * max_token_history;                 // 28 * 2 * 128, Remember edit the value if using others param size model.
const int64_t image_pad_len = WIDTH_FACTOR * HEIGHT_FACTOR;
const int64_t pos_factor_v = 1 - image_pad_len + WIDTH_FACTOR;
const int single_chat_limit = 341 + image_pad_len;                        // It is recommended to set it to max_token_history/3, and use phrases like 'go ahead', 'go on', or 'and then?' to continue answering."
const int next_chat_buffer = max_token_history - single_chat_limit;
std::vector<int> input_ids(max_token_history, 0);
std::vector<int> ids_exclude_image(max_token_history, 0);
std::vector<int> accumulate_num_ids(30, 0); // Just make sure the size is enough before reaching max_token_history.
std::vector<int> num_ids_per_chat(30, 0);   // Same size with accumulate_num_ids.
std::vector<int> save_max_logit_position(max_token_history, 0);
std::vector<int> image_pad_ids(image_pad_len, 151655);
const std::string storage_path = "/storage/emulated/0/Android/data/com.example.myapplication/";
const std::string vocab_file = "/data/user/0/com.example.myapplication/cache/vocab_Qwen.txt"; // We have moved the vocab.txt from assets to the cache folder in Java process.
const char* ctx_model_A = "/storage/emulated/0/Android/data/com.example.myapplication/ctx_model_A.onnx";
const char* ctx_model_B = "/storage/emulated/0/Android/data/com.example.myapplication/ctx_model_B.onnx";
const char* ctx_model_C = "/storage/emulated/0/Android/data/com.example.myapplication/ctx_model_C.onnx";
const char* ctx_model_D = "/storage/emulated/0/Android/data/com.example.myapplication/ctx_model_D.onnx";
const char* ctx_model_E = "/storage/emulated/0/Android/data/com.example.myapplication/ctx_model_E.onnx";
const char* cache_path = "/data/user/0/com.example.myapplication/cache";
const char* qnn_htp_so = "/data/user/0/com.example.myapplication/cache/libQnnHtp.so";  //  If use (std::string + "libQnnHtp.so").c_str() instead, it will open failed.
const char* qnn_cpu_so = "/data/user/0/com.example.myapplication/cache/libQnnCpu.so";  //  If use (std::string + "libQnnCpu.so").c_str() instead, it will open failed.
// Just specify the path for qnn_*_so, and the code will automatically locate the other required libraries.
