#include <jni.h>
#include <iostream>
#include <fstream>
#include <android/asset_manager_jni.h>
#include "onnxruntime_cxx_api.h"
#include <GLES3/gl32.h>
#include "tokenizer.hpp"


const char* computeShaderSource = "#version 320 es\n"
                                  "#extension GL_OES_EGL_image_external_essl3 : require\n"
                                  "precision mediump float;\n"
                                  "layout(binding = 0) uniform samplerExternalOES yuvTex;\n"
                                  "layout(local_size_x = 16, local_size_y = 16) in;\n"  // gpu_num_group=16, Customize it to fit your device's specifications.
                                  "const int camera_width = 960;\n"                     //  camera_width, Remember edit the value if using custom param in export_config.py.
                                  "const int camera_height = 960;\n"                    //  camera_height, Remember edit the value if using custom param in export_config.py.
                                  "const int camera_height_minus = camera_height - 1;\n"
                                  "layout(std430, binding = 1) buffer Output {\n"
                                  "    int result[camera_height * camera_width];\n"     // pixelCount
                                  "} outputData;\n"
                                  "const vec3 bias = vec3(-0.15, -0.5, -0.2);\n"
                                  "const mat3 YUVtoRGBMatrix = mat3(127.5, 0.0, 1.402 * 127.5, "
                                  "                                 127.5, -0.344136 * 127.5, -0.714136 * 127.5, "
                                  "                                 127.5, 1.772 * 127.5, 0.0);\n"
                                  "void main() {\n"
                                  "    ivec2 texelPos = ivec2(gl_GlobalInvocationID.xy);\n"
                                  "    ivec3 rgb = clamp(ivec3(YUVtoRGBMatrix * (texelFetch(yuvTex, texelPos, 0).rgb + bias)), -128, 127) + 128;\n"
                                  // Use int8 packing the pixels, it would be 1.6 times faster than using float32 buffer.
                                  "    outputData.result[texelPos.x * camera_height + (camera_height_minus - texelPos.y)] = (rgb.b << 16) | (rgb.r << 8) | (rgb.g);\n"
                                  "}";
// OpenGL Setting
GLuint pbo_A = 0;
GLuint computeProgram = 0;
GLint yuvTexLoc = 0;
const GLsizei camera_width = 960;   // Remember edit the value if using custom param in export_config.py.
const GLsizei camera_height = 960;  // Remember edit the value if using custom param in export_config.py.
const GLsizei pixelCount = camera_width * camera_height;
const int pixelCount_rgb = 3 * pixelCount;
const int gpu_num_group = 16;       // Customize it to fit your device's specifications.
const GLsizei rgbSize = pixelCount_rgb * sizeof(float);
const GLsizei rgbSize_int = pixelCount * sizeof(int);
const GLsizei rgbSize_i8 = pixelCount_rgb * sizeof(uint8_t);
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
Tokenizer* tokenizer;
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
const std::string file_name_A_external = "";                               // If using external data to load the model, provide the file name; otherwise, set to "". If contains many parts, please modify the project.cpp line 357-360.
const std::string file_name_B_external = "";                               // If using external data to load the model, provide the file name; otherwise, set to "". If contains many parts, please modify the project.cpp line 537-540.
const std::string file_name_C_external = "";                               // If using external data to load the model, provide the file name; otherwise, set to "". If contains many parts, please modify the project.cpp line 730-733.
const std::string file_name_D_external = "";                               // If using external data to load the model, provide the file name; otherwise, set to "". If contains many parts, please modify the project.cpp line 918-921.
const std::string file_name_E_external = "";                               // If using external data to load the model, provide the file name; otherwise, set to "". If contains many parts, please modify the project.cpp line 1129-1132.
const int prompt_head_len = 5;
const int WIDTH_FACTOR = 10;                                               // Please set this value to match the exported model.
const int HEIGHT_FACTOR = 10;                                              // Please set this value to match the exported model.
const int max_token_history = 1024;                                        // Please set this value to match the exported model.
const int end_id_0 = 151643;
const int end_id_1 = 151645;
const int hidden_size = 1536;                                              // Qwen2VL-2B = 1536;  Qwen2.5VL-3B = 2048;
const size_t past_key_value_size = 28 * 2 * 128 * max_token_history;       // Qwen2VL-2B = 28 * 2 * 128;  Qwen2.5VL-3B = 36 * 2 * 128; Remember edit the value if using others param size model.
const int64_t image_pad_len = WIDTH_FACTOR * HEIGHT_FACTOR;
const int64_t pos_factor_v = 1 - image_pad_len + WIDTH_FACTOR;
std::vector<int> input_ids(max_token_history, 0);
const std::string storage_path = "/storage/emulated/0/Android/data/com.example.myapplication/";
const std::string vocab_file = "/data/user/0/com.example.myapplication/cache/vocab_Qwen.txt"; // We have moved the vocab.txt from assets to the cache folder in Java process.
const char* qualcomm_soc_id = "43";                                                           // 0 for unknown, Find your device from here: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/overview.html#supported-snapdragon-devices
const char* ctx_model_A = "/storage/emulated/0/Android/data/com.example.myapplication/ctx_model_A.onnx";
const char* cache_path = "/data/user/0/com.example.myapplication/cache/";
const char* qnn_htp_so = "/data/user/0/com.example.myapplication/cache/libQnnHtp.so";         //  If use (std::string + "libQnnHtp.so").c_str() instead, it will open failed.
const char* qnn_cpu_so = "/data/user/0/com.example.myapplication/cache/libQnnCpu.so";         //  If use (std::string + "libQnnCpu.so").c_str() instead, it will open failed.
// Just specify the path for qnn_*_so, and the code will automatically locate the other required libraries.
