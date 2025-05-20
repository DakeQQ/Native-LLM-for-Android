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
                                  "const int camera_width = 960;\n"                     // camera_width, Please set this value to match the exported model.
                                  "const int camera_height = 960;\n"                    // camera_height, Please set this value to match the exported model.
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
                                  // Use uint8 packing the pixels, it would be 1.6 times faster than using float32 buffer.
                                  "    outputData.result[texelPos.x * camera_height + (camera_height_minus - texelPos.y)] = (rgb.b << 16) | (rgb.r << 8) | (rgb.g);\n"
                                  "}";
//------------------------------------------------------------------------------
// OpenGL Configuration
//------------------------------------------------------------------------------
// Camera/Image Dimensions
const GLsizei camera_width = 960;   // Please set this value to match the exported model.
const GLsizei camera_height = 960;  // Please set this value to match the exported model.
const GLsizei pixelCount = camera_width * camera_height;
const int pixelCount_rgb = 3 * pixelCount;

// GPU Configuration
const int gpu_num_group = 16;       // Customize it to fit your device's specifications.
const GLsizei workGroupCountX = camera_width / gpu_num_group;
const GLsizei workGroupCountY = camera_height / gpu_num_group;

// Memory Sizes
const GLsizei rgbSize = pixelCount_rgb * sizeof(float);
const GLsizei rgbSize_int = pixelCount * sizeof(int);
const GLsizei rgbSize_i8 = pixelCount_rgb * sizeof(uint8_t);

// OpenGL Objects/Handles Init.
GLuint pbo_A = 0;
GLuint computeProgram = 0;
GLint yuvTexLoc = 0;

//------------------------------------------------------------------------------
// Model Configuration
//------------------------------------------------------------------------------
// Core Model Parameters
const int num_layers = 36;                              // Transformer layers. Refer to config.json for the value. Qwen2.5VL-3B = 36;
const int max_seq_len = 4096;                           // Please set this value to match the exported model.
const int end_id_0 = 151643;                            // The stop id in Qwen series.
const int end_id_1 = 151645;                            // The stop id in Qwen series.
const int WIDTH_FACTOR = 10;                            // Please set this value to match the exported model.
const int HEIGHT_FACTOR = 10;                           // Please set this value to match the exported model.
const int64_t num_image_token = WIDTH_FACTOR * HEIGHT_FACTOR;

// Rotary Configuration
const int rotary_outputs_len = 4;
const int num_keys_values = num_layers + num_layers;
const int num_keys_values_plus = num_keys_values + 1;

//------------------------------------------------------------------------------
// ONNX Runtime & LLM Files
//------------------------------------------------------------------------------
// Model Files
const std::string file_name_A = "QwenVL_A.onnx";
const std::string file_name_B = "QwenVL_B.onnx";
const std::string file_name_C = "QwenVL_C.onnx";
const std::string file_name_D = "QwenVL_D.onnx";
const std::string file_name_E = "QwenVL_E.onnx";
const std::string file_name_F = "QwenVL_F.onnx";

// External Data Files
const std::string file_name_A_external = "5021b928-34bd-11f0-9e43-bc091bee2d5c.data";  // If using external data to load the model, provide the file name; otherwise, set to "". If contains many parts, please modify the project.cpp Load_Model_A().
const std::string file_name_B_external = "3d945f40-34bd-11f0-ac93-bc091bee2d5c.data";  // If using external data to load the model, provide the file name; otherwise, set to "". If contains many parts, please modify the project.cpp Load_Model_B().
const std::string file_name_C_external = "";                                           // Default to using a single file to load the model.
const std::string file_name_D_external = "";                                           // Default to using a single file to load the model.
const std::string file_name_E_external = "";                                           // Default to using a single file to load the model.
const std::string file_name_F_external = "c865c110-34ca-11f0-a617-bc091bee2d5c.data";  // If using external data to load the model, provide the file name; otherwise, set to "". If contains many parts, please modify the project.cpp Load_Model_F().

//------------------------------------------------------------------------------
// Path Configuration
//------------------------------------------------------------------------------
const std::string storage_path = "/storage/emulated/0/Android/data/com.example.myapplication/";
const std::string vocab_file = "/data/user/0/com.example.myapplication/cache/vocab_Qwen.txt"; // We have moved the vocab.txt from assets to the cache folder in Java process.
const char* cache_path = "/data/user/0/com.example.myapplication/cache/";
const char* ctx_model_B = "/storage/emulated/0/Android/data/com.example.myapplication/ctx_model_B.onnx";

// QNN Library Paths
const char* qnn_cpu_so = "/data/user/0/com.example.myapplication/cache/libQnnCpu.so";  // If use (std::string + "libQnnCpu.so").c_str() instead, it will open failed.
const char* qnn_gpu_so = "/data/user/0/com.example.myapplication/cache/libQnnGpu.so";  // If use (std::string + "libQnnGpu.so").c_str() instead, it will open failed.
const char* qnn_htp_so = "/data/user/0/com.example.myapplication/cache/libQnnHtp.so";  // If use (std::string + "libQnnHtp.so").c_str() instead, it will open failed.
// Just specify the path for qnn_*_so, and the code will automatically locate the other required libraries.

// Device Configuration
const char* qualcomm_soc_id = "43";  // 0 for unknown, Find your device from here: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/overview.html#supported-snapdragon-devices

//------------------------------------------------------------------------------
// Runtime State
//------------------------------------------------------------------------------
// Counters and Indices
bool chatting = true;
int token_id = 0;
int buffer_index_F = 0;
int64_t ids_len = 0;
int64_t kv_seq_len = 0;
int64_t history_len = 0;
size_t amount_of_output_F;
int8_t attention_mask = 1;

// Arrays and Vectors
std::vector<int> rotary_indices(rotary_outputs_len, 0);
std::vector<size_t> input_ids_buffer_size(max_seq_len, 0);
std::vector<float> past_key_values_init(1, 0.f);


// ONNX Runtime API Components
OrtAllocator* allocator;
OrtMemoryInfo *memory_info;

const OrtApi *ort_runtime_A;
OrtSession *session_model_A;
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

const OrtApi* ort_runtime_F;
OrtSession* session_model_F;
OrtRunOptions *run_options_F;
std::vector<const char*> input_names_F;
std::vector<const char*> output_names_F;
std::vector<std::vector<std::int64_t>> input_dims_F;
std::vector<std::vector<std::int64_t>> output_dims_F;
std::vector<ONNXTensorElementDataType> input_types_F;
std::vector<ONNXTensorElementDataType> output_types_F;
std::vector<OrtValue*> input_tensors_F;
std::vector<OrtValue*> input_tensors_kv_init_F(num_keys_values);
std::vector<std::vector<OrtValue*>> output_tensors_F(2);

// Tokenizer
MNN::Transformer::Tokenizer* tokenizer;
