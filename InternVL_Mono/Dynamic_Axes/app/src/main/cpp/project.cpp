#include "project.h"

inline static void correctUtfBytes(char* bytes) {
    char three;
    while (*bytes != '\0') {
        unsigned char utf8 = *(bytes++);
        three = 0;
        // Switch on the high four bits.
        switch (utf8 >> 4) {
            case 0x00:
            case 0x01:
            case 0x02:
            case 0x03:
            case 0x04:
            case 0x05:
            case 0x06:
            case 0x07:
                // Bit pattern 0xxx. No need for any extra bytes.
                break;
            case 0x08:
            case 0x09:
            case 0x0a:
            case 0x0b:
            case 0x0f:
                /*
                 * Bit pattern 10xx or 1111, which are illegal start bytes.
                 * Note: 1111 is valid for normal UTF-8, but not the
                 * modified UTF-8 used here.
                 */
                *(bytes-1) = '?';
                break;
            case 0x0e:
                // Bit pattern 1110, so there are two additional bytes.
                utf8 = *(bytes++);
                if ((utf8 & 0xc0) != 0x80) {
                    --bytes;
                    *(bytes-1) = '?';
                    break;
                }
                three = 1;
                // Fall through to take care of the final byte.
            case 0x0c:
            case 0x0d:
                // Bit pattern 110x, so there is one additional byte.
                utf8 = *(bytes++);
                if ((utf8 & 0xc0) != 0x80) {
                    --bytes;
                    if(three)--bytes;
                    *(bytes-1)='?';
                }
                break;
        }
    }
}

inline static std::string get_output_words(const int &id) {
    std::string words = tokenizer->decode(id);
    if (words.length() == 6 && words[0] == '<' && words[words.length() - 1] == '>' && words[1] == '0' && words[2] == 'x')
    {
        words = static_cast<char>(std::stoi(words.substr(3, 2), nullptr, 16));

    }
    correctUtfBytes(words.data());
    return words;
}

inline static void clear_history() {
    history_len = 0;
    attention_mask = 1;
    split_factor = prompt_head_len;
    for (int i = 0; i < num_keys_values; i++) {
        input_tensors_D[i] = input_tensors_kv_init_D[i];
    }
}

extern "C"
JNIEXPORT jintArray JNICALL
Java_com_example_myapplication_MainActivity_Process_1Texture(JNIEnv *env, jclass clazz) {
    glUseProgram(computeProgram);
    glDispatchCompute(workGroupCountX, workGroupCountY, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    jintArray final_results = env->NewIntArray(pixelCount);
    env->SetIntArrayRegion(final_results, 0, pixelCount, (jint*) glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, rgbSize_int, GL_MAP_READ_BIT));
    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
    return final_results;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Process_1Init(JNIEnv *env, jclass clazz, jint texture_id) {
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(computeShader, 1, &computeShaderSource, nullptr);
    glCompileShader(computeShader);
    computeProgram = glCreateProgram();
    glAttachShader(computeProgram, computeShader);
    glLinkProgram(computeProgram);
    glDeleteShader(computeShader);
    yuvTexLoc = glGetUniformLocation(computeProgram, "yuvTex");
    glUniform1i(yuvTexLoc, 0);
    glGenBuffers(1, &pbo_A);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_A);
    glBufferData(GL_PIXEL_PACK_BUFFER, rgbSize, nullptr, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, pbo_A);
    glBindImageTexture(0, static_cast<GLuint> (texture_id), 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA);
    tokenizer = MNN::Transformer::Tokenizer::createTokenizer(vocab_file);
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Run_1LLM_1ABC(JNIEnv *env, jclass clazz,
                                                          jstring jquery,
                                                          jbyteArray pixel_values,
                                                          jboolean use_vision) {
    // We haven't supported the chat mode for InternVL yet.
    clear_history();
    const char *query = env->GetStringUTFChars(jquery, nullptr);
    std::vector<int> input_ids = tokenizer->encode(query);
    input_ids.insert(input_ids.begin(), {1, 92543, 1008, 364, 92544, 92545, 364});         // Chat prompt head
    input_ids.insert(input_ids.end(), {92542, 92543, 525, 11353, 364});                    // Chat prompt tail
    ids_len = static_cast<int> (input_ids.size());
    input_dims_C[0][1] = ids_len;
    ort_runtime_C->CreateTensorWithDataAsOrtValue(
            memory_info_C,
            reinterpret_cast<void*>(input_ids.data()), input_ids_buffer_size[ids_len],
            input_dims_C[0].data(), input_dims_C[0].size(), input_types_C[0],
            &input_tensors_C[0]);
    if (output_tensors_C[buffer_index_C][0] != nullptr) {
        ort_runtime_C->ReleaseValue(output_tensors_C[buffer_index_C][0]);
        buffer_index_C += 1;
        if (buffer_index_C >= output_tensors_C.size()) {
            return JNI_FALSE;
        }
    }
    ort_runtime_C->Run(session_model_C, run_options_C, input_names_C.data(),
                       (const OrtValue* const*) input_tensors_C.data(),
                       input_tensors_C.size(), output_names_C.data(), output_names_C.size(),
                       output_tensors_C[buffer_index_C].data());
    if (use_vision) {
        ids_len += num_image_token;
        split_factor = prompt_head_len + num_image_token;
        jbyte* pixels = env->GetByteArrayElements(pixel_values, nullptr);
        ort_runtime_A->CreateTensorWithDataAsOrtValue(
                memory_info_A, reinterpret_cast<void*>(pixels), rgbSize_i8,
                input_dims_A[0].data(), input_dims_A[0].size(), input_types_A[0], &input_tensors_A[0]);
        ort_runtime_A->Run(session_model_A, run_options_A, input_names_A.data(),
                           (const OrtValue* const*) input_tensors_A.data(),
                           input_tensors_A.size(), output_names_A.data(), output_names_A.size(),
                           output_tensors_A.data());
        input_tensors_B[0] = output_tensors_C[buffer_index_C][0];
        input_tensors_B[1] = output_tensors_A[0];
        if (output_tensors_B[buffer_index_B][0] != nullptr) {
            ort_runtime_B->ReleaseValue(output_tensors_B[buffer_index_B][0]);
            buffer_index_B += 1;
            if (buffer_index_B >= output_tensors_B.size()) {
                return JNI_FALSE;
            }
        }
        ort_runtime_B->Run(session_model_B, run_options_B, input_names_B.data(),
                           (const OrtValue* const*) input_tensors_B.data(),
                           input_tensors_B.size(), output_names_B.data(), output_names_B.size(),
                           output_tensors_B[buffer_index_B].data());
        input_tensors_D[last_indices] = output_tensors_B[buffer_index_B][0];
    } else {
        input_tensors_D[last_indices] = output_tensors_C[buffer_index_C][0];
    }
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_Run_1LLM_1CD(JNIEnv *env, jclass clazz,
                                                        jboolean add_prompt) {
    ort_runtime_D->Run(session_model_D, run_options_D, input_names_D.data(),
                       (const OrtValue *const *)input_tensors_D.data(),
                       input_tensors_D.size(), output_names_D.data(), output_names_D.size(),
                       output_tensors_D[buffer_index_D].data());
    int token_id = end_id_0;
    if (chatting) {  // Java multithreading may not stop immediately. Therefore, use a switch to prevent incorrect saves.
        void* max_logit_id;
        ort_runtime_D->GetTensorMutableData(output_tensors_D[buffer_index_D][0], &max_logit_id);
        token_id = reinterpret_cast<int*>(max_logit_id)[0];
        input_tensors_C[0] = output_tensors_D[buffer_index_D][0];
        for (int i = 0; i < num_keys_values; i++) {
            input_tensors_D[i] = output_tensors_D[buffer_index_D][layer_indices[i]];
        }
    }
    if (buffer_index_D > 0) {
        int clear_idx = buffer_index_D - 1;
        if (output_tensors_D[clear_idx][0] != nullptr) {
            for (int i = 0; i < amount_of_output_D; i++) {
                ort_runtime_D->ReleaseValue(output_tensors_D[clear_idx][i]);
            }
        }
    }
    buffer_index_D += 1;
    if (buffer_index_D >= output_tensors_D.size()) {
        return env->NewStringUTF("Out_of_Buffer");
    }
    if (chatting) {  // Java multithreading may not stop immediately. Therefore, use a switch to prevent incorrect saves.
        if ((token_id != end_id_0) && (token_id != end_id_1) && (history_len < max_seq_len)) {
            ort_runtime_C->ReleaseValue(output_tensors_C[buffer_index_C][0]);
            buffer_index_C += 1;
            if (buffer_index_C >= output_tensors_C.size()) {
                return env->NewStringUTF("Out_of_Buffer");
            }
            ort_runtime_C->Run(session_model_C, run_options_C, input_names_C.data(),
                               (const OrtValue* const*) input_tensors_C.data(),
                               input_tensors_C.size(), output_names_C.data(), output_names_C.size(),
                               output_tensors_C[buffer_index_C].data());
            input_tensors_D[last_indices] = output_tensors_C[buffer_index_C][0];
            if (add_prompt) {
                history_len += ids_len;
                split_factor = prompt_head_len;
                attention_mask = 0;
            } else {
                history_len += 1;
            }
            return env->NewStringUTF(get_output_words(token_id).c_str());
        } else {
            return env->NewStringUTF("END");
        }
    }
    return env->NewStringUTF("PASS");
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1A(JNIEnv *env, jclass clazz,
                                                            jobject asset_manager,
                                                            jboolean use_float_model,
                                                            jboolean use_qnn,
                                                            jboolean use_dsp_npu,
                                                            jboolean use_xnnpack,
                                                            jboolean low_memory_mode) {
    OrtStatus *status;
    OrtAllocator *allocator;
    OrtEnv *ort_env_A;
    OrtSessionOptions *session_options_A;
    {
        std::vector<char> fileBuffer;
        std::vector<char> fileBuffer_external;
        off_t fileSize;
        off_t fileSize_external;
        bool use_storage_path = false;
        if (!low_memory_mode) {
            if (asset_manager != nullptr) {
                AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
                AAsset *asset = AAssetManager_open(mgr, file_name_A.c_str(), AASSET_MODE_BUFFER);
                fileSize = AAsset_getLength(asset);
                fileBuffer.resize(fileSize);
                AAsset_read(asset, fileBuffer.data(), fileSize);
                if (!file_name_A_external.empty()) {
                    // Load external data using AAsset_read. For models with multiple external files, manually load additional files as needed.
                    AAsset* asset_ex = AAssetManager_open(mgr, file_name_A_external.c_str(), AASSET_MODE_BUFFER);
                    fileSize_external = AAsset_getLength(asset_ex);
                    fileBuffer_external.resize(fileSize_external);
                    AAsset_read(asset_ex, fileBuffer_external.data(), fileSize_external);
                }
            } else {
                use_storage_path = true;
                low_memory_mode = true;
            }
        }
        ort_runtime_A = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        ort_runtime_A->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "myapplication", &ort_env_A);
        ort_runtime_A->CreateSessionOptions(&session_options_A);
        ort_runtime_A->CreateRunOptions(&run_options_A);
        ort_runtime_A->AddRunConfigEntry(run_options_A, "memory.enable_memory_arena_shrinkage", "");            // Keep empty for performance; "cpu:0" for low memory usage.
        ort_runtime_A->AddRunConfigEntry(run_options_A, "disable_synchronize_execution_providers", "1");        // 1 for aggressive performance.
        ort_runtime_A->DisableProfiling(session_options_A);
        ort_runtime_A->EnableCpuMemArena(session_options_A);
        ort_runtime_A->EnableMemPattern(session_options_A);
        ort_runtime_A->SetSessionExecutionMode(session_options_A, ORT_SEQUENTIAL);
        ort_runtime_A->SetInterOpNumThreads(session_options_A, 4);
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.dynamic_block_base",
                                             "2");                                                              // One block can contain 1 or more cores, and sharing 1 job.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.intra_op_thread_affinities",           // Binding the #cpu to run the model. 'A;B' means A & B work respectively. 'A,B' means A & B work cooperatively.
                                             "1,3;2,4");                                                        // It is the best cost/performance (C/P) value setting for running the InternVL on my device.
        ort_runtime_A->SetIntraOpNumThreads(session_options_A, 3);                                              // dynamic_block_base + 1
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.inter_op.allow_spinning",
                                             "1");                                                              // 0 for low power
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.intra_op.allow_spinning",
                                             "1");                                                              // 0 for low power
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.force_spinning_stop",
                                             "0");                                                              // 1 for low power
        ort_runtime_A->SetSessionGraphOptimizationLevel(session_options_A, ORT_ENABLE_ALL);                     // CPU backend would failed on some FP16 operators with latest opset. Hence, use ORT_ENABLE_EXTENDED instead of ORT_ENABLE_ALL.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "optimization.minimal_build_optimizations",
                                             "");                                                               // Keep empty for full optimization
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "optimization.disable_specified_optimizers",
                                             "NchwcTransformer");                                               // For Arm
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.disable_prepacking",
                                             "0");                                                              // 0 for enable
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "optimization.enable_gelu_approximation",
                                             "1");                                                              // Set 1 to enable
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "mlas.enable_gemm_fastmath_arm64_bfloat16",
                                             "1");                                                              //
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.disable_aot_function_inlining",
                                             "0");                                                              // 0 for speed
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.qdqisint8allowed",
                                             "1");                                                              // 1 for Arm
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.enable_quant_qdq_cleanup",
                                             "1");                                                              // 0 for precision, 1 for performance
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.disable_double_qdq_remover",
                                             "0");                                                              // 1 for precision, 0 for performance
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.disable_quant_qdq",
                                             "0");                                                              // 0 for use Int8
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.use_ort_model_bytes_directly",
                                             "1");                                                              // Use this option to lower the peak memory during loading.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.use_ort_model_bytes_for_initializers",
                                             "0");                                                              // If set use_ort_model_bytes_directly=1, use_ort_model_bytes_for_initializers should be 0.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.set_denormal_as_zero",
                                             "1");                                                              // Use 0 instead of NaN or Inf.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.use_env_allocators",
                                             "1");                                                              // Use it to lower memory usage.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.use_device_allocator_for_initializers",
                                             "1");                                                              // Use it to lower memory usage.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.qdq_matmulnbits_accuracy_level",
                                             "0");                                                              // 0:default, 1:FP32, 2:FP16, 3:BF16, 4:INT8
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "ep.dynamic.workload_type",
                                             "Default");                                                        // Default = Performance; Efficient = Save Power
        std::vector<const char*> option_keys = {};
        std::vector<const char*> option_values = {};
        if (use_qnn) {
            setenv("LD_LIBRARY_PATH", cache_path, 1);
            setenv("ADSP_LIBRARY_PATH", cache_path, 1);
            if (use_dsp_npu) {
                option_keys.push_back("backend_path");
                option_values.push_back(qnn_htp_so);
                ort_runtime_A->AddRunConfigEntry(run_options_A, "qnn.htp_perf_mode", "burst");                  // Do not use "option_keys.push_back("htp_performance_mode")", it not work now. (demo version=1.20.1)
                ort_runtime_A->AddRunConfigEntry(run_options_A, "qnn.htp_perf_mode_post_run", "burst");
                ort_runtime_A->AddRunConfigEntry(run_options_A, "qnn.rpc_control_latency", "0");
                option_keys.push_back("htp_graph_finalization_optimization_mode");
                option_values.push_back("3");
                option_keys.push_back("soc_model");
                option_values.push_back(qualcomm_soc_id);
                option_keys.push_back("device_id");
                option_values.push_back("0");                                                                   // 0 for single device
                option_keys.push_back("vtcm_mb");
                option_values.push_back("0");                                                                   // 0 for auto
                option_keys.push_back("qnn_context_priority");
                option_values.push_back("high");
                if (use_float_model) {
                    option_keys.push_back("enable_htp_fp16_precision");
                    option_values.push_back("1");
                } else {
                    option_keys.push_back("enable_htp_fp16_precision");
                    option_values.push_back("0");
                    ort_runtime_A->AddSessionConfigEntry(session_options_A, "ep.context_enable", "1");
                    ort_runtime_A->AddSessionConfigEntry(session_options_A, "ep.context_embed_mode", "1");
                    ort_runtime_A->AddSessionConfigEntry(session_options_A, "ep.context_file_path", ctx_model_A);
                }
            } else {
                option_keys.push_back("backend_path");
                option_values.push_back(qnn_cpu_so);
            }
            ort_runtime_A->SessionOptionsAppendExecutionProvider(session_options_A, "QNN", option_keys.data(), option_values.data(), option_keys.size());
        } else if (use_xnnpack) {
            option_keys.push_back("intra_op_num_threads");
            option_values.push_back("4");
            ort_runtime_A->SetInterOpNumThreads(session_options_A, 4);                                                // Keep the same value as above.
            ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.intra_op.allow_spinning", "0");          // Set to 0.
            ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.dynamic_block_base", "1");               // Set to 1.
            ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.intra_op_thread_affinities", "1,2,3,4"); // Use ',' to split the #core
            ort_runtime_A->SetIntraOpNumThreads(session_options_A, 1);                                                // // Set to 1.
            ort_runtime_A->SessionOptionsAppendExecutionProvider(session_options_A, "XNNPACK", option_keys.data(), option_values.data(), option_keys.size());
        }
        if (low_memory_mode) {
            if (use_storage_path) {
                status = ort_runtime_A->CreateSession(ort_env_A, (storage_path + file_name_A).c_str(), session_options_A, &session_model_A);
            } else {
                status = ort_runtime_A->CreateSession(ort_env_A, (cache_path + file_name_A).c_str(), session_options_A, &session_model_A);
            }
        } else {
            if (!file_name_A_external.empty()) {
                const char* external_file_names[] = {file_name_A_external.c_str()};                         // Add all external data file names here if your model uses multiple external files.
                const char* external_file_buffers[] = {fileBuffer_external.data()};                         // Read external data into fileBuffers and add them here.
                size_t external_file_sizes[] = {fileBuffer_external.size()};                                // Store the size of each fileBuffer here for multiple external data files.
                ort_runtime_A->AddExternalInitializersFromFilesInMemory(session_options_A, external_file_names, const_cast<char**>(external_file_buffers), external_file_sizes, 1);  // '1' indicates a single external file.
            }
            status = ort_runtime_A->CreateSessionFromArray(ort_env_A, fileBuffer.data(), fileSize, session_options_A, &session_model_A);
        }
    }
    if (status != nullptr) {
        return JNI_FALSE;
    }
    std::size_t amount_of_input;
    ort_runtime_A->GetAllocatorWithDefaultOptions(&allocator);
    ort_runtime_A->SessionGetInputCount(session_model_A, &amount_of_input);
    input_names_A.resize(amount_of_input);
    input_dims_A.resize(amount_of_input);
    input_types_A.resize(amount_of_input);
    input_tensors_A.resize(amount_of_input);
    for (int i = 0; i < amount_of_input; i++) {
        char* name;
        OrtTypeInfo* typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_A->SessionGetInputName(session_model_A, i, allocator, &name);
        input_names_A[i] = name;
        ort_runtime_A->SessionGetInputTypeInfo(session_model_A, i, &typeinfo);
        ort_runtime_A->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_A->GetTensorElementType(tensor_info, &type);
        input_types_A[i] = type;
        ort_runtime_A->GetDimensionsCount(tensor_info, &dimensions);
        input_dims_A[i].resize(dimensions);
        ort_runtime_A->GetDimensions(tensor_info, input_dims_A[i].data(), dimensions);
        ort_runtime_A->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) {
            ort_runtime_A->ReleaseTypeInfo(typeinfo);
        }
    }
    std::size_t amount_of_output;
    ort_runtime_A->SessionGetOutputCount(session_model_A, &amount_of_output);
    output_names_A.resize(amount_of_output);
    output_dims_A.resize(amount_of_output);
    output_types_A.resize(amount_of_output);
    output_tensors_A.resize(amount_of_output);
    for (int i = 0; i < amount_of_output; i++) {
        char* name;
        OrtTypeInfo* typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_A->SessionGetOutputName(session_model_A, i, allocator, &name);
        output_names_A[i] = name;
        ort_runtime_A->SessionGetOutputTypeInfo(session_model_A, i, &typeinfo);
        ort_runtime_A->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_A->GetTensorElementType(tensor_info, &type);
        output_types_A[i] = type;
        ort_runtime_A->GetDimensionsCount(tensor_info, &dimensions);
        output_dims_A[i].resize(dimensions);
        ort_runtime_A->GetDimensions(tensor_info, output_dims_A[i].data(), dimensions);
        ort_runtime_A->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) {
            ort_runtime_A->ReleaseTypeInfo(typeinfo);
        }
    }
    ort_runtime_A->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info_A);
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1B(JNIEnv *env, jclass clazz,
                                                            jobject asset_manager,
                                                            jboolean use_xnnpack,
                                                            jboolean low_memory_mode) {
    OrtStatus *status;
    OrtAllocator *allocator;
    OrtEnv *ort_env_B;
    OrtSessionOptions *session_options_B;
    {
        std::vector<char> fileBuffer;
        std::vector<char> fileBuffer_external;
        off_t fileSize;
        off_t fileSize_external;
        bool use_storage_path = false;
        if (!low_memory_mode) {
            if (asset_manager != nullptr) {
                AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
                AAsset *asset = AAssetManager_open(mgr, file_name_B.c_str(), AASSET_MODE_BUFFER);
                fileSize = AAsset_getLength(asset);
                fileBuffer.resize(fileSize);
                AAsset_read(asset, fileBuffer.data(), fileSize);
                if (!file_name_B_external.empty()) {
                    // Load external data using AAsset_read. For models with multiple external files, manually load additional files as needed.
                    AAsset* asset_ex = AAssetManager_open(mgr, file_name_B_external.c_str(), AASSET_MODE_BUFFER);
                    fileSize_external = AAsset_getLength(asset_ex);
                    fileBuffer_external.resize(fileSize_external);
                    AAsset_read(asset_ex, fileBuffer_external.data(), fileSize_external);
                }
            } else {
                use_storage_path = true;
                low_memory_mode = true;
            }
        }
        ort_runtime_B = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        ort_runtime_B->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "myapplication", &ort_env_B);
        ort_runtime_B->CreateSessionOptions(&session_options_B);
        ort_runtime_B->CreateRunOptions(&run_options_B);
        ort_runtime_B->AddRunConfigEntry(run_options_B, "memory.enable_memory_arena_shrinkage", "");            // Keep empty for performance; "cpu:0" for low memory usage.
        ort_runtime_B->AddRunConfigEntry(run_options_B, "disable_synchronize_execution_providers", "1");        // 1 for aggressive performance.
        ort_runtime_B->DisableProfiling(session_options_B);
        ort_runtime_B->EnableCpuMemArena(session_options_B);
        ort_runtime_B->EnableMemPattern(session_options_B);
        ort_runtime_B->SetSessionExecutionMode(session_options_B, ORT_SEQUENTIAL);
        ort_runtime_B->SetInterOpNumThreads(session_options_B, 4);
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.dynamic_block_base",
                                             "2");                                                              // One block can contain 1 or more cores, and sharing 1 job.
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.intra_op_thread_affinities",           // Binding the #cpu to run the model. 'A;B' means A & B work respectively. 'A,B' means A & B work cooperatively.
                                             "1,3;2,4");                                                        // It is the best cost/performance (C/P) value setting for running the InternVL on my device.
        ort_runtime_B->SetIntraOpNumThreads(session_options_B, 3);                                              // dynamic_block_base + 1
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.inter_op.allow_spinning",
                                             "1");                                                              // 0 for low power
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.intra_op.allow_spinning",
                                             "1");                                                              // 0 for low power
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.force_spinning_stop",
                                             "0");                                                              // 1 for low power
        ort_runtime_B->SetSessionGraphOptimizationLevel(session_options_B, ORT_ENABLE_ALL);                     // CPU backend would failed on some FP16 operators with latest opset. Hence, use ORT_ENABLE_EXTENDED instead of ORT_ENABLE_BLL.
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "optimization.minimal_build_optimizations",
                                             "");                                                               // Keep empty for full optimization
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "optimization.disable_specified_optimizers",
                                             "NchwcTransformer");                                               // For Arm
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.disable_prepacking",
                                             "0");                                                              // 0 for enable
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "optimization.enable_gelu_approximation",
                                             "1");                                                              // Set 1 is better for this model
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "mlas.enable_gemm_fastmath_arm64_bfloat16",
                                             "1");                                                              //
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.disable_aot_function_inlining",
                                             "0");                                                              // 0 for speed
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.qdqisint8allowed",
                                             "1");                                                              // 1 for Arm
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.enable_quant_qdq_cleanup",
                                             "1");                                                              // 0 for precision, 1 for performance
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.disable_double_qdq_remover",
                                             "0");                                                              // 1 for precision, 0 for performance
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.disable_quant_qdq",
                                             "0");                                                              // 0 for use Int8
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.use_ort_model_bytes_directly",
                                             "1");                                                              // Use this option to lower the peak memory during loading.
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.use_ort_model_bytes_for_initializers",
                                             "0");                                                              // If set use_ort_model_bytes_directly=1, use_ort_model_bytes_for_initializers should be 0.
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.set_denormal_as_zero",
                                             "1");                                                              // Use 0 instead of NaN or Inf.
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.use_env_allocators",
                                             "1");                                                              // Use it to lower memory usage.
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.use_device_allocator_for_initializers",
                                             "1");                                                              // Use it to lower memory usage.
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.qdq_matmulnbits_accuracy_level",
                                             "0");                                                              // 0:default, 1:FP32, 2:FP16, 3:BF16, 4:INT8
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "ep.dynamic.workload_type",
                                             "Default");                                                        // Default = Performance; Efficient = Save Power
        std::vector<const char*> option_keys = {};
        std::vector<const char*> option_values = {};
        if (use_xnnpack) {
            option_keys.push_back("intra_op_num_threads");
            option_values.push_back("4");
            ort_runtime_B->SetInterOpNumThreads(session_options_B, 4);                                                // Keep the same value as above.
            ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.intra_op.allow_spinning", "0");          // Set to 0.
            ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.dynamic_block_base", "1");               // Set to 1.
            ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.intra_op_thread_affinities", "1,2,3,4"); // Use ',' to split the #core
            ort_runtime_B->SetIntraOpNumThreads(session_options_B, 1);                                                // Set to 1.
            ort_runtime_B->SessionOptionsAppendExecutionProvider(session_options_B, "XNNPACK", option_keys.data(), option_values.data(), option_keys.size());
        }
        if (low_memory_mode) {
            if (use_storage_path) {
                status = ort_runtime_B->CreateSession(ort_env_B, (storage_path + file_name_B).c_str(), session_options_B, &session_model_B);
            } else {
                status = ort_runtime_B->CreateSession(ort_env_B, (cache_path + file_name_B).c_str(), session_options_B, &session_model_B);
            }
        } else {
            if (!file_name_B_external.empty()) {
                const char* external_file_names[] = {file_name_B_external.c_str()};                         // Add all external data file names here if your model uses multiple external files.
                const char* external_file_buffers[] = {fileBuffer_external.data()};                         // Read external data into fileBuffers and add them here.
                size_t external_file_sizes[] = {fileBuffer_external.size()};                                // Store the size of each fileBuffer here for multiple external data files.
                ort_runtime_B->AddExternalInitializersFromFilesInMemory(session_options_B, external_file_names, const_cast<char**>(external_file_buffers), external_file_sizes, 1);  // '1' indicates a single external file.
            }
            status = ort_runtime_B->CreateSessionFromArray(ort_env_B, fileBuffer.data(), fileSize, session_options_B, &session_model_B);
        }
    }
    if (status != nullptr) {
        return JNI_FALSE;
    }
    std::size_t amount_of_input;
    ort_runtime_B->GetAllocatorWithDefaultOptions(&allocator);
    ort_runtime_B->SessionGetInputCount(session_model_B, &amount_of_input);
    input_names_B.resize(amount_of_input);
    input_dims_B.resize(amount_of_input);
    input_types_B.resize(amount_of_input);
    input_tensors_B.resize(amount_of_input);
    for (int i = 0; i < amount_of_input; i++) {
        char* name;
        OrtTypeInfo* typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_B->SessionGetInputName(session_model_B, i, allocator, &name);
        input_names_B[i] = name;
        ort_runtime_B->SessionGetInputTypeInfo(session_model_B, i, &typeinfo);
        ort_runtime_B->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_B->GetTensorElementType(tensor_info, &type);
        input_types_B[i] = type;
        ort_runtime_B->GetDimensionsCount(tensor_info, &dimensions);
        input_dims_B[i].resize(dimensions);
        ort_runtime_B->GetDimensions(tensor_info, input_dims_B[i].data(), dimensions);
        ort_runtime_B->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) {
            ort_runtime_B->ReleaseTypeInfo(typeinfo);
        }
    }
    std::size_t amount_of_output;
    ort_runtime_B->SessionGetOutputCount(session_model_B, &amount_of_output);
    output_names_B.resize(amount_of_output);
    output_dims_B.resize(amount_of_output);
    output_types_B.resize(amount_of_output);
    for (auto & i : output_tensors_B) {
        i.resize(amount_of_output);
    }
    for (int i = 0; i < amount_of_output; i++) {
        char* name;
        OrtTypeInfo* typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_B->SessionGetOutputName(session_model_B, i, allocator, &name);
        output_names_B[i] = name;
        ort_runtime_B->SessionGetOutputTypeInfo(session_model_B, i, &typeinfo);
        ort_runtime_B->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_B->GetTensorElementType(tensor_info, &type);
        output_types_B[i] = type;
        ort_runtime_B->GetDimensionsCount(tensor_info, &dimensions);
        output_dims_B[i].resize(dimensions);
        ort_runtime_B->GetDimensions(tensor_info, output_dims_B[i].data(), dimensions);
        ort_runtime_B->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) {
            ort_runtime_B->ReleaseTypeInfo(typeinfo);
        }
    }
    ort_runtime_B->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info_B);
    input_dims_B[0][1] = 0;
    std::vector<float> hidden_state(1);
    ort_runtime_B->CreateTensorWithDataAsOrtValue(
            memory_info_B,
            reinterpret_cast<void*>(hidden_state.data()), 0,
            input_dims_B[0].data(), input_dims_B[0].size(), input_types_B[0],
            &input_tensors_B[0]);
    std::vector<float> vision_embed(num_image_token * hidden_size);
    ort_runtime_B->CreateTensorWithDataAsOrtValue(
            memory_info_B,
            reinterpret_cast<void*>(vision_embed.data()), vision_embed.size() * sizeof(float),
            input_dims_B[1].data(), input_dims_B[1].size(), input_types_B[1],
            &input_tensors_B[1]);
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1C(JNIEnv *env, jclass clazz,
                                                            jobject asset_manager,
                                                            jboolean use_xnnpack,
                                                            jboolean low_memory_mode) {
    OrtStatus *status;
    OrtAllocator *allocator;
    OrtEnv *ort_env_C;
    OrtSessionOptions *session_options_C;
    {
        std::vector<char> fileBuffer;
        std::vector<char> fileBuffer_external;
        off_t fileSize;
        off_t fileSize_external;
        bool use_storage_path = false;
        if (!low_memory_mode) {
            if (asset_manager != nullptr) {
                AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
                AAsset *asset = AAssetManager_open(mgr, file_name_C.c_str(), AASSET_MODE_BUFFER);
                fileSize = AAsset_getLength(asset);
                fileBuffer.resize(fileSize);
                AAsset_read(asset, fileBuffer.data(), fileSize);
                if (!file_name_C_external.empty()) {
                    // Load external data using AAsset_read. For models with multiple external files, manually load additional files as needed.
                    AAsset* asset_ex = AAssetManager_open(mgr, file_name_C_external.c_str(), AASSET_MODE_BUFFER);
                    fileSize_external = AAsset_getLength(asset_ex);
                    fileBuffer_external.resize(fileSize_external);
                    AAsset_read(asset_ex, fileBuffer_external.data(), fileSize_external);
                }
            } else {
                use_storage_path = true;
                low_memory_mode = true;
            }
        }
        ort_runtime_C = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        ort_runtime_C->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "myapplication", &ort_env_C);
        ort_runtime_C->CreateSessionOptions(&session_options_C);
        ort_runtime_C->CreateRunOptions(&run_options_C);
        ort_runtime_C->AddRunConfigEntry(run_options_C, "memory.enable_memory_arena_shrinkage", "");            // Keep empty for performance; "cpu:0" for low memory usage.
        ort_runtime_C->AddRunConfigEntry(run_options_C, "disable_synchronize_execution_providers", "1");        // 1 for aggressive performance.
        ort_runtime_C->DisableProfiling(session_options_C);
        ort_runtime_C->EnableCpuMemArena(session_options_C);
        ort_runtime_C->EnableMemPattern(session_options_C);
        ort_runtime_C->SetSessionExecutionMode(session_options_C, ORT_SEQUENTIAL);
        ort_runtime_C->SetInterOpNumThreads(session_options_C, 4);
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.dynamic_block_base",
                                             "2");                                                              // One block can contain 1 or more cores, and sharing 1 job.
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.intra_op_thread_affinities",           // Binding the #cpu to run the model. 'A;B' means A & B work respectively. 'A,B' means A & B work cooperatively.
                                             "1,3;2,4");                                                        // It is the best cost/performance (C/P) value setting for running the InternVL on my device.
        ort_runtime_C->SetIntraOpNumThreads(session_options_C, 3);                                              // dynamic_block_base + 1
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.inter_op.allow_spinning",
                                             "1");                                                              // 0 for low power
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.intra_op.allow_spinning",
                                             "1");                                                              // 0 for low power
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.force_spinning_stop",
                                             "0");                                                              // 1 for low power
        ort_runtime_C->SetSessionGraphOptimizationLevel(session_options_C, ORT_ENABLE_ALL);                     // CPU backend would failed on some FP16 operators with latest opset. Hence, use ORT_ENABLE_EXTENDED instead of ORT_ENABLE_CLL.
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "optimization.minimal_build_optimizations",
                                             "");                                                               // Keep empty for full optimization
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "optimization.disable_specified_optimizers",
                                             "NchwcTransformer");                                               // For Arm
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.disable_prepacking",
                                             "0");                                                              // 0 for enable
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "optimization.enable_gelu_approximation",
                                             "1");                                                              // Set 1 is better for this model
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "mlas.enable_gemm_fastmath_arm64_bfloat16",
                                             "1");                                                              //
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.disable_aot_function_inlining",
                                             "0");                                                              // 0 for speed
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.qdqisint8allowed",
                                             "1");                                                              // 1 for Arm
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.enable_quant_qdq_cleanup",
                                             "1");                                                              // 0 for precision, 1 for performance
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.disable_double_qdq_remover",
                                             "0");                                                              // 1 for precision, 0 for performance
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.disable_quant_qdq",
                                             "0");                                                              // 0 for use Int8
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.use_ort_model_bytes_directly",
                                             "1");                                                              // Use this option to lower the peak memory during loading.
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.use_ort_model_bytes_for_initializers",
                                             "0");                                                              // If set use_ort_model_bytes_directly=1, use_ort_model_bytes_for_initializers should be 0.
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.set_denormal_as_zero",
                                             "1");                                                              // Use 0 instead of NaN or Inf.
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.use_env_allocators",
                                             "1");                                                              // Use it to lower memory usage.
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.use_device_allocator_for_initializers",
                                             "1");                                                              // Use it to lower memory usage.
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.qdq_matmulnbits_accuracy_level",
                                             "0");                                                              // 0:default, 1:FP32, 2:FP16, 3:BF16, 4:INT8
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "ep.dynamic.workload_type",
                                             "Default");                                                        // Default = Performance; Efficient = Save Power
        std::vector<const char*> option_keys = {};
        std::vector<const char*> option_values = {};
        if (use_xnnpack) {
            option_keys.push_back("intra_op_num_threads");
            option_values.push_back("4");
            ort_runtime_C->SetInterOpNumThreads(session_options_C, 4);                                                // Keep the same value as above.
            ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.intra_op.allow_spinning", "0");          // Set to 0.
            ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.dynamic_block_base", "1");               // Set to 1.
            ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.intra_op_thread_affinities", "1,2,3,4"); // Use ',' to split the #core
            ort_runtime_C->SetIntraOpNumThreads(session_options_C, 1);                                                // // Set to 1.
            ort_runtime_C->SessionOptionsAppendExecutionProvider(session_options_C, "XNNPACK", option_keys.data(), option_values.data(), option_keys.size());
        }
        if (low_memory_mode) {
            if (use_storage_path) {
                status = ort_runtime_C->CreateSession(ort_env_C, (storage_path + file_name_C).c_str(), session_options_C, &session_model_C);
            } else {
                status = ort_runtime_C->CreateSession(ort_env_C, (cache_path + file_name_C).c_str(), session_options_C, &session_model_C);
            }
        } else {
            if (!file_name_C_external.empty()) {
                const char* external_file_names[] = {file_name_C_external.c_str()};                         // Add all external data file names here if your model uses multiple external files.
                const char* external_file_buffers[] = {fileBuffer_external.data()};                         // Read external data into fileBuffers and add them here.
                size_t external_file_sizes[] = {fileBuffer_external.size()};                                // Store the size of each fileBuffer here for multiple external data files.
                ort_runtime_C->AddExternalInitializersFromFilesInMemory(session_options_C, external_file_names, const_cast<char**>(external_file_buffers), external_file_sizes, 1);  // '1' indicates a single external file.
            }
            status = ort_runtime_C->CreateSessionFromArray(ort_env_C, fileBuffer.data(), fileSize, session_options_C, &session_model_C);
        }
    }
    if (status != nullptr) {
        return JNI_FALSE;
    }
    std::size_t amount_of_input;
    ort_runtime_C->GetAllocatorWithDefaultOptions(&allocator);
    ort_runtime_C->SessionGetInputCount(session_model_C, &amount_of_input);
    input_names_C.resize(amount_of_input);
    input_dims_C.resize(amount_of_input);
    input_types_C.resize(amount_of_input);
    input_tensors_C.resize(amount_of_input);
    for (int i = 0; i < amount_of_input; i++) {
        char* name;
        OrtTypeInfo* typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_C->SessionGetInputName(session_model_C, i, allocator, &name);
        input_names_C[i] = name;
        ort_runtime_C->SessionGetInputTypeInfo(session_model_C, i, &typeinfo);
        ort_runtime_C->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_C->GetTensorElementType(tensor_info, &type);
        input_types_C[i] = type;
        ort_runtime_C->GetDimensionsCount(tensor_info, &dimensions);
        input_dims_C[i].resize(dimensions);
        ort_runtime_C->GetDimensions(tensor_info, input_dims_C[i].data(), dimensions);
        ort_runtime_C->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) {
            ort_runtime_C->ReleaseTypeInfo(typeinfo);
        }
    }
    std::size_t amount_of_output;
    ort_runtime_C->SessionGetOutputCount(session_model_C, &amount_of_output);
    output_names_C.resize(amount_of_output);
    output_dims_C.resize(amount_of_output);
    output_types_C.resize(amount_of_output);
    for (auto & i : output_tensors_C) {
        i.resize(amount_of_output);
    }
    for (int i = 0; i < amount_of_output; i++) {
        char* name;
        OrtTypeInfo* typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_C->SessionGetOutputName(session_model_C, i, allocator, &name);
        output_names_C[i] = name;
        ort_runtime_C->SessionGetOutputTypeInfo(session_model_C, i, &typeinfo);
        ort_runtime_C->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_C->GetTensorElementType(tensor_info, &type);
        output_types_C[i] = type;
        ort_runtime_C->GetDimensionsCount(tensor_info, &dimensions);
        output_dims_C[i].resize(dimensions);
        ort_runtime_C->GetDimensions(tensor_info, output_dims_C[i].data(), dimensions);
        ort_runtime_C->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) {
            ort_runtime_C->ReleaseTypeInfo(typeinfo);
        }
    }
    ort_runtime_C->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info_C);
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1D(JNIEnv *env, jclass clazz,
                                                            jobject asset_manager,
                                                            jboolean use_xnnpack,
                                                            jboolean low_memory_mode) {
    OrtStatus *status;
    OrtAllocator *allocator;
    OrtEnv *ort_env_D;
    OrtSessionOptions *session_options_D;
    {
        std::vector<char> fileBuffer;
        std::vector<char> fileBuffer_external;
        off_t fileSize;
        off_t fileSize_external;
        bool use_storage_path = false;
        if (!low_memory_mode) {
            if (asset_manager != nullptr) {
                AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
                AAsset *asset = AAssetManager_open(mgr, file_name_D.c_str(), AASSET_MODE_BUFFER);
                fileSize = AAsset_getLength(asset);
                fileBuffer.resize(fileSize);
                AAsset_read(asset, fileBuffer.data(), fileSize);
                if (!file_name_D_external.empty()) {
                    // Load external data using AAsset_read. For models with multiple external files, manually load additional files as needed.
                    AAsset* asset_ex = AAssetManager_open(mgr, file_name_D_external.c_str(), AASSET_MODE_BUFFER);
                    fileSize_external = AAsset_getLength(asset_ex);
                    fileBuffer_external.resize(fileSize_external);
                    AAsset_read(asset_ex, fileBuffer_external.data(), fileSize_external);
                }
            } else {
                use_storage_path = true;
                low_memory_mode = true;
            }
        }
        ort_runtime_D = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        ort_runtime_D->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "myapplication", &ort_env_D);
        ort_runtime_D->CreateSessionOptions(&session_options_D);
        ort_runtime_D->CreateRunOptions(&run_options_D);
        ort_runtime_D->AddRunConfigEntry(run_options_D, "memory.enable_memory_arena_shrinkage", "");            // Keep empty for performance; "cpu:0" for low memory usage.
        ort_runtime_D->AddRunConfigEntry(run_options_D, "disable_synchronize_execution_providers", "1");        // 1 for aggressive performance.
        ort_runtime_D->DisableProfiling(session_options_D);
        ort_runtime_D->EnableCpuMemArena(session_options_D);
        ort_runtime_D->EnableMemPattern(session_options_D);
        ort_runtime_D->SetSessionExecutionMode(session_options_D, ORT_SEQUENTIAL);
        ort_runtime_D->SetInterOpNumThreads(session_options_D, 6);
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.dynamic_block_base",
                                             "2");                                                              // One block can contain 1 or more cores, and sharing 1 job.
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.intra_op_thread_affinities",           // Binding the #cpu to run the model. 'A;B' means A & B work respectively. 'A,B' means A & B work cooperatively.
                                             "1,3,5;2,4,6");                                                        // It is the best cost/performance (C/P) value setting for running the InternVL on my device.
        ort_runtime_D->SetIntraOpNumThreads(session_options_D, 3);                                              // dynamic_block_base + 1
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.inter_op.allow_spinning",
                                             "1");                                                              // 0 for low power
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.intra_op.allow_spinning",
                                             "1");                                                              // 0 for low power
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.force_spinning_stop",
                                             "0");                                                              // 1 for low power
        ort_runtime_D->SetSessionGraphOptimizationLevel(session_options_D, ORT_ENABLE_ALL);                     // CPU backend would failed on some FP16 operators with latest opset. Hence, use ORT_ENABLE_EXTENDED instead of ORT_ENABLE_DLL.
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "optimization.minimal_build_optimizations",
                                             "");                                                               // Keep empty for full optimization
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "optimization.disable_specified_optimizers",
                                             "NchwcTransformer");                                               // For Arm
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.disable_prepacking",
                                             "0");                                                              // 0 for enable
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "optimization.enable_gelu_approximation",
                                             "1");                                                              // Set 1 is better for this model
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "mlas.enable_gemm_fastmath_arm64_bfloat16",
                                             "1");                                                              //
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.disable_aot_function_inlining",
                                             "0");                                                              // 0 for speed
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.qdqisint8allowed",
                                             "1");                                                              // 1 for Arm
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.enable_quant_qdq_cleanup",
                                             "1");                                                              // 0 for precision, 1 for performance
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.disable_double_qdq_remover",
                                             "0");                                                              // 1 for precision, 0 for performance
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.disable_quant_qdq",
                                             "0");                                                              // 0 for use Int8
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.use_ort_model_bytes_directly",
                                             "1");                                                              // Use this option to lower the peak memory during loading.
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.use_ort_model_bytes_for_initializers",
                                             "0");                                                              // If set use_ort_model_bytes_directly=1, use_ort_model_bytes_for_initializers should be 0.
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.set_denormal_as_zero",
                                             "1");                                                              // Use 0 instead of NaN or Inf.
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.use_env_allocators",
                                             "1");                                                              // Use it to lower memory usage.
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.use_device_allocator_for_initializers",
                                             "1");                                                              // Use it to lower memory usage.
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.qdq_matmulnbits_accuracy_level",
                                             "0");                                                              // 0:default, 1:FP32, 2:FP16, 3:BF16, 4:INT8
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "ep.dynamic.workload_type",
                                             "Default");                                                        // Default = Performance; Efficient = Save Power
        std::vector<const char*> option_keys = {};
        std::vector<const char*> option_values = {};
        if (use_xnnpack) {
            option_keys.push_back("intra_op_num_threads");
            option_values.push_back("4");
            ort_runtime_D->SetInterOpNumThreads(session_options_D, 4);                                                // Keep the same value as above.
            ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.intra_op.allow_spinning", "0");          // Set to 0.
            ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.dynamic_block_base", "1");               // Set to 1.
            ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.intra_op_thread_affinities", "1,2,3,4"); // Use ',' to split the #core
            ort_runtime_D->SetIntraOpNumThreads(session_options_D, 1);                                                // // Set to 1.
            ort_runtime_D->SessionOptionsAppendExecutionProvider(session_options_D, "XNNPACK", option_keys.data(), option_values.data(), option_keys.size());
        }
        if (low_memory_mode) {
            if (use_storage_path) {
                status = ort_runtime_D->CreateSession(ort_env_D, (storage_path + file_name_D).c_str(), session_options_D, &session_model_D);
            } else {
                status = ort_runtime_D->CreateSession(ort_env_D, (cache_path + file_name_D).c_str(), session_options_D, &session_model_D);
            }
        } else {
            if (!file_name_D_external.empty()) {
                const char* external_file_names[] = {file_name_D_external.c_str()};                         // Add all external data file names here if your model uses multiple external files.
                const char* external_file_buffers[] = {fileBuffer_external.data()};                         // Read external data into fileBuffers and add them here.
                size_t external_file_sizes[] = {fileBuffer_external.size()};                                // Store the size of each fileBuffer here for multiple external data files.
                ort_runtime_D->AddExternalInitializersFromFilesInMemory(session_options_D, external_file_names, const_cast<char**>(external_file_buffers), external_file_sizes, 1);  // '1' indicates a single external file.
            }
            status = ort_runtime_D->CreateSessionFromArray(ort_env_D, fileBuffer.data(), fileSize, session_options_D, &session_model_D);
        }
    }
    if (status != nullptr) {
        return JNI_FALSE;
    }
    std::size_t amount_of_input;
    ort_runtime_D->GetAllocatorWithDefaultOptions(&allocator);
    ort_runtime_D->SessionGetInputCount(session_model_D, &amount_of_input);
    input_names_D.resize(amount_of_input);
    input_dims_D.resize(amount_of_input);
    input_types_D.resize(amount_of_input);
    input_tensors_D.resize(amount_of_input);
    for (int i = 0; i < amount_of_input; i++) {
        char* name;
        OrtTypeInfo* typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_D->SessionGetInputName(session_model_D, i, allocator, &name);
        input_names_D[i] = name;
        ort_runtime_D->SessionGetInputTypeInfo(session_model_D, i, &typeinfo);
        ort_runtime_D->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_D->GetTensorElementType(tensor_info, &type);
        input_types_D[i] = type;
        ort_runtime_D->GetDimensionsCount(tensor_info, &dimensions);
        input_dims_D[i].resize(dimensions);
        ort_runtime_D->GetDimensions(tensor_info, input_dims_D[i].data(), dimensions);
        ort_runtime_D->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) {
            ort_runtime_D->ReleaseTypeInfo(typeinfo);
        }
    }
    ort_runtime_D->SessionGetOutputCount(session_model_D, &amount_of_output_D);
    output_names_D.resize(amount_of_output_D);
    output_dims_D.resize(amount_of_output_D);
    output_types_D.resize(amount_of_output_D);
    for (auto & i : output_tensors_D) {
        i.resize(amount_of_output_D);
    }
    for (int i = 0; i < amount_of_output_D; i++) {
        char* name;
        OrtTypeInfo* typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo* tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_D->SessionGetOutputName(session_model_D, i, allocator, &name);
        output_names_D[i] = name;
        ort_runtime_D->SessionGetOutputTypeInfo(session_model_D, i, &typeinfo);
        ort_runtime_D->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_D->GetTensorElementType(tensor_info, &type);
        output_types_D[i] = type;
        ort_runtime_D->GetDimensionsCount(tensor_info, &dimensions);
        output_dims_D[i].resize(dimensions);
        ort_runtime_D->GetDimensions(tensor_info, output_dims_D[i].data(), dimensions);
        ort_runtime_D->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) {
            ort_runtime_D->ReleaseTypeInfo(typeinfo);
        }
    }
    ort_runtime_D->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info_D);
    for (int i = 0; i < num_layers; i++) {
        input_dims_D[i][2] = 0;
        ort_runtime_D->CreateTensorWithDataAsOrtValue(
                memory_info_D,
                reinterpret_cast<void*>(past_key_values_init.data()), 0,
                input_dims_D[i].data(), input_dims_D[i].size(), input_types_D[i],
                &input_tensors_kv_init_D[i]);
    }
    for (int i = num_layers; i < num_keys_values; i++) {
        input_dims_D[i][1] = 0;
        ort_runtime_D->CreateTensorWithDataAsOrtValue(
                memory_info_D,
                reinterpret_cast<void*>(past_key_values_init.data()), 0,
                input_dims_D[i].data(), input_dims_D[i].size(), input_types_D[i],
                &input_tensors_kv_init_D[i]);
    }
    ort_runtime_D->CreateTensorWithDataAsOrtValue(
            memory_info_D,
            reinterpret_cast<void*>(&attention_mask), sizeof(int8_t),
            input_dims_D[num_keys_values].data(), input_dims_D[num_keys_values].size(), input_types_D[num_keys_values],
            &input_tensors_D[num_keys_values]);
    ort_runtime_D->CreateTensorWithDataAsOrtValue(
            memory_info_D,
            reinterpret_cast<void*>(&split_factor), sizeof(int64_t),
            input_dims_D[split_factor_indices].data(), input_dims_D[split_factor_indices].size(), input_types_D[split_factor_indices],
            &input_tensors_D[split_factor_indices]);
    input_dims_D[last_indices][1] = 0;
    std::vector<float> hidden_state(1);
    ort_runtime_D->CreateTensorWithDataAsOrtValue(
            memory_info_D,
            reinterpret_cast<void*>(hidden_state.data()), 0,
            input_dims_D[last_indices].data(), input_dims_D[last_indices].size(), input_types_D[last_indices],
            &input_tensors_D[last_indices]);
    for (int i = 1; i < num_keys_values; i++) {
        layer_indices[i] += i;
    }
    input_ids_buffer_size[1] = sizeof(int);
    for (int i = 2; i < max_seq_len; i++) {
        input_ids_buffer_size[i] = input_ids_buffer_size[i - 1] + sizeof(int);
    }
    return JNI_TRUE;
}
