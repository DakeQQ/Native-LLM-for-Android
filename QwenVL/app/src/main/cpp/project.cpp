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
    if (words.length() == 6 && words[0] == '<' && words[words.length() - 1] == '>' && words[1] == '0' && words[2] == 'x') {
        words = static_cast<char>(std::stoi(words.substr(3, 2), nullptr, 16));

    }
    correctUtfBytes(words.data());
    return words;
}

inline static void clear_history(bool use_vision) {
    ids_len = 0;
    kv_seq_len = 0;
    history_len = 0;
    attention_mask = 1;
    if (output_tensors_A[0] != nullptr) {
        ort_runtime_A->ReleaseValue(output_tensors_A[0]);
        output_tensors_A[0] = nullptr;
    }
    if (use_vision) {
        if (output_tensors_C[0] != nullptr) {
            ort_runtime_C->ReleaseValue(output_tensors_C[0]);
            output_tensors_C[0] = nullptr;
        }
        for (int i = 0; i < rotary_outputs_len; i++) {
            if (output_tensors_D[i] != nullptr) {
                ort_runtime_D->ReleaseValue(output_tensors_D[i]);
                output_tensors_D[i] = nullptr;
            }
        }
    } else {
        for (int i = 0; i < rotary_outputs_len; i++) {
            if (output_tensors_E[i] != nullptr) {
                ort_runtime_E->ReleaseValue(output_tensors_E[i]);
                output_tensors_E[i] = nullptr;
            }
        }
    }
}

GLuint createComputeProgram(const char* shaderSource) {
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(shader, 1, &shaderSource, nullptr);
    glCompileShader(shader);

    // In production, add compile log checks
    GLint status = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
        GLint logLen = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logLen);
        if (logLen > 1) {
            std::vector<char> log(logLen);
            glGetShaderInfoLog(shader, logLen, nullptr, log.data());
            // Log/print the error as appropriate in your environment
        }
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glDeleteShader(shader);

    // In production, add link log checks
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
        GLint logLen = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLen);
        if (logLen > 1) {
            std::vector<char> log(logLen);
            glGetProgramInfoLog(program, logLen, nullptr, log.data());
            // Log/print the error as appropriate in your environment
        }
    }

    return program;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_Process_1Texture(JNIEnv *env, jclass clazz, jbyteArray output_buffer) {
    const int write_index = current_index;
    const int read_index  = (current_index + 1) % NUM_BUFFERS;

    // Read back last completed buffer (triple-buffered)
    if (fences[read_index] != 0) {
        glClientWaitSync(fences[read_index], GL_SYNC_FLUSH_COMMANDS_BIT, GLuint64(1000000000));
        glDeleteSync(fences[read_index]);
        fences[read_index] = 0;

        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos[read_index]);
        void* mapped_buffer = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, rgbSize_i8, GL_MAP_READ_BIT);
        if (mapped_buffer) {
            env->SetByteArrayRegion(output_buffer, 0, rgbSize_i8, static_cast<jbyte*>(mapped_buffer));
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
        }
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    }

    // Bind the SSBO for writing the next frame (no clear needed; shader fully overwrites)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, pbos[write_index]);

    // Dispatch the fused processing shader
    glUseProgram(processProgram);
    glDispatchCompute(workGroupCountX, workGroupCountY, 1);

    // Ensure SSBO writes are visible before we fence
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Fence this frame's GPU work
    fences[write_index] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

    // Advance ring buffer index
    current_index = (current_index + 1) % NUM_BUFFERS;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_Process_1Init(JNIEnv *env, jclass clazz, jint texture_id) {
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // Create the processing shader program
    processProgram = createComputeProgram(computeShaderSource);

    // Setup uniforms for the processing program
    glUseProgram(processProgram);
    yuvTexLoc = glGetUniformLocation(processProgram, "yuvTex");
    glUniform1i(yuvTexLoc, 0);

    // Create triple buffers for SSBO writes and CPU readback
    glGenBuffers(NUM_BUFFERS, pbos);
    for (int i = 0; i < NUM_BUFFERS; ++i) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, pbos[i]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, rgbSize_i8, nullptr, GL_DYNAMIC_DRAW);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

    // Bind the external texture to an image unit is not required for samplerExternalOES,
    // but this line is kept as in original code to preserve behavior.
    glBindImageTexture(0, static_cast<GLuint>(texture_id), 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA);

    // Tokenizer Init.
    tokenizer = MNN::Transformer::Tokenizer::createTokenizer(vocab_file);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_Run_1LLM_1Part_10(JNIEnv *env, jclass clazz, jbyteArray pixel_values, jstring jquery, jboolean use_vision)
{
    // We haven't supported the chat mode for QwenVL yet.
    clear_history(use_vision);
    const char *query = env->GetStringUTFChars(jquery, nullptr);
    std::vector<int32_t> input_ids = tokenizer->encode(query);
    input_ids.insert(input_ids.begin(), {151644, 872, 198, 151652, 151653});         // Chat prompt head
    input_ids.insert(input_ids.end(), {151645, 198, 151644, 77091, 198});            // Chat prompt tail
    ids_len = static_cast<int64_t> (input_ids.size());
    input_dims_A[0][1] = ids_len;
    ort_runtime_A->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void*>(input_ids.data()), input_ids_buffer_size[ids_len],
            input_dims_A[0].data(), input_dims_A[0].size(), input_types_A[0],
            &input_tensors_A[0]);
    ort_runtime_A->Run(session_model_A, run_options_A, input_names_A.data(),
                       (const OrtValue* const*) input_tensors_A.data(),
                       input_tensors_A.size(), output_names_A.data(), output_names_A.size(),
                       output_tensors_A.data());
    if (use_vision) {
        ids_len += num_image_token;
        kv_seq_len = ids_len;
        jbyte* pixels = env->GetByteArrayElements(pixel_values, nullptr);
        ort_runtime_B->CreateTensorWithDataAsOrtValue(
                memory_info,
                reinterpret_cast<void*>(pixels), rgbSize_i8,
                input_dims_B[0].data(), input_dims_B[0].size(), input_types_B[0],
                &input_tensors_B[0]);
        ort_runtime_B->Run(session_model_B, run_options_B, input_names_B.data(),
                           (const OrtValue* const*) input_tensors_B.data(),
                           input_tensors_B.size(), output_names_B.data(), output_names_B.size(),
                           output_tensors_B.data());
        input_tensors_C[0] = output_tensors_A[0];
        input_tensors_C[1] = output_tensors_B[0];
        ort_runtime_C->Run(session_model_C, run_options_C, input_names_C.data(),
                           (const OrtValue* const*) input_tensors_C.data(),
                           input_tensors_C.size(), output_names_C.data(), output_names_C.size(),
                           output_tensors_C.data());
        input_tensors_F[num_keys_values_plus] = output_tensors_C[0];
        ort_runtime_D->CreateTensorWithDataAsOrtValue(
                memory_info,
                reinterpret_cast<void*>(&history_len), sizeof(int64_t),
                input_dims_D[0].data(), input_dims_D[0].size(), input_types_D[0],
                &input_tensors_D[0]);
        ort_runtime_D->CreateTensorWithDataAsOrtValue(
                memory_info,
                reinterpret_cast<void*>(&kv_seq_len), sizeof(int64_t),
                input_dims_D[1].data(), input_dims_D[1].size(), input_types_D[1],
                &input_tensors_D[1]);
        ort_runtime_D->Run(session_model_D, run_options_D, input_names_D.data(),
                           (const OrtValue* const*) input_tensors_D.data(),
                           input_tensors_D.size(), output_names_D.data(), output_names_D.size(),
                           output_tensors_D.data());
        for (int i = 0; i < rotary_outputs_len; i++) {
            input_tensors_F[rotary_indices[i]] = output_tensors_D[i];
        }
    } else {
        kv_seq_len = ids_len;
        input_tensors_F[num_keys_values_plus] = output_tensors_A[0];
        ort_runtime_E->CreateTensorWithDataAsOrtValue(
                memory_info,
                reinterpret_cast<void*>(&history_len), sizeof(int64_t),
                input_dims_E[0].data(), input_dims_E[0].size(), input_types_E[0],
                &input_tensors_E[0]);
        ort_runtime_E->CreateTensorWithDataAsOrtValue(
                memory_info,
                reinterpret_cast<void*>(&kv_seq_len), sizeof(int64_t),
                input_dims_E[1].data(), input_dims_E[1].size(), input_types_E[1],
                &input_tensors_E[1]);
        ort_runtime_E->Run(session_model_E, run_options_E, input_names_E.data(),
                           (const OrtValue* const*) input_tensors_E.data(),
                           input_tensors_E.size(), output_names_E.data(), output_names_E.size(),
                           output_tensors_E.data());
        for (int i = 0; i < rotary_outputs_len; i++) {
            input_tensors_F[rotary_indices[i]] = output_tensors_E[i];
        }
    }
    for (int i = 0; i < num_keys_values; i++) {
        input_tensors_F[i] = input_tensors_kv_init_F[i];
    }
    ort_runtime_F->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void*>(&kv_seq_len), sizeof(int64_t),
            input_dims_F[num_keys_values].data(), input_dims_F[num_keys_values].size(), input_types_F[num_keys_values],
            &input_tensors_F[num_keys_values]);
    chatting = true;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_Run_1LLM_1Part_11(JNIEnv *env, jclass clazz, jboolean add_prompt, jboolean use_vision)
{
    if (chatting) {
        for (int i = 0; i < num_keys_values; i++) {
            if (output_tensors_F[buffer_index_F][i] != nullptr) {
                ort_runtime_F->ReleaseValue(output_tensors_F[buffer_index_F][i]);
                output_tensors_F[buffer_index_F][i] = nullptr;
            }
        }
        ort_runtime_F->Run(session_model_F, run_options_F, input_names_F.data(),
                           (const OrtValue *const *)input_tensors_F.data(),
                           input_tensors_F.size(), output_names_F.data(), output_names_F.size(),
                           output_tensors_F[buffer_index_F].data());
        void *max_logit_id;
        input_tensors_A[0] = output_tensors_F[buffer_index_F][num_keys_values_plus];
        ort_runtime_F->GetTensorMutableData(input_tensors_A[0], &max_logit_id);
        token_id = reinterpret_cast<int*>(max_logit_id)[0];
        if ((token_id != end_id_0) && (token_id != end_id_1)) {
            if (add_prompt) {
                ids_len = 1;
                attention_mask = 0;
                if (output_tensors_A[0] != nullptr) {
                    ort_runtime_A->ReleaseValue(output_tensors_A[0]);
                    output_tensors_A[0] = nullptr;
                }
                if (use_vision) {
                    for (int i = 0; i < rotary_outputs_len; i++) {
                        if (output_tensors_D[i] != nullptr) {
                            ort_runtime_D->ReleaseValue(output_tensors_D[i]);
                            output_tensors_D[i] = nullptr;
                        }
                    }
                } else {
                    for (int i = 0; i < rotary_outputs_len; i++) {
                        if (output_tensors_E[i] != nullptr) {
                            ort_runtime_E->ReleaseValue(output_tensors_E[i]);
                            output_tensors_E[i] = nullptr;
                        }
                    }
                }
            }
            ort_runtime_A->Run(session_model_A, run_options_A, input_names_A.data(),
                               (const OrtValue* const*) input_tensors_A.data(),
                               input_tensors_A.size(), output_names_A.data(), output_names_A.size(),
                               output_tensors_A.data());
            input_tensors_F[num_keys_values_plus] = output_tensors_A[0];
            if (use_vision) {
                input_tensors_D[0] = input_tensors_F[num_keys_values];
                input_tensors_D[1] = output_tensors_F[buffer_index_F][num_keys_values];
                ort_runtime_D->Run(session_model_D, run_options_D, input_names_D.data(),
                                   (const OrtValue* const*) input_tensors_D.data(),
                                   input_tensors_D.size(), output_names_D.data(), output_names_D.size(),
                                   output_tensors_D.data());
                for (int i = 0; i < rotary_outputs_len; i++) {
                    input_tensors_F[rotary_indices[i]] = output_tensors_D[i];
                }
            } else {
                input_tensors_E[0] = input_tensors_F[num_keys_values];
                input_tensors_E[1] = output_tensors_F[buffer_index_F][num_keys_values];
                ort_runtime_E->Run(session_model_E, run_options_E, input_names_E.data(),
                                   (const OrtValue* const*) input_tensors_E.data(),
                                   input_tensors_E.size(), output_names_E.data(), output_names_E.size(),
                                   output_tensors_E.data());
                for (int i = 0; i < rotary_outputs_len; i++) {
                    input_tensors_F[rotary_indices[i]] = output_tensors_E[i];
                }
            }
            for (int i = 0; i < num_keys_values_plus; i++) {
                input_tensors_F[i] = output_tensors_F[buffer_index_F][i];
            }
            buffer_index_F = (buffer_index_F != 0) ? 0 : 1;
            return env->NewStringUTF(get_output_words(token_id).c_str());
        } else {
            chatting = false;
            return env->NewStringUTF("END");
        }
    } else {
        return env->NewStringUTF("END");
    }
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1A(JNIEnv *env,  jobject thiz,
                                                            jobject asset_manager,
                                                            jboolean use_xnnpack,
                                                            jboolean low_memory_mode)
{
    OrtStatus *status;
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
        ort_runtime_A->SetInterOpNumThreads(session_options_A, 6);
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.dynamic_block_base",                   // One block can contain 1 or more cores, and sharing 1 job.
                                             "3");
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.intra_op_thread_affinities",           // Binding the #cpu to run the model. 'A;B' means A & B work respectively. 'A,B' means A & B work cooperatively.
                                             "1,3;2,4;5,7");                                                    // It is the best cost/performance (C/P) value setting for running the Qwen 1.8B LLM on my device, due to limitations imposed by the RAM bandwidth.
        ort_runtime_A->SetIntraOpNumThreads(session_options_A, 4);                                              // dynamic_block_base + 1
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.inter_op.allow_spinning",
                                             "1");                                                              // 0 for low power
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.intra_op.allow_spinning",
                                             "1");                                                              // 0 for low power
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.force_spinning_stop",
                                             "0");                                                              // 1 for low power
        ort_runtime_A->SetSessionGraphOptimizationLevel(session_options_A, ORT_ENABLE_ALL);
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "optimization.minimal_build_optimizations",
                                             "");                                                               // Keep empty for full optimization
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "optimization.disable_specified_optimizers",
                                             "NchwcTransformer");                                               // For Arm
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.disable_prepacking",
                                             "0");                                                              // 0 for enable
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "optimization.enable_gelu_approximation",
                                             "1");                                                              // Set 1 is better for this model
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "mlas.enable_gemm_fastmath_arm64_bfloat16",
                                             "1");
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
        std::vector<const char *> option_keys = {};
        std::vector<const char *> option_values = {};
        if (use_xnnpack)
        {
            option_keys.push_back("intra_op_num_threads");
            option_values.push_back("4");
            ort_runtime_A->SetInterOpNumThreads(session_options_A, 4);                                                // Keep the same value as above.
            ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.intra_op.allow_spinning", "0");          // Set to 0.
            ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.dynamic_block_base", "1");               // Set to 1.
            ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.intra_op_thread_affinities", "1,2,3,4"); // Use ',' to split the #core
            ort_runtime_A->SetIntraOpNumThreads(session_options_A, 1);                                                // Set to 1.
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
    std::size_t amount_of_output;
    ort_runtime_A->GetAllocatorWithDefaultOptions(&allocator);
    ort_runtime_A->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    ort_runtime_A->SessionGetInputCount(session_model_A, &amount_of_input);
    input_names_A.resize(amount_of_input);
    input_dims_A.resize(amount_of_input);
    input_types_A.resize(amount_of_input);
    input_tensors_A.resize(amount_of_input);
    for (int i = 0; i < amount_of_input; i++) {
        char *name;
        OrtTypeInfo *typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo *tensor_info;
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
    ort_runtime_A->SessionGetOutputCount(session_model_A, &amount_of_output);
    output_names_A.resize(amount_of_output);
    output_dims_A.resize(amount_of_output);
    output_types_A.resize(amount_of_output);
    output_tensors_A.resize(amount_of_output);
    for (int i = 0; i < amount_of_output; i++) {
        char *name;
        OrtTypeInfo *typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo *tensor_info;
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
    input_ids_buffer_size[1] = sizeof(int);
    for (int i = 2; i < max_seq_len; i++) {
        input_ids_buffer_size[i] = input_ids_buffer_size[i - 1] + sizeof(int);
    }
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1B(JNIEnv *env,  jobject thiz,
                                                            jobject asset_manager,
                                                            jboolean use_float_model,
                                                            jboolean use_qnn_cpu,
                                                            jboolean use_qnn_gpu,
                                                            jboolean use_qnn_npu,
                                                            jboolean use_xnnpack,
                                                            jboolean low_memory_mode)
{
    OrtStatus *status;
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
        ort_runtime_B->SetInterOpNumThreads(session_options_B, 6);
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.dynamic_block_base",                   // One block can contain 1 or more cores, and sharing 1 job.
                                             "3");
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.intra_op_thread_affinities",           // Binding the #cpu to run the model. 'A;B' means A & B work respectively. 'A,B' means A & B work cooperatively.
                                             "1,3;2,4;5,7");                                                    // It is the best cost/performance (C/P) value setting for running the Qwen 1.8B LLM on my device, due to limitations imposed by the RAM bandwidth.
        ort_runtime_B->SetIntraOpNumThreads(session_options_B, 4);                                              // dynamic_block_base + 1
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.inter_op.allow_spinning",
                                             "1");                                                              // 0 for low power
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.intra_op.allow_spinning",
                                             "1");                                                              // 0 for low power
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.force_spinning_stop",
                                             "0");                                                              // 1 for low power
        ort_runtime_B->SetSessionGraphOptimizationLevel(session_options_B, ORT_ENABLE_ALL);
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "optimization.minimal_build_optimizations",
                                             "");                                                               // Keep empty for full optimization
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "optimization.disable_specified_optimizers",
                                             "NchwcTransformer");                                               // For Arm
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.disable_prepacking",
                                             "0");                                                              // 0 for enable
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "optimization.enable_gelu_approximation",
                                             "1");                                                              // Set 1 is better for this model
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "mlas.enable_gemm_fastmath_arm64_bfloat16",
                                             "1");
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
        if (use_qnn_cpu || use_qnn_npu || use_qnn_gpu) {
            setenv("LD_LIBRARY_PATH", cache_path, 1);
            setenv("ADSP_LIBRARY_PATH", cache_path, 1);
            if (use_qnn_cpu) {
                option_keys.push_back("backend_path");
                option_values.push_back(qnn_cpu_so);
                // option_keys.push_back("backend_type");
                // option_values.push_back("cpu");
                option_keys.push_back("profiling_level");
                option_values.push_back("off");
                option_keys.push_back("offload_graph_io_quantization");       // Offload quantization and dequantization of graph I/O to CPU EP else handle by QNN EP .
                option_values.push_back("1");                                 // Default. Enabled.
            } else {
                option_keys.push_back("backend_path");
                if (use_qnn_npu) {
                    option_values.push_back(qnn_htp_so);
                    ort_runtime_B->AddRunConfigEntry(run_options_B, "qnn.htp_perf_mode", "burst");  // Use run_options instead of "option_keys.push_back("htp_performance_mode")"
                    ort_runtime_B->AddRunConfigEntry(run_options_B, "qnn.htp_perf_mode_post_run", "burst");
                    ort_runtime_B->AddRunConfigEntry(run_options_B, "qnn.rpc_control_latency", "0");
                    option_keys.push_back("htp_performance_mode");  // It not work now.
                    option_values.push_back("burst");
                    // option_keys.push_back("backend_type");
                    // option_values.push_back("htp"); 
                    option_keys.push_back("htp_graph_finalization_optimization_mode");
                    option_values.push_back("3");
                    option_keys.push_back("enable_htp_shared_memory_allocator");  // QNN HTP shared memory allocator.
                    option_values.push_back("0");                                 // Default. Disabled.
                    if (use_float_model) {
                        option_keys.push_back("enable_htp_fp16_precision");
                        option_values.push_back("1");
                    } else {
                        option_keys.push_back("enable_htp_fp16_precision");
                        option_values.push_back("0");
                        ort_runtime_A->AddSessionConfigEntry(session_options_B, "ep.context_enable", "1");
                        ort_runtime_A->AddSessionConfigEntry(session_options_B, "ep.context_embed_mode", "1");
                        ort_runtime_A->AddSessionConfigEntry(session_options_B, "ep.context_file_path", ctx_model_B);
                    }
                } else {
                    option_values.push_back(qnn_gpu_so);
                    // option_keys.push_back("backend_type");
                    // option_values.push_back("gpu");
                }
                option_keys.push_back("profiling_level");
                option_values.push_back("off");
                option_keys.push_back("offload_graph_io_quantization");       // Offload quantization and dequantization of graph I/O to CPU EP else handle by QNN EP .
                option_values.push_back("1");                                 // Default. Enabled.
                option_keys.push_back("soc_model");
                option_values.push_back(qualcomm_soc_id);  // 0 for unknown, Find your device from here: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/overview.html#supported-snapdragon-devices
                option_keys.push_back("device_id");
                option_values.push_back("0");              // 0 for single device
                option_keys.push_back("vtcm_mb");
                option_values.push_back("8");              // 0 for auto
                option_keys.push_back("qnn_context_priority");
                option_values.push_back("high");
            }
            ort_runtime_A->SessionOptionsAppendExecutionProvider(session_options_B, "QNN", option_keys.data(), option_values.data(), option_keys.size());
        } else if (use_xnnpack) {
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
    std::size_t amount_of_output;
    ort_runtime_B->GetAllocatorWithDefaultOptions(&allocator);
    ort_runtime_B->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    ort_runtime_B->SessionGetInputCount(session_model_B, &amount_of_input);
    input_names_B.resize(amount_of_input);
    input_dims_B.resize(amount_of_input);
    input_types_B.resize(amount_of_input);
    input_tensors_B.resize(amount_of_input);
    for (int i = 0; i < amount_of_input; i++) {
        char *name;
        OrtTypeInfo *typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo *tensor_info;
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
    ort_runtime_B->SessionGetOutputCount(session_model_B, &amount_of_output);
    output_names_B.resize(amount_of_output);
    output_dims_B.resize(amount_of_output);
    output_types_B.resize(amount_of_output);
    output_tensors_B.resize(amount_of_output);
    for (int i = 0; i < amount_of_output; i++) {
        char *name;
        OrtTypeInfo *typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo *tensor_info;
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
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1C(JNIEnv *env,  jobject thiz,
                                                            jobject asset_manager,
                                                            jboolean use_xnnpack,
                                                            jboolean low_memory_mode)
{
    OrtStatus *status;
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
        ort_runtime_C->SetInterOpNumThreads(session_options_C, 6);
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.dynamic_block_base",                   // One block can contain 1 or more cores, and sharing 1 job.
                                             "3");
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.intra_op_thread_affinities",           // Binding the #cpu to run the model. 'A;B' means A & B work respectively. 'A,B' means A & B work cooperatively.
                                             "1,3;2,4;5,7");                                                    // It is the best cost/performance (C/P) value setting for running the Qwen 1.8B LLM on my device, due to limitations imposed by the RAM bandwidth.
        ort_runtime_C->SetIntraOpNumThreads(session_options_C, 4);                                              // dynamic_block_base + 1
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.inter_op.allow_spinning",
                                             "1");                                                              // 0 for low power
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.intra_op.allow_spinning",
                                             "1");                                                              // 0 for low power
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.force_spinning_stop",
                                             "0");                                                              // 1 for low power
        ort_runtime_C->SetSessionGraphOptimizationLevel(session_options_C, ORT_ENABLE_ALL);
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "optimization.minimal_build_optimizations",
                                             "");                                                               // Keep empty for full optimization
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "optimization.disable_specified_optimizers",
                                             "NchwcTransformer");                                               // For Arm
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.disable_prepacking",
                                             "0");                                                              // 0 for enable
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "optimization.enable_gelu_approximation",
                                             "1");                                                              // Set 1 is better for this model
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "mlas.enable_gemm_fastmath_arm64_bfloat16",
                                             "1");
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
        std::vector<const char *> option_keys = {};
        std::vector<const char *> option_values = {};
        if (use_xnnpack)
        {
            option_keys.push_back("intra_op_num_threads");
            option_values.push_back("4");
            ort_runtime_C->SetInterOpNumThreads(session_options_C, 4);                                                // Keep the same value as above.
            ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.intra_op.allow_spinning", "0");          // Set to 0.
            ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.dynamic_block_base", "1");               // Set to 1.
            ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.intra_op_thread_affinities", "1,2,3,4"); // Use ',' to split the #core
            ort_runtime_C->SetIntraOpNumThreads(session_options_C, 1);                                                // Set to 1.
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
    std::size_t amount_of_output;
    ort_runtime_C->GetAllocatorWithDefaultOptions(&allocator);
    ort_runtime_C->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    ort_runtime_C->SessionGetInputCount(session_model_C, &amount_of_input);
    input_names_C.resize(amount_of_input);
    input_dims_C.resize(amount_of_input);
    input_types_C.resize(amount_of_input);
    input_tensors_C.resize(amount_of_input);
    for (int i = 0; i < amount_of_input; i++) {
        char *name;
        OrtTypeInfo *typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo *tensor_info;
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
    ort_runtime_C->SessionGetOutputCount(session_model_C, &amount_of_output);
    output_names_C.resize(amount_of_output);
    output_dims_C.resize(amount_of_output);
    output_types_C.resize(amount_of_output);
    output_tensors_C.resize(amount_of_output);
    for (int i = 0; i < amount_of_output; i++) {
        char *name;
        OrtTypeInfo *typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo *tensor_info;
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
    input_dims_C[0][1] = 1;
    std::vector<float> temp(input_dims_C[0][1], 0.f);
    ort_runtime_C->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void*>(temp.data()), sizeof(float),
            input_dims_C[0].data(), input_dims_C[0].size(), input_types_C[0],
            &input_tensors_C[0]);
    ort_runtime_C->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void*>(temp.data()), rgbSize_i8,
            input_dims_C[1].data(), input_dims_C[1].size(), input_types_C[1],
            &input_tensors_C[1]);
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1D(JNIEnv *env,  jobject thiz,
                                                            jobject asset_manager,
                                                            jboolean use_xnnpack,
                                                            jboolean low_memory_mode)
{
    OrtStatus *status;
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
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.dynamic_block_base",                   // One block can contain 1 or more cores, and sharing 1 job.
                                             "3");
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.intra_op_thread_affinities",           // Binding the #cpu to run the model. 'A;B' means A & B work respectively. 'A,B' means A & B work cooperatively.
                                             "1,3;2,4;5,7");                                                    // It is the best cost/performance (C/P) value setting for running the Qwen 1.8B LLM on my device, due to limitations imposed by the RAM bandwidth.
        ort_runtime_D->SetIntraOpNumThreads(session_options_D, 4);                                              // dynamic_block_base + 1
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.inter_op.allow_spinning",
                                             "1");                                                              // 0 for low power
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.intra_op.allow_spinning",
                                             "1");                                                              // 0 for low power
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.force_spinning_stop",
                                             "0");                                                              // 1 for low power
        ort_runtime_D->SetSessionGraphOptimizationLevel(session_options_D, ORT_ENABLE_ALL);
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "optimization.minimal_build_optimizations",
                                             "");                                                               // Keep empty for full optimization
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "optimization.disable_specified_optimizers",
                                             "NchwcTransformer");                                               // For Arm
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.disable_prepacking",
                                             "0");                                                              // 0 for enable
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "optimization.enable_gelu_approximation",
                                             "1");                                                              // Set 1 is better for this model
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "mlas.enable_gemm_fastmath_arm64_bfloat16",
                                             "1");
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
        std::vector<const char *> option_keys = {};
        std::vector<const char *> option_values = {};
        if (use_xnnpack)
        {
            option_keys.push_back("intra_op_num_threads");
            option_values.push_back("4");
            ort_runtime_D->SetInterOpNumThreads(session_options_D, 4);                                                // Keep the same value as above.
            ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.intra_op.allow_spinning", "0");          // Set to 0.
            ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.dynamic_block_base", "1");               // Set to 1.
            ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.intra_op_thread_affinities", "1,2,3,4"); // Use ',' to split the #core
            ort_runtime_D->SetIntraOpNumThreads(session_options_D, 1);                                                // Set to 1.
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
    std::size_t amount_of_output;
    ort_runtime_D->GetAllocatorWithDefaultOptions(&allocator);
    ort_runtime_D->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    ort_runtime_D->SessionGetInputCount(session_model_D, &amount_of_input);
    input_names_D.resize(amount_of_input);
    input_dims_D.resize(amount_of_input);
    input_types_D.resize(amount_of_input);
    input_tensors_D.resize(amount_of_input);
    for (int i = 0; i < amount_of_input; i++) {
        char *name;
        OrtTypeInfo *typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo *tensor_info;
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
    ort_runtime_D->SessionGetOutputCount(session_model_D, &amount_of_output);
    output_names_D.resize(amount_of_output);
    output_dims_D.resize(amount_of_output);
    output_types_D.resize(amount_of_output);
    output_tensors_D.resize(amount_of_output);
    for (int i = 0; i < amount_of_output; i++) {
        char *name;
        OrtTypeInfo *typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo *tensor_info;
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
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1E(JNIEnv *env,  jobject thiz,
                                                            jobject asset_manager,
                                                            jboolean use_xnnpack,
                                                            jboolean low_memory_mode)
{
    OrtStatus *status;
    OrtEnv *ort_env_E;
    OrtSessionOptions *session_options_E;
    {
        std::vector<char> fileBuffer;
        std::vector<char> fileBuffer_external;
        off_t fileSize;
        off_t fileSize_external;
        bool use_storage_path = false;
        if (!low_memory_mode) {
            if (asset_manager != nullptr) {
                AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
                AAsset *asset = AAssetManager_open(mgr, file_name_E.c_str(), AASSET_MODE_BUFFER);
                fileSize = AAsset_getLength(asset);
                fileBuffer.resize(fileSize);
                AAsset_read(asset, fileBuffer.data(), fileSize);
                if (!file_name_E_external.empty()) {
                    // Load external data using AAsset_read. For models with multiple external files, manually load additional files as needed.
                    AAsset* asset_ex = AAssetManager_open(mgr, file_name_E_external.c_str(), AASSET_MODE_BUFFER);
                    fileSize_external = AAsset_getLength(asset_ex);
                    fileBuffer_external.resize(fileSize_external);
                    AAsset_read(asset_ex, fileBuffer_external.data(), fileSize_external);
                }
            } else {
                use_storage_path = true;
                low_memory_mode = true;
            }
        }
        ort_runtime_E = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        ort_runtime_E->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "myapplication", &ort_env_E);
        ort_runtime_E->CreateSessionOptions(&session_options_E);
        ort_runtime_E->CreateRunOptions(&run_options_E);
        ort_runtime_E->AddRunConfigEntry(run_options_E, "memory.enable_memory_arena_shrinkage", "");            // Keep empty for performance; "cpu:0" for low memory usage.
        ort_runtime_E->AddRunConfigEntry(run_options_E, "disable_synchronize_execution_providers", "1");        // 1 for aggressive performance.
        ort_runtime_E->DisableProfiling(session_options_E);
        ort_runtime_E->EnableCpuMemArena(session_options_E);
        ort_runtime_E->EnableMemPattern(session_options_E);
        ort_runtime_E->SetSessionExecutionMode(session_options_E, ORT_SEQUENTIAL);
        ort_runtime_E->SetInterOpNumThreads(session_options_E, 6);
        ort_runtime_E->AddSessionConfigEntry(session_options_E, "session.dynamic_block_base",                   // One block can contain 1 or more cores, and sharing 1 job.
                                             "3");
        ort_runtime_E->AddSessionConfigEntry(session_options_E, "session.intra_op_thread_affinities",           // Binding the #cpu to run the model. 'A;B' means A & B work respectively. 'A,B' means A & B work cooperatively.
                                             "1,3;2,4;5,7");                                                    // It is the best cost/performance (C/P) value setting for running the Qwen 1.8B LLM on my device, due to limitations imposed by the RAM bandwidth.
        ort_runtime_E->SetIntraOpNumThreads(session_options_E, 4);                                              // dynamic_block_base + 1
        ort_runtime_E->AddSessionConfigEntry(session_options_E, "session.inter_op.allow_spinning",
                                             "1");                                                              // 0 for low power
        ort_runtime_E->AddSessionConfigEntry(session_options_E, "session.intra_op.allow_spinning",
                                             "1");                                                              // 0 for low power
        ort_runtime_E->AddSessionConfigEntry(session_options_E, "session.force_spinning_stop",
                                             "0");                                                              // 1 for low power
        ort_runtime_E->SetSessionGraphOptimizationLevel(session_options_E, ORT_ENABLE_ALL);
        ort_runtime_E->AddSessionConfigEntry(session_options_E, "optimization.minimal_build_optimizations",
                                             "");                                                               // Keep empty for full optimization
        ort_runtime_E->AddSessionConfigEntry(session_options_E, "optimization.disable_specified_optimizers",
                                             "NchwcTransformer");                                               // For Arm
        ort_runtime_E->AddSessionConfigEntry(session_options_E, "session.disable_prepacking",
                                             "0");                                                              // 0 for enable
        ort_runtime_E->AddSessionConfigEntry(session_options_E, "optimization.enable_gelu_approximation",
                                             "1");                                                              // Set 1 is better for this model
        ort_runtime_E->AddSessionConfigEntry(session_options_E, "mlas.enable_gemm_fastmath_arm64_bfloat16",
                                             "1");
        ort_runtime_E->AddSessionConfigEntry(session_options_E, "session.disable_aot_function_inlining",
                                             "0");                                                              // 0 for speed
        ort_runtime_E->AddSessionConfigEntry(session_options_E, "session.qdqisint8allowed",
                                             "1");                                                              // 1 for Arm
        ort_runtime_E->AddSessionConfigEntry(session_options_E, "session.enable_quant_qdq_cleanup",
                                             "1");                                                              // 0 for precision, 1 for performance
        ort_runtime_E->AddSessionConfigEntry(session_options_E, "session.disable_double_qdq_remover",
                                             "0");                                                              // 1 for precision, 0 for performance
        ort_runtime_E->AddSessionConfigEntry(session_options_E, "session.disable_quant_qdq",
                                             "0");                                                              // 0 for use Int8
        ort_runtime_E->AddSessionConfigEntry(session_options_E, "session.use_ort_model_bytes_directly",
                                             "1");                                                              // Use this option to lower the peak memory during loading.
        ort_runtime_E->AddSessionConfigEntry(session_options_E, "session.use_ort_model_bytes_for_initializers",
                                             "0");                                                              // If set use_ort_model_bytes_directly=1, use_ort_model_bytes_for_initializers should be 0.
        ort_runtime_E->AddSessionConfigEntry(session_options_E, "session.set_denormal_as_zero",
                                             "1");                                                              // Use 0 instead of NaN or Inf.
        ort_runtime_E->AddSessionConfigEntry(session_options_E, "session.use_env_allocators",
                                             "1");                                                              // Use it to lower memory usage.
        ort_runtime_E->AddSessionConfigEntry(session_options_E, "session.use_device_allocator_for_initializers",
                                             "1");                                                              // Use it to lower memory usage.
        ort_runtime_E->AddSessionConfigEntry(session_options_E, "session.qdq_matmulnbits_accuracy_level",
                                             "0");                                                              // 0:default, 1:FP32, 2:FP16, 3:BF16, 4:INT8
        ort_runtime_E->AddSessionConfigEntry(session_options_E, "ep.dynamic.workload_type",
                                             "Default");                                                        // Default = Performance; Efficient = Save Power
        std::vector<const char *> option_keys = {};
        std::vector<const char *> option_values = {};
        if (use_xnnpack)
        {
            option_keys.push_back("intra_op_num_threads");
            option_values.push_back("4");
            ort_runtime_E->SetInterOpNumThreads(session_options_E, 4);                                                // Keep the same value as above.
            ort_runtime_E->AddSessionConfigEntry(session_options_E, "session.intra_op.allow_spinning", "0");          // Set to 0.
            ort_runtime_E->AddSessionConfigEntry(session_options_E, "session.dynamic_block_base", "1");               // Set to 1.
            ort_runtime_E->AddSessionConfigEntry(session_options_E, "session.intra_op_thread_affinities", "1,2,3,4"); // Use ',' to split the #core
            ort_runtime_E->SetIntraOpNumThreads(session_options_E, 1);                                                // Set to 1.
            ort_runtime_E->SessionOptionsAppendExecutionProvider(session_options_E, "XNNPACK", option_keys.data(), option_values.data(), option_keys.size());
        }
        if (low_memory_mode) {
            if (use_storage_path) {
                status = ort_runtime_E->CreateSession(ort_env_E, (storage_path + file_name_E).c_str(), session_options_E, &session_model_E);
            } else {
                status = ort_runtime_E->CreateSession(ort_env_E, (cache_path + file_name_E).c_str(), session_options_E, &session_model_E);
            }
        } else {
            if (!file_name_E_external.empty()) {
                const char* external_file_names[] = {file_name_E_external.c_str()};                         // Add all external data file names here if your model uses multiple external files.
                const char* external_file_buffers[] = {fileBuffer_external.data()};                         // Read external data into fileBuffers and add them here.
                size_t external_file_sizes[] = {fileBuffer_external.size()};                                // Store the size of each fileBuffer here for multiple external data files.
                ort_runtime_E->AddExternalInitializersFromFilesInMemory(session_options_E, external_file_names, const_cast<char**>(external_file_buffers), external_file_sizes, 1);  // '1' indicates a single external file.
            }
            status = ort_runtime_E->CreateSessionFromArray(ort_env_E, fileBuffer.data(), fileSize, session_options_E, &session_model_E);
        }
    }
    if (status != nullptr) {
        return JNI_FALSE;
    }
    std::size_t amount_of_input;
    std::size_t amount_of_output;
    ort_runtime_E->GetAllocatorWithDefaultOptions(&allocator);
    ort_runtime_E->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    ort_runtime_E->SessionGetInputCount(session_model_E, &amount_of_input);
    input_names_E.resize(amount_of_input);
    input_dims_E.resize(amount_of_input);
    input_types_E.resize(amount_of_input);
    input_tensors_E.resize(amount_of_input);
    for (int i = 0; i < amount_of_input; i++) {
        char *name;
        OrtTypeInfo *typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo *tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_E->SessionGetInputName(session_model_E, i, allocator, &name);
        input_names_E[i] = name;
        ort_runtime_E->SessionGetInputTypeInfo(session_model_E, i, &typeinfo);
        ort_runtime_E->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_E->GetTensorElementType(tensor_info, &type);
        input_types_E[i] = type;
        ort_runtime_E->GetDimensionsCount(tensor_info, &dimensions);
        input_dims_E[i].resize(dimensions);
        ort_runtime_E->GetDimensions(tensor_info, input_dims_E[i].data(), dimensions);
        ort_runtime_E->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) {
            ort_runtime_E->ReleaseTypeInfo(typeinfo);
        }
    }
    ort_runtime_E->SessionGetOutputCount(session_model_E, &amount_of_output);
    output_names_E.resize(amount_of_output);
    output_dims_E.resize(amount_of_output);
    output_types_E.resize(amount_of_output);
    output_tensors_E.resize(amount_of_output);
    for (int i = 0; i < amount_of_output; i++) {
        char *name;
        OrtTypeInfo *typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo *tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_E->SessionGetOutputName(session_model_E, i, allocator, &name);
        output_names_E[i] = name;
        ort_runtime_E->SessionGetOutputTypeInfo(session_model_E, i, &typeinfo);
        ort_runtime_E->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_E->GetTensorElementType(tensor_info, &type);
        output_types_E[i] = type;
        ort_runtime_E->GetDimensionsCount(tensor_info, &dimensions);
        output_dims_E[i].resize(dimensions);
        ort_runtime_E->GetDimensions(tensor_info, output_dims_E[i].data(), dimensions);
        ort_runtime_E->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) {
            ort_runtime_E->ReleaseTypeInfo(typeinfo);
        }
    }
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1F(JNIEnv *env, jobject thiz,
                                                            jobject asset_manager,
                                                            jboolean use_xnnpack,
                                                            jboolean low_memory_mode)
{
    OrtStatus *status;
    OrtEnv *ort_env_F;
    OrtSessionOptions *session_options_F;
    {
        std::vector<char> fileBuffer;
        std::vector<char> fileBuffer_external;
        off_t fileSize;
        off_t fileSize_external;
        bool use_storage_path = false;
        if (!low_memory_mode) {
            if (asset_manager != nullptr) {
                AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
                AAsset *asset = AAssetManager_open(mgr, file_name_F.c_str(), AASSET_MODE_BUFFER);
                fileSize = AAsset_getLength(asset);
                fileBuffer.resize(fileSize);
                AAsset_read(asset, fileBuffer.data(), fileSize);
                if (!file_name_F_external.empty()) {
                    // Load external data using AAsset_read. For models with multiple external files, manually load additional files as needed.
                    AAsset* asset_ex = AAssetManager_open(mgr, file_name_F_external.c_str(), AASSET_MODE_BUFFER);
                    fileSize_external = AAsset_getLength(asset_ex);
                    fileBuffer_external.resize(fileSize_external);
                    AAsset_read(asset_ex, fileBuffer_external.data(), fileSize_external);
                }
            } else {
                use_storage_path = true;
                low_memory_mode = true;
            }
        }
        ort_runtime_F = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        ort_runtime_F->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "myapplication", &ort_env_F);
        ort_runtime_F->CreateSessionOptions(&session_options_F);
        ort_runtime_F->CreateRunOptions(&run_options_F);
        ort_runtime_F->AddRunConfigEntry(run_options_F, "memory.enable_memory_arena_shrinkage", "");            // Keep empty for performance; "cpu:0" for low memory usage.
        ort_runtime_F->AddRunConfigEntry(run_options_F, "disable_synchronize_execution_providers", "1");        // 1 for aggressive performance.
        ort_runtime_F->DisableProfiling(session_options_F);
        ort_runtime_F->EnableCpuMemArena(session_options_F);
        ort_runtime_F->EnableMemPattern(session_options_F);
        ort_runtime_F->SetSessionExecutionMode(session_options_F, ORT_SEQUENTIAL);
        ort_runtime_F->SetInterOpNumThreads(session_options_F, 6);
        ort_runtime_F->AddSessionConfigEntry(session_options_F, "session.dynamic_block_base",                   // One block can contain 1 or more cores, and sharing 1 job.
                                             "3");
        ort_runtime_F->AddSessionConfigEntry(session_options_F, "session.intra_op_thread_affinities",           // Binding the #cpu to run the model. 'A;B' means A & B work respectively. 'A,B' means A & B work cooperatively.
                                             "1,3;2,4;5,7");                                                    // It is the best cost/performance (C/P) value setting for running the Qwen 1.8B LLM on my device, due to limitations imposed by the RAM bandwidth.
        ort_runtime_F->SetIntraOpNumThreads(session_options_F, 4);                                              // dynamic_block_base + 1
        ort_runtime_F->AddSessionConfigEntry(session_options_F, "session.inter_op.allow_spinning",
                                             "1");                                                              // 0 for low power
        ort_runtime_F->AddSessionConfigEntry(session_options_F, "session.intra_op.allow_spinning",
                                             "1");                                                              // 0 for low power
        ort_runtime_F->AddSessionConfigEntry(session_options_F, "session.force_spinning_stop",
                                             "0");                                                              // 1 for low power
        ort_runtime_F->SetSessionGraphOptimizationLevel(session_options_F, ORT_ENABLE_ALL);
        ort_runtime_F->AddSessionConfigEntry(session_options_F, "optimization.minimal_build_optimizations",
                                             "");                                                               // Keep empty for full optimization
        ort_runtime_F->AddSessionConfigEntry(session_options_F, "optimization.disable_specified_optimizers",
                                             "NchwcTransformer");                                               // For Arm
        ort_runtime_F->AddSessionConfigEntry(session_options_F, "session.disable_prepacking",
                                             "0");                                                              // 0 for enable
        ort_runtime_F->AddSessionConfigEntry(session_options_F, "optimization.enable_gelu_approximation",
                                             "1");                                                              // Set 1 is better for this model
        ort_runtime_F->AddSessionConfigEntry(session_options_F, "mlas.enable_gemm_fastmath_arm64_bfloat16",
                                             "1");
        ort_runtime_F->AddSessionConfigEntry(session_options_F, "session.disable_aot_function_inlining",
                                             "0");                                                              // 0 for speed
        ort_runtime_F->AddSessionConfigEntry(session_options_F, "session.qdqisint8allowed",
                                             "1");                                                              // 1 for Arm
        ort_runtime_F->AddSessionConfigEntry(session_options_F, "session.enable_quant_qdq_cleanup",
                                             "1");                                                              // 0 for precision, 1 for performance
        ort_runtime_F->AddSessionConfigEntry(session_options_F, "session.disable_double_qdq_remover",
                                             "0");                                                              // 1 for precision, 0 for performance
        ort_runtime_F->AddSessionConfigEntry(session_options_F, "session.disable_quant_qdq",
                                             "0");                                                              // 0 for use Int8
        ort_runtime_F->AddSessionConfigEntry(session_options_F, "session.use_ort_model_bytes_directly",
                                             "1");                                                              // Use this option to lower the peak memory during loading.
        ort_runtime_F->AddSessionConfigEntry(session_options_F, "session.use_ort_model_bytes_for_initializers",
                                             "0");                                                              // If set use_ort_model_bytes_directly=1, use_ort_model_bytes_for_initializers should be 0.
        ort_runtime_F->AddSessionConfigEntry(session_options_F, "session.set_denormal_as_zero",
                                             "1");                                                              // Use 0 instead of NaN or Inf.
        ort_runtime_F->AddSessionConfigEntry(session_options_F, "session.use_env_allocators",
                                             "1");                                                              // Use it to lower memory usage.
        ort_runtime_F->AddSessionConfigEntry(session_options_F, "session.use_device_allocator_for_initializers",
                                             "1");                                                              // Use it to lower memory usage.
        ort_runtime_F->AddSessionConfigEntry(session_options_F, "session.qdq_matmulnbits_accuracy_level",
                                             "0");                                                              // 0:default, 1:FP32, 2:FP16, 3:BF16, 4:INT8
        ort_runtime_F->AddSessionConfigEntry(session_options_F, "ep.dynamic.workload_type",
                                             "Default");                                                        // Default = Performance; Efficient = Save Power
        std::vector<const char *> option_keys = {};
        std::vector<const char *> option_values = {};
        if (use_xnnpack)
        {
            option_keys.push_back("intra_op_num_threads");
            option_values.push_back("4");
            ort_runtime_F->SetInterOpNumThreads(session_options_F, 4);                                                // Keep the same value as above.
            ort_runtime_F->AddSessionConfigEntry(session_options_F, "session.intra_op.allow_spinning", "0");          // Set to 0.
            ort_runtime_F->AddSessionConfigEntry(session_options_F, "session.dynamic_block_base", "1");               // Set to 1.
            ort_runtime_F->AddSessionConfigEntry(session_options_F, "session.intra_op_thread_affinities", "1,2,3,4"); // Use ',' to split the #core
            ort_runtime_F->SetIntraOpNumThreads(session_options_F, 1);                                                // Set to 1.
            ort_runtime_F->SessionOptionsAppendExecutionProvider(session_options_F, "XNNPACK", option_keys.data(), option_values.data(), option_keys.size());
        }
        if (low_memory_mode) {
            if (use_storage_path) {
                status = ort_runtime_F->CreateSession(ort_env_F, (storage_path + file_name_F).c_str(), session_options_F, &session_model_F);
            } else {
                status = ort_runtime_F->CreateSession(ort_env_F, (cache_path + file_name_F).c_str(), session_options_F, &session_model_F);
            }
        } else {
            if (!file_name_F_external.empty()) {
                const char* external_file_names[] = {file_name_F_external.c_str()};                         // Add all external data file names here if your model uses multiple external files.
                const char* external_file_buffers[] = {fileBuffer_external.data()};                         // Read external data into fileBuffers and add them here.
                size_t external_file_sizes[] = {fileBuffer_external.size()};                                // Store the size of each fileBuffer here for multiple external data files.
                ort_runtime_F->AddExternalInitializersFromFilesInMemory(session_options_F, external_file_names, const_cast<char**>(external_file_buffers), external_file_sizes, 1);  // '1' indicates a single external file.
            }
            status = ort_runtime_F->CreateSessionFromArray(ort_env_F, fileBuffer.data(), fileSize, session_options_F, &session_model_F);
        }
    }
    if (status != nullptr) {
        auto a = ort_runtime_F->GetErrorMessage(status);
        int b = 1;
        return JNI_FALSE;
    }
    std::size_t amount_of_input;
    ort_runtime_F->GetAllocatorWithDefaultOptions(&allocator);
    ort_runtime_F->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    ort_runtime_F->SessionGetInputCount(session_model_F, &amount_of_input);
    input_names_F.resize(amount_of_input);
    input_dims_F.resize(amount_of_input);
    input_types_F.resize(amount_of_input);
    input_tensors_F.resize(amount_of_input);
    for (int i = 0; i < amount_of_input; i++) {
        char *name;
        OrtTypeInfo *typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo *tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_F->SessionGetInputName(session_model_F, i, allocator, &name);
        input_names_F[i] = name;
        ort_runtime_F->SessionGetInputTypeInfo(session_model_F, i, &typeinfo);
        ort_runtime_F->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_F->GetTensorElementType(tensor_info, &type);
        input_types_F[i] = type;
        ort_runtime_F->GetDimensionsCount(tensor_info, &dimensions);
        input_dims_F[i].resize(dimensions);
        ort_runtime_F->GetDimensions(tensor_info, input_dims_F[i].data(), dimensions);
        ort_runtime_F->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) {
            ort_runtime_F->ReleaseTypeInfo(typeinfo);
        }
    }
    ort_runtime_F->SessionGetOutputCount(session_model_F, &amount_of_output_F);
    output_names_F.resize(amount_of_output_F);
    output_dims_F.resize(amount_of_output_F);
    output_types_F.resize(amount_of_output_F);
    for (auto & i : output_tensors_F) {
        i.resize(amount_of_output_F);
    }
    for (int i = 0; i < amount_of_output_F; i++) {
        char *name;
        OrtTypeInfo *typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo *tensor_info;
        ONNXTensorElementDataType type;
        ort_runtime_F->SessionGetOutputName(session_model_F, i, allocator, &name);
        output_names_F[i] = name;
        ort_runtime_F->SessionGetOutputTypeInfo(session_model_F, i, &typeinfo);
        ort_runtime_F->CastTypeInfoToTensorInfo(typeinfo, &tensor_info);
        ort_runtime_F->GetTensorElementType(tensor_info, &type);
        output_types_F[i] = type;
        ort_runtime_F->GetDimensionsCount(tensor_info, &dimensions);
        output_dims_F[i].resize(dimensions);
        ort_runtime_F->GetDimensions(tensor_info, output_dims_F[i].data(), dimensions);
        ort_runtime_F->GetTensorShapeElementCount(tensor_info, &tensor_size);
        if (typeinfo) {
            ort_runtime_F->ReleaseTypeInfo(typeinfo);
        }
    }
    std::vector<float> temp(1, 0.f);
    int indices = num_keys_values_plus;  // hidden_states
    input_dims_F[indices][1] = 0;
    ort_runtime_F->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void*>(temp.data()), 0,
            input_dims_F[indices].data(), input_dims_F[indices].size(), input_types_F[indices],
            &input_tensors_F[indices]);
    indices += 1;
    rotary_indices[0] = indices;
    for (int i = 1; i < rotary_outputs_len; i++) {
        rotary_indices[i] = rotary_indices[0] + i;
    }
    input_dims_F[indices][1] = 0;        // rotary_cos_q
    ort_runtime_F->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void*>(temp.data()), 0,
            input_dims_F[indices].data(), input_dims_F[indices].size(), input_types_F[indices],
            &input_tensors_F[indices]);
    indices += 1;
    input_dims_F[indices][1] = 0;       // rotary_sin_q
    ort_runtime_F->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void*>(temp.data()), 0,
            input_dims_F[indices].data(), input_dims_F[indices].size(), input_types_F[indices],
            &input_tensors_F[indices]);
    indices += 1;
    input_dims_F[indices][2] = 0;       // rotary_cos_k
    ort_runtime_F->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void*>(temp.data()), 0,
            input_dims_F[indices].data(), input_dims_F[indices].size(), input_types_F[indices],
            &input_tensors_F[indices]);
    indices += 1;
    input_dims_F[indices][2] = 0;      // rotary_sin_k
    ort_runtime_F->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void*>(temp.data()), 0,
            input_dims_F[indices].data(), input_dims_F[indices].size(), input_types_F[indices],
            &input_tensors_F[indices]);
    indices += 1;
    ort_runtime_F->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void*>(&ids_len), sizeof(int64_t),
            input_dims_F[indices].data(), input_dims_F[indices].size(), input_types_F[indices],
            &input_tensors_F[indices]);
    indices += 1;
    ort_runtime_F->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void*>(&attention_mask), sizeof(int8_t),
            input_dims_F[indices].data(), input_dims_F[indices].size(), input_types_F[indices],
            &input_tensors_F[indices]);
    for (int i = 0; i < num_layers; i++) {
        input_dims_F[i][3] = 0;
        ort_runtime_F->CreateTensorWithDataAsOrtValue(
                memory_info,
                reinterpret_cast<void*>(past_key_values_init.data()), 0,
                input_dims_F[i].data(), input_dims_F[i].size(), input_types_F[i],
                &input_tensors_kv_init_F[i]);
    }
    for (int i = num_layers; i < num_keys_values; i++) {
        input_dims_F[i][2] = 0;
        ort_runtime_F->CreateTensorWithDataAsOrtValue(
                memory_info,
                reinterpret_cast<void*>(past_key_values_init.data()), 0,
                input_dims_F[i].data(), input_dims_F[i].size(), input_types_F[i],
                &input_tensors_kv_init_F[i]);
    }
    return JNI_TRUE;
}
