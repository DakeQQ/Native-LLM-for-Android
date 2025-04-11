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
    save_index = 0;
    history_len = 0;
    response_count = 0;
    num_ids_per_chat[0] = 0;
    accumulate_num_ids[0] = 0;
    attention_mask = 1;
    token_id = end_id_0;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Pre_1Process(JNIEnv *env, jobject clazz)
{
    if (use_deepseek) {
        start_id = 151646;
        end_id_1 = end_id_0;
        tokenizer = MNN::Transformer::Tokenizer::createTokenizer(cache_path + "vocab_DeepSeek_Qwen.txt");
    } else {
        tokenizer = MNN::Transformer::Tokenizer::createTokenizer(cache_path + "vocab_Qwen.txt");
    }
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_Run_1LLM(JNIEnv *env, jclass clazz, jstring jquery,
                                                     jboolean add_prompt,
                                                     jboolean clear) {
    if (add_prompt) {
        chatting = true;
        if (clear) {
            clear_history();
        } else {
            response_count = 0;
        }
        const char *query = env->GetStringUTFChars(jquery, nullptr);
        std::vector<int> get_ids = tokenizer->encode(query);
        if (use_deepseek) {
            get_ids.insert(get_ids.begin(), {151646, 151644});             // DeepSeek-Distill-Qwen Chat prompt head
            get_ids.insert(get_ids.end(), {151645});                       // DeepSeek-Distill-Qwen Chat prompt tail
        } else {
            get_ids.insert(get_ids.begin(), {151644, 872, 198});                // Qwen Chat prompt head
            get_ids.insert(get_ids.end(), {151645, 198, 151644, 77091, 198});   // Qwen Chat prompt tail
        }
        ids_len = static_cast<int> (get_ids.size());
        num_ids_per_chat[save_index] = ids_len;
        if (save_index > 0) {
            accumulate_num_ids[save_index] = num_ids_per_chat[save_index] + accumulate_num_ids[save_index - 1];
            if (accumulate_num_ids[save_index] > next_chat_buffer) {
                bool over_inputs = true;
                for (int i = 0; i < save_index; i++) {
                    if (accumulate_num_ids[save_index] - accumulate_num_ids[i] <= next_chat_buffer) {
                        std::move(input_ids.begin() + accumulate_num_ids[i], input_ids.end(), input_ids.begin());
                        int k = i + 1;
                        for (int j = i; j <= save_index; j++) {
                            accumulate_num_ids[j] -= accumulate_num_ids[i];
                        }
                        ids_len = accumulate_num_ids[save_index];
                        std::move(get_ids.begin(), get_ids.end(), input_ids.begin() + accumulate_num_ids[save_index - 1]);
                        std::move(num_ids_per_chat.begin() + k, num_ids_per_chat.end(), num_ids_per_chat.begin());
                        std::move(accumulate_num_ids.begin() + k, accumulate_num_ids.end(), accumulate_num_ids.begin());
                        save_index -= k;
                        over_inputs = false;
                        break;
                    }
                }
                if (over_inputs) {
                    clear_history();
                    return env->NewStringUTF("Over_Inputs");
                }
            } else {
                std::move(get_ids.begin(), get_ids.end(), input_ids.begin() + accumulate_num_ids[save_index - 1]);
                ids_len = accumulate_num_ids[save_index];
            }
        } else {
            if (num_ids_per_chat[0] >= max_seq_len) {
                clear_history();
                return env->NewStringUTF("Over_Inputs");
            } else {
                accumulate_num_ids[0] = num_ids_per_chat[0];
                std::move(get_ids.begin(), get_ids.end(), input_ids.begin());
            }
        }
        input_dims_A[num_keys_values][1] = ids_len;
        ort_runtime_A->CreateTensorWithDataAsOrtValue(
                memory_info,
                reinterpret_cast<void*>(input_ids.data()), input_ids_buffer_size[ids_len],
                input_dims_A[num_keys_values].data(), input_dims_A[num_keys_values].size(), input_types_A[num_keys_values],
                &input_tensors_A[num_keys_values]);
        for (int i = 0; i < num_keys_values; i++) {
            input_tensors_A[i] = input_tensors_kv_init_A[i];
        }
        if (output_tensors_A[0][0] != nullptr) {
            for (int i = 0; i < amount_of_output; i++) {
                ort_runtime_A->ReleaseValue(output_tensors_A[0][i]);
                output_tensors_A[0][i] = nullptr;
            }
        }
        if (output_tensors_A[1][0] != nullptr) {
            for (int i = 0; i < amount_of_output; i++) {
                ort_runtime_A->ReleaseValue(output_tensors_A[1][i]);
                output_tensors_A[1][i] = nullptr;
            }
        }
    }
    if (chatting) {  // Java multithreading may not stop immediately. Therefore, use a switch to prevent incorrect saves.
        ort_runtime_A->Run(session_model_A, run_options_A, input_names_A.data(),
                           (const OrtValue *const *)input_tensors_A.data(),
                           input_tensors_A.size(), output_names_A.data(), output_names_A.size(),
                           output_tensors_A[buffer_index].data());
        void *max_logit_id;
        ort_runtime_A->GetTensorMutableData(output_tensors_A[buffer_index][num_keys_values], &max_logit_id);
        token_id = reinterpret_cast<int*>(max_logit_id)[0];
        if ((token_id != end_id_0) && (token_id != end_id_1) && (response_count < single_chat_limit) && (history_len < max_seq_len)) {
            for (int i = 0; i < amount_of_output; i++) {
                input_tensors_A[i] = output_tensors_A[buffer_index][i];
            }
            buffer_index = (buffer_index != 0) ? 0 : 1;
            if (add_prompt) {
                attention_mask = 0;
                history_len += ids_len;
            } else {
                for (int i = 0; i < amount_of_output; i++) {
                    ort_runtime_A->ReleaseValue(output_tensors_A[buffer_index][i]);
                    output_tensors_A[buffer_index][i] = nullptr;
                }
                history_len += 1;
            }
            save_max_logit_position[response_count] = token_id;
            response_count += 1;
            return env->NewStringUTF(get_output_words(token_id).c_str());
        } else {
            chatting = false;
            save_max_logit_position[response_count] = end_id_1;
            response_count += 1;
            num_ids_per_chat[save_index] += response_count;
            attention_mask = 1;
            history_len = 0;
            if (save_index > 0) {
                accumulate_num_ids[save_index] = num_ids_per_chat[save_index] + accumulate_num_ids[save_index - 1];
                if (accumulate_num_ids[save_index] > next_chat_buffer) {
                    for (int i = 0; i < save_index; i++) {
                        if (accumulate_num_ids[save_index] - accumulate_num_ids[i] <= next_chat_buffer) {
                            std::move(input_ids.begin() + accumulate_num_ids[i], input_ids.end(), input_ids.begin());
                            int k = i + 1;
                            for (int j = i; j <= save_index; j++) {
                                accumulate_num_ids[j] -= accumulate_num_ids[i];
                            }
                            std::move(save_max_logit_position.begin(), save_max_logit_position.begin() + response_count, input_ids.begin() + accumulate_num_ids[save_index] - response_count);
                            std::move(num_ids_per_chat.begin() + k, num_ids_per_chat.end(), num_ids_per_chat.begin());
                            std::move(accumulate_num_ids.begin() + k, accumulate_num_ids.end(), accumulate_num_ids.begin());
                            save_index -= i;
                            return env->NewStringUTF("END");
                        }
                    }
                    clear_history();
                } else {
                    std::move(save_max_logit_position.begin(), save_max_logit_position.begin() + response_count, input_ids.begin() + accumulate_num_ids[save_index] - response_count);
                    save_index += 1;
                }
            } else {
                if (num_ids_per_chat[0] > next_chat_buffer) {
                    clear_history();
                } else {
                    std::move(save_max_logit_position.begin(), save_max_logit_position.begin() + response_count, input_ids.begin() + accumulate_num_ids[0]);
                    accumulate_num_ids[0] = num_ids_per_chat[0];
                    save_index += 1;
                }
            }
            return env->NewStringUTF("END");
        }
    }
    return env->NewStringUTF("END");
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1A(JNIEnv *env, jobject clazz,
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
        ort_runtime_A->SetInterOpNumThreads(session_options_A, 4);
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.dynamic_block_base",                   // One block can contain 1 or more cores, and sharing 1 job.
                                             "2");
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.intra_op_thread_affinities",           // Binding the #cpu to run the model. 'A;B' means A & B work respectively. 'A,B' means A & B work cooperatively.
                                             "1,3;2,4");                                                        // It is the best cost/performance (C/P) value setting for running the Qwen 1.8B LLM on my device, due to limitations imposed by the RAM bandwidth.
        ort_runtime_A->SetIntraOpNumThreads(session_options_A, 3);                                              // dynamic_block_base + 1
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
    for (auto & i : output_tensors_A) {
        i.resize(amount_of_output);
    }
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
    ort_runtime_A->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(&attention_mask), sizeof(int8_t),
            input_dims_A[last_indices].data(), input_dims_A[last_indices].size(), input_types_A[last_indices],
            &input_tensors_A[last_indices]);

    for (int i = 0; i < num_layers; i++) {
        input_dims_A[i][2] = 0;
        ort_runtime_A->CreateTensorWithDataAsOrtValue(
                memory_info,
                reinterpret_cast<void *>(past_key_values_init.data()), 0,
                input_dims_A[i].data(), input_dims_A[i].size(), input_types_A[i],
                &input_tensors_kv_init_A[i]);
    }
    for (int i = num_layers; i < num_keys_values; i++) {
        input_dims_A[i][1] = 0;
        ort_runtime_A->CreateTensorWithDataAsOrtValue(
                memory_info,
                reinterpret_cast<void *>(past_key_values_init.data()), 0,
                input_dims_A[i].data(), input_dims_A[i].size(), input_types_A[i],
                &input_tensors_kv_init_A[i]);
    }
    input_ids_buffer_size[1] = sizeof(int);
    for (int i = 2; i < max_seq_len; i++) {
        input_ids_buffer_size[i] = input_ids_buffer_size[i - 1] + sizeof(int);
    }
    return JNI_TRUE;
}
