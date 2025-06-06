#include "project.h"

inline static void correctUtfBytes(char* bytes) {
    char three = 0;
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

inline static std::string get_output_words(const int &id)
{
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
//    attention_mask = Ort::Float16_t(-65504.f);
    attention_mask = -65504.f;
    accumulate_num_ids[0] = 0;
    num_ids_per_chat[0] = 0;
    std::fill(input_ids.begin(), input_ids.end(), 0);
    response_count = 0;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Pre_1Process(JNIEnv *env, jobject clazz) {
    std::unique_ptr<Tokenizer> temp = std::make_unique<Sentencepiece>();
    tokenizer = temp->createTokenizer(vocab_file);
    return JNI_TRUE;
}


extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_Run_1LLM(JNIEnv *env, jclass clazz, jstring jquery,
                                                     jboolean add_prompt,
                                                     jboolean clear) {
    if (add_prompt) {
        if (clear) {
            clear_history();
        }
        response_count = 0;
        const char *query = env->GetStringUTFChars(jquery, nullptr);
        std::vector<int32_t> get_ids = tokenizer->encode(query);
        get_ids.insert(get_ids.begin(), {1, 1786, 4194, 95388});  // Chat prompt head
        get_ids.insert(get_ids.end(), {95396, 10850, 95388});  // Chat prompt tail
        ids_len = static_cast<int64_t> (get_ids.size());
        num_ids_per_chat[save_index] = static_cast<int> (ids_len);
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
                std::move(get_ids.begin(), get_ids.end(),input_ids.begin() + accumulate_num_ids[save_index - 1]);
                ids_len = accumulate_num_ids[save_index];
            }
        } else {
            if (num_ids_per_chat[0] >= max_token_history) {
                clear_history();
                return env->NewStringUTF("Over_Inputs");
            } else {
                accumulate_num_ids[0] = num_ids_per_chat[0];
                std::move(get_ids.begin(), get_ids.end(),input_ids.begin());
            }
        }
    }
    ort_runtime_A->Run(session_model_A, nullptr, input_names_A.data(),
                       (const OrtValue *const *)input_tensors_A.data(),
                       input_tensors_A.size(), output_names_A.data(), output_names_A.size(),
                       output_tensors_A.data());
    input_tensors_B[0] = output_tensors_A[0];
    input_tensors_A[2] = output_tensors_A[1];
    input_tensors_A[3] = output_tensors_A[2];
    ort_runtime_B->Run(session_model_B, nullptr, input_names_B.data(),
                       (const OrtValue *const *) input_tensors_B.data(),
                       input_tensors_B.size(), output_names_B.data(), output_names_B.size(),
                       output_tensors_B.data());
    void *max_logit_id;
    ort_runtime_B->GetTensorMutableData(output_tensors_B[0], &max_logit_id);
    input_ids[0] = reinterpret_cast<int32_t*>(max_logit_id)[0];
    if ((input_ids[0] != end_id_0) && (response_count < single_chat_limit) && (history_len < max_token_history)) {
        input_tensors_B[2] = output_tensors_B[1];
        input_tensors_B[3] = output_tensors_B[2];
        history_len += ids_len;
        if (add_prompt) {
            ids_len = 1;
//            attention_mask = Ort::Float16_t(0.f);
            attention_mask = 0.f;
        }
        save_max_logit_position[response_count] = input_ids[0];
        response_count += 1;
        return env->NewStringUTF(get_output_words(input_ids[0]).c_str());
    } else {
        if (history_len != 0) {
            save_max_logit_position[response_count] = end_id_0;
            response_count += 1;
            num_ids_per_chat[save_index] += response_count;
    //        attention_mask = Ort::Float16_t(-65504.f);
            attention_mask = -65504.f;
            history_len = 0;
            input_ids[0] = start_id;
            if (save_index > 0) {
                accumulate_num_ids[save_index] = num_ids_per_chat[save_index] + accumulate_num_ids[save_index - 1];
                if (accumulate_num_ids[save_index] > next_chat_buffer) {
                    for (int i = 0; i < save_index; i++) {
                        if (accumulate_num_ids[save_index] - accumulate_num_ids[i] <= next_chat_buffer) {
                            std::move(input_ids.begin() + accumulate_num_ids[i],input_ids.end(),input_ids.begin());
                            int k = i + 1;
                            for (int j = i; j <= save_index; j++) {
                                accumulate_num_ids[j] -= accumulate_num_ids[i];
                            }
                            std::move(save_max_logit_position.begin(),save_max_logit_position.begin() + response_count,input_ids.begin() + accumulate_num_ids[save_index] - response_count);
                            std::move(num_ids_per_chat.begin() + k,num_ids_per_chat.end(),num_ids_per_chat.begin());
                            std::move(accumulate_num_ids.begin() + k,accumulate_num_ids.end(),accumulate_num_ids.begin());
                            save_index -= i;
                            return env->NewStringUTF("END");
                        }
                    }
                    clear_history();
                } else {
                    std::move(save_max_logit_position.begin(),save_max_logit_position.begin() + response_count,input_ids.begin() + accumulate_num_ids[save_index] - response_count);
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
        }
        return env->NewStringUTF("END");
    }
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1A(JNIEnv *env, jobject clazz,
                                                            jobject asset_manager,
                                                            jboolean use_xnnpack) {
    OrtStatus *status;
    OrtAllocator *allocator;
    OrtEnv *ort_env_A;
    OrtSessionOptions *session_options_A;
    {
        std::vector<char> fileBuffer;
        off_t fileSize;
        if (asset_manager != nullptr) {
            AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
            AAsset* asset = AAssetManager_open(mgr,file_name_A.c_str(), AASSET_MODE_BUFFER);
            fileSize = AAsset_getLength(asset);
            fileBuffer.resize(fileSize);
            AAsset_read(asset,fileBuffer.data(),fileSize);
        } else {
            std::ifstream model_file(storage_path + file_name_A, std::ios::binary | std::ios::ate);
            if (!model_file.is_open()) {
                return JNI_FALSE;
            }
            fileSize = model_file.tellg();
            model_file.seekg(0, std::ios::beg);
            fileBuffer.resize(fileSize);
            if (!model_file.read(fileBuffer.data(), fileSize)) {
                return JNI_FALSE;
            }
            model_file.close();
        }
        ort_runtime_A = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        ort_runtime_A->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "myapplication", &ort_env_A);
        ort_runtime_A->CreateSessionOptions(&session_options_A);
        ort_runtime_A->CreateRunOptions(&run_options_A);
        ort_runtime_A->AddRunConfigEntry(run_options_A, "memory.enable_memory_arena_shrinkage", "");  // Keep empty for performance; "cpu:0" for low memory usage.
        ort_runtime_A->AddRunConfigEntry(run_options_A, "disable_synchronize_execution_providers", "1");  // 1 for aggressive performance.
        ort_runtime_A->DisableProfiling(session_options_A);
        ort_runtime_A->EnableCpuMemArena(session_options_A);
        ort_runtime_A->EnableMemPattern(session_options_A);
        ort_runtime_A->SetSessionExecutionMode(session_options_A, ORT_SEQUENTIAL);
        ort_runtime_A->SetInterOpNumThreads(session_options_A, 2);
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.dynamic_block_base", "2");  // One block can contain 1 or more cores, and sharing 1 job.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, // Binding the #cpu to run the model. 'A;B;' means A & B work respectively. 'A,B' means A & B work cooperatively.
                                             "session.intra_op_thread_affinities",
                                             "1;2");  // It is the best cost/performance (C/P) value setting for running the MiniCPM 2B LLM on the Kirin 990 5G, due to limitations imposed by the RAM bandwidth.
        ort_runtime_A->SetIntraOpNumThreads(session_options_A, 3); // dynamic_block_base + 1
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.inter_op.allow_spinning",
                                             "1");  // 0 for low power
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.intra_op.allow_spinning",
                                             "1");  // 0 for low power
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.force_spinning_stop",
                                             "0");  // 1 for low power
        ort_runtime_A->SetSessionGraphOptimizationLevel(session_options_A, ORT_ENABLE_ALL);
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "optimization.minimal_build_optimizations",
                                             "");   // Keep empty for full optimization
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "optimization.disable_specified_optimizers",
                                             "NchwcTransformer");   // For Arm
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.disable_prepacking",
                                             "0");  // 0 for enable
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "optimization.enable_gelu_approximation",
                                             "1");  // Set 1 is better for this model
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "mlas.enable_gemm_fastmath_arm64_bfloat16",
                                             "1");  //
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.disable_aot_function_inlining",
                                             "0");  // 0 for speed
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.qdqisint8allowed",
                                             "1");  // 1 for Arm
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.enable_quant_qdq_cleanup",
                                             "1");  // 0 for precision, 1 for performance
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.disable_double_qdq_remover",
                                             "0");  // 1 for precision, 0 for performance
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.disable_quant_qdq",
                                             "0");  // 0 for use Int8
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.use_ort_model_bytes_directly",
                                             "1");  // Use this option to lower the peak memory during loading.
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.use_ort_model_bytes_for_initializers",
                                             "0");  // If set use_ort_model_bytes_directly=1, use_ort_model_bytes_for_initializers should be 0.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.set_denormal_as_zero",
                                             "1");  // // Use 0 instead of NaN or Inf.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.use_env_allocators",
                                             "1");  // Use it to lower memory usage.
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.use_device_allocator_for_initializers",
                                             "1");  // Use it to lower memory usage.
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.qdq_matmulnbits_accuracy_level",
                                             "4");  // 0:default, 1:FP32, 2:FP16, 3:BF16, 4:INT8
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "ep.dynamic.workload_type",
                                             "Default");  // Default = Performance; Efficient = Save Power
        std::vector<const char*> option_keys = {};
        std::vector<const char*> option_values = {};
        if (use_xnnpack) {
            option_keys.push_back("intra_op_num_threads");
            option_values.push_back("4");
            ort_runtime_A->SetInterOpNumThreads(session_options_A, 4);  // Keep the same value as above.
            ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.intra_op.allow_spinning", "0");  // Set to 0.
            ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.dynamic_block_base", "1");  // Set to 1.
            ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.intra_op_thread_affinities", "1,2,3,4");  // Use ',' to split the #core
            ort_runtime_A->SetIntraOpNumThreads(session_options_A, 1);  // Set to 1.
            ort_runtime_A->SessionOptionsAppendExecutionProvider(session_options_A, "XNNPACK", option_keys.data(), option_values.data(), option_keys.size());
        }
        status = ort_runtime_A->CreateSessionFromArray(ort_env_A, fileBuffer.data(), fileSize, session_options_A, &session_model_A);
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
    for (size_t i = 0; i < amount_of_input; i++) {
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
        if (typeinfo) ort_runtime_A->ReleaseTypeInfo(typeinfo);
    }
    std::size_t amount_of_output;
    ort_runtime_A->SessionGetOutputCount(session_model_A, &amount_of_output);
    output_names_A.resize(amount_of_output);
    output_dims_A.resize(amount_of_output);
    output_types_A.resize(amount_of_output);
    output_tensors_A.resize(amount_of_output);
    for (size_t i = 0; i < amount_of_output; i++) {
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
        if (typeinfo) ort_runtime_A->ReleaseTypeInfo(typeinfo);
    }
    OrtMemoryInfo *memory_info;
    ort_runtime_A->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    ort_runtime_A->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(input_ids.data()), static_cast<size_t> (max_token_history) * sizeof(int32_t),
            input_dims_A[0].data(), input_dims_A[0].size(), input_types_A[0],
            &input_tensors_A[0]);
    ort_runtime_A->CreateTensorWithDataAsOrtValue(
            memory_info,
//        reinterpret_cast<void *>(&attention_mask), sizeof(Ort::Float16_t),
            reinterpret_cast<void *>(&attention_mask), sizeof(float),
            input_dims_A[1].data(), input_dims_A[1].size(), input_types_A[1],
            &input_tensors_A[1]);
    std::vector<Ort::Float16_t> past_key_values(past_key_value_size, Ort::Float16_t(0.f));
    size_t buffer_size = past_key_value_size * sizeof(Ort::Float16_t);
    ort_runtime_A->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(past_key_values.data()), buffer_size,
            input_dims_A[2].data(), input_dims_A[2].size(), input_types_A[2],
            &input_tensors_A[2]);
    ort_runtime_A->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(past_key_values.data()), buffer_size,
            input_dims_A[3].data(), input_dims_A[3].size(), input_types_A[3],
            &input_tensors_A[3]);
    ort_runtime_A->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(&history_len), sizeof(int64_t),
            input_dims_A[4].data(), input_dims_A[4].size(), input_types_A[4],
            &input_tensors_A[4]);
    ort_runtime_A->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(&ids_len), sizeof(int64_t),
            input_dims_A[5].data(), input_dims_A[5].size(), input_types_A[5],
            &input_tensors_A[5]);
    ort_runtime_A->ReleaseMemoryInfo(memory_info);
    return JNI_TRUE;
}


extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1B(JNIEnv *env, jobject clazz,
                                                            jobject asset_manager,
                                                            jboolean use_xnnpack) {
    OrtStatus *status;
    OrtAllocator *allocator;
    OrtEnv *ort_env_B;
    OrtSessionOptions *session_options_B;
    {
        std::vector<char> fileBuffer;
        off_t fileSize;
        if (asset_manager != nullptr) {
            AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
            AAsset* asset = AAssetManager_open(mgr,file_name_B.c_str(), AASSET_MODE_BUFFER);
            fileSize = AAsset_getLength(asset);
            fileBuffer.resize(fileSize);
            AAsset_read(asset,fileBuffer.data(),fileSize);
        } else {
            std::ifstream model_file(storage_path + file_name_B, std::ios::binary | std::ios::ate);
            if (!model_file.is_open()) {
                return JNI_FALSE;
            }
            fileSize = model_file.tellg();
            model_file.seekg(0, std::ios::beg);
            fileBuffer.resize(fileSize);
            if (!model_file.read(fileBuffer.data(), fileSize)) {
                return JNI_FALSE;
            }
            model_file.close();
        }
        ort_runtime_B = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        ort_runtime_B->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "myapplication", &ort_env_B);
        ort_runtime_B->CreateSessionOptions(&session_options_B);
        ort_runtime_B->CreateRunOptions(&run_options_B);
        ort_runtime_B->AddRunConfigEntry(run_options_B, "memory.enable_memory_arena_shrinkage", "");  // Keep empty for performance; "cpu:0" for low memory usage.
        ort_runtime_B->AddRunConfigEntry(run_options_B, "disable_synchronize_execution_providers", "1");  // 1 for aggressive performance.
        ort_runtime_B->DisableProfiling(session_options_B);
        ort_runtime_B->EnableCpuMemArena(session_options_B);
        ort_runtime_B->EnableMemPattern(session_options_B);
        ort_runtime_B->SetSessionExecutionMode(session_options_B, ORT_SEQUENTIAL);
        ort_runtime_B->SetInterOpNumThreads(session_options_B, 2);
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.dynamic_block_base", "2");
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.intra_op_thread_affinities",
                                             "1;2");
        ort_runtime_B->SetIntraOpNumThreads(session_options_B, 3); // dynamic_block_base + 1
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.inter_op.allow_spinning",
                                             "1");  // 0 for low power
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.intra_op.allow_spinning",
                                             "1");  // 0 for low power
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.force_spinning_stop",
                                             "0");  // 1 for low power
        ort_runtime_B->SetSessionGraphOptimizationLevel(session_options_B, ORT_ENABLE_ALL);
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "optimization.minimal_build_optimizations",
                                             "");   // Keep empty for full optimization
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "optimization.disable_specified_optimizers",
                                             "NchwcTransformer");   // For Arm
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.disable_prepacking",
                                             "0");  // 0 for enable
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "optimization.enable_gelu_approximation",
                                             "1");  // Set 1 is better for this model
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "mlas.enable_gemm_fastmath_arm64_bfloat16",
                                             "1");  //
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.disable_aot_function_inlining",
                                             "0");  // 0 for speed
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.qdqisint8allowed",
                                             "1");  // 1 for Arm
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.enable_quant_qdq_cleanup",
                                             "1");  // 0 for precision, 1 for performance
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.disable_double_qdq_remover",
                                             "0");  // 1 for precision, 0 for performance
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.disable_quant_qdq",
                                             "0");  // 0 for use Int8
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.use_ort_model_bytes_directly",
                                             "1");  // Use this option to lower the peak memory during loading.
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.use_ort_model_bytes_for_initializers",
                                             "0");  // If set use_ort_model_bytes_directly=1, use_ort_model_bytes_for_initializers should be 0.
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.set_denormal_as_zero",
                                             "1");  // // Use 0 instead of NaN or Inf.
        ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.use_env_allocators",
                                             "1");  // Use it to lower memory usage.
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.use_device_allocator_for_initializers",
                                             "1");  // Use it to lower memory usage.
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.qdq_matmulnbits_accuracy_level",
                                             "4");  // 0:default, 1:FP32, 2:FP16, 3:BF16, 4:INT8
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "ep.dynamic.workload_type",
                                             "Default");  // Default = Performance; Efficient = Save Power
        std::vector<const char*> option_keys = {};
        std::vector<const char*> option_values = {};
        if (use_xnnpack) {
            option_keys.push_back("intra_op_num_threads");
            option_values.push_back("4");
            ort_runtime_B->SetInterOpNumThreads(session_options_B, 4);  // Keep the same value as above.
            ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.intra_op.allow_spinning", "0");  // Set to 0.
            ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.dynamic_block_base", "1");  // Set to 1.
            ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.intra_op_thread_affinities", "1,2,3,4");  // Use ',' to split the #core
            ort_runtime_B->SetIntraOpNumThreads(session_options_B, 1);  // Set to 1.
            ort_runtime_B->SessionOptionsAppendExecutionProvider(session_options_B, "XNNPACK", option_keys.data(), option_values.data(), option_keys.size());
        }
        status = ort_runtime_B->CreateSessionFromArray(ort_env_B, fileBuffer.data(), fileSize, session_options_B, &session_model_B);
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
    for (size_t i = 0; i < amount_of_input; i++) {
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
        if (typeinfo) ort_runtime_B->ReleaseTypeInfo(typeinfo);
    }
    std::size_t amount_of_output;
    ort_runtime_B->SessionGetOutputCount(session_model_B, &amount_of_output);
    output_names_B.resize(amount_of_output);
    output_dims_B.resize(amount_of_output);
    output_types_B.resize(amount_of_output);
    output_tensors_B.resize(amount_of_output);
    for (size_t i = 0; i < amount_of_output; i++) {
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
        if (typeinfo) ort_runtime_B->ReleaseTypeInfo(typeinfo);
    }
    OrtMemoryInfo *memory_info;
    ort_runtime_B->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    std::vector<Ort::Float16_t> hidden_state_B(max_token_history * hidden_size, Ort::Float16_t(0.f));
    ort_runtime_B->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void*>(hidden_state_B.data()), hidden_state_B.size() * sizeof(Ort::Float16_t),
            input_dims_B[0].data(), input_dims_B[0].size(), input_types_B[0],
            &input_tensors_B[0]);
    ort_runtime_B->CreateTensorWithDataAsOrtValue(
            memory_info,
//        reinterpret_cast<void *>(&attention_mask), sizeof(Ort::Float16_t),
            reinterpret_cast<void *>(&attention_mask), sizeof(float),
            input_dims_B[1].data(), input_dims_B[1].size(), input_types_B[1],
            &input_tensors_B[1]);
    std::vector<Ort::Float16_t> past_key_values(past_key_value_size, Ort::Float16_t(0.f));
    size_t buffer_size = past_key_value_size * sizeof(Ort::Float16_t);
    ort_runtime_B->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(past_key_values.data()), buffer_size,
            input_dims_B[2].data(), input_dims_B[2].size(), input_types_B[2],
            &input_tensors_B[2]);
    ort_runtime_B->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(past_key_values.data()), buffer_size,
            input_dims_B[3].data(), input_dims_B[3].size(), input_types_B[3],
            &input_tensors_B[3]);
    ort_runtime_B->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(&history_len), sizeof(int64_t),
            input_dims_B[4].data(), input_dims_B[4].size(), input_types_B[4],
            &input_tensors_B[4]);
    ort_runtime_B->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(&ids_len), sizeof(int64_t),
            input_dims_B[5].data(), input_dims_B[5].size(), input_types_B[5],
            &input_tensors_B[5]);
    ort_runtime_B->ReleaseMemoryInfo(memory_info);
    return JNI_TRUE;
}
