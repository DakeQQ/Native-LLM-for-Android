#include "project.h"

inline static std::string get_output_words(const int &id)
{
    std::string words = tokenizer->decode(id);
    if (words.length() == 6 && words[0] == '<' && words[words.length() - 1] == '>' && words[1] == '0' && words[2] == 'x')
    {
        words = static_cast<char>(std::stoi(words.substr(3, 2), nullptr, 16));
    }
    return words;
}

inline static void clear_history()
{
    save_index = 0;
    history_len = 0;
    attention_mask = -65504.f;
//    attention_mask = Ort::Float16_t(-65504.f);
    accumulate_num_ids[0] = 0;
    num_ids_per_chat[0] = 0;
    std::fill(input_ids.begin(), input_ids.end(), 0);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Pre_1Process(JNIEnv *env, jobject clazz)
{
    std::unique_ptr<Tokenizer> temp = std::make_unique<Tiktoken>();
    tokenizer = temp->createTokenizer(vocab_file); 
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_Run_1LLM(JNIEnv *env, jclass clazz, jstring jquery,
                                                     jboolean add_prompt,
                                                     jboolean clear)
{
    if (add_prompt)
    {
        // if (clear) {
        //     clear_history();  // Open for "Chat" model.
        // }
        clear_history(); // Do clear every time for "Instruct" model.
        const char *query = env->GetStringUTFChars(jquery, nullptr);
        std::vector<int32_t> get_ids = tokenizer->encode(query);
        get_ids.insert(get_ids.begin(), {128000, 128006, 882, 128007, 271});              // Chat prompt head
        get_ids.insert(get_ids.end(), {128009, 271, 128006, 78191, 128007, 271});         // Chat prompt tail
        ids_len = static_cast<int64_t> (get_ids.size());
        num_ids_per_chat[save_index] = static_cast<int> (ids_len);
        if (save_index > 0)
        {
            accumulate_num_ids[save_index] = num_ids_per_chat[save_index] + accumulate_num_ids[save_index - 1];
            if (accumulate_num_ids[save_index] > next_chat_buffer)
            {
                bool over_inputs = true;
                for (int i = 0; i < save_index; i++)
                {
                    if (accumulate_num_ids[save_index] - accumulate_num_ids[i] <= next_chat_buffer)
                    {
                        std::move(input_ids.begin() + accumulate_num_ids[i], input_ids.end(), input_ids.begin());
                        int k = i + 1;
                        for (int j = k; j <= save_index; j++)
                        {
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
                if (over_inputs)
                {
                    clear_history();
                    return env->NewStringUTF("Over_Inputs");
                }
            }
            else
            {
                std::move(get_ids.begin(), get_ids.end(), input_ids.begin() + accumulate_num_ids[save_index - 1]);
                ids_len = accumulate_num_ids[save_index];
            }
        }
        else
        {
            if (num_ids_per_chat[0] >= max_token_history)
            {
                clear_history();
                return env->NewStringUTF("Over_Inputs");
            }
            else
            {
                accumulate_num_ids[0] = num_ids_per_chat[0];
                std::move(get_ids.begin(), get_ids.end(), input_ids.begin());
            }
        }
    }
    ort_runtime_A->Run(session_model_A, run_options_A, input_names_A.data(),
                       (const OrtValue *const *)input_tensors_A.data(),
                       input_tensors_A.size(), output_names_A.data(), output_names_A.size(),
                       output_tensors_A.data());
    void *max_logit_id;
    ort_runtime_A->GetTensorMutableData(output_tensors_A[0], &max_logit_id);
    input_ids[0] = reinterpret_cast<int32_t*>(max_logit_id)[0];
    if ((input_ids[0] != end_id_0) && (input_ids[0] != end_id_1) && (response_count < single_chat_limit) && (history_len < max_token_history))
    {
        history_len += ids_len;
        if (add_prompt)
        {
            ids_len = 1;
            response_count = 0;
            attention_mask = 0.f;
//        attention_mask = Ort::Float16_t(0.f);
        }
        input_tensors_A[2] = output_tensors_A[1];
        input_tensors_A[3] = output_tensors_A[2];
        save_max_logit_position[response_count] = input_ids[0];
        response_count += 1;
        return env->NewStringUTF(get_output_words(input_ids[0]).c_str());
    }
    else
    {
        save_max_logit_position[response_count] = end_id_1;
        response_count += 1;
        num_ids_per_chat[save_index] += response_count;
        attention_mask = -65504.f;
//        attention_mask = Ort::Float16_t(-65504.f);
        history_len = 0;
        input_ids[0] = start_id;
        if (save_index > 0)
        {
            accumulate_num_ids[save_index] = num_ids_per_chat[save_index] + accumulate_num_ids[save_index - 1];
            if (accumulate_num_ids[save_index] > next_chat_buffer)
            {
                for (int i = 0; i < save_index; i++)
                {
                    if (accumulate_num_ids[save_index] - accumulate_num_ids[i] <= next_chat_buffer)
                    {
                        std::move(input_ids.begin() + accumulate_num_ids[i], input_ids.end(), input_ids.begin());
                        int k = i + 1;
                        for (int j = k; j <= save_index; j++)
                        {
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
            }
            else
            {
                std::move(save_max_logit_position.begin(), save_max_logit_position.begin() + response_count, input_ids.begin() + accumulate_num_ids[save_index] - response_count);
                save_index += 1;
            }
        }
        else
        {
            std::move(save_max_logit_position.begin(), save_max_logit_position.begin() + response_count, input_ids.begin() + accumulate_num_ids[0]);
            accumulate_num_ids[0] = num_ids_per_chat[0];
            save_index += 1;
        }
        return env->NewStringUTF("END");
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1A(JNIEnv *env, jobject clazz,
                                                            jobject asset_manager,
                                                            jboolean use_xnnpack)
{
    OrtStatus *status;
    OrtAllocator *allocator;
    OrtEnv *ort_env_A;
    OrtSessionOptions *session_options_A;
    {
        std::vector<char> fileBuffer;
        std::vector<char> fileBuffer_external;
        off_t fileSize;
        off_t fileSize_external;
        if (asset_manager != nullptr)
        {
            AAssetManager *mgr = AAssetManager_fromJava(env, asset_manager);
            AAsset *asset = AAssetManager_open(mgr, file_name_A.c_str(), AASSET_MODE_BUFFER);
            fileSize = AAsset_getLength(asset);
            fileBuffer.resize(fileSize);
            AAsset_read(asset, fileBuffer.data(), fileSize);
            if (file_name_A_external != "NONE") {
                // Load external data using AAsset_read. For models with multiple external files, manually load additional files as needed.
                AAsset* asset_ex = AAssetManager_open(mgr, file_name_A_external.c_str(), AASSET_MODE_BUFFER);
                fileSize_external = AAsset_getLength(asset_ex);
                fileBuffer_external.resize(fileSize_external);
                AAsset_read(asset_ex, fileBuffer_external.data(), fileSize_external);
            }
        }
        else
        {
            std::ifstream model_file(storage_path + file_name_A, std::ios::binary | std::ios::ate);
            if (!model_file.is_open())
            {
                return JNI_FALSE;
            }
            fileSize = model_file.tellg();
            model_file.seekg(0, std::ios::beg);
            fileBuffer.resize(fileSize);
            if (!model_file.read(fileBuffer.data(), fileSize))
            {
                return JNI_FALSE;
            }
            model_file.close();
            if (file_name_A_external != "NONE") {
                // Load external data using std::ifstream. For models with multiple external files, manually load additional files as needed.
                std::ifstream model_file_external(storage_path + file_name_A_external, std::ios::binary | std::ios::ate);
                if (!model_file_external.is_open()) {
                    return JNI_FALSE;
                }
                fileSize_external = model_file_external.tellg();
                model_file_external.seekg(0, std::ios::beg);
                fileBuffer_external.resize(fileSize_external);
                if (!model_file_external.read(fileBuffer_external.data(), fileSize_external)) {
                    return JNI_FALSE;
                }
                model_file_external.close();
            }
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
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.dynamic_block_base", "2"); // One block can contain 1 or more cores, and sharing 1 job.
        ort_runtime_A->AddSessionConfigEntry(session_options_A,                                     // Binding the #cpu to run the model. 'A;B;' means A & B work respectively. 'A,B' means A & B work cooperatively.
                                             "session.intra_op_thread_affinities",
                                             "1;2");               // It is the best cost/performance (C/P) value setting for running the Qwen 1.8B LLM on my device, due to limitations imposed by the RAM bandwidth.
        ort_runtime_A->SetIntraOpNumThreads(session_options_A, 3); // dynamic_block_base + 1
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.inter_op.allow_spinning",
                                             "1"); // 0 for low power
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.intra_op.allow_spinning",
                                             "1"); // 0 for low power
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.force_spinning_stop",
                                             "0"); // 1 for low power
        ort_runtime_A->SetSessionGraphOptimizationLevel(session_options_A, ORT_ENABLE_ALL);
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "optimization.minimal_build_optimizations",
                                             "");   // Keep empty for full optimization
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "optimization.disable_specified_optimizers",
                                             "NchwcTransformer");   // For Arm
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.disable_prepacking",
                                             "0"); // 0 for enable
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "optimization.enable_gelu_approximation",
                                             "1"); // Set 1 is better for this model
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "mlas.enable_gemm_fastmath_arm64_bfloat16",
                                             "1"); //
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.disable_aot_function_inlining",
                                             "0"); // 0 for speed
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.qdqisint8allowed",
                                             "1"); // 1 for Arm
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.enable_quant_qdq_cleanup",
                                             "1"); // 0 for precision, 1 for performance
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.disable_double_qdq_remover",
                                             "0"); // 1 for precision, 0 for performance
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.disable_quant_qdq",
                                             "0"); // 0 for use Int8
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.use_ort_model_bytes_directly",
                                             "1"); // Use this option to lower the peak memory during loading.
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.use_ort_model_bytes_for_initializers",
                                             "0"); // If set use_ort_model_bytes_directly=1, use_ort_model_bytes_for_initializers should be 0.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.set_denormal_as_zero",
                                             "0"); // // Use 0 instead of NaN or Inf.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.use_env_allocators",
                                             "1"); // Use it to lower memory usage.
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.use_device_allocator_for_initializers",
                                             "1"); // Use it to lower memory usage.
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.qdq_matmulnbits_accuracy_level",
                                             "2");  // 0:default, 1:FP32, 2:FP16, 3:BF16, 4:INT8
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "ep.dynamic.workload_type",
                                             "Default");  // Default = Performance; Efficient = Save Power
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
            ort_runtime_A->SetIntraOpNumThreads(session_options_A, 1);                                                // // Set to 1.
            ort_runtime_A->SessionOptionsAppendExecutionProvider(session_options_A, "XNNPACK", option_keys.data(), option_values.data(), option_keys.size());
        }
        if (file_name_A_external != "NONE") {
            const char* external_file_names[] = {file_name_A_external.c_str()};     // Add all external data file names here if your model uses multiple external files.
            const char* external_file_buffers[] = {fileBuffer_external.data()};     // Read external data into fileBuffers and add them here.
            size_t external_file_sizes[] = {fileBuffer_external.size()};            // Store the size of each fileBuffer here for multiple external data files.
            ort_runtime_A->AddExternalInitializersFromFilesInMemory(session_options_A, external_file_names, const_cast<char**>(external_file_buffers), external_file_sizes, 1);  // '1' indicates a single external file.
        }
        status = ort_runtime_A->CreateSessionFromArray(ort_env_A, fileBuffer.data(), fileSize, session_options_A, &session_model_A);
    }
    if (status != nullptr)
    {
        return JNI_FALSE;
    }
    std::size_t amount_of_input;
    ort_runtime_A->GetAllocatorWithDefaultOptions(&allocator);
    ort_runtime_A->SessionGetInputCount(session_model_A, &amount_of_input);
    input_names_A.resize(amount_of_input);
    input_dims_A.resize(amount_of_input);
    input_types_A.resize(amount_of_input);
    input_tensors_A.resize(amount_of_input);
    for (size_t i = 0; i < amount_of_input; i++)
    {
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
        if (typeinfo)
            ort_runtime_A->ReleaseTypeInfo(typeinfo);
    }
    std::size_t amount_of_output;
    ort_runtime_A->SessionGetOutputCount(session_model_A, &amount_of_output);
    output_names_A.resize(amount_of_output);
    output_dims_A.resize(amount_of_output);
    output_types_A.resize(amount_of_output);
    output_tensors_A.resize(amount_of_output);
    for (size_t i = 0; i < amount_of_output; i++)
    {
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
        if (typeinfo)
            ort_runtime_A->ReleaseTypeInfo(typeinfo);
    }
    OrtMemoryInfo *memory_info;
    ort_runtime_A->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    ort_runtime_A->CreateTensorWithDataAsOrtValue(
        memory_info,
        reinterpret_cast<void *>(input_ids.data()), max_token_history * sizeof(int32_t),
        input_dims_A[0].data(), input_dims_A[0].size(), input_types_A[0],
        &input_tensors_A[0]);
    ort_runtime_A->CreateTensorWithDataAsOrtValue(
        memory_info,
        reinterpret_cast<void *>(&attention_mask), sizeof(float),
//        reinterpret_cast<void *>(&attention_mask), sizeof(Ort::Float16_t),
        input_dims_A[1].data(), input_dims_A[1].size(), input_types_A[1],
        &input_tensors_A[1]);
    std::vector<Ort::Float16_t> past_key_values(past_key_value_size, Ort::Float16_t(0.f));
    int buffer_size = past_key_value_size * sizeof(Ort::Float16_t);
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
