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
    attention_mask = Ort::Float16_t(-65504.f);
    pos_factor = Ort::Float16_t(0.f);
    accumulate_num_ids[0] = 0;
    num_ids_per_chat[0] = 0;
    std::fill(input_ids.begin(), input_ids.end(), 0);
}

extern "C"
JNIEXPORT jintArray JNICALL
Java_com_example_myapplication_MainActivity_Process_1Texture(JNIEnv *env, jclass clazz) {
    glUseProgram(computeProgram);
    glDispatchCompute(workGroupCountX, workGroupCountY, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    jintArray final_results = env->NewIntArray(pixelCount);
    env->SetIntArrayRegion(final_results, 0, pixelCount, (jint*) glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, rgbSize_i8, GL_MAP_READ_BIT));
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
    std::unique_ptr<Tokenizer> temp = std::make_unique<HuggingfaceTokenizer>();
    tokenizer = temp->createTokenizer(vocab_file);
    return JNI_TRUE;
}


extern "C"
JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_Run_1LLM_1AD(JNIEnv *env, jclass clazz, jfloatArray pixel_values) {
    jfloat* pixels = env->GetFloatArrayElements(pixel_values, nullptr);
    OrtMemoryInfo *memory_info;
    ort_runtime_A->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    ort_runtime_A->CreateTensorWithDataAsOrtValue(
            memory_info, reinterpret_cast<void*>(pixels), rgbSize,
            input_dims_A[0].data(), input_dims_A[0].size(), input_types_A[0], &input_tensors_A[0]);
    ort_runtime_A->ReleaseMemoryInfo(memory_info);
    ort_runtime_A->Run(session_model_A, run_options_A, input_names_A.data(),
                       (const OrtValue* const*) input_tensors_A.data(),
                       input_tensors_A.size(), output_names_A.data(), output_names_A.size(),
                       output_tensors_A.data());
    input_tensors_D[0] = output_tensors_B[0];
    input_tensors_D[1] = output_tensors_A[0];
    ids_len += image_pad_len;
    ids_len_minus = static_cast<int> (ids_len) - prompt_head_len;
    split_factor = max_token_history - static_cast<int> (ids_len + image_pad_len);
    ort_runtime_D->Run(session_model_D, run_options_D, input_names_D.data(),
                       (const OrtValue* const*) input_tensors_D.data(),
                       input_tensors_D.size(), output_names_D.data(), output_names_D.size(),
                       output_tensors_D.data());
    input_tensors_E[0] = output_tensors_D[0];
    input_tensors_E[6] = output_tensors_D[1];
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_Run_1LLM_1BC(JNIEnv *env, jclass clazz, jstring jquery,
                                                         jboolean clear,
                                                         jboolean use_vision) {
//     if (clear) {
//         clear_history();  // Open for "Chat" model.
//     }
    clear_history(); // Do clear every time for "Instruct" model. We haven't supported the chat mode for QwenVL yet.
    const char *query = env->GetStringUTFChars(jquery, nullptr);
    std::vector<int32_t> get_ids = tokenizer->encode(query);
    get_ids.insert(get_ids.begin(), {198, 151644, 872, 198, 151652, 151653});              // Chat prompt head
    get_ids.insert(get_ids.end(), {151645, 198, 151644, 77091, 198});                      // Chat prompt tail
    ids_len = static_cast<int64_t> (get_ids.size());
    int64_t ids_len_plus;
    std::copy(get_ids.begin(), get_ids.end(), ids_exclude_image.begin());
    if (use_vision) {
        get_ids.insert(get_ids.begin() + prompt_head_len, image_pad_ids.begin(), image_pad_ids.end());
        ids_len_plus = ids_len + image_pad_len;
    } else {
        ids_len_plus = ids_len;
    }
    num_ids_per_chat[save_index] = static_cast<int> (ids_len_plus);
    if (save_index > 0) {
        accumulate_num_ids[save_index] = num_ids_per_chat[save_index] + accumulate_num_ids[save_index - 1];
        if (accumulate_num_ids[save_index] > next_chat_buffer) {
            bool over_inputs = true;
            for (int i = 0; i < save_index; i++) {
                if (accumulate_num_ids[save_index] - accumulate_num_ids[i] <= next_chat_buffer) {
                    std::move(input_ids.begin() + accumulate_num_ids[i], input_ids.end(), input_ids.begin());
                    int k = i + 1;
                    for (int j = k; j <= save_index; j++) {
                        accumulate_num_ids[j] -= accumulate_num_ids[i];
                    }
                    ids_len_plus = accumulate_num_ids[save_index];
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
            ids_len_plus = accumulate_num_ids[save_index];
        }
    } else {
        if (num_ids_per_chat[0] >= max_token_history) {
            clear_history();
            return env->NewStringUTF("Over_Inputs");
        } else {
            accumulate_num_ids[0] = num_ids_per_chat[0];
            std::move(get_ids.begin(), get_ids.end(), input_ids.begin());
        }
    }
    ort_runtime_B->Run(session_model_B, run_options_B, input_names_B.data(),
                       (const OrtValue* const*) input_tensors_B.data(),
                       input_tensors_B.size(), output_names_B.data(), output_names_B.size(),
                       output_tensors_B.data());
    ort_runtime_C->Run(session_model_C, run_options_C, input_names_C.data(),
                       (const OrtValue* const*) input_tensors_C.data(),
                       input_tensors_C.size(), output_names_C.data(), output_names_C.size(),
                       output_tensors_C.data());
    input_tensors_E[0] = output_tensors_B[0];
    input_tensors_E[6] = output_tensors_C[0];
    return env->NewStringUTF("Text_Embed_Done");
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_Run_1LLM_1E(JNIEnv *env, jclass clazz,
                                                        jboolean add_prompt,
                                                        jboolean use_vision) {
    ort_runtime_E->Run(session_model_E, run_options_E, input_names_E.data(),
                       (const OrtValue *const *)input_tensors_E.data(),
                       input_tensors_E.size(), output_names_E.data(), output_names_E.size(),
                       output_tensors_E.data());
    void* max_logit_id;
    ort_runtime_E->GetTensorMutableData(output_tensors_E[0], &max_logit_id);
    ids_exclude_image[0] = reinterpret_cast<int*>(max_logit_id)[0];
    if ((ids_exclude_image[0] != end_id_0) && (ids_exclude_image[0] != end_id_1) && (response_count < single_chat_limit) && (history_len < max_token_history)) {
        history_len += ids_len;
        if (add_prompt) {
            ids_len = 1;
            response_count = 0;
            attention_mask = Ort::Float16_t(0.f);
            if (use_vision) {
                pos_factor = Ort::Float16_t(static_cast<float> (pos_factor_v + ids_len));
            } else {
                pos_factor = Ort::Float16_t(static_cast<float> (history_len + 1));
            }
        } else {
            pos_factor = Ort::Float16_t(1.f + static_cast<float> (pos_factor));
        }
        ort_runtime_B->Run(session_model_B, run_options_B, input_names_B.data(),
                           (const OrtValue* const*) input_tensors_B.data(),
                           input_tensors_B.size(), output_names_B.data(), output_names_B.size(),
                           output_tensors_B.data());
        input_tensors_E[0] = output_tensors_B[0];
        input_tensors_E[2] = output_tensors_E[1];
        input_tensors_E[3] = output_tensors_E[2];
        save_max_logit_position[response_count] = ids_exclude_image[0];
        response_count += 1;
        return env->NewStringUTF(get_output_words(ids_exclude_image[0]).c_str());
    } else {
        save_max_logit_position[response_count] = end_id_1;
        response_count += 1;
        num_ids_per_chat[save_index] += response_count;
        history_len = 0;
        attention_mask = Ort::Float16_t(-65504.f);
        pos_factor = Ort::Float16_t(0.f);
        if (save_index > 0) {
            accumulate_num_ids[save_index] = num_ids_per_chat[save_index] + accumulate_num_ids[save_index - 1];
            if (accumulate_num_ids[save_index] > next_chat_buffer) {
                for (int i = 0; i < save_index; i++) {
                    if (accumulate_num_ids[save_index] - accumulate_num_ids[i] <= next_chat_buffer) {
                        std::move(input_ids.begin() + accumulate_num_ids[i], input_ids.end(), input_ids.begin());
                        int k = i + 1;
                        for (int j = k; j <= save_index; j++) {
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
            std::move(save_max_logit_position.begin(), save_max_logit_position.begin() + response_count, input_ids.begin() + accumulate_num_ids[0]);
            accumulate_num_ids[0] = num_ids_per_chat[0];
            save_index += 1;
        }
        return env->NewStringUTF("END");
    }
}


extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1A(JNIEnv *env, jclass clazz,
                                                            jobject asset_manager,
                                                            jboolean use_float_model,
                                                            jboolean use_qnn,
                                                            jboolean use_dsp_npu,
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
        ort_runtime_A->SetInterOpNumThreads(session_options_A, 4);
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.dynamic_block_base",
                                             "2");  // One block can contain 1 or more cores, and sharing 1 job.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, // Binding the #cpu to run the model. 'A;B' means A & B work respectively. 'A,B' means A & B work cooperatively.
                                             "session.intra_op_thread_affinities",
                                             "1,3;2,4");  // It is the best cost/performance (C/P) value setting for running the QwenVL on my device.
        ort_runtime_A->SetIntraOpNumThreads(session_options_A, 3); // dynamic_block_base + 1
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.inter_op.allow_spinning",
                                             "1");  // 0 for low power
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.intra_op.allow_spinning",
                                             "1");  // 0 for low power
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.force_spinning_stop",
                                             "0");  // 1 for low power
        ort_runtime_A->SetSessionGraphOptimizationLevel(session_options_A, ORT_ENABLE_ALL);  // CPU backend would failed on some FP16 operators with latest opset. Hence, use ORT_ENABLE_ALL instead of ORT_ENABLE_ALL.
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
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.qdqisint8allowed",
                                             "1");  // 1 for Arm
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.enable_quant_qdq_cleanup",
                                             "1");  // 0 for precision, 1 for performance
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.disable_double_qdq_remover",
                                             "0");  // 1 for precision, 0 for performance
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.disable_quant_qdq",
                                             "0");  // 0 for use Int8
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.use_ort_model_bytes_directly",
                                             "1");  // Use this option to lower the peak memory during loading.
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.use_ort_model_bytes_for_initializers",
                                             "0");  // If set use_ort_model_bytes_directly=1, use_ort_model_bytes_for_initializers should be 0.
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.set_denormal_as_zero",
                                             "1");  // // Use 0 instead of NaN or Inf.
        ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                             "session.use_env_allocators",
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
        if (use_qnn) {
            setenv("LD_LIBRARY_PATH", cache_path, 1);
            setenv("ADSP_LIBRARY_PATH", cache_path, 1);
            if (use_dsp_npu) {
                option_keys.push_back("backend_path");
                option_values.push_back(qnn_htp_so);
                ort_runtime_A->AddRunConfigEntry(run_options_A, "qnn.htp_perf_mode", "burst");  // Do not use "option_keys.push_back("htp_performance_mode")", it not work now. (demo version=1.18.1)
                ort_runtime_A->AddRunConfigEntry(run_options_A, "qnn.htp_perf_mode_post_run", "burst");
                ort_runtime_A->AddRunConfigEntry(run_options_A, "qnn.rpc_control_latency", "0");
                option_keys.push_back("htp_graph_finalization_optimization_mode");
                option_values.push_back("3");
                option_keys.push_back("soc_model");
                option_values.push_back("43");  // 0 for unknown, Find your device from here: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/overview.html#supported-snapdragon-devices
                option_keys.push_back("device_id");
                option_values.push_back("0");  // 0 for single device
                option_keys.push_back("vtcm_mb");
                option_values.push_back("8");  // 0 for auto
                option_keys.push_back("qnn_context_priority");
                option_values.push_back("high");
                if (use_float_model) {
                    option_keys.push_back("enable_htp_fp16_precision");
                    option_values.push_back("1");
                } else {
                    option_keys.push_back("enable_htp_fp16_precision");
                    option_values.push_back("0");
                    ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                                         "ep.context_enable", "1");
                    ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                                         "ep.context_embed_mode", "1");
                    ort_runtime_A->AddSessionConfigEntry(session_options_A,
                                                         "ep.context_file_path", ctx_model_A);
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
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1B(JNIEnv *env, jclass clazz,
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
        ort_runtime_B->SetInterOpNumThreads(session_options_B, 4);
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.dynamic_block_base",
                                             "2");  // One block can contain 1 or more cores, and sharing 1 job.
        ort_runtime_B->AddSessionConfigEntry(session_options_B, // Binding the #cpu to run the model. 'A;B' means A & B work respectively. 'A,B' means A & B work cooperatively.
                                             "session.intra_op_thread_affinities",
                                             "1,3;2,4");  // It is the best cost/performance (C/P) value setting for running the QwenVL on my device.
        ort_runtime_B->SetIntraOpNumThreads(session_options_B, 3); // dynamic_block_base + 1
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.inter_op.allow_spinning",
                                             "1");  // 0 for low power
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.intra_op.allow_spinning",
                                             "1");  // 0 for low power
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.force_spinning_stop",
                                             "0");  // 1 for low power
        ort_runtime_B->SetSessionGraphOptimizationLevel(session_options_B, ORT_ENABLE_ALL);  // CPU backend would failed on some FP16 operators with latest opset. Hence, use ORT_ENABLE_ALL instead of ORT_ENABLE_BLL.
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
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.qdqisint8allowed",
                                             "1");  // 1 for Arm
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.enable_quant_qdq_cleanup",
                                             "1");  // 0 for precision, 1 for performance
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.disable_double_qdq_remover",
                                             "0");  // 1 for precision, 0 for performance
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.disable_quant_qdq",
                                             "0");  // 0 for use Int8
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.use_ort_model_bytes_directly",
                                             "1");  // Use this option to lower the peak memory during loading.
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.use_ort_model_bytes_for_initializers",
                                             "0");  // If set use_ort_model_bytes_directly=1, use_ort_model_bytes_for_initializers should be 0.
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.set_denormal_as_zero",
                                             "1");  // // Use 0 instead of NaN or Inf.
        ort_runtime_B->AddSessionConfigEntry(session_options_B,
                                             "session.use_env_allocators",
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
            ort_runtime_B->SetInterOpNumThreads(session_options_B, 4);                                                // Keep the same value as above.
            ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.intra_op.allow_spinning", "0");          // Set to 0.
            ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.dynamic_block_base", "1");               // Set to 1.
            ort_runtime_B->AddSessionConfigEntry(session_options_B, "session.intra_op_thread_affinities", "1,2,3,4"); // Use ',' to split the #core
            ort_runtime_B->SetIntraOpNumThreads(session_options_B, 1);                                                // // Set to 1.
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
    ort_runtime_B->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(ids_exclude_image.data()), max_token_history * sizeof(int),
            input_dims_B[0].data(), input_dims_B[0].size(), input_types_B[0],
            &input_tensors_B[0]);
    ort_runtime_B->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(&ids_len), sizeof(int64_t),
            input_dims_B[1].data(), input_dims_B[1].size(), input_types_B[1],
            &input_tensors_B[1]);
    ort_runtime_B->ReleaseMemoryInfo(memory_info);
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1C(JNIEnv *env, jclass clazz,
                                                            jobject asset_manager,
                                                            jboolean use_xnnpack) {
    OrtStatus *status;
    OrtAllocator *allocator;
    OrtEnv *ort_env_C;
    OrtSessionOptions *session_options_C;
    {
        std::vector<char> fileBuffer;
        off_t fileSize;
        if (asset_manager != nullptr) {
            AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
            AAsset* asset = AAssetManager_open(mgr,file_name_C.c_str(), AASSET_MODE_BUFFER);
            fileSize = AAsset_getLength(asset);
            fileBuffer.resize(fileSize);
            AAsset_read(asset,fileBuffer.data(),fileSize);
        } else {
            std::ifstream model_file(storage_path + file_name_C, std::ios::binary | std::ios::ate);
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
        ort_runtime_C = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        ort_runtime_C->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "myapplication", &ort_env_C);
        ort_runtime_C->CreateSessionOptions(&session_options_C);
        ort_runtime_C->CreateRunOptions(&run_options_C);
        ort_runtime_C->AddRunConfigEntry(run_options_C, "memory.enable_memory_arena_shrinkage", "");  // Keep empty for performance; "cpu:0" for low memory usage.
        ort_runtime_C->AddRunConfigEntry(run_options_C, "disable_synchronize_execution_providers", "1");  // 1 for aggressive performance.
        ort_runtime_C->DisableProfiling(session_options_C);
        ort_runtime_C->EnableCpuMemArena(session_options_C);
        ort_runtime_C->EnableMemPattern(session_options_C);
        ort_runtime_C->SetSessionExecutionMode(session_options_C, ORT_SEQUENTIAL);
        ort_runtime_C->SetInterOpNumThreads(session_options_C, 4);
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "session.dynamic_block_base",
                                             "2");  // One block can contain 1 or more cores, and sharing 1 job.
        ort_runtime_C->AddSessionConfigEntry(session_options_C, // Binding the #cpu to run the model. 'A;B' means A & B work respectively. 'A,B' means A & B work cooperatively.
                                             "session.intra_op_thread_affinities",
                                             "1,3;2,4");  // It is the best cost/performance (C/P) value setting for running the QwenVL on my device.
        ort_runtime_C->SetIntraOpNumThreads(session_options_C, 3); // dynamic_block_base + 1
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "session.inter_op.allow_spinning",
                                             "1");  // 0 for low power
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "session.intra_op.allow_spinning",
                                             "1");  // 0 for low power
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "session.force_spinning_stop",
                                             "0");  // 1 for low power
        ort_runtime_C->SetSessionGraphOptimizationLevel(session_options_C, ORT_ENABLE_ALL);  // CPU backend would failed on some FP16 operators with latest opset. Hence, use ORT_ENABLE_ALL instead of ORT_ENABLE_CLL.
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "optimization.minimal_build_optimizations",
                                             "");   // Keep empty for full optimization
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "optimization.disable_specified_optimizers",
                                             "NchwcTransformer");   // For Arm
        ort_runtime_C->AddSessionConfigEntry(session_options_C, "session.disable_prepacking",
                                             "0");  // 0 for enable
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "optimization.enable_gelu_approximation",
                                             "1");  // Set 1 is better for this model
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "mlas.enable_gemm_fastmath_arm64_bfloat16",
                                             "1");  //
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "session.disable_aot_function_inlining",
                                             "0");  // 0 for speed
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "session.qdqisint8allowed",
                                             "1");  // 1 for Arm
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "session.enable_quant_qdq_cleanup",
                                             "1");  // 0 for precision, 1 for performance
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "session.disable_double_qdq_remover",
                                             "0");  // 1 for precision, 0 for performance
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "session.disable_quant_qdq",
                                             "0");  // 0 for use Int8
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "session.use_ort_model_bytes_directly",
                                             "1");  // Use this option to lower the peak memory during loading.
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "session.use_ort_model_bytes_for_initializers",
                                             "0");  // If set use_ort_model_bytes_directly=1, use_ort_model_bytes_for_initializers should be 0.
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "session.set_denormal_as_zero",
                                             "1");  // // Use 0 instead of NaN or Inf.
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "session.use_env_allocators",
                                             "1");  // Use it to lower memory usage.
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "session.use_device_allocator_for_initializers",
                                             "1");  // Use it to lower memory usage.
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "session.qdq_matmulnbits_accuracy_level",
                                             "4");  // 0:default, 1:FP32, 2:FP16, 3:BF16, 4:INT8
        ort_runtime_C->AddSessionConfigEntry(session_options_C,
                                             "ep.dynamic.workload_type",
                                             "Default");  // Default = Performance; Efficient = Save Power
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
        status = ort_runtime_C->CreateSessionFromArray(ort_env_C, fileBuffer.data(), fileSize, session_options_C, &session_model_C);
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
    for (size_t i = 0; i < amount_of_input; i++) {
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
        if (typeinfo) ort_runtime_C->ReleaseTypeInfo(typeinfo);
    }
    std::size_t amount_of_output;
    ort_runtime_C->SessionGetOutputCount(session_model_C, &amount_of_output);
    output_names_C.resize(amount_of_output);
    output_dims_C.resize(amount_of_output);
    output_types_C.resize(amount_of_output);
    output_tensors_C.resize(amount_of_output);
    for (size_t i = 0; i < amount_of_output; i++) {
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
        if (typeinfo) ort_runtime_C->ReleaseTypeInfo(typeinfo);
    }
    OrtMemoryInfo *memory_info;
    ort_runtime_C->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    ort_runtime_C->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(&dummy), sizeof(int),
            input_dims_C[0].data(), input_dims_C[0].size(), input_types_C[0],
            &input_tensors_C[0]);
    ort_runtime_C->ReleaseMemoryInfo(memory_info);
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1D(JNIEnv *env, jclass clazz,
                                                            jobject asset_manager,
                                                            jboolean use_xnnpack) {
    OrtStatus *status;
    OrtAllocator *allocator;
    OrtEnv *ort_env_D;
    OrtSessionOptions *session_options_D;
    {
        std::vector<char> fileBuffer;
        off_t fileSize;
        if (asset_manager != nullptr) {
            AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
            AAsset* asset = AAssetManager_open(mgr,file_name_D.c_str(), AASSET_MODE_BUFFER);
            fileSize = AAsset_getLength(asset);
            fileBuffer.resize(fileSize);
            AAsset_read(asset,fileBuffer.data(),fileSize);
        } else {
            std::ifstream model_file(storage_path + file_name_D, std::ios::binary | std::ios::ate);
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
        ort_runtime_D = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        ort_runtime_D->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "myapplication", &ort_env_D);
        ort_runtime_D->CreateSessionOptions(&session_options_D);
        ort_runtime_D->CreateRunOptions(&run_options_D);
        ort_runtime_D->AddRunConfigEntry(run_options_D, "memory.enable_memory_arena_shrinkage", "");  // Keep empty for performance; "cpu:0" for low memory usage.
        ort_runtime_D->AddRunConfigEntry(run_options_D, "disable_synchronize_execution_providers", "1");  // 1 for aggressive performance.
        ort_runtime_D->DisableProfiling(session_options_D);
        ort_runtime_D->EnableCpuMemArena(session_options_D);
        ort_runtime_D->EnableMemPattern(session_options_D);
        ort_runtime_D->SetSessionExecutionMode(session_options_D, ORT_SEQUENTIAL);
        ort_runtime_D->SetInterOpNumThreads(session_options_D, 4);
        ort_runtime_D->AddSessionConfigEntry(session_options_D,
                                             "session.dynamic_block_base",
                                             "2");  // One block can contain 1 or more cores, and sharing 1 job.
        ort_runtime_D->AddSessionConfigEntry(session_options_D, // Binding the #cpu to run the model. 'A;B' means A & B work respectively. 'A,B' means A & B work cooperatively.
                                             "session.intra_op_thread_affinities",
                                             "1,3;2,4");  // It is the best cost/performance (C/P) value setting for running the QwenVL on my device.
        ort_runtime_D->SetIntraOpNumThreads(session_options_D, 3); // dynamic_block_base + 1
        ort_runtime_D->AddSessionConfigEntry(session_options_D,
                                             "session.inter_op.allow_spinning",
                                             "1");  // 0 for low power
        ort_runtime_D->AddSessionConfigEntry(session_options_D,
                                             "session.intra_op.allow_spinning",
                                             "1");  // 0 for low power
        ort_runtime_D->AddSessionConfigEntry(session_options_D,
                                             "session.force_spinning_stop",
                                             "0");  // 1 for low power
        ort_runtime_D->SetSessionGraphOptimizationLevel(session_options_D, ORT_ENABLE_ALL);  // CPU backend would failed on some FP16 operators with latest opset. Hence, use ORT_ENABLE_ALL instead of ORT_ENABLE_DLL.
        ort_runtime_D->AddSessionConfigEntry(session_options_D,
                                             "optimization.minimal_build_optimizations",
                                             "");   // Keep empty for full optimization
        ort_runtime_D->AddSessionConfigEntry(session_options_D,
                                             "optimization.disable_specified_optimizers",
                                             "NchwcTransformer");   // For Arm
        ort_runtime_D->AddSessionConfigEntry(session_options_D, "session.disable_prepacking",
                                             "0");  // 0 for enable
        ort_runtime_D->AddSessionConfigEntry(session_options_D,
                                             "optimization.enable_gelu_approximation",
                                             "1");  // Set 1 is better for this model
        ort_runtime_D->AddSessionConfigEntry(session_options_D,
                                             "mlas.enable_gemm_fastmath_arm64_bfloat16",
                                             "1");  //
        ort_runtime_D->AddSessionConfigEntry(session_options_D,
                                             "session.disable_aot_function_inlining",
                                             "0");  // 0 for speed
        ort_runtime_D->AddSessionConfigEntry(session_options_D,
                                             "session.qdqisint8allowed",
                                             "1");  // 1 for Arm
        ort_runtime_D->AddSessionConfigEntry(session_options_D,
                                             "session.enable_quant_qdq_cleanup",
                                             "1");  // 0 for precision, 1 for performance
        ort_runtime_D->AddSessionConfigEntry(session_options_D,
                                             "session.disable_double_qdq_remover",
                                             "0");  // 1 for precision, 0 for performance
        ort_runtime_D->AddSessionConfigEntry(session_options_D,
                                             "session.disable_quant_qdq",
                                             "0");  // 0 for use Int8
        ort_runtime_D->AddSessionConfigEntry(session_options_D,
                                             "session.use_ort_model_bytes_directly",
                                             "1");  // Use this option to lower the peak memory during loading.
        ort_runtime_D->AddSessionConfigEntry(session_options_D,
                                             "session.use_ort_model_bytes_for_initializers",
                                             "0");  // If set use_ort_model_bytes_directly=1, use_ort_model_bytes_for_initializers should be 0.
        ort_runtime_D->AddSessionConfigEntry(session_options_D,
                                             "session.set_denormal_as_zero",
                                             "1");  // // Use 0 instead of NaN or Inf.
        ort_runtime_D->AddSessionConfigEntry(session_options_D,
                                             "session.use_env_allocators",
                                             "1");  // Use it to lower memory usage.
        ort_runtime_D->AddSessionConfigEntry(session_options_D,
                                             "session.use_device_allocator_for_initializers",
                                             "1");  // Use it to lower memory usage.
        ort_runtime_D->AddSessionConfigEntry(session_options_D,
                                             "session.qdq_matmulnbits_accuracy_level",
                                             "4");  // 0:default, 1:FP32, 2:FP16, 3:BF16, 4:INT8
        ort_runtime_D->AddSessionConfigEntry(session_options_D,
                                             "ep.dynamic.workload_type",
                                             "Default");  // Default = Performance; Efficient = Save Power
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
        status = ort_runtime_D->CreateSessionFromArray(ort_env_D, fileBuffer.data(), fileSize, session_options_D, &session_model_D);
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
    for (size_t i = 0; i < amount_of_input; i++) {
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
        if (typeinfo) ort_runtime_D->ReleaseTypeInfo(typeinfo);
    }
    std::size_t amount_of_output;
    ort_runtime_D->SessionGetOutputCount(session_model_D, &amount_of_output);
    output_names_D.resize(amount_of_output);
    output_dims_D.resize(amount_of_output);
    output_types_D.resize(amount_of_output);
    output_tensors_D.resize(amount_of_output);
    for (size_t i = 0; i < amount_of_output; i++) {
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
        if (typeinfo) ort_runtime_D->ReleaseTypeInfo(typeinfo);
    }
    OrtMemoryInfo *memory_info;
    ort_runtime_D->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    std::vector<Ort::Float16_t> hidden_state(max_token_history * hidden_size, Ort::Float16_t(0.f));
    std::vector<Ort::Float16_t> image_embed(WIDTH_FACTOR * HEIGHT_FACTOR * hidden_size, Ort::Float16_t(0.f));
    ort_runtime_D->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(hidden_state.data()), hidden_state.size() * sizeof(Ort::Float16_t),
            input_dims_D[0].data(), input_dims_D[0].size(), input_types_D[0],
            &input_tensors_D[0]);
    ort_runtime_D->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(image_embed.data()), image_embed.size() * sizeof(Ort::Float16_t),
            input_dims_D[1].data(), input_dims_D[1].size(), input_types_D[1],
            &input_tensors_D[1]);
    ort_runtime_D->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(&ids_len), sizeof(int64_t),
            input_dims_D[2].data(), input_dims_D[2].size(), input_types_D[2],
            &input_tensors_D[2]);
    ort_runtime_D->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(&ids_len_minus), sizeof(int),
            input_dims_D[3].data(), input_dims_D[3].size(), input_types_D[3],
            &input_tensors_D[3]);
    ort_runtime_D->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(&split_factor), sizeof(int),
            input_dims_D[4].data(), input_dims_D[4].size(), input_types_D[4],
            &input_tensors_D[4]);
    ort_runtime_D->ReleaseMemoryInfo(memory_info);
    return JNI_TRUE;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1E(JNIEnv *env, jclass clazz,
                                                            jobject asset_manager,
                                                            jboolean use_xnnpack) {
    OrtStatus *status;
    OrtAllocator *allocator;
    OrtEnv *ort_env_E;
    OrtSessionOptions *session_options_E;
    {
        std::vector<char> fileBuffer;
        std::vector<char> fileBuffer_external;
        off_t fileSize;
        off_t fileSize_external;
        if (asset_manager != nullptr) {
            AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
            AAsset* asset = AAssetManager_open(mgr,file_name_E.c_str(), AASSET_MODE_BUFFER);
            fileSize = AAsset_getLength(asset);
            fileBuffer.resize(fileSize);
            AAsset_read(asset,fileBuffer.data(),fileSize);
            if (file_name_E_external != "NONE") {
                // Load external data using AAsset_read. For models with multiple external files, manually load additional files as needed.
                AAsset* asset_ex = AAssetManager_open(mgr, file_name_E_external.c_str(), AASSET_MODE_BUFFER);
                fileSize_external = AAsset_getLength(asset_ex);
                fileBuffer_external.resize(fileSize_external);
                AAsset_read(asset_ex, fileBuffer_external.data(), fileSize_external);
            }
        } else {
            std::ifstream model_file(storage_path + file_name_E, std::ios::binary | std::ios::ate);
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
            if (file_name_E_external != "NONE") {
                // Load external data using std::ifstream. For models with multiple external files, manually load additional files as needed.
                std::ifstream model_file_external(storage_path + file_name_E_external, std::ios::binary | std::ios::ate);
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
        ort_runtime_E = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        ort_runtime_E->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "myapplication", &ort_env_E);
        ort_runtime_E->CreateSessionOptions(&session_options_E);
        ort_runtime_E->CreateRunOptions(&run_options_E);
        ort_runtime_E->AddRunConfigEntry(run_options_E, "memory.enable_memory_arena_shrinkage", "");  // Keep empty for performance; "cpu:0" for low memory usage.
        ort_runtime_E->AddRunConfigEntry(run_options_E, "disable_synchronize_execution_providers", "1");  // 1 for aggressive performance.
        ort_runtime_E->DisableProfiling(session_options_E);
        ort_runtime_E->EnableCpuMemArena(session_options_E);
        ort_runtime_E->EnableMemPattern(session_options_E);
        ort_runtime_E->SetSessionExecutionMode(session_options_E, ORT_SEQUENTIAL);
        ort_runtime_E->SetInterOpNumThreads(session_options_E, 4);
        ort_runtime_E->AddSessionConfigEntry(session_options_E,
                                             "session.dynamic_block_base",
                                             "2");  // One block can contain 1 or more cores, and sharing 1 job.
        ort_runtime_E->AddSessionConfigEntry(session_options_E, // Binding the #cpu to run the model. 'A;B' means A & B work respectively. 'A,B' means A & B work cooperatively.
                                             "session.intra_op_thread_affinities",
                                             "1,3;2,4");  // It is the best cost/performance (C/P) value setting for running the QwenVL on my device.
        ort_runtime_E->SetIntraOpNumThreads(session_options_E, 3); // dynamic_block_base + 1
        ort_runtime_E->AddSessionConfigEntry(session_options_E,
                                             "session.inter_op.allow_spinning",
                                             "1");  // 0 for low power
        ort_runtime_E->AddSessionConfigEntry(session_options_E,
                                             "session.intra_op.allow_spinning",
                                             "1");  // 0 for low power
        ort_runtime_E->AddSessionConfigEntry(session_options_E,
                                             "session.force_spinning_stop",
                                             "0");  // 1 for low power
        ort_runtime_E->SetSessionGraphOptimizationLevel(session_options_E, ORT_ENABLE_ALL);  // CPU backend would failed on some FP16 operators with latest opset. Hence, use ORT_ENABLE_ALL instead of ORT_ENABLE_ELL.
        ort_runtime_E->AddSessionConfigEntry(session_options_E,
                                             "optimization.minimal_build_optimizations",
                                             "");   // Keep empty for full optimization
        ort_runtime_E->AddSessionConfigEntry(session_options_E,
                                             "optimization.disable_specified_optimizers",
                                             "NchwcTransformer");   // For Arm
        ort_runtime_E->AddSessionConfigEntry(session_options_E, "session.disable_prepacking",
                                             "0");  // 0 for enable
        ort_runtime_E->AddSessionConfigEntry(session_options_E,
                                             "optimization.enable_gelu_approximation",
                                             "1");  // Set 1 is better for this model
        ort_runtime_E->AddSessionConfigEntry(session_options_E,
                                             "mlas.enable_gemm_fastmath_arm64_bfloat16",
                                             "1");  //
        ort_runtime_E->AddSessionConfigEntry(session_options_E,
                                             "session.disable_aot_function_inlining",
                                             "0");  // 0 for speed
        ort_runtime_E->AddSessionConfigEntry(session_options_E,
                                             "session.qdqisint8allowed",
                                             "1");  // 1 for Arm
        ort_runtime_E->AddSessionConfigEntry(session_options_E,
                                             "session.enable_quant_qdq_cleanup",
                                             "1");  // 0 for precision, 1 for performance
        ort_runtime_E->AddSessionConfigEntry(session_options_E,
                                             "session.disable_double_qdq_remover",
                                             "0");  // 1 for precision, 0 for performance
        ort_runtime_E->AddSessionConfigEntry(session_options_E,
                                             "session.disable_quant_qdq",
                                             "0");  // 0 for use Int8
        ort_runtime_E->AddSessionConfigEntry(session_options_E,
                                             "session.use_ort_model_bytes_directly",
                                             "1");  // Use this option to lower the peak memory during loading.
        ort_runtime_E->AddSessionConfigEntry(session_options_E,
                                             "session.use_ort_model_bytes_for_initializers",
                                             "0");  // If set use_ort_model_bytes_directly=1, use_ort_model_bytes_for_initializers should be 0.
        ort_runtime_E->AddSessionConfigEntry(session_options_E,
                                             "session.set_denormal_as_zero",
                                             "1");  // // Use 0 instead of NaN or Inf.
        ort_runtime_E->AddSessionConfigEntry(session_options_E,
                                             "session.use_env_allocators",
                                             "1");  // Use it to lower memory usage.
        ort_runtime_E->AddSessionConfigEntry(session_options_E,
                                             "session.use_device_allocator_for_initializers",
                                             "1");  // Use it to lower memory usage.
        ort_runtime_E->AddSessionConfigEntry(session_options_E,
                                             "session.qdq_matmulnbits_accuracy_level",
                                             "4");  // 0:default, 1:FP32, 2:FP16, 3:BF16, 4:INT8
        ort_runtime_E->AddSessionConfigEntry(session_options_E,
                                             "ep.dynamic.workload_type",
                                             "Default");  // Default = Performance; Efficient = Save Power
        std::vector<const char*> option_keys = {};
        std::vector<const char*> option_values = {};
        if (use_xnnpack) {
            option_keys.push_back("intra_op_num_threads");
            option_values.push_back("4");
            ort_runtime_E->SetInterOpNumThreads(session_options_E, 4);                                                // Keep the same value as above.
            ort_runtime_E->AddSessionConfigEntry(session_options_E, "session.intra_op.allow_spinning", "0");          // Set to 0.
            ort_runtime_E->AddSessionConfigEntry(session_options_E, "session.dynamic_block_base", "1");               // Set to 1.
            ort_runtime_E->AddSessionConfigEntry(session_options_E, "session.intra_op_thread_affinities", "1,2,3,4"); // Use ',' to split the #core
            ort_runtime_E->SetIntraOpNumThreads(session_options_E, 1);                                                // // Set to 1.
            ort_runtime_E->SessionOptionsAppendExecutionProvider(session_options_E, "XNNPACK", option_keys.data(), option_values.data(), option_keys.size());
        }
        if (file_name_E_external != "NONE") {
            const char* external_file_names[] = {file_name_E_external.c_str()};     // Add all external data file names here if your model uses multiple external files.
            const char* external_file_buffers[] = {fileBuffer_external.data()};     // Read external data into fileBuffers and add them here.
            size_t external_file_sizes[] = {fileBuffer_external.size()};            // Store the size of each fileBuffer here for multiple external data files.
            ort_runtime_E->AddExternalInitializersFromFilesInMemory(session_options_E, external_file_names, const_cast<char**>(external_file_buffers), external_file_sizes, 1);  // '1' indicates a single external file.
        }
        status = ort_runtime_E->CreateSessionFromArray(ort_env_E, fileBuffer.data(), fileSize, session_options_E, &session_model_E);
    }
    if (status != nullptr) {
        return JNI_FALSE;
    }
    std::size_t amount_of_input;
    ort_runtime_E->GetAllocatorWithDefaultOptions(&allocator);
    ort_runtime_E->SessionGetInputCount(session_model_E, &amount_of_input);
    input_names_E.resize(amount_of_input);
    input_dims_E.resize(amount_of_input);
    input_types_E.resize(amount_of_input);
    input_tensors_E.resize(amount_of_input);
    for (size_t i = 0; i < amount_of_input; i++) {
        char* name;
        OrtTypeInfo* typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo* tensor_info;
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
        if (typeinfo) ort_runtime_E->ReleaseTypeInfo(typeinfo);
    }
    std::size_t amount_of_output;
    ort_runtime_E->SessionGetOutputCount(session_model_E, &amount_of_output);
    output_names_E.resize(amount_of_output);
    output_dims_E.resize(amount_of_output);
    output_types_E.resize(amount_of_output);
    output_tensors_E.resize(amount_of_output);
    for (size_t i = 0; i < amount_of_output; i++) {
        char* name;
        OrtTypeInfo* typeinfo;
        size_t dimensions;
        size_t tensor_size;
        const OrtTensorTypeAndShapeInfo* tensor_info;
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
        if (typeinfo) ort_runtime_E->ReleaseTypeInfo(typeinfo);
    }
    OrtMemoryInfo *memory_info;
    ort_runtime_E->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    std::vector<Ort::Float16_t> past_key_values(past_key_value_size, Ort::Float16_t(0.f));
    std::vector<Ort::Float16_t> hidden_state(max_token_history * hidden_size, Ort::Float16_t(0.f));
    std::vector<Ort::Float16_t> position_ids(max_token_history * 3, Ort::Float16_t(0.f));
    ort_runtime_E->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(hidden_state.data()), hidden_state.size() * sizeof(Ort::Float16_t),
            input_dims_E[0].data(), input_dims_E[0].size(), input_types_E[0],
            &input_tensors_E[0]);
    ort_runtime_E->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(&attention_mask), sizeof(Ort::Float16_t),
            input_dims_E[1].data(), input_dims_E[1].size(), input_types_E[1],
            &input_tensors_E[1]);
    ort_runtime_E->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(past_key_values.data()), past_key_value_size * sizeof(Ort::Float16_t),
            input_dims_E[2].data(), input_dims_E[2].size(), input_types_E[2],
            &input_tensors_E[2]);
    ort_runtime_E->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(past_key_values.data()), past_key_value_size * sizeof(Ort::Float16_t),
            input_dims_E[3].data(), input_dims_E[3].size(), input_types_E[3],
            &input_tensors_E[3]);
    ort_runtime_E->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(&history_len), sizeof(int64_t),
            input_dims_E[4].data(), input_dims_E[4].size(), input_types_E[4],
            &input_tensors_E[4]);
    ort_runtime_E->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(&ids_len), sizeof(int64_t),
            input_dims_E[5].data(), input_dims_E[5].size(), input_types_E[5],
            &input_tensors_E[5]);
    ort_runtime_E->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(position_ids.data()), position_ids.size() * sizeof(Ort::Float16_t),
            input_dims_E[6].data(), input_dims_E[6].size(), input_types_E[6],
            &input_tensors_E[6]);
    ort_runtime_E->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void *>(&pos_factor), sizeof(Ort::Float16_t),
            input_dims_E[7].data(), input_dims_E[7].size(), input_types_E[7],
            &input_tensors_E[7]);
    ort_runtime_E->ReleaseMemoryInfo(memory_info);
    return JNI_TRUE;
}

