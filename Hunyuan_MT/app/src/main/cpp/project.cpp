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

// Lookup table mapping an ASCII byte to its hexadecimal value (0-15); non-hex bytes map to 0.
// Built once on first use (thread-safe static init) and reused to decode "<0xHH>" byte-fallback
// tokens without the per-token heap allocation of substr() or the heavier std::stoi() parse.
inline static const unsigned char* hex_value_lut() {
    static const struct HexLut {
        unsigned char t[256];
        HexLut() {
            for (int i = 0; i < 256; ++i) t[i] = 0;
            for (int c = '0'; c <= '9'; ++c) t[c] = static_cast<unsigned char>(c - '0');
            for (int c = 'a'; c <= 'f'; ++c) t[c] = static_cast<unsigned char>(c - 'a' + 10);
            for (int c = 'A'; c <= 'F'; ++c) t[c] = static_cast<unsigned char>(c - 'A' + 10);
        }
    } lut;
    return lut.t;
}

inline static std::string get_output_words(const int &id) {
    std::string words = tokenizer->decode(id);
    // Byte-fallback tokens are emitted as the 6-char form "<0xHH>"; collapse them to the single raw
    // byte they represent. Parse the two hex digits via the lookup table (no substr/stoi allocation).
    if (words.length() == 6 && words[0] == '<' && words[5] == '>' && words[1] == '0' && words[2] == 'x')
    {
        const unsigned char* hex = hex_value_lut();
        words = static_cast<char>((hex[static_cast<unsigned char>(words[3])] << 4) |
                                   hex[static_cast<unsigned char>(words[4])]);
    }
    correctUtfBytes(words.data());
    return words;
}

inline static bool is_stop_token(int id) {
    return (id == end_id_0) || (id == end_id_1) || (id == end_id_2);
}

inline static void clear_history() {
    ids_len = 0;
    save_index = 0;
    history_len = 0;
    response_count = 0;
    attention_mask = 1;
    token_id = end_id_0;
    num_ids_per_chat[0] = 0;
    accumulate_num_ids[0] = 0;
}

inline static void bind_kv_outputs_to_device() {
    // Only the KV cache outputs need re-binding each step: they grow by ids_len every run (dynamic
    // length) so ORT must allocate fresh device buffers for them, which also stops the KV tensors
    // just bound as inputs from being overwritten in place. The max_logit_id output is fixed (1,1)
    // and stays bound to max_idx_buf for the whole session (see Load_Models_A), so it is skipped.
    for (int i = 0; i < num_keys_values; i++) {
        ort_runtime_A->BindOutputToDevice(io_binding_A, output_names_A[i], memory_info);
    }
}

inline static void release_carryover() {
    // Release the OrtValues handed back by GetBoundOutputValues for the previous step once they are
    // no longer bound as inputs (mirrors Python letting the old get_outputs() references drop).
    if (carryover_outputs != nullptr) {
        for (size_t i = 0; i < carryover_count; i++) {
            if (carryover_outputs[i] != nullptr) {
                ort_runtime_A->ReleaseValue(carryover_outputs[i]);
            }
        }
        ort_runtime_A->AllocatorFree(allocator, carryover_outputs);
        carryover_outputs = nullptr;
        carryover_count = 0;
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Pre_1Process(JNIEnv *env, jobject clazz)
{
    tokenizer = MNN::Transformer::Tokenizer::createTokenizer(voacb_path);
    return JNI_TRUE;
}

inline static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline static void flush_stream(JNIEnv* env) {
    // Deliver the accumulated batch to the UI through the cached static callback. This runs on the
    // Java-attached LLMThread, so the JNIEnv handed to Run_LLM is valid here (no AttachCurrentThread).
    if (stream_buf.empty()) {
        return;
    }
    if (g_main_cls != nullptr && g_on_token != nullptr) {
        jstring js = env->NewStringUTF(stream_buf.c_str());
        if (js != nullptr) {
            env->CallStaticVoidMethod(g_main_cls, g_on_token, js);
            // MUST free every per-flush local ref: a long reply would otherwise overflow the ~512-entry
            // local reference table and crash. Then clear any pending Java exception so it cannot leak
            // across the next JNI call (this build is compiled -fno-exceptions).
            env->DeleteLocalRef(js);
            if (env->ExceptionCheck()) {
                env->ExceptionClear();
            }
        }
    }
    stream_buf.clear();
    tokens_since_flush = 0;
    last_flush_ms = now_ms();
}

inline static void end_bookkeeping() {
    // Post-generation history bookkeeping, lifted VERBATIM from the original stop branch (multi-turn
    // input_ids buffer maintenance via save_index / num_ids_per_chat / accumulate_num_ids). Logic is
    // unchanged; the former `return env->NewStringUTF("END")` short-circuit is now a plain `return`.
    chatting = false;
    save_max_logit_position[response_count] = end_id_0;
    response_count += 1;
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
                    return;
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
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_Run_1LLM(JNIEnv *env, jclass clazz, jstring jquery,
                                                     jboolean clear) {
    // Non-reentrancy guard for the global, non-thread-safe inference state. The Java side joins the
    // previous LLMThread before launching a new one, so g_busy is expected to be false here; if it is
    // somehow still held, bail out safely rather than corrupting the in-flight generation's state.
    if (g_busy.exchange(true, std::memory_order_acq_rel)) {
        return env->NewStringUTF("");
    }
    // Scope-exit clear of g_busy on every return below. Safe under -fno-exceptions (normal returns only).
    struct BusyGuard { ~BusyGuard() { g_busy.store(false, std::memory_order_release); } } busy_guard;
    g_cancel.store(false, std::memory_order_relaxed);

    // =============================== Prefill bookkeeping (runs once) ===============================
    // Not a chat model: every call starts a fresh context (clear_history), exactly as the original did.
    chatting = true;
    clear_history();  // It is not a chat model.
//        if (clear) {
//            clear_history();
//        } else {
//            response_count = 0;
//        }
    const char *query = env->GetStringUTFChars(jquery, nullptr);
    std::vector<int> get_ids = tokenizer->encode(query);
    env->ReleaseStringUTFChars(jquery, query);
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
    // === IOBinding setup for the prefill run ===
    // Drop any buffers carried over from a previous (possibly interrupted) generation.
    release_carryover();
    // (Re)create the input_ids tensor for this prompt and bind it. history_len / ids_len /
    // attention_mask are bound once in Load_Models_A and updated in place (their OrtValues wrap
    // the C++ variables), so only input_ids needs rebinding when its sequence length changes.
    input_dims_A[num_keys_values][1] = ids_len;
    if (input_tensors_A[num_keys_values] != nullptr) {
        ort_runtime_A->ReleaseValue(input_tensors_A[num_keys_values]);
        input_tensors_A[num_keys_values] = nullptr;
    }
    ort_runtime_A->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void*>(input_ids.data()), input_ids_buffer_size[ids_len],
            input_dims_A[num_keys_values].data(), input_dims_A[num_keys_values].size(), input_types_A[num_keys_values],
            &input_tensors_A[num_keys_values]);
    ort_runtime_A->BindInput(io_binding_A, input_names_A[num_keys_values], input_tensors_A[num_keys_values]);
    // Auto-bind the empty (zero-length) KV cache as the initial inputs for the prefill.
    for (int i = 0; i < num_keys_values; i++) {
        ort_runtime_A->BindInput(io_binding_A, input_names_A[i], input_tensors_kv_init_A[i]);
    }
    // Let ORT allocate the dynamic-length KV cache outputs on-device. The fixed (1,1)
    // max_logit_id output is already bound once to max_idx_buf and must not be rebound here.
    bind_kv_outputs_to_device();

    // =============================== Decode loop (entirely in C++) ===============================
    // Iteration 0 is the prefill forward pass (attention_mask = 1, full prompt); every later iteration
    // is a single-token decode step. The whole reply is produced in ONE Run_LLM call and streamed to
    // the UI in batches via flush_stream(), instead of one JNI round-trip + UI rebind per token.
    stream_buf.clear();
    tokens_since_flush = 0;
    // Decode-rate clock. It is (re)started AFTER prefill's forward pass below, so the reported token/s
    // measures DECODE throughput only. The prefill pass is a single batched forward over the entire
    // prompt and is far slower than one decode step; folding it into the denominator was making the
    // rate look "very small". This matches the original Java timing, which captured start_time only
    // after the first Run_LLM (prefill) returned. Initialized here just for the empty-reply edge
    // (first token is a stop token => generated == 0 => rate is 0 regardless of this value).
    int64_t decode_start_ms = now_ms();
    last_flush_ms = decode_start_ms;
    bool timing_started = false;
    bool first_decode = true;
    int generated = 0;
    while (true) {
        ort_runtime_A->RunWithBinding(session_model_A, run_options_A, io_binding_A);
        if (!timing_started) {
            // Prefill just finished: start the decode clock here so prefill latency is excluded from
            // the token/s figure (the prompt forward pass is not part of decode throughput).
            decode_start_ms = now_ms();
            last_flush_ms = decode_start_ms;
            timing_started = true;
        }
        OrtValue** output_values = nullptr;
        size_t output_values_count = 0;
        ort_runtime_A->GetBoundOutputValues(io_binding_A, allocator, &output_values, &output_values_count);
        // max_logit_id is permanently bound to max_idx_buf, so the argmax token is already sitting in
        // its stable backing store; no per-step output lookup or rebinding is needed for it.
        token_id = argmax_id;
        const bool cancelled = g_cancel.load(std::memory_order_relaxed);
        if (!cancelled && !is_stop_token(token_id) && (response_count < single_chat_limit) && (history_len < max_seq_len)) {
            // Carry the freshly produced KV cache forward: rebind this step's KV outputs as the next
            // step's KV inputs. Only the dynamic-length KV pairs are rebound here.
            for (int i = 0; i < num_keys_values; i++) {
                ort_runtime_A->BindInput(io_binding_A, input_names_A[i], output_values[i]);
            }
            // Re-bind the KV outputs to fresh device buffers so the next run cannot overwrite the KV
            // tensors just bound as inputs. max_logit_id stays on its fixed max_idx_buf binding.
            bind_kv_outputs_to_device();
            // The buffers carried over from the previous step are no longer referenced as inputs;
            // free them, then keep this step's outputs alive until the next step rebinds past them.
            release_carryover();
            carryover_outputs = output_values;
            carryover_count = output_values_count;
            history_len += ids_len;
            if (first_decode) {
                // Prefill -> decode transition (done exactly once): input_ids becomes a fixed (1,1)
                // tensor from now on, so bind it ONCE to the shared decode buffer and never rebind it.
                attention_mask = 0;
                ids_len = 1;
                ort_runtime_A->BindInput(io_binding_A, input_names_A[num_keys_values], decode_ids_buf);
                first_decode = false;
            }
            // Feed the just-produced token back as the next input_ids by writing it into the shared
            // decode buffer's stable backing store (no rebind: the binding to decode_ids_buf holds).
            decode_input_id = token_id;
            save_max_logit_position[response_count] = token_id;
            response_count += 1;
            // Append the decoded word to the batch buffer (no per-token std::string return / JNI call),
            // then flush per the batching rule: every STREAM_BATCH tokens OR every STREAM_FLUSH_MS ms.
            stream_buf.append(get_output_words(token_id));
            tokens_since_flush += 1;
            if (tokens_since_flush >= STREAM_BATCH || (now_ms() - last_flush_ms) >= STREAM_FLUSH_MS) {
                flush_stream(env);
            }
        } else {
            // Stop token / per-chat limit / max_seq_len reached / cancellation: release this step's
            // outputs and finish the generation. (Identical teardown to the original stop branch.)
            for (size_t i = 0; i < output_values_count; i++) {
                ort_runtime_A->ReleaseValue(output_values[i]);
            }
            ort_runtime_A->AllocatorFree(allocator, output_values);
            release_carryover();
            generated = response_count;  // real tokens streamed, before end_bookkeeping adds the marker
            end_bookkeeping();
            break;
        }
    }
    // Deliver any remaining batched tail text so the final tokens are never dropped.
    flush_stream(env);

    // Native owns the timing now: return the final decode-rate line for the UI to append once.
    const int64_t elapsed_ms = now_ms() - decode_start_ms;
    const double rate = (elapsed_ms > 0) ? (1000.0 * static_cast<double>(generated) / static_cast<double>(elapsed_ms)) : 0.0;
    char stats[64];
    std::snprintf(stats, sizeof(stats), "\n\nDecode: %.4f token/s", rate);
    return env->NewStringUTF(stats);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_Stop_1LLM(JNIEnv *env, jclass clazz) {
    // Cooperative cancel: the in-flight decode loop polls g_cancel each iteration and stops within a
    // token or two. Lock-free and cheap; safe to call from the UI thread (Clear button / new query).
    g_cancel.store(true, std::memory_order_relaxed);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1A(JNIEnv *env, jobject clazz,
                                                            jobject asset_manager,
                                                            jint ep_type,
                                                            jboolean low_memory_mode)
{
    // ep_type: 0 = CPU (default), 1 = XNNPACK, 2 = QNN
    // Cache the C++ -> Java streaming callback ONCE (process-lifetime singleton). `clazz` is the
    // MainActivity instance; resolve its class to a global ref and look up the static onTokenStream
    // method id, so the per-token decode loop never pays a FindClass / GetStaticMethodID cost.
    // DeleteGlobalRef is intentionally omitted: this lives for the whole process (like the session).
    if (g_main_cls == nullptr) {
        jclass local_cls = env->GetObjectClass(clazz);
        if (local_cls != nullptr) {
            g_main_cls = static_cast<jclass>(env->NewGlobalRef(local_cls));
            env->DeleteLocalRef(local_cls);
            g_on_token = env->GetStaticMethodID(g_main_cls, "onTokenStream", "(Ljava/lang/String;)V");
        }
    }
    // One-time reservation for the streaming batch buffer (reused for the whole process; the decode
    // loop only appends/clears it, so the heap is never churned per token).
    stream_buf.reserve(256);
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

        // ==================== Run Options Configuration ====================
        // Memory arena shrinkage: Keep empty for performance; "cpu:0" for low memory usage.
        // Format: "device_0:device_id_0;device_1:device_id_1". Supported devices: "cpu", "gpu" (case sensitive).
        ort_runtime_A->AddRunConfigEntry(run_options_A, "memory.enable_memory_arena_shrinkage", "");
        // Disable EP synchronization: "1" to not synchronize EPs with CPU at end of session run.
        // Per default "0". For CUDA EP, omits triggering cudaStreamSynchronize on compute stream.
        ort_runtime_A->AddRunConfigEntry(run_options_A, "disable_synchronize_execution_providers", "1");
        // CUDA graph annotation ID: Use with enable_cuda_graph=true. "-1" disables cuda graph capture/replay.
        // "0" is reserved for internal use. Leave unset for default single cuda graph behavior.
        // ort_runtime_A->AddRunConfigEntry(run_options_A, "gpu_graph_id", "");

        // QNN-specific run options (auto-enabled when using QNN EP)
        if (ep_type == EP_QNN) {
            // QNN HTP performance mode: "burst", "balanced", "default", "high_performance", "high_power_saver",
            // "low_balanced", "extreme_power_saver", "low_power_saver", "power_saver", "sustained_high_performance"
            ort_runtime_A->AddRunConfigEntry(run_options_A, "qnn.htp_perf_mode", "burst");
            // QNN HTP performance mode applied AFTER each Run() returns. Kept at "burst" to avoid the
            // overhead of switching perf modes between the many per-token Run() calls of LLM decoding.
            // Set a lower mode (e.g. "balanced"/"power_saver") instead if you prefer to save power after inference.
            ort_runtime_A->AddRunConfigEntry(run_options_A, "qnn.htp_perf_mode_post_run", "burst");
            // QNN RPC control latency (microseconds)
            ort_runtime_A->AddRunConfigEntry(run_options_A, "qnn.rpc_control_latency", "100");
            // QNN LoRA Config File: Path to LoRA config file for applying LoRA in QNN context binary
            // ort_runtime_A->AddRunConfigEntry(run_options_A, "qnn.lora_config", "/path/to/lora_config.json");
        }

        // ==================== Basic Session Configuration ====================
        ort_runtime_A->DisableProfiling(session_options_A);
        ort_runtime_A->EnableCpuMemArena(session_options_A);
        ort_runtime_A->EnableMemPattern(session_options_A);
        ort_runtime_A->SetSessionExecutionMode(session_options_A, ORT_SEQUENTIAL);

        // ==================== Thread Configuration ====================
        // Inter-op threads: max threads used to run independent graph branches in parallel.
        ort_runtime_A->SetInterOpNumThreads(session_options_A, 6);
        // Dynamic block-sizing for intra-op parallelism. An N-iteration task is split into blocks of
        // ~N/(intra_op_threads * dynamic_block_base), shrinking as work drains for finer load-balancing.
        // "0" disables; any positive integer enables (higher value = finer granularity).
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.dynamic_block_base", "3");
        // Pin intra-op threads to logical CPUs. Format: "t0;t1;t2" where ';' separates threads and ',' lists
        // the CPUs a thread may run on (e.g. "1,3" = CPUs 1 and 3 cooperatively). The number of thread groups
        // MUST equal intra_op_num_threads - 1 (ORT never pins the caller's main thread).
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.intra_op_thread_affinities", "1,3;2,4;5,7");
        // Intra-op threads (parallelism within a single op). Heuristic here: dynamic_block_base + 1 = 4,
        // which matches the 3 affinity groups above (4 - 1 = 3).
        ort_runtime_A->SetIntraOpNumThreads(session_options_A, 4);

        // ==================== Thread Spinning Configuration ====================
        // AGGRESSIVE PERFORMANCE: keep worker threads hot so the frequent per-token Run() calls never pay a
        // thread wake-up cost. 1: spin before blocking (max performance); 0: block immediately (low power).
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.inter_op.allow_spinning", "1");        // 0 for low power
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.intra_op.allow_spinning", "1");        // 0 for low power
        // Spin duration (us) before a thread blocks. Intentionally LEFT UNSET: when unset, ORT uses its
        // iteration-count spin loop which "provides the best throughput" (ORT docs). Setting any positive value
        // switches to calibrated spinning that targets a shorter duration => lower throughput. Uncomment and set
        // e.g. "1000" only for power-sensitive builds.
        // ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.intra_op.spin_duration_us", "1000");
        // ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.inter_op.spin_duration_us", "1000");
        // Exponential-backoff cap for the spin loop. "1" = no backoff = densest spinning = lowest latency
        // (highest CPU/power). Use 4 (hybrid/E-core friendly) or 8 (desktop/server) to trade latency for power.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.intra_op.spin_backoff_max", "1");      // 1 = no backoff (max performance)
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.inter_op.spin_backoff_max", "1");      // 1 = no backoff (max performance)
        // Keep threads spinning after the last concurrent Run() returns (do NOT force-stop). 1 for low power.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.force_spinning_stop", "0");

        // ==================== Graph Optimization Configuration ====================
        ort_runtime_A->SetSessionGraphOptimizationLevel(session_options_A, ORT_ENABLE_ALL);
        // Graph optimization loop level: "0" disable, "1" loop depending on Level 4, "2" loop on Level 2+
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.graph_optimizations_loop_level", "2");
        // Keep empty for full optimization; "save" or "apply" for minimal build optimizations
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "optimization.minimal_build_optimizations", "");
        // Comma-separated optimizer names to disable.
        // For FP16 models (ORT_FP16), disable the FP16->FP32 cast/initializer optimizers so the graph stays in FP16.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "optimization.disable_specified_optimizers",
                                             ORT_FP16 ? "CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer" : "");
        // Maximum total output size in bytes for constant folding. Default "1073741824" (1 GB). "0" to disable limit.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "optimization.constant_folding_max_output_size_in_bytes", "0");
        // Enable GELU approximation: may change inference results. 0: disable, 1: enable
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "optimization.enable_gelu_approximation", "1");
        // Enable Cast chain elimination: may change inference results. 0: disable, 1: enable
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "optimization.enable_cast_chain_elimination", "1");

        // ==================== Model Format and External Data Configuration ====================
        // Load model format: Set to 'ORT' to load an ORT format model. Inferred from '.ort' extension otherwise.
        // ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.load_model_format", "ORT");
        // Save model format: Set to 'ORT' to save optimized model in ORT format.
        // ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.save_model_format", "ORT");
        // External initializers folder path: Set when loading model from memory with external initializers
        // ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.model_external_initializers_file_folder_path", "/path/to/external_data");
        // Optimized model external initializers file name: Used when serializing large optimized model
        // ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.optimized_model_external_initializers_file_name", "external_data.bin");
        // Minimum size for externalizing initializers (bytes) during serialization
        // ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.optimized_model_external_initializers_min_size_in_bytes", "1024");
        // Save pre-packed constant initializers to external file. "0": default (heap), "1": external file (memory-mapped)
        // ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.save_external_prepacked_constant_initializers", "0");

        // ==================== Prepacking and Memory Configuration ====================
        // Disable prepacking: 0 for enable, 1 for disable
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.disable_prepacking", "0");
        // Use the ORT-format model bytes in place instead of copying them (lowers peak memory during load).
        // Applies to ORT-format (.ort) models only; the supplied buffer must stay valid for the session lifetime.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.use_ort_model_bytes_directly", "1");
        // Point initializers directly at the ORT model bytes (no copy, lower peak memory). Requires
        // use_ort_model_bytes_directly=1 OR use_memory_mapped_ort_model=1, and the buffer must outlive the
        // session. "0" keeps a safe private copy. Note: ORT-format (.ort) models only; no-op for plain .onnx.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.use_ort_model_bytes_for_initializers", "0");
        // Use memory-mapped I/O to load ORT format model files (when loading from file path)
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.use_memory_mapped_ort_model", "1");    // 1 for memory-mapped loading
        // Use environment allocators to lower memory usage
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.use_env_allocators", "1");
        // Use device allocator for initializers to lower memory usage
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.use_device_allocator_for_initializers", "1");
        // Flush-to-zero and denormal-as-zero: may hurt model accuracy. 1 to enable.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.set_denormal_as_zero", "1");

        // ==================== Quantization Configuration ====================
        // Disable QDQ (QuantizeLinear-DeQuantizeLinear) fusion. 0: enable fusion, 1: disable
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.disable_quant_qdq", "0");
        // Prevent constant folding from folding DequantizeLinear nodes. 0: default, 1: preserve DQ nodes
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.disable_qdq_constant_folding", "0");
        // Disable Double QDQ remover. 0: not disable (remove middle 2 nodes), 1: disable
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.disable_double_qdq_remover", "0");
        // Enable removal of Q/DQ pairs after QDQ handling. 0: disable, 1: enable (may impact accuracy)
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.enable_quant_qdq_cleanup", "1");
        // QDQ is Int8 allowed for ARM platforms. 0: no, 1: yes
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.qdqisint8allowed", "1");
        // MatMulNBits accuracy level: 0:default, 1:FP32, 2:FP16, 3:BF16, 4:INT8.
        // AGGRESSIVE PERFORMANCE: force "4" (INT8 accumulation) - the fastest MLAS path on ARM dot-product
        // (SDOT/UDOT) hardware for 4-bit (MatMulNBits) weights. Switch to "2" (FP16) if INT8 hurts output quality.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.qdq_matmulnbits_accuracy_level", "4");
        // MatMulNBits block size. Positive: explicit (power-of-2, >= 16), "0": default 32, "-1": heuristic
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.qdq_matmulnbits_block_size", "0");
        // Enable DQ->MatMulNBits fusion: 0: disabled (default), 1: enabled
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.enable_dq_matmulnbits_fusion", "1");

        // ==================== Function Inlining Configuration ====================
        // Disable AOT function inlining. 0: enable (speed), 1: disable
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.disable_aot_function_inlining", "0");

        // ==================== MLAS Configuration ====================
        // Enable GEMM FastMath for ARM64 with bfloat16: accelerates FP32 GEMM via bf16 matmul. 0: disable, 1: enable.
        // AGGRESSIVE PERFORMANCE: "1" (faster FP32 GEMM; bf16 has fewer mantissa bits so a tiny accuracy trade-off).
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "mlas.enable_gemm_fastmath_arm64_bfloat16", "0");
        // Use LUT (Lookup Table) based GEMM for quantized models. 0: disable, 1: enable.
        // Left "0": LUT GEMM is hardware-dependent and can regress; benchmark before enabling on your target SoC.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "mlas.use_lut_gemm", "0");
        // Disable KleidiAI kernels. 0: use when available (default), 1: disable
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "mlas.disable_kleidiai", "0");

        // ==================== Model Validation Configuration ====================
        // Strict shape/type inference. 0: warnings with continue, 1: fail on inconsistencies
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.strict_shape_type_inference", "0");
        // Allow only released opsets. 0: may work with newer opsets, 1: fail on unreleased opsets
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.allow_released_opsets_only", "0");

        // ==================== Execution Provider Fallback Configuration ====================
        // Disable CPU EP fallback for unsupported nodes. 0: allow fallback (default), 1: fail if EP can't fully support
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.disable_cpu_ep_fallback", "0");

        // ==================== EP Context Configuration (for pre-compiled models) ====================
        // Enable EP context feature for compiled models. "0": disable (default), "1": enable
        // The dumped ONNX model with EP context can be used to avoid EP graph partitioning/compile overhead.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "ep.context_enable", "0");
        // EP context embed mode. "0": separate file (default), "1": embed in ONNX model
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "ep.context_embed_mode", "0");
        // EP context file path: Specify path for ONNX model with EP context. Default: original_file_name_ctx.onnx
        // ort_runtime_A->AddSessionConfigEntry(session_options_A, "ep.context_file_path", "/path/to/model_ctx.onnx");
        // EP context node name prefix: Make EPContext node names unique for merging multiple EPContext nodes
        // ort_runtime_A->AddSessionConfigEntry(session_options_A, "ep.context_node_name_prefix", "model_a_");
        // Share EP contexts across sessions. Enables sharing EP related resources.
        // ort_runtime_A->AddSessionConfigEntry(session_options_A, "ep.share_ep_contexts", "0");
        // Stop sharing EP contexts from then on
        // ort_runtime_A->AddSessionConfigEntry(session_options_A, "ep.stop_share_ep_contexts", "0");
        // EP context model external initializers file name: Used when generating EP context model
        // ort_runtime_A->AddSessionConfigEntry(session_options_A, "ep.context_model_external_initializers_file_name", "external_data.bin");
        // Enable weightless EP context nodes: EPContext nodes without internal weights. "0": disable (default), "1": enable
        // ort_runtime_A->AddSessionConfigEntry(session_options_A, "ep.enable_weightless_ep_context_nodes", "0");

        // ==================== Model Compilation Configuration ====================
        // Disable model compilation during session init. "0": allow (default), "1": fail if compilation required
        // If "1", session creation fails with ORT_MODEL_REQUIRES_COMPILATION if compilation is required.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.disable_model_compile", "0");
        // Fail on suboptimal compiled model. "0": allow with suboptimal perf (default), "1": fail to require recompile
        // Controls behavior when compiled model compatibility is SUPPORTED_PREFER_RECOMPILATION.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.fail_on_suboptimal_compiled_model", "0");
        // Compile-only session (internal flag, set by OrtCompileAPI::CompileModel). DO NOT set directly.
        // "0": normal session (default), "1": compile-only session
        // ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.compile_only", "0");

        // ==================== Dynamic EP Configuration ====================
        // Workload type: "Default" = Performance, "Efficient" = Save Power
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "ep.dynamic.workload_type", "Default");
        // QNN HTP performance mode (dynamic option)
        // ort_runtime_A->AddSessionConfigEntry(session_options_A, "ep.dynamic.qnn_htp_performance_mode", "default");

        // ==================== NNAPI EP Configuration ====================
        // NNAPI partitioning stop ops: comma-delimited list of op types to exclude. Empty to disable stop op exclusion.
        // ort_runtime_A->AddSessionConfigEntry(session_options_A, "ep.nnapi.partitioning_stop_ops", "");

        // ==================== Debug Configuration ====================
        // Debug layout transformation (for NNAPI, XNNPACK, QNN). "0": off (default), "1": enable
        // Dumps model after layout transformation steps for debugging.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.debug_layout_transformation", "0");
        // Record EP graph assignment info. "0": disabled (default), "1": enabled
        // When enabled, call Session_GetEpGraphAssignmentInfo() to retrieve assignment information.
        ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.record_ep_graph_assignment_info", "0");
        // Collect node memory stats to CSV file. Useful for estimating memory requirements.
        // Disable memory patterns when using this option for accurate measurements.
        // ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.collect_node_memory_stats_to_file", "/path/to/memory_stats.csv");
        // Node partition config file: Configuration for partitioning nodes among logic streams
        // ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.node_partition_config_file", "/path/to/partition_config.json");

        // ==================== Execution Provider Configuration ====================
        std::vector<const char *> option_keys = {};
        std::vector<const char *> option_values = {};

        switch (ep_type) {
            case EP_XNNPACK:
                // ==================== XNNPACK Execution Provider ====================
                // XNNPACK is optimized for ARM and x86 CPU inference with NHWC layout.
                // Supported keys:
                // - "intra_op_num_threads": Number of thread-pool size for XNNPACK.
                //   "0" = use session thread-pool size (default).
                option_keys.push_back("intra_op_num_threads");
                option_values.push_back("4");

                // Adjust session options for XNNPACK. XNNPACK runs its partitioned subgraph on its OWN thread
                // pool (sized by the "intra_op_num_threads" provider option above), so ORT's intra-op pool is
                // kept at a single thread to avoid oversubscribing the CPU.
                ort_runtime_A->SetInterOpNumThreads(session_options_A, 4);
                ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.intra_op.allow_spinning", "0");
                ort_runtime_A->AddSessionConfigEntry(session_options_A, "session.dynamic_block_base", "1");
                // NOTE: do NOT set "session.intra_op_thread_affinities" here. With intra_op_num_threads = 1,
                // ORT expects (1 - 1) = 0 affinity groups, so supplying any would fail session creation.
                ort_runtime_A->SetIntraOpNumThreads(session_options_A, 1);

                ort_runtime_A->SessionOptionsAppendExecutionProvider(session_options_A, "XNNPACK",
                    option_keys.data(), option_values.data(), option_keys.size());
                break;

            case EP_QNN:
                // ==================== QNN Execution Provider (Qualcomm NPU) ====================
                // QNN Backend Type: one of "cpu", "gpu", "htp", "saver", "ir".
                // Selects the QNN backend by type; ORT resolves it to the matching backend library
                // (e.g. "htp" -> libQnnHtp.so). Mutually exclusive with "backend_path".
                option_keys.push_back("backend_type");
                option_values.push_back("htp");

                // QNN Backend Path: File path to QNN backend library. Mutually exclusive with "backend_type".
                // option_keys.push_back("backend_path");
                // option_values.push_back("/path/to/libQnnHtp.so");

                // QNN Profiling Level: "off" (default), "basic", "detailed"
                option_keys.push_back("profiling_level");
                option_values.push_back("off");

                // QNN Profiling File Path: QNN profiling file path if ETW not enabled.
                // option_keys.push_back("profiling_file_path");
                // option_values.push_back("/path/to/profiling.json");

                // QNN HTP Performance Mode: "burst", "balanced", "default", "high_performance",
                // "high_power_saver", "low_balanced", "extreme_power_saver", "low_power_saver",
                // "power_saver", "sustained_high_performance"
                option_keys.push_back("htp_performance_mode");
                option_values.push_back("burst");

                // QNN RPC Control Latency (microseconds)
                option_keys.push_back("rpc_control_latency");
                option_values.push_back("100");

                // QNN VTCM Size in MB. "0" = not set (default)
                option_keys.push_back("vtcm_mb");
                option_values.push_back("0");

                // QNN Context Priority: "low", "normal" (default), "normal_high", "high".
                // AGGRESSIVE PERFORMANCE: "high" so the HTP scheduler prioritizes this context's workloads.
                option_keys.push_back("qnn_context_priority");
                option_values.push_back("high");

                // HTP Graph Finalization Optimization Mode:
                // "0" (default), "1" (faster prep, less optimal),
                // "2" (longer prep, more optimal), "3" (longest prep, most optimal).
                // AGGRESSIVE PERFORMANCE: "3" maximizes runtime (per-token) HTP throughput. It increases the
                // one-time graph-finalization/load time; cache it with EPContext (ep.context_enable) to amortize.
                option_keys.push_back("htp_graph_finalization_optimization_mode");
                option_values.push_back("3");

                // SoC Model Number. "0" = unknown (default). Refer to QNN SDK documentation for valid values.
                option_keys.push_back("soc_model");
                option_values.push_back("0");

                // HTP Architecture: Minimum HTP architecture for driver to select compatible QNN operators.
                // Available options: "0" (default/none), "68", "69", "73", "75", "81"
                option_keys.push_back("htp_arch");
                option_values.push_back("0");

                // Device ID for HTP architecture. "0" = default (for single device)
                option_keys.push_back("device_id");
                option_values.push_back("0");

                // Enable HTP FP16 Precision for FP32 models: "0" (FP32), "1" (FP16, default)
                option_keys.push_back("enable_htp_fp16_precision");
                option_values.push_back("1");

                // Offload Graph I/O Quantization to CPU EP: "0" (QNN handles), "1" (offload, default)
                option_keys.push_back("offload_graph_io_quantization");
                option_values.push_back("1");

                // Enable HTP Spill Fill Buffer: "0" (disabled, default), "1" (enabled)
                // Used while generating a context binary to reuse one spill/fill buffer across graphs.
                option_keys.push_back("enable_htp_spill_fill_buffer");
                option_values.push_back("0");

                // Enable HTP Weight Sharing: "0" (disabled, default), "1" (enabled)
                // Lets multiple graphs within one context binary share weights (context-binary generation only).
                // option_keys.push_back("enable_htp_weight_sharing");
                // option_values.push_back("0");

                // Enable HTP Shared Memory Allocator: "0" (disabled, default), "1" (enabled)
                // Requires libcdsprpc.so/dll to be available.
                option_keys.push_back("enable_htp_shared_memory_allocator");
                option_values.push_back("0");

                // Dump QNN Graph as JSON: "0" (disabled), "1" (enabled for debugging)
                // Each graph partition assigned to QNN EP is dumped to a separate file.
                option_keys.push_back("dump_json_qnn_graph");
                option_values.push_back("0");

                // JSON QNN Graph Directory: Directory to dump QNN JSON graphs.
                // If not specified, dumps in current working directory. Ignored if dump_json_qnn_graph is not set.
                // option_keys.push_back("json_qnn_graph_dir");
                // option_values.push_back("/path/to/json_graphs");

                // Dump QNN IR DLC: Use QnnIr backend to write .dlc files for debugging. Inference results will be incorrect!
                // "0" (disabled, default), "1" (enabled)
                // option_keys.push_back("dump_qnn_ir_dlc");
                // option_values.push_back("0");

                // Dump QNN IR DLC Directory: Directory to write QNN .dlc files. Default is current working directory.
                // option_keys.push_back("dump_qnn_ir_dlc_dir");
                // option_values.push_back("/path/to/dlc_files");

                // QNN IR Backend Path: File path to QnnIr backend library (used with dump_qnn_ir_dlc).
                // option_keys.push_back("qnn_ir_backend_path");
                // option_values.push_back("/path/to/libQnnIr.so");

                // QNN Saver Path: File path to QNN Saver backend library for replay/debugging.
                // Produces incorrect inference results and may alter partitioning. Use only for debugging.
                // option_keys.push_back("qnn_saver_path");
                // option_values.push_back("/path/to/libQnnSaver.so");

                // QNN UDO Op Packages: Format: "<op_type>:<op_package_path>:<interface>[:<target>],..."
                // option_keys.push_back("op_packages");
                // option_values.push_back("");

                // Skip QNN Version Check: "1" to allow different QNN version than compiled into ORT.
                // May cause crashes, inaccurate results, and poor performance. Use with caution!
                // option_keys.push_back("skip_qnn_version_check");
                // option_values.push_back("0");

                ort_runtime_A->SessionOptionsAppendExecutionProvider(session_options_A, "QNN",
                    option_keys.data(), option_values.data(), option_keys.size());
                break;

            case EP_CPU:
            default:
                // ==================== Default CPU Execution Provider ====================
                // CPU EP is always available and used as fallback.
                // No additional configuration needed - uses session thread settings defined above.
                break;
        }

        // Clear EP options after use
        option_keys.clear();
        option_values.clear();

       
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
            reinterpret_cast<void*>(&history_len), sizeof(int64_t),
            input_dims_A[num_keys_values_plus_1].data(), input_dims_A[num_keys_values_plus_1].size(), input_types_A[num_keys_values_plus_1],
            &input_tensors_A[num_keys_values_plus_1]);
    ort_runtime_A->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void*>(&ids_len), sizeof(int64_t),
            input_dims_A[num_keys_values_plus_2].data(), input_dims_A[num_keys_values_plus_2].size(), input_types_A[num_keys_values_plus_2],
            &input_tensors_A[num_keys_values_plus_2]);
    ort_runtime_A->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void*>(&attention_mask), sizeof(int8_t),
            input_dims_A[num_keys_values + 3].data(), input_dims_A[num_keys_values + 3].size(), input_types_A[num_keys_values + 3],
            &input_tensors_A[num_keys_values + 3]);
    for (int i = 0; i < num_layers; i++) {
        input_dims_A[i][4] = 0;
        ort_runtime_A->CreateTensorWithDataAsOrtValue(
                memory_info,
                reinterpret_cast<void*>(past_key_values_init.data()), 0,
                input_dims_A[i].data(), input_dims_A[i].size(), input_types_A[i],
                &input_tensors_kv_init_A[i]);
    }
    for (int i = num_layers; i < num_keys_values; i++) {
        input_dims_A[i][3] = 0;
        ort_runtime_A->CreateTensorWithDataAsOrtValue(
                memory_info,
                reinterpret_cast<void*>(past_key_values_init.data()), 0,
                input_dims_A[i].data(), input_dims_A[i].size(), input_types_A[i],
                &input_tensors_kv_init_A[i]);
    }
    // === Create the IOBinding and bind the persistent fixed-shape tensors once ===
    // history_len, ids_len and attention_mask each wrap a C++ variable, so RunWithBinding reads
    // their current value on every call; only the value changes between steps, never the binding.
    ort_runtime_A->CreateIoBinding(session_model_A, &io_binding_A);
    ort_runtime_A->BindInput(io_binding_A, input_names_A[num_keys_values_plus_1], input_tensors_A[num_keys_values_plus_1]);
    ort_runtime_A->BindInput(io_binding_A, input_names_A[num_keys_values_plus_2], input_tensors_A[num_keys_values_plus_2]);
    ort_runtime_A->BindInput(io_binding_A, input_names_A[num_keys_values + 3], input_tensors_A[num_keys_values + 3]);
    // Fixed (1,1) shared buffers for the decode token round-trip. Both the model's argmax output
    // (max_logit_id) and the decode-phase input_ids are static-shape, so each is bound ONCE here.
    // argmax_id / decode_input_id give stable backing storage; only their value changes per step.
    std::vector<int64_t> max_idx_dims = output_dims_A[num_keys_values];
    for (auto &d : max_idx_dims) { if (d < 1) { d = 1; } }
    ort_runtime_A->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void*>(&argmax_id), sizeof(int),
            max_idx_dims.data(), max_idx_dims.size(), output_types_A[num_keys_values],
            &max_idx_buf);
    // Establish the output binding in MODEL ORDER so GetBoundOutputValues returns values in the same
    // order as input_names_A: KV outputs [0, num_keys_values) first, then max_logit_id at index
    // num_keys_values. GetBoundOutputValues yields values in BIND order, so binding max_logit_id
    // first (as before) would shift every KV by one slot and corrupt the carried-forward KV cache.
    bind_kv_outputs_to_device();
    // Bind the argmax output to its fixed buffer LAST; it is never rebound for the session lifetime.
    ort_runtime_A->BindOutput(io_binding_A, output_names_A[num_keys_values], max_idx_buf);
    std::vector<int64_t> decode_ids_dims = input_dims_A[num_keys_values];
    for (auto &d : decode_ids_dims) { if (d < 1) { d = 1; } }
    ort_runtime_A->CreateTensorWithDataAsOrtValue(
            memory_info,
            reinterpret_cast<void*>(&decode_input_id), sizeof(int),
            decode_ids_dims.data(), decode_ids_dims.size(), input_types_A[num_keys_values],
            &decode_ids_buf);
    // decode_ids_buf is bound to input_ids only after prefill (in the decode transition), because
    // during prefill input_ids must carry the full (1, num_prefill) prompt.
    input_ids_buffer_size[1] = sizeof(int);
    for (int i = 2; i < max_seq_len; i++) {
        input_ids_buffer_size[i] = input_ids_buffer_size[i - 1] + sizeof(int);
    }
    return JNI_TRUE;
}
