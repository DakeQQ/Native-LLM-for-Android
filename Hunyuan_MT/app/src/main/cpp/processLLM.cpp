#include "configs.h"

// Single combined Hunyuan-MT graph (reference ModelRuntime style; one model in the pipeline).
static ModelRuntime& mLLM = getModel(LLM_Decoder);

// ===================== ONNX Runtime status / IOBinding helpers =====================
// Mirrors the reference processASR.cpp idioms (logOrtStatus / bindIn / bindOut / bindOutDevice /
// runBinding) adapted to this project's single session. ORT C-API calls return nullptr on success, so
// logOrtStatus is branch-free in the common case; on failure it logs the EP's message, releases the
// status and returns false so the caller can bail instead of running on garbage.
static inline bool logOrtStatus(const OrtApi* api, OrtStatus* status, const char* op) {
    if (status == nullptr) {
        return true;
    }
    LOGE("ORT: %s failed: %s", op, api->GetErrorMessage(status));
    api->ReleaseStatus(status);
    return false;
}

static inline bool bindIn(ModelRuntime& m, int idx, OrtValue* v) {
    return logOrtStatus(m.api, m.api->BindInput(m.binding, m.inputNames[idx], v), "BindInput");
}

static inline bool bindOut(ModelRuntime& m, int idx, OrtValue* v) {
    return logOrtStatus(m.api, m.api->BindOutput(m.binding, m.outputNames[idx], v), "BindOutput");
}

static inline bool bindOutDevice(ModelRuntime& m, int idx) {
    // ioMemoryInfo == memoryInfo on the CPU/XNNPACK paths (unchanged), or the QNN HTP shared-memory
    // allocator when active, keeping the kv_out -> kv_in recursion zero-copy on-device.
    return logOrtStatus(m.api, m.api->BindOutputToDevice(m.binding, m.outputNames[idx], m.ioMemoryInfo),
                        "BindOutputToDevice");
}

static inline bool runBinding(ModelRuntime& m) {
    return logOrtStatus(m.api, m.api->RunWithBinding(m.session, m.runOptions, m.binding), "RunWithBinding");
}

static inline void releaseOutputArray(ModelRuntime& m, OrtValue** values, size_t count) {
    if (values == nullptr) {
        return;
    }
    for (size_t i = 0; i < count; i++) {
        if (values[i] != nullptr) {
            m.api->ReleaseValue(values[i]);
        }
    }
    logOrtStatus(m.api, m.api->AllocatorFree(m.allocator, values), "AllocatorFree");
}

// Refresh all outputs in deterministic bind order for GetBoundOutputValues(): KV outputs first, then
// the fixed max_logit_id buffer. The KV outputs grow every step, so ORT must allocate fresh device
// buffers for them; rebinding the fixed argmax wrapper is cheap and keeps the output array order
// independent of ORT's replacement semantics.
static inline bool bindStepOutputs(ModelRuntime& m) {
    m.api->ClearBoundOutputs(m.binding);
    for (int i = 0; i < num_keys_values; i++) {
        if (!bindOutDevice(m, i)) {
            return false;
        }
    }
    return bindOut(m, maxLogitOutIdx, max_idx_buf);
}

// Release the OrtValues handed back by GetBoundOutputValues for the previous step once they are no
// longer bound as inputs (mirrors Python letting the old get_outputs() references drop).
static inline void releaseCarryover(ModelRuntime& m) {
    if (carryover_outputs != nullptr) {
        for (size_t i = 0; i < carryover_count; i++) {
            if (carryover_outputs[i] != nullptr) {
                m.api->ReleaseValue(carryover_outputs[i]);
            }
        }
        logOrtStatus(m.api, m.api->AllocatorFree(m.allocator, carryover_outputs), "AllocatorFree");
        carryover_outputs = nullptr;
        carryover_count = 0;
    }
}

// ===================== Token decode helpers =====================
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

// Lookup table mapping an ASCII byte to its hexadecimal value (0-15); non-hex bytes map to 0. Built
// once on first use (thread-safe static init) and reused to decode "<0xHH>" byte-fallback tokens
// without the per-token heap allocation of substr() or the heavier std::stoi() parse.
inline static const unsigned char* hex_value_lut() {
    static const struct HexLut {
        unsigned char t[256]{};
        HexLut() {
            for (unsigned char & i : t) i = 0;
            for (int c = '0'; c <= '9'; ++c) t[c] = static_cast<unsigned char>(c - '0');
            for (int c = 'a'; c <= 'f'; ++c) t[c] = static_cast<unsigned char>(c - 'a' + 10);
            for (int c = 'A'; c <= 'F'; ++c) t[c] = static_cast<unsigned char>(c - 'A' + 10);
        }
    } lut;
    return lut.t;
}

// Append the decoded text for one token id DIRECTLY into the reusable streaming batch buffer (no
// per-token std::string return value, no intermediate allocation). The token's raw vocab bytes are
// obtained zero-copy through decode_id() -- a const reference straight into the Tiktoken decoder table
// for Hunyuan (a thread-local scratch only for backends that synthesize text) -- so the single
// remaining copy is the unavoidable append into the batch we are accumulating.
inline static void append_output_words(std::string& out, int id) {
    static thread_local std::string scratch;
    const std::string& words = tokenizer->decode_id(id, scratch);
    const size_t start = out.size();
    // Byte-fallback tokens are emitted as the 6-char form "<0xHH>"; collapse them to the single raw
    // byte they represent. Parse the two hex digits via the lookup table (no substr/stoi allocation).
    if (words.length() == 6 && words[0] == '<' && words[5] == '>' && words[1] == '0' && words[2] == 'x') {
        const unsigned char* hex = hex_value_lut();
        out.push_back(static_cast<char>((hex[static_cast<unsigned char>(words[3])] << 4) |
                                        hex[static_cast<unsigned char>(words[4])]));
    } else {
        out.append(words);
    }
    // Fix only the just-appended region in place for modified-UTF8. correctUtfBytes walks to out's NUL
    // terminator -- exactly the bytes we appended -- and never rewinds before `start`, so earlier
    // batched tokens are untouched. (&out[start] == the NUL terminator when nothing was appended.)
    correctUtfBytes(&out[start]);
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
    // input_ids buffer maintenance via save_index / num_ids_per_chat / accumulate_num_ids). The former
    // `return env->NewStringUTF("END")` short-circuit is now a plain `return`.
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

// ===================== JNI entry points =====================
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Pre_1Process(JNIEnv* env, jobject clazz) {
    if (tokenizer != nullptr) {
        delete tokenizer;
        tokenizer = nullptr;
    }
    tokenizer = Tokenizer::createTokenizer(vocab_path);
    return tokenizer != nullptr ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MainActivity_Run_1LLM(JNIEnv* env, jclass clazz, jstring jquery) {
    // Non-reentrancy guard for the global, non-thread-safe inference state. The Java side joins the
    // previous LLMThread before launching a new one, so g_busy is expected to be false here; if it is
    // somehow still held, bail out safely rather than corrupting the in-flight generation's state.
    if (g_busy.exchange(true, std::memory_order_acq_rel)) {
        return env->NewStringUTF("");
    }
    // Scope-exit clear of g_busy on every return below. Safe under -fno-exceptions (normal returns only).
    struct BusyGuard { ~BusyGuard() { g_busy.store(false, std::memory_order_release); } } busy_guard;
    g_cancel.store(false, std::memory_order_relaxed);

    const OrtApi* api = mLLM.api;
    if (api == nullptr || mLLM.session == nullptr || mLLM.binding == nullptr || tokenizer == nullptr ||
        max_idx_buf == nullptr || decode_ids_buf == nullptr) {
        return env->NewStringUTF("");
    }

    // =============================== Prefill bookkeeping (runs once) ===============================
    // Not a chat model: every call starts a fresh context (clear_history), exactly as the original did.
    clear_history();  // It is not a chat model.
    const char* query = env->GetStringUTFChars(jquery, nullptr);
    std::vector<int> get_ids = tokenizer->encode(query);
    env->ReleaseStringUTFChars(jquery, query);
    ids_len = static_cast<int64_t>(get_ids.size());
    num_ids_per_chat[save_index] = static_cast<int>(ids_len);
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
    releaseCarryover(mLLM);
    // (Re)create the input_ids tensor for this prompt and bind it. history_len / ids_len /
    // attention_mask are bound once in Load_Models_A and updated in place (their OrtValues wrap the
    // C++ variables), so only input_ids needs rebinding when its sequence length changes.
    mLLM.inputDims[inputIdsIdx][1] = ids_len;
    if (mLLM.inputTensors[inputIdsIdx] != nullptr) {
        api->ReleaseValue(mLLM.inputTensors[inputIdsIdx]);
        mLLM.inputTensors[inputIdsIdx] = nullptr;
    }
    // input_ids is int32, so the (1, ids_len) tensor is ids_len * 4 == ids_len << 2 bytes (computed
    // inline; no lookup table). ids_len < max_seq_len is already guaranteed by the over-inputs guards.
    if (!logOrtStatus(api, api->CreateTensorWithDataAsOrtValue(
            mLLM.memoryInfo, reinterpret_cast<void*>(input_ids.data()),
            static_cast<size_t>(ids_len) << input_ids_elem_shift,
            mLLM.inputDims[inputIdsIdx].data(), mLLM.inputDims[inputIdsIdx].size(), mLLM.inputTypes[inputIdsIdx],
            &mLLM.inputTensors[inputIdsIdx]), "CreateTensorWithDataAsOrtValue(input_ids)")) {
        return env->NewStringUTF("");
    }
    if (!bindIn(mLLM, inputIdsIdx, mLLM.inputTensors[inputIdsIdx])) {
        return env->NewStringUTF("");
    }
    // Auto-bind the empty (zero-length) KV cache as the initial inputs for the prefill.
    bool bound_init_kv = true;
    for (int i = 0; i < num_keys_values; i++) {
        bound_init_kv = bindIn(mLLM, i, input_tensors_kv_init[i]) && bound_init_kv;
    }
    if (!bound_init_kv) {
        return env->NewStringUTF("");
    }
    // Let ORT allocate the dynamic-length KV cache outputs on-device while writing max_logit_id into
    // the reusable fixed buffer.
    if (!bindStepOutputs(mLLM)) {
        return env->NewStringUTF("");
    }

    // =============================== Decode loop (entirely in C++) ===============================
    // Iteration 0 is the prefill forward pass (attention_mask = 1, full prompt); every later iteration
    // is a single-token decode step. The whole reply is produced in ONE Run_LLM call and streamed to
    // the UI in batches via flush_stream(), instead of one JNI round-trip + UI rebind per token.
    stream_buf.clear();
    tokens_since_flush = 0;
    // Decode-rate clock. It is (re)started AFTER prefill's forward pass below, so the reported token/s
    // measures DECODE throughput only (prefill is a single batched forward over the whole prompt and is
    // far slower than one decode step; folding it in made the rate look "very small"). Initialized here
    // just for the empty-reply edge (first token is a stop token => generated == 0 => rate is 0).
    int64_t decode_start_ms = now_ms();
    last_flush_ms = decode_start_ms;
    bool timing_started = false;
    bool first_decode = true;
    int generated = 0;
    while (true) {
        if (!runBinding(mLLM)) {
            // Inference failed: finish like the stop path. No outputs were produced this iteration, so
            // there is nothing to fetch/release here; just drop the carry-over and run the bookkeeping.
            releaseCarryover(mLLM);
            generated = response_count;
            end_bookkeeping();
            break;
        }
        if (!timing_started) {
            // Prefill just finished: start the decode clock here so prefill latency is excluded from
            // the token/s figure (the prompt forward pass is not part of decode throughput).
            decode_start_ms = now_ms();
            last_flush_ms = decode_start_ms;
            timing_started = true;
        }
        OrtValue** output_values = nullptr;
        size_t output_values_count = 0;
        if (!logOrtStatus(api, api->GetBoundOutputValues(mLLM.binding, mLLM.allocator, &output_values, &output_values_count),
                          "GetBoundOutputValues") || output_values_count < static_cast<size_t>(num_keys_values)) {
            releaseOutputArray(mLLM, output_values, output_values_count);
            releaseCarryover(mLLM);
            generated = response_count;
            end_bookkeeping();
            break;
        }
        // max_logit_id is bound to the reusable max_idx_buf for this step, so the argmax token is
        // already sitting in stable backing storage; no per-step tensor-data lookup is needed.
        token_id = static_cast<int>(argmax_id);
        const bool cancelled = g_cancel.load(std::memory_order_relaxed);
        if (!cancelled && !is_stop_token(token_id) && (response_count < single_chat_limit) && (history_len < max_seq_len)) {
            // Carry the freshly produced KV cache forward: rebind this step's KV outputs as the next
            // step's KV inputs. Only the dynamic-length KV pairs are rebound here.
            bool bound_kv_inputs = true;
            for (int i = 0; i < num_keys_values; i++) {
                bound_kv_inputs = bindIn(mLLM, i, output_values[i]) && bound_kv_inputs;
            }
            if (!bound_kv_inputs) {
                releaseOutputArray(mLLM, output_values, output_values_count);
                releaseCarryover(mLLM);
                generated = response_count;
                end_bookkeeping();
                break;
            }
            // GetBoundOutputValues also returns a wrapper for the fixed max_logit_id binding. It is read
            // through max_idx_buf, not carried as KV, so release that wrapper immediately and keep only
            // [0, num_keys_values) alive for the next step's inputs.
            for (auto i = static_cast<size_t>(num_keys_values); i < output_values_count; i++) {
                if (output_values[i] != nullptr) {
                    api->ReleaseValue(output_values[i]);
                    output_values[i] = nullptr;
                }
            }
            const int64_t completed_ids_len = ids_len;
            if (first_decode) {
                // Prefill -> decode transition (done exactly once): input_ids becomes a fixed (1,1)
                // tensor from now on, so bind it ONCE to the shared decode buffer and never rebind it.
                attention_mask = 0;
                ids_len = 1;
                if (!bindIn(mLLM, inputIdsIdx, decode_ids_buf)) {
                    releaseOutputArray(mLLM, output_values, output_values_count);
                    releaseCarryover(mLLM);
                    generated = response_count;
                    end_bookkeeping();
                    break;
                }
                first_decode = false;
            }
            // Re-bind outputs to fresh device buffers so the next run cannot overwrite the KV tensors
            // just bound as inputs; max_logit_id is rebound last to preserve GetBoundOutputValues order.
            if (!bindStepOutputs(mLLM)) {
                releaseOutputArray(mLLM, output_values, output_values_count);
                releaseCarryover(mLLM);
                generated = response_count;
                end_bookkeeping();
                break;
            }
            // The buffers carried over from the previous step are no longer referenced as inputs; free
            // them, then keep this step's outputs alive until the next step rebinds past them.
            releaseCarryover(mLLM);
            carryover_outputs = output_values;
            carryover_count = static_cast<size_t>(num_keys_values);
            history_len += completed_ids_len;
            // Feed the just-produced token back as the next input_ids by writing it into the shared
            // decode buffer's stable backing store (no rebind: the binding to decode_ids_buf holds).
            decode_input_id = static_cast<int32_t>(token_id);
            save_max_logit_position[response_count] = token_id;
            response_count += 1;
            // Append the decoded word to the batch buffer (zero-copy decode, written straight into the
            // reused buffer), then flush per the batching rule: every STREAM_BATCH tokens OR every
            // STREAM_FLUSH_MS ms.
            append_output_words(stream_buf, token_id);
            tokens_since_flush += 1;
            if (tokens_since_flush >= STREAM_BATCH || (now_ms() - last_flush_ms) >= STREAM_FLUSH_MS) {
                flush_stream(env);
            }
        } else {
            // Stop token / per-chat limit / max_seq_len reached / cancellation: release this step's
            // outputs and finish the generation. (Identical teardown to the original stop branch.)
            releaseOutputArray(mLLM, output_values, output_values_count);
            releaseCarryover(mLLM);
            generated = response_count;  // real tokens streamed, before end_bookkeeping adds the marker
            end_bookkeeping();
            break;
        }
    }
    // Deliver any remaining batched tail text so the final tokens are never dropped.
    flush_stream(env);

    // Native owns the timing now: return the final decode-rate line for the UI to append once.
    // Absolute timestamps stay int64 (steady_clock epoch ms overflows int32), but their delta is only
    // the decode duration, so int32 is enough bits for elapsed_ms; token/s fits comfortably in a float.
    const auto elapsed_ms = static_cast<int32_t>(now_ms() - decode_start_ms);
    const float rate = (elapsed_ms > 0) ? (1000.0f * static_cast<float>(generated) / static_cast<float>(elapsed_ms)) : 0.0f;
    char stats[64];
    std::snprintf(stats, sizeof(stats), "\n\nDecode: %.4f token/s", rate);
    return env->NewStringUTF(stats);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_myapplication_MainActivity_Stop_1LLM(JNIEnv* env, jclass clazz) {
    // Cooperative cancel: the in-flight decode loop polls g_cancel each iteration and stops within a
    // token or two. Lock-free and cheap; safe to call from the UI thread (Clear button / new query).
    g_cancel.store(true, std::memory_order_relaxed);
}
