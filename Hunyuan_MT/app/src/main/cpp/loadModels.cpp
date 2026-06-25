#include "configs.h"

#include <sys/stat.h>
#include <cstdio>

// ===================== Shared environment =====================
// Hunyuan-MT runs a single session, but the reference keeps the OrtEnv in ONE process-wide, lazily
// created, never-released singleton so the env always outlives every session (the original port leaked
// a function-local OrtEnv to achieve the same lifetime). Created on the first loadModel().
static OrtEnv* gSharedEnv = nullptr;

static OrtEnv* getSharedEnv(const OrtApi* api) {
    if (gSharedEnv == nullptr) {
        OrtStatus* status = api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "myapplication", &gSharedEnv);
        if (status != nullptr) {
            LOGE("ORT: CreateEnv failed: %s", api->GetErrorMessage(status));
            api->ReleaseStatus(status);
            gSharedEnv = nullptr;
        }
    }
    return gSharedEnv;
}

// ===================== Internal helpers =====================

// True if a regular file exists at path with a non-trivial size (>10 bytes). Mirrors the Java
// Copy_from_Asset_to_Cache `outFile.length() > 10` guard used to decide whether a model is staged.
inline static bool cacheFileExists(const std::string& path) {
    struct stat st{};
    return (stat(path.c_str(), &st) == 0) && (st.st_size > 10);
}

inline static bool readAssetFully(AAsset* asset, std::vector<char>& out, const std::string& name) {
    if (asset == nullptr) {
        LOGE("Asset open failed: %s", name.c_str());
        return false;
    }
    const off_t fileSize = AAsset_getLength(asset);
    if (fileSize <= 0) {
        LOGE("Asset is empty: %s", name.c_str());
        return false;
    }
    out.resize(static_cast<size_t>(fileSize));
    size_t offset = 0;
    while (offset < out.size()) {
        const size_t remaining = out.size() - offset;
        const size_t chunk = remaining > (1u << 20) ? (1u << 20) : remaining;
        const int readBytes = AAsset_read(asset, out.data() + offset, chunk);
        if (readBytes <= 0) {
            LOGE("Asset read failed: %s", name.c_str());
            out.clear();
            return false;
        }
        offset += static_cast<size_t>(readBytes);
    }
    return true;
}

// Resolve the .onnx path to open for a path-based load, auto-adapting to single- vs two-part models.
// ORT resolves a model's *.data sibling from the SAME directory as the .onnx, so a two-part model must
// be opened from whichever directory holds BOTH files. Prefer a complete pair in the cache (staged by
// the Java Copy_from_Asset_to_Cache on launch), then in external storage; otherwise fall back to
// wherever the .onnx lives (single-file, or let ORT surface a precise error if a required *.data is
// truly absent).
inline static std::string resolveModelPath(const ModelRuntime& m) {
    const std::string cacheOnnx   = cache_path + m.fileName;
    const std::string storageOnnx = storage_path + m.fileName;
    if (!m.externalFileName.empty()) {
        if (cacheFileExists(cacheOnnx) && cacheFileExists(cache_path + m.externalFileName)) {
            return cacheOnnx;
        }
        if (cacheFileExists(storageOnnx) && cacheFileExists(storage_path + m.externalFileName)) {
            return storageOnnx;
        }
    }
    return cacheFileExists(cacheOnnx) ? cacheOnnx : storageOnnx;
}

// ORT C-API calls return nullptr on success; log + release + false on failure so callers can bail.
inline static bool ok_status(const OrtApi* api, OrtStatus* status, const char* op) {
    if (status == nullptr) {
        return true;
    }
    LOGE("ORT: %s failed: %s", op, api->GetErrorMessage(status));
    api->ReleaseStatus(status);
    return false;
}

inline static bool createTensorWithData(const OrtApi* api,
                                        const OrtMemoryInfo* memInfo,
                                        void* dataPtr,
                                        size_t sizeBytes,
                                        const std::vector<int64_t>& dims,
                                        ONNXTensorElementDataType type,
                                        OrtValue** out,
                                        const char* op) {
    return ok_status(api,
                     api->CreateTensorWithDataAsOrtValue(
                             memInfo, dataPtr, sizeBytes, dims.data(), dims.size(), type, out),
                     op);
}

inline static std::vector<int64_t> emptyKVDims(const std::vector<int64_t>& modelDims,
                                               size_t sequenceAxis) {
    std::vector<int64_t> dims = modelDims;
    for (int64_t& dim : dims) {
        if (dim < 1) {
            dim = 1;
        }
    }
    if (sequenceAxis < dims.size()) {
        dims[sequenceAxis] = 0;
    }
    return dims;
}

inline static bool validateDecoderIO(const ModelRuntime& m) {
    if (m.inputNames.size() <= static_cast<size_t>(attentionMaskIdx) ||
        m.outputNames.size() <= static_cast<size_t>(maxLogitOutIdx)) {
        LOGE("ORT: unexpected model I/O count: inputs=%zu outputs=%zu", m.inputNames.size(), m.outputNames.size());
        return false;
    }
    for (int i = 0; i < num_layers; i++) {
        if (m.inputDims[i].size() <= 4) {
            LOGE("ORT: key KV input %d has invalid rank %zu", i, m.inputDims[i].size());
            return false;
        }
    }
    for (int i = num_layers; i < num_keys_values; i++) {
        if (m.inputDims[i].size() <= 3) {
            LOGE("ORT: value KV input %d has invalid rank %zu", i, m.inputDims[i].size());
            return false;
        }
    }
    return true;
}

// ===================== I/O metadata extraction =====================
// Enumerate the session's inputs/outputs into the ModelRuntime, set up the default (CPU) allocator and
// memory info, and — when the QNN HTP shared-memory allocator is active — repoint the dynamic-output
// residency at it so the autoregressive KV cache stays zero-copy on-device across kv_out -> kv_in.
inline static bool extractIO(ModelRuntime& m, int epType) {
    if (!ok_status(m.api, m.api->GetAllocatorWithDefaultOptions(&m.allocator), "GetAllocatorWithDefaultOptions")) {
        return false;
    }
    OrtMemoryInfo* cpuMemInfo = nullptr;
    if (!ok_status(m.api, m.api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpuMemInfo),
                   "CreateCpuMemoryInfo") || cpuMemInfo == nullptr) {
        return false;
    }
    m.memoryInfo = cpuMemInfo;
    // Default the IOBinding residency for the dynamic KV-cache outputs to the CPU allocator/memInfo so
    // the CPU and XNNPACK paths are byte-for-byte unchanged.
    m.ioMemoryInfo = m.memoryInfo;
    m.ioAllocator = m.allocator;

    // QNN HTP zero-copy: when the HTP backend exposes its CPU<->HTP shared-memory allocator (EP option
    // enable_htp_shared_memory_allocator=1) repoint the dynamic-output residency at it so the KV cache
    // stays resident on the accelerator across the kv_out -> kv_in recursion (no per-step transfer).
    // The shared-memory memInfo name must match what the QNN EP registers ("QnnHtpShared"). On any
    // failure (older ORT/QNN, option rejected) we keep the CPU default: correctness preserved.
    if (epType == EP_QNN) {
        OrtMemoryInfo* shared_info = nullptr;
        if (ok_status(m.api, m.api->CreateMemoryInfo("QnnHtpShared", OrtDeviceAllocator, 0, OrtMemTypeDefault, &shared_info),
                      "CreateMemoryInfo(QnnHtpShared)") && shared_info != nullptr) {
            OrtAllocator* dev_alloc = nullptr;
            if (ok_status(m.api, m.api->CreateAllocator(m.session, shared_info, &dev_alloc),
                          "CreateAllocator(QnnHtpShared)") && dev_alloc != nullptr) {
                m.ioAllocator = dev_alloc;   // session-scoped; freed with the session
                if (ok_status(m.api, m.api->AllocatorGetInfo(dev_alloc, &m.ioMemoryInfo),
                              "AllocatorGetInfo(QnnHtpShared)")) {
                    LOGI("QNN HTP shared-memory allocator active: KV cache is zero-copy on-device");
                } else {
                    m.ioAllocator = m.allocator;
                    m.ioMemoryInfo = m.memoryInfo;
                }
            }
            m.api->ReleaseMemoryInfo(shared_info);
        }
    }

    size_t inCount = 0;
    if (!ok_status(m.api, m.api->SessionGetInputCount(m.session, &inCount), "SessionGetInputCount")) {
        return false;
    }
    m.inputNames.resize(inCount);
    m.inputDims.resize(inCount);
    m.inputTypes.resize(inCount);
    m.inputTensors.assign(inCount, nullptr);
    for (size_t i = 0; i < inCount; i++) {
        char* name = nullptr;
        OrtTypeInfo* typeInfo = nullptr;
        const OrtTensorTypeAndShapeInfo* tensorInfo = nullptr;
        ONNXTensorElementDataType type;
        size_t dimensions = 0;
        if (!ok_status(m.api, m.api->SessionGetInputName(m.session, i, m.allocator, &name), "SessionGetInputName")) {
            return false;
        }
        m.inputNames[i] = name;
        if (!ok_status(m.api, m.api->SessionGetInputTypeInfo(m.session, i, &typeInfo), "SessionGetInputTypeInfo") ||
            !ok_status(m.api, m.api->CastTypeInfoToTensorInfo(typeInfo, &tensorInfo), "CastTypeInfoToTensorInfo") ||
            tensorInfo == nullptr ||
            !ok_status(m.api, m.api->GetTensorElementType(tensorInfo, &type), "GetTensorElementType")) {
            if (typeInfo) {
                m.api->ReleaseTypeInfo(typeInfo);
            }
            return false;
        }
        m.inputTypes[i] = type;
        if (!ok_status(m.api, m.api->GetDimensionsCount(tensorInfo, &dimensions), "GetDimensionsCount")) {
            m.api->ReleaseTypeInfo(typeInfo);
            return false;
        }
        m.inputDims[i].resize(dimensions);
        if (!ok_status(m.api, m.api->GetDimensions(tensorInfo, m.inputDims[i].data(), dimensions), "GetDimensions")) {
            m.api->ReleaseTypeInfo(typeInfo);
            return false;
        }
        if (typeInfo) {
            m.api->ReleaseTypeInfo(typeInfo);
        }
    }

    size_t outCount = 0;
    if (!ok_status(m.api, m.api->SessionGetOutputCount(m.session, &outCount), "SessionGetOutputCount")) {
        return false;
    }
    m.outputNames.resize(outCount);
    m.outputDims.resize(outCount);
    m.outputTypes.resize(outCount);
    for (size_t i = 0; i < outCount; i++) {
        char* name = nullptr;
        OrtTypeInfo* typeInfo = nullptr;
        const OrtTensorTypeAndShapeInfo* tensorInfo = nullptr;
        ONNXTensorElementDataType type;
        size_t dimensions = 0;
        if (!ok_status(m.api, m.api->SessionGetOutputName(m.session, i, m.allocator, &name), "SessionGetOutputName")) {
            return false;
        }
        m.outputNames[i] = name;
        if (!ok_status(m.api, m.api->SessionGetOutputTypeInfo(m.session, i, &typeInfo), "SessionGetOutputTypeInfo") ||
            !ok_status(m.api, m.api->CastTypeInfoToTensorInfo(typeInfo, &tensorInfo), "CastTypeInfoToTensorInfo") ||
            tensorInfo == nullptr ||
            !ok_status(m.api, m.api->GetTensorElementType(tensorInfo, &type), "GetTensorElementType")) {
            if (typeInfo) {
                m.api->ReleaseTypeInfo(typeInfo);
            }
            return false;
        }
        m.outputTypes[i] = type;
        if (!ok_status(m.api, m.api->GetDimensionsCount(tensorInfo, &dimensions), "GetDimensionsCount")) {
            m.api->ReleaseTypeInfo(typeInfo);
            return false;
        }
        m.outputDims[i].resize(dimensions);
        if (!ok_status(m.api, m.api->GetDimensions(tensorInfo, m.outputDims[i].data(), dimensions), "GetDimensions")) {
            m.api->ReleaseTypeInfo(typeInfo);
            return false;
        }
        if (typeInfo) {
            m.api->ReleaseTypeInfo(typeInfo);
        }
    }
    return true;
}

// ===================== Core model loading =====================
// Builds the session options (full original Load_Models_A config block), selects the execution
// provider, creates the session from an asset buffer (non-low-memory) or a cache/storage path
// (low-memory), and extracts the I/O metadata into the ModelRuntime.
bool loadModel(ModelId id, JNIEnv* env, jobject assetManager, int epType, bool lowMemoryMode) {
    ModelRuntime& m = getModel(id);
    m.id = id;
    m.fileName = kModelFileNames[id];
    if (kModelExternalFileNames[id] && kModelExternalFileNames[id][0] != '\0') {
        m.externalFileName = kModelExternalFileNames[id];
    }
    m.api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (m.session != nullptr) {
        LOGI("Model %d already loaded, skipping reload", static_cast<int>(id));
        return true;
    }
    m.env = getSharedEnv(m.api);
    if (m.env == nullptr) {
        return false;
    }

    OrtStatus* status = nullptr;
    OrtSessionOptions* session_options = nullptr;
    {
        std::vector<char> fileBuffer;
        std::vector<char> fileBuffer_external;
        off_t fileSize = 0;
        bool use_storage_path = false;
        if (!lowMemoryMode) {
            if (assetManager != nullptr) {
                AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
                if (mgr == nullptr) {
                    LOGE("AAssetManager_fromJava failed");
                    return false;
                }
                AAsset* asset = AAssetManager_open(mgr, m.fileName.c_str(), AASSET_MODE_BUFFER);
                if (!readAssetFully(asset, fileBuffer, m.fileName)) {
                    if (asset != nullptr) {
                        AAsset_close(asset);
                    }
                    return false;
                }
                fileSize = static_cast<off_t>(fileBuffer.size());
                AAsset_close(asset);
                if (!m.externalFileName.empty()) {
                    AAsset* asset_ex = AAssetManager_open(mgr, m.externalFileName.c_str(), AASSET_MODE_BUFFER);
                    if (asset_ex != nullptr) {
                        // Load external data via AAsset_read. For models with multiple external files,
                        // load additional siblings here as needed.
                        if (!readAssetFully(asset_ex, fileBuffer_external, m.externalFileName)) {
                            AAsset_close(asset_ex);
                            return false;
                        }
                        AAsset_close(asset_ex);
                    } else {
                        LOGI("External data asset '%s' not found; loading model as single-file", m.externalFileName.c_str());
                    }
                }
            } else {
                use_storage_path = true;
                lowMemoryMode = true;
            }
        }

        if (!ok_status(m.api, m.api->CreateSessionOptions(&session_options), "CreateSessionOptions") ||
            !ok_status(m.api, m.api->CreateRunOptions(&m.runOptions), "CreateRunOptions")) {
            if (session_options != nullptr) {
                m.api->ReleaseSessionOptions(session_options);
            }
            return false;
        }

        bool configOk = true;
        auto check = [&](OrtStatus* st, const char* op) {
            configOk = ok_status(m.api, st, op) && configOk;
        };
        auto addRunConfig = [&](const char* key, const char* value) {
            check(m.api->AddRunConfigEntry(m.runOptions, key, value), key);
        };
        auto addSessionConfig = [&](const char* key, const char* value) {
            check(m.api->AddSessionConfigEntry(session_options, key, value), key);
        };

        // ==================== Run Options Configuration ====================
        // Memory arena shrinkage: empty for performance; "cpu:0" for low memory usage.
        addRunConfig("memory.enable_memory_arena_shrinkage", "");
        // "1": do not synchronize EPs with CPU at the end of each Run() (lower per-call overhead).
        addRunConfig("disable_synchronize_execution_providers", "1");
        // QNN-specific run options (auto-enabled when using QNN EP)
        if (epType == EP_QNN) {
            addRunConfig("qnn.htp_perf_mode", "burst");
            addRunConfig("qnn.htp_perf_mode_post_run", "burst");
            addRunConfig("qnn.rpc_control_latency", "100");
        }

        // ==================== Basic Session Configuration ====================
        check(m.api->DisableProfiling(session_options), "DisableProfiling");
        check(m.api->EnableCpuMemArena(session_options), "EnableCpuMemArena");
        check(m.api->EnableMemPattern(session_options), "EnableMemPattern");
        check(m.api->SetSessionExecutionMode(session_options, ORT_SEQUENTIAL), "SetSessionExecutionMode");

        // ==================== Thread Configuration ====================
        // Inter-op threads: max threads used to run independent graph branches in parallel.
        check(m.api->SetInterOpNumThreads(session_options, 6), "SetInterOpNumThreads");
        // Dynamic block-sizing for intra-op parallelism (higher value = finer granularity; "0" disables).
        addSessionConfig("session.dynamic_block_base", "3");
        // Pin intra-op threads to logical CPUs. The number of ';'-groups MUST equal intra_op_num_threads - 1
        // (ORT never pins the caller's thread). NOT valid for XNNPACK (intra = 1 -> 0 groups), so it is gated.
        if (epType != EP_XNNPACK) {
            addSessionConfig("session.intra_op_thread_affinities", "1,3;2,4;5,7");
        }
        check(m.api->SetIntraOpNumThreads(session_options, 4), "SetIntraOpNumThreads");

        // ==================== Thread Spinning Configuration ====================
        // Keep worker threads hot so the frequent per-token Run() calls never pay a wake-up cost.
        addSessionConfig("session.inter_op.allow_spinning", "1");
        addSessionConfig("session.intra_op.allow_spinning", "1");
        // spin_duration_us intentionally LEFT UNSET (ORT's iteration-count spin loop gives best throughput).
        addSessionConfig("session.intra_op.spin_backoff_max", "1");
        addSessionConfig("session.inter_op.spin_backoff_max", "1");
        addSessionConfig("session.force_spinning_stop", "0");

        // ==================== Graph Optimization Configuration ====================
        check(m.api->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL), "SetSessionGraphOptimizationLevel");
        addSessionConfig("session.graph_optimizations_loop_level", "2");
        addSessionConfig("optimization.minimal_build_optimizations", "");
        // For FP16 models, disable the FP16->FP32 cast/initializer optimizers so the graph stays in FP16.
        addSessionConfig("optimization.disable_specified_optimizers",
                         ORT_FP16 ? "CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer" : "");
        addSessionConfig("optimization.constant_folding_max_output_size_in_bytes", "0");
        addSessionConfig("optimization.enable_gelu_approximation", "1");
        addSessionConfig("optimization.enable_cast_chain_elimination", "1");

        // ==================== Prepacking and Memory Configuration ====================
        addSessionConfig("session.disable_prepacking", "0");
        addSessionConfig("session.use_ort_model_bytes_directly", "1");
        addSessionConfig("session.use_ort_model_bytes_for_initializers", "0");
        addSessionConfig("session.use_memory_mapped_ort_model", "1");
        addSessionConfig("session.use_env_allocators", "1");
        addSessionConfig("session.use_device_allocator_for_initializers", "1");
        addSessionConfig("session.set_denormal_as_zero", "1");

        // ==================== Quantization Configuration ====================
        addSessionConfig("session.disable_quant_qdq", "0");
        addSessionConfig("session.disable_qdq_constant_folding", "0");
        addSessionConfig("session.disable_double_qdq_remover", "0");
        addSessionConfig("session.enable_quant_qdq_cleanup", "1");
        addSessionConfig("session.qdqisint8allowed", "1");
        // MatMulNBits accuracy level: "4" (INT8 accumulation) = fastest MLAS path on ARM SDOT/UDOT for
        // 4-bit weights. Switch to "2" (FP16) if INT8 hurts output quality.
        addSessionConfig("session.qdq_matmulnbits_accuracy_level", "4");
        addSessionConfig("session.qdq_matmulnbits_block_size", "16");
        addSessionConfig("session.enable_dq_matmulnbits_fusion", "1");

        // ==================== Function Inlining Configuration ====================
        addSessionConfig("session.disable_aot_function_inlining", "0");

        // ==================== MLAS Configuration ====================
        addSessionConfig("mlas.enable_gemm_fastmath_arm64_bfloat16", "0");
        addSessionConfig("mlas.use_lut_gemm", "0");
        addSessionConfig("mlas.disable_kleidiai", "0");

        // ==================== Model Validation Configuration ====================
        addSessionConfig("session.strict_shape_type_inference", "0");
        addSessionConfig("session.allow_released_opsets_only", "0");

        // ==================== Execution Provider Fallback Configuration ====================
        addSessionConfig("session.disable_cpu_ep_fallback", "0");

        // ==================== EP Context Configuration (pre-compiled models) ====================
        addSessionConfig("ep.context_enable", "0");
        addSessionConfig("ep.context_embed_mode", "0");

        // ==================== Model Compilation Configuration ====================
        addSessionConfig("session.disable_model_compile", "0");
        addSessionConfig("session.fail_on_suboptimal_compiled_model", "0");

        // ==================== Dynamic EP Configuration ====================
        addSessionConfig("ep.dynamic.workload_type", "Default");
        if (epType == EP_QNN) {
            addSessionConfig("ep.dynamic.qnn_htp_performance_mode", "burst");
        }

        // ==================== Debug Configuration ====================
        addSessionConfig("session.debug_layout_transformation", "0");
        addSessionConfig("session.record_ep_graph_assignment_info", "0");

        // ==================== Execution Provider Configuration ====================
        std::vector<const char*> option_keys;
        std::vector<const char*> option_values;
        switch (epType) {
            case EP_XNNPACK:
                // XNNPACK runs its partitioned subgraph on its OWN thread pool (sized by this provider
                // option), so ORT's intra-op pool is kept at a single thread to avoid oversubscription.
                option_keys.push_back("intra_op_num_threads");
                option_values.push_back("4");
                check(m.api->SetInterOpNumThreads(session_options, 4), "SetInterOpNumThreads(XNNPACK)");
                addSessionConfig("session.intra_op.allow_spinning", "0");
                addSessionConfig("session.dynamic_block_base", "1");
                // With intra_op_num_threads = 1 ORT expects (1 - 1) = 0 affinity groups, so none are set.
                check(m.api->SetIntraOpNumThreads(session_options, 1), "SetIntraOpNumThreads(XNNPACK)");
                check(m.api->SessionOptionsAppendExecutionProvider(session_options, "XNNPACK",
                    option_keys.data(), option_values.data(), option_keys.size()), "SessionOptionsAppendExecutionProvider(XNNPACK)");
                break;

            case EP_QNN:
                // ==================== QNN Execution Provider (Qualcomm NPU) ====================
                option_keys.push_back("backend_type");                                 option_values.push_back("htp");
                option_keys.push_back("profiling_level");                              option_values.push_back("off");
                option_keys.push_back("htp_performance_mode");                         option_values.push_back("burst");
                option_keys.push_back("rpc_control_latency");                          option_values.push_back("100");
                option_keys.push_back("vtcm_mb");                                      option_values.push_back("0");
                option_keys.push_back("qnn_context_priority");                         option_values.push_back("high");
                option_keys.push_back("htp_graph_finalization_optimization_mode");     option_values.push_back("3");
                option_keys.push_back("soc_model");                                    option_values.push_back("0");
                option_keys.push_back("htp_arch");                                     option_values.push_back("0");
                option_keys.push_back("device_id");                                    option_values.push_back("0");
                option_keys.push_back("enable_htp_fp16_precision");                    option_values.push_back("1");
                option_keys.push_back("offload_graph_io_quantization");                option_values.push_back("1");
                option_keys.push_back("enable_htp_spill_fill_buffer");                 option_values.push_back("0");
                // enable_htp_shared_memory_allocator = "1": expose the CPU<->HTP shared-memory allocator
                // so extractIO can bind the KV cache into it for zero-copy device residency.
                option_keys.push_back("enable_htp_shared_memory_allocator");           option_values.push_back("1");
                option_keys.push_back("dump_json_qnn_graph");                          option_values.push_back("0");
                check(m.api->SessionOptionsAppendExecutionProvider(session_options, "QNN",
                    option_keys.data(), option_values.data(), option_keys.size()), "SessionOptionsAppendExecutionProvider(QNN)");
                break;

            case EP_CPU:
            default:
                // CPU EP is always available; no extra configuration needed.
                break;
        }
        option_keys.clear();
        option_values.clear();

        if (!configOk) {
            m.api->ReleaseSessionOptions(session_options);
            return false;
        }

        // ==================== Create the session ====================
        if (lowMemoryMode) {
            const std::string path = use_storage_path ? (storage_path + m.fileName) : resolveModelPath(m);
            status = m.api->CreateSession(m.env, path.c_str(), session_options, &m.session);
        } else {
            if (!fileBuffer_external.empty()) {
                const char* external_file_names[]   = {m.externalFileName.c_str()};
                char*       external_file_buffers[] = {fileBuffer_external.data()};
                size_t      external_file_sizes[]   = {fileBuffer_external.size()};
                if (!ok_status(m.api, m.api->AddExternalInitializersFromFilesInMemory(
                        session_options, external_file_names, external_file_buffers, external_file_sizes, 1),
                        "AddExternalInitializersFromFilesInMemory")) {
                    m.api->ReleaseSessionOptions(session_options);
                    return false;
                }
            }
            status = m.api->CreateSessionFromArray(m.env, fileBuffer.data(), fileSize, session_options, &m.session);
        }
    }
    m.api->ReleaseSessionOptions(session_options);
    if (!ok_status(m.api, status, "CreateSession")) {
        return false;
    }
    return extractIO(m, epType);
}

// ===================== JNI: load + one-time IOBinding setup =====================
extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_myapplication_MainActivity_Load_1Models_1A(JNIEnv* env, jobject clazz,
                                                            jobject asset_manager,
                                                            jint ep_type,
                                                            jboolean low_memory_mode) {
    // ep_type: 0 = CPU (default), 1 = XNNPACK, 2 = QNN
    // Cache the C++ -> Java streaming callback ONCE (process-lifetime singleton). `clazz` is the
    // MainActivity instance; resolve its class to a global ref and look up the static onTokenStream
    // method id so the per-token decode loop never pays a FindClass / GetStaticMethodID cost.
    // DeleteGlobalRef is intentionally omitted: this lives for the whole process (like the session).
    if (g_main_cls == nullptr) {
        jclass local_cls = env->GetObjectClass(clazz);
        if (local_cls != nullptr) {
            g_main_cls = static_cast<jclass>(env->NewGlobalRef(local_cls));
            env->DeleteLocalRef(local_cls);
            g_on_token = env->GetStaticMethodID(g_main_cls, "onTokenStream", "(Ljava/lang/String;)V");
        }
    }
    // One-time reservation for the streaming batch buffer (reused for the whole process).
    stream_buf.reserve(256);

    if (!loadModel(LLM_Decoder, env, asset_manager, ep_type, low_memory_mode == JNI_TRUE)) {
        return JNI_FALSE;
    }
    ModelRuntime& m = getModel(LLM_Decoder);
    const OrtApi* api = m.api;

    if (m.ioPrepared && m.binding != nullptr && max_idx_buf != nullptr && decode_ids_buf != nullptr) {
        return JNI_TRUE;
    }

    if (!validateDecoderIO(m)) {
        return JNI_FALSE;
    }

    // === Create the persistent fixed-shape scalar tensors (wrap C++ variables; value updated in place) ===
    if (!createTensorWithData(api, m.memoryInfo, reinterpret_cast<void*>(&history_len), sizeof(int64_t),
                              m.inputDims[historyLenIdx], m.inputTypes[historyLenIdx],
                              &m.inputTensors[historyLenIdx], "CreateTensorWithDataAsOrtValue(history_len)") ||
        !createTensorWithData(api, m.memoryInfo, reinterpret_cast<void*>(&ids_len), sizeof(int64_t),
                              m.inputDims[idsLenIdx], m.inputTypes[idsLenIdx],
                              &m.inputTensors[idsLenIdx], "CreateTensorWithDataAsOrtValue(ids_len)") ||
        !createTensorWithData(api, m.memoryInfo, reinterpret_cast<void*>(&attention_mask), sizeof(int8_t),
                              m.inputDims[attentionMaskIdx], m.inputTypes[attentionMaskIdx],
                              &m.inputTensors[attentionMaskIdx], "CreateTensorWithDataAsOrtValue(attention_mask)")) {
        return JNI_FALSE;
    }

    // === Empty (zero-length) prefill KV inputs ===
    // Live on ioMemoryInfo -- the CPU arena by default, or the QNN HTP shared memory when active -- so
    // the very first prefill concat (empty_kv + new_kv) starts on the same device the model writes its
    // KV outputs to (no host staging). Keys carry the dynamic seq axis at dim 4, values at dim 3.
    for (int i = 0; i < num_layers; i++) {
        std::vector<int64_t> kvDims = emptyKVDims(m.inputDims[i], 4);
        if (!createTensorWithData(api, m.ioMemoryInfo, reinterpret_cast<void*>(past_key_values_init.data()), 0,
                                  kvDims, m.inputTypes[i], &input_tensors_kv_init[i],
                                  "CreateTensorWithDataAsOrtValue(empty_key_kv)")) {
            return JNI_FALSE;
        }
    }
    for (int i = num_layers; i < num_keys_values; i++) {
        std::vector<int64_t> kvDims = emptyKVDims(m.inputDims[i], 3);
        if (!createTensorWithData(api, m.ioMemoryInfo, reinterpret_cast<void*>(past_key_values_init.data()), 0,
                                  kvDims, m.inputTypes[i], &input_tensors_kv_init[i],
                                  "CreateTensorWithDataAsOrtValue(empty_value_kv)")) {
            return JNI_FALSE;
        }
    }

    // === Create the IOBinding and bind the persistent fixed-shape tensors once ===
    if (!ok_status(api, api->CreateIoBinding(m.session, &m.binding), "CreateIoBinding")) {
        return JNI_FALSE;
    }
    if (!ok_status(api, api->BindInput(m.binding, m.inputNames[historyLenIdx],    m.inputTensors[historyLenIdx]),    "BindInput(history_len)") ||
        !ok_status(api, api->BindInput(m.binding, m.inputNames[idsLenIdx],        m.inputTensors[idsLenIdx]),        "BindInput(ids_len)") ||
        !ok_status(api, api->BindInput(m.binding, m.inputNames[attentionMaskIdx], m.inputTensors[attentionMaskIdx]), "BindInput(attention_mask)")) {
        return JNI_FALSE;
    }

    // Fixed (1,1) shared buffers for the decode token round-trip. argmax_id / decode_input_id give the
    // stable backing storage; only their value changes per step. The output binding is refreshed in
    // processLLM.cpp together with the dynamic KV outputs to keep GetBoundOutputValues order stable.
    std::vector<int64_t> max_idx_dims = m.outputDims[maxLogitOutIdx];
    for (auto& d : max_idx_dims) { if (d < 1) { d = 1; } }
    if (!ok_status(api, api->CreateTensorWithDataAsOrtValue(
            m.memoryInfo, reinterpret_cast<void*>(&argmax_id), sizeof(int32_t),
            max_idx_dims.data(), max_idx_dims.size(), m.outputTypes[maxLogitOutIdx], &max_idx_buf),
            "CreateTensorWithDataAsOrtValue(max_logit_id)")) {
        return JNI_FALSE;
    }
    // Establish the initial output binding in MODEL ORDER so GetBoundOutputValues returns values in the
    // same order as outputNames: KV outputs [0, num_keys_values) first, then max_logit_id. The decode
    // loop refreshes this same order whenever it asks ORT for fresh dynamic KV buffers.
    for (int i = 0; i < num_keys_values; i++) {
        if (!ok_status(api, api->BindOutputToDevice(m.binding, m.outputNames[i], m.ioMemoryInfo), "BindOutputToDevice(KV)")) {
            return JNI_FALSE;
        }
    }
    if (!ok_status(api, api->BindOutput(m.binding, m.outputNames[maxLogitOutIdx], max_idx_buf), "BindOutput(max_logit_id)")) {
        return JNI_FALSE;
    }

    std::vector<int64_t> decode_ids_dims = m.inputDims[inputIdsIdx];
    for (auto& d : decode_ids_dims) { if (d < 1) { d = 1; } }
    if (!ok_status(api, api->CreateTensorWithDataAsOrtValue(
            m.memoryInfo, reinterpret_cast<void*>(&decode_input_id), sizeof(int32_t),
            decode_ids_dims.data(), decode_ids_dims.size(), m.inputTypes[inputIdsIdx], &decode_ids_buf),
            "CreateTensorWithDataAsOrtValue(decode_input_ids)")) {
        return JNI_FALSE;
    }
    // decode_ids_buf is bound to input_ids only after prefill (in the decode transition), because
    // during prefill input_ids must carry the full (1, num_prefill) prompt.

    m.ioPrepared = true;
    return JNI_TRUE;
}
