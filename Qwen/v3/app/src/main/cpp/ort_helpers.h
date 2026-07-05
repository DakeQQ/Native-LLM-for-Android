#pragma once

// Model-agnostic ONNX Runtime helpers: session loading, I/O binding, tensors, and UTF-8 text assembly.

#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <jni.h>

#include "onnxruntime_cxx_api.h"
#include "onnxruntime_float16.h"
#include "user_settings.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <utility>
#include <vector>

#define LOG_TAG "NativeLLM"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#if defined(__GNUC__) || defined(__clang__)
#define ALWAYS_INLINE inline __attribute__((always_inline))
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define ALWAYS_INLINE inline
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

enum ExecutionProviderType {
    EP_CPU     = 0,
    EP_XNNPACK = 1,
    EP_QNN     = 2
};

struct ModelRuntime {
    int id = 0;
    std::string fileName;
    std::string externalFileName;
    const OrtApi* api = nullptr;
    OrtEnv* env = nullptr;
    OrtSession* session = nullptr;
    OrtRunOptions* runOptions = nullptr;
    OrtIoBinding* binding = nullptr;
    OrtAllocator* allocator = nullptr;
    const OrtMemoryInfo* memoryInfo = nullptr;

    const OrtMemoryInfo* ioMemoryInfo = nullptr;
    OrtAllocator* ioAllocator = nullptr;

    std::vector<const char*> inputNames;
    std::vector<std::vector<int64_t>> inputDims;
    std::vector<ONNXTensorElementDataType> inputTypes;
    std::vector<const char*> outputNames;
    std::vector<std::vector<int64_t>> outputDims;
    std::vector<ONNXTensorElementDataType> outputTypes;
};

// Status / IOBinding
inline bool logOrtStatus(const OrtApi* api, OrtStatus* status, const char* op) {
    if (status == nullptr) {
        return true;
    }
    LOGE("ORT: %s failed: %s", op, api->GetErrorMessage(status));
    api->ReleaseStatus(status);
    return false;
}

inline bool createTensorWithData(const OrtApi* api,
                                 const OrtMemoryInfo* memInfo,
                                 void* dataPtr,
                                 size_t sizeBytes,
                                 const std::vector<int64_t>& dims,
                                 ONNXTensorElementDataType type,
                                 OrtValue** out,
                                 const char* op) {
    return logOrtStatus(api, api->CreateTensorWithDataAsOrtValue(memInfo, dataPtr, sizeBytes, dims.data(), dims.size(), type, out), op);
}

ALWAYS_INLINE bool ensureBinding(ModelRuntime& m) {
    if (!m.binding && m.session) {
        return logOrtStatus(m.api, m.api->CreateIoBinding(m.session, &m.binding), "CreateIoBinding");
    }
    return m.binding != nullptr;
}

ALWAYS_INLINE void resetBinding(ModelRuntime& m) {
    m.api->ClearBoundInputs(m.binding);
    m.api->ClearBoundOutputs(m.binding);
}

ALWAYS_INLINE bool bindIn(ModelRuntime& m, int idx, OrtValue* v) {
    return logOrtStatus(m.api, m.api->BindInput(m.binding, m.inputNames[idx], v), "BindInput");
}

ALWAYS_INLINE bool bindOut(ModelRuntime& m, int idx, OrtValue* v) {
    return logOrtStatus(m.api, m.api->BindOutput(m.binding, m.outputNames[idx], v), "BindOutput");
}

ALWAYS_INLINE bool bindOutDevice(ModelRuntime& m, int idx) {
    return logOrtStatus(m.api, m.api->BindOutputToDevice(m.binding, m.outputNames[idx], m.ioMemoryInfo),
                        "BindOutputToDevice");
}

ALWAYS_INLINE bool runBinding(ModelRuntime& m) {
    return logOrtStatus(m.api, m.api->RunWithBinding(m.session, m.runOptions, m.binding), "RunWithBinding");
}

// OrtValue lifetime
ALWAYS_INLINE void releaseOne(const OrtApi* api, OrtValue*& v) {
    if (v) {
        api->ReleaseValue(v);
        v = nullptr;
    }
}

ALWAYS_INLINE void releaseValues(const OrtApi* api, std::vector<OrtValue*>& v) {
    for (OrtValue*& x : v) {
        if (x) {
            api->ReleaseValue(x);
            x = nullptr;
        }
    }
    v.clear();
}

inline bool fetchOutputsInto(ModelRuntime& m, std::vector<OrtValue*>& out) {
    out.clear();
    OrtValue** raw = nullptr;
    size_t count = 0;
    OrtStatus* st = m.api->GetBoundOutputValues(m.binding, m.allocator, &raw, &count);
    if (st) {
        LOGE("ORT: GetBoundOutputValues(%s) failed: %s", m.fileName.c_str(), m.api->GetErrorMessage(st));
        m.api->ReleaseStatus(st);
        return false;
    }
    if (raw) {
        out.insert(out.end(), raw, raw + count);
        m.allocator->Free(m.allocator, raw);
    }
    return true;
}

inline bool takeFirstOutput(const OrtApi* api, std::vector<OrtValue*>& outputs, OrtValue*& out) {
    if (outputs.empty() || !outputs[0]) {
        releaseValues(api, outputs);
        return false;
    }
    out = outputs[0];
    outputs[0] = nullptr;
    releaseValues(api, outputs);
    return true;
}

// Tensor utilities
inline size_t tensorElementSize(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            return 8;
        default:
            return 4;
    }
}

inline int64_t tensorDim(const OrtApi* api, OrtValue* t, int axis) {
    if (t == nullptr) {
        return -1;
    }
    OrtTensorTypeAndShapeInfo* info = nullptr;
    if (!logOrtStatus(api, api->GetTensorTypeAndShape(t, &info), "GetTensorTypeAndShape") || info == nullptr) {
        return -1;
    }
    size_t rank = 0;
    int64_t dim = -1;
    if (logOrtStatus(api, api->GetDimensionsCount(info, &rank), "GetDimensionsCount") &&
        axis >= 0 && static_cast<size_t>(axis) < rank) {
        std::vector<int64_t> dims(rank, 0);
        if (logOrtStatus(api, api->GetDimensions(info, dims.data(), rank), "GetDimensions")) {
            dim = dims[axis];
        }
    }
    api->ReleaseTensorTypeAndShapeInfo(info);
    return dim;
}

inline std::vector<int> extractFirstRowInt32(const OrtApi* api, OrtValue* tensor) {
    std::vector<int> ids;
    if (!tensor) {
        return ids;
    }
    OrtTensorTypeAndShapeInfo* info = nullptr;
    if (!logOrtStatus(api, api->GetTensorTypeAndShape(tensor, &info), "GetTensorTypeAndShape") || !info) {
        return ids;
    }
    size_t dimCount = 0;
    if (!logOrtStatus(api, api->GetDimensionsCount(info, &dimCount), "GetDimensionsCount")) {
        api->ReleaseTensorTypeAndShapeInfo(info);
        return ids;
    }
    std::vector<int64_t> dims(dimCount, 0);
    if (dimCount > 0 &&
        !logOrtStatus(api, api->GetDimensions(info, dims.data(), dimCount), "GetDimensions")) {
        api->ReleaseTensorTypeAndShapeInfo(info);
        return ids;
    }
    api->ReleaseTensorTypeAndShapeInfo(info);
    size_t rowLen = (dimCount >= 2) ? static_cast<size_t>(dims[dimCount - 1]) : 0;
    if (rowLen == 0) {
        return ids;
    }
    const int32_t* data = nullptr;
    if (!logOrtStatus(api, api->GetTensorMutableData(tensor, (void**)&data), "GetTensorMutableData") || !data) {
        return ids;
    }
    ids.reserve(rowLen);
    for (size_t i = 0; i < rowLen; ++i) {
        ids.push_back(static_cast<int>(data[i]));
    }
    return ids;
}

inline uint16_t floatToHalf(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mant = x & 0x7FFFFFu;
    if (exp <= 0)    return static_cast<uint16_t>(sign);
    if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

// Fixed-shape IOBinding buffer; owns its OrtValue and optional device allocation.
struct OrtBuffer {
    std::vector<uint8_t> cpuData;
    void* devData = nullptr;
    OrtAllocator* devAllocator = nullptr;
    OrtValue* value = nullptr;
    const OrtApi* api = nullptr;

    OrtBuffer() = default;
    OrtBuffer(const OrtBuffer&) = delete;
    OrtBuffer& operator=(const OrtBuffer&) = delete;
    OrtBuffer(OrtBuffer&& o) noexcept { steal(o); }
    OrtBuffer& operator=(OrtBuffer&& o) noexcept {
        if (this != &o) { destroy(); steal(o); }
        return *this;
    }
    ~OrtBuffer() { destroy(); }

    uint8_t* data() { return devAllocator ? static_cast<uint8_t*>(devData) : cpuData.data(); }

private:
    void steal(OrtBuffer& o) {
        cpuData = std::move(o.cpuData);
        devData = o.devData;             o.devData = nullptr;
        devAllocator = o.devAllocator;   o.devAllocator = nullptr;
        value = o.value;                 o.value = nullptr;
        api = o.api;
    }
    void destroy() {
        if (api == nullptr) return;
        if (value != nullptr) { api->ReleaseValue(value); value = nullptr; }
        if (devAllocator != nullptr && devData != nullptr) {
            logOrtStatus(api, api->AllocatorFree(devAllocator, devData), "AllocatorFree");
            devData = nullptr;
            devAllocator = nullptr;
        }
    }
};

inline OrtBuffer makeBuffer(ModelRuntime& m,
                            const std::vector<int64_t>& dims,
                            ONNXTensorElementDataType type) {
    const OrtApi* api = m.api;
    OrtBuffer buf;
    buf.api = api;
    size_t count = 1;
    bool empty = false;
    for (int64_t d : dims) {
        if (d <= 0) { empty = true; break; }
        count *= static_cast<size_t>(d);
    }
    const size_t bytes = empty ? 0 : count * tensorElementSize(type);
    const size_t allocBytes = (bytes == 0) ? 1u : bytes;

    const OrtMemoryInfo* mem = m.memoryInfo;
    void* backing = nullptr;
    if (m.ioAllocator && m.ioAllocator != m.allocator) {
        if (logOrtStatus(api, api->AllocatorAlloc(m.ioAllocator, allocBytes, &backing), "AllocatorAlloc") && backing) {
            std::memset(backing, 0, allocBytes);
            buf.devData = backing;
            buf.devAllocator = m.ioAllocator;
            mem = m.ioMemoryInfo;
        } else {
            backing = nullptr;
        }
    }
    if (!backing) {
        buf.cpuData.resize(allocBytes, 0);
        backing = buf.cpuData.data();
    }

    logOrtStatus(api,
                 api->CreateTensorWithDataAsOrtValue(mem, backing, bytes, dims.data(), dims.size(), type, &buf.value),
                 "CreateTensorWithDataAsOrtValue(buffer)");
    return buf;
}

ALWAYS_INLINE void releaseBuffer(const OrtApi* api, OrtBuffer& buf) {
    releaseOne(api, buf.value);
    if (buf.devAllocator && buf.devData) {
        logOrtStatus(api, api->AllocatorFree(buf.devAllocator, buf.devData), "AllocatorFree");
        buf.devData = nullptr;
        buf.devAllocator = nullptr;
    }
}

ALWAYS_INLINE void writeScalarFloat(OrtBuffer& buf, ONNXTensorElementDataType type, float v) {
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        *reinterpret_cast<float*>(buf.data()) = v;
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        *reinterpret_cast<uint16_t*>(buf.data()) = floatToHalf(v);
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
        uint32_t x;
        std::memcpy(&x, &v, sizeof(x));
        *reinterpret_cast<uint16_t*>(buf.data()) = static_cast<uint16_t>(x >> 16);
    }
}

ALWAYS_INLINE void writeScalarInt64(OrtBuffer& buf, ONNXTensorElementDataType type, int64_t v) {
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        *reinterpret_cast<int64_t*>(buf.data()) = v;
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        *reinterpret_cast<int32_t*>(buf.data()) = static_cast<int32_t>(v);
    }
}

// Text / time utilities
inline int64_t now_ms() {
    static const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    return static_cast<int64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count());
}

inline const unsigned char* hex_value_lut() {
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

inline std::string& pending_utf8_tail() {
    static thread_local std::string tail;
    return tail;
}

inline void reset_output_words_decoder() {
    pending_utf8_tail().clear();
}

// Keeps split trailing UTF-8 bytes until a later token completes them.
inline void append_utf8_complete(std::string& out, const std::string& bytes) {
    std::string& tail = pending_utf8_tail();
    std::string joined;
    const std::string* text = &bytes;
    if (!tail.empty()) {
        joined.reserve(tail.size() + bytes.size());
        joined.append(tail);
        joined.append(bytes);
        tail.clear();
        text = &joined;
    }

    const char* data = text->data();
    const size_t len = text->size();
    size_t pos = 0;
    while (pos < len) {
        const unsigned char c0 = static_cast<unsigned char>(data[pos]);
        if (c0 == 0) {
            out.push_back('?');
            ++pos;
            continue;
        }
        if (c0 < 0x80) {
            out.push_back(static_cast<char>(c0));
            ++pos;
            continue;
        }

        size_t need = 0;
        bool validLead = true;
        if (c0 >= 0xC2 && c0 <= 0xDF) {
            need = 2;
        } else if (c0 >= 0xE0 && c0 <= 0xEF) {
            need = 3;
        } else if (c0 >= 0xF0 && c0 <= 0xF4) {
            need = 4;
        } else {
            validLead = false;
        }
        if (!validLead) {
            out.push_back('?');
            ++pos;
            continue;
        }
        bool valid = true;
        if (pos + 1 < len) {
            const unsigned char c1 = static_cast<unsigned char>(data[pos + 1]);
            valid = (c1 & 0xC0) == 0x80;
            if (valid && c0 == 0xE0) valid = c1 >= 0xA0;
            if (valid && c0 == 0xED) valid = c1 <= 0x9F;
            if (valid && c0 == 0xF0) valid = c1 >= 0x90;
            if (valid && c0 == 0xF4) valid = c1 <= 0x8F;
        }
        if (valid && need >= 3 && pos + 2 < len) {
            const unsigned char c2 = static_cast<unsigned char>(data[pos + 2]);
            valid = (c2 & 0xC0) == 0x80;
        }
        if (valid && need == 4 && pos + 3 < len) {
            const unsigned char c3 = static_cast<unsigned char>(data[pos + 3]);
            valid = (c3 & 0xC0) == 0x80;
        }
        if (!valid) {
            out.push_back('?');
            ++pos;
            continue;
        }
        if (pos + need > len) {
            tail.assign(data + pos, len - pos);
            break;
        }
        out.append(data + pos, need);
        pos += need;
    }
}

// Model loading
inline bool ok_status(const OrtApi* api, OrtStatus* status, const char* op) {
    return logOrtStatus(api, status, op);
}

inline bool cacheFileExists(const std::string& path) {
    struct stat st{};
    return (stat(path.c_str(), &st) == 0) && (st.st_size > 10);
}

inline bool readAssetFully(AAsset* asset, std::vector<char>& out, const std::string& name) {
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

// External data must live beside its .onnx file.
inline std::string resolveModelPath(const ModelRuntime& m) {
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

// Exported metadata carries model geometry, KV layout, chat-template IDs, and activation precision.
inline std::string lookupModelMetadata(const OrtApi* api, const OrtModelMetadata* metadata,
                                       OrtAllocator* allocator, const char* key) {
    char* value = nullptr;
    OrtStatus* status = api->ModelMetadataLookupCustomMetadataMap(metadata, allocator, key, &value);
    std::string result;
    if (status != nullptr) {
        api->ReleaseStatus(status);
    } else if (value != nullptr) {
        result.assign(value);
    }
    if (value != nullptr) {
        allocator->Free(allocator, value);
    }
    return result;
}

// Inline globals share one OrtEnv across translation units.
inline OrtEnv* gSharedEnv = nullptr;
inline bool    gSharedEnvUsesGlobalPools = false;
inline bool    gSharedEnvArenaRegistered = false;

// Registers ONE CPU arena on the shared env so session.use_env_allocators="1" actually shares a single
// arena across all sessions (otherwise the flag silently no-ops and every session builds its own arena).
// Must run before any session is created; idempotent and null-safe.
inline void registerSharedEnvArena(const OrtApi* api, OrtEnv* env) {
    if (!kRegisterSharedEnvAllocator || env == nullptr || gSharedEnvArenaRegistered) {
        return;
    }
    OrtMemoryInfo* cpuArenaInfo = nullptr;
    if (!logOrtStatus(api, api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpuArenaInfo),
                      "CreateCpuMemoryInfo(shared env arena)") || cpuArenaInfo == nullptr) {
        return;
    }
    OrtArenaCfg* arenaCfg = nullptr;
    const bool wantCustomCfg = kEnvArenaExtendStrategy >= 0 || kEnvArenaInitialChunkBytes >= 0 ||
                               kEnvArenaMaxDeadBytesPerChunk >= 0 || kEnvArenaMaxMemBytes > 0;
    if (wantCustomCfg &&
        !logOrtStatus(api, api->CreateArenaCfg(kEnvArenaMaxMemBytes, kEnvArenaExtendStrategy,
                                               kEnvArenaInitialChunkBytes, kEnvArenaMaxDeadBytesPerChunk,
                                               &arenaCfg),
                      "CreateArenaCfg(shared env arena)")) {
        arenaCfg = nullptr;   // registration falls back to ORT default arena config
    }
    if (logOrtStatus(api, api->CreateAndRegisterAllocator(env, cpuArenaInfo, arenaCfg),
                     "CreateAndRegisterAllocator(shared env cpu arena)")) {
        gSharedEnvArenaRegistered = true;
        LOGI("ORT: shared env CPU arena registered \u2014 use_env_allocators active (extend_strategy=%d, max_mem=%zu)",
             kEnvArenaExtendStrategy, static_cast<size_t>(kEnvArenaMaxMemBytes));
    }
    if (arenaCfg != nullptr) {
        api->ReleaseArenaCfg(arenaCfg);
    }
    api->ReleaseMemoryInfo(cpuArenaInfo);
}

inline OrtEnv* getSharedEnv(const OrtApi* api) {
    if (gSharedEnv != nullptr) {
        return gSharedEnv;
    }

    if (kUseGlobalThreadPool) {
        OrtThreadingOptions* threadingOptions = nullptr;
        OrtStatus* status = api->CreateThreadingOptions(&threadingOptions);
        if (status == nullptr && threadingOptions != nullptr) {
            bool threadingOk = true;
            auto checkThreading = [&](OrtStatus* st, const char* op) {
                if (st == nullptr) {
                    return true;
                }
                LOGE("ORT: %s failed: %s", op, api->GetErrorMessage(st));
                api->ReleaseStatus(st);
                return false;
            };
            threadingOk = checkThreading(api->SetGlobalIntraOpNumThreads(threadingOptions, kGlobalIntraOpThreads),
                                         "SetGlobalIntraOpNumThreads") && threadingOk;
            threadingOk = checkThreading(api->SetGlobalInterOpNumThreads(threadingOptions, kGlobalInterOpThreads),
                                         "SetGlobalInterOpNumThreads") && threadingOk;
            threadingOk = checkThreading(api->SetGlobalSpinControl(threadingOptions, kGlobalAllowSpinning),
                                         "SetGlobalSpinControl") && threadingOk;
            threadingOk = checkThreading(api->SetGlobalDenormalAsZero(threadingOptions),
                                         "SetGlobalDenormalAsZero") && threadingOk;
            if (kGlobalIntraAffinities && kGlobalIntraAffinities[0] != '\0') {
                OrtStatus* affinityStatus = api->SetGlobalIntraOpThreadAffinity(threadingOptions, kGlobalIntraAffinities);
                if (affinityStatus != nullptr) {
                    LOGE("ORT: invalid kGlobalIntraAffinities \"%s\": %s",
                         kGlobalIntraAffinities, api->GetErrorMessage(affinityStatus));
                    api->ReleaseStatus(affinityStatus);
                    threadingOk = false;
                }
            }
            status = threadingOk
                ? api->CreateEnvWithGlobalThreadPools(ORT_LOGGING_LEVEL_ERROR, "myapplication", threadingOptions, &gSharedEnv)
                : nullptr;
            api->ReleaseThreadingOptions(threadingOptions);
            if (status == nullptr && gSharedEnv != nullptr) {
                gSharedEnvUsesGlobalPools = true;
                LOGI("ORT: shared global thread pool ready (intra=%d, inter=%d, spin=%d)",
                     kGlobalIntraOpThreads, kGlobalInterOpThreads, kGlobalAllowSpinning);
                registerSharedEnvArena(api, gSharedEnv);
                return gSharedEnv;
            }
        }
        if (status != nullptr) {
            LOGE("ORT: global thread pool env failed (%s); falling back to per-session threads",
                 api->GetErrorMessage(status));
            api->ReleaseStatus(status);
        }
    }

    OrtStatus* status = api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "myapplication", &gSharedEnv);
    if (status != nullptr) {
        LOGE("ORT: CreateEnv failed: %s", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        gSharedEnv = nullptr;
    }
    gSharedEnvUsesGlobalPools = false;
    registerSharedEnvArena(api, gSharedEnv);
    return gSharedEnv;
}

inline bool enumerateSessionIO(ModelRuntime& m, bool isInput) {
    std::vector<const char*>&               names = isInput ? m.inputNames : m.outputNames;
    std::vector<std::vector<int64_t>>&      dims  = isInput ? m.inputDims  : m.outputDims;
    std::vector<ONNXTensorElementDataType>& types = isInput ? m.inputTypes : m.outputTypes;
    size_t count = 0;
    OrtStatus* countSt = isInput ? m.api->SessionGetInputCount(m.session, &count)
                                 : m.api->SessionGetOutputCount(m.session, &count);
    if (!ok_status(m.api, countSt, "SessionGetIOCount")) {
        return false;
    }
    names.resize(count);
    dims.resize(count);
    types.resize(count);
    for (size_t i = 0; i < count; i++) {
        char* name = nullptr;
        OrtStatus* nameSt = isInput ? m.api->SessionGetInputName(m.session, i, m.allocator, &name)
                                    : m.api->SessionGetOutputName(m.session, i, m.allocator, &name);
        if (!ok_status(m.api, nameSt, "SessionGetIOName")) {
            return false;
        }
        names[i] = name;
        OrtTypeInfo* typeInfo = nullptr;
        const OrtTensorTypeAndShapeInfo* tensorInfo = nullptr;
        ONNXTensorElementDataType type;
        size_t dimensions = 0;
        OrtStatus* infoSt = isInput ? m.api->SessionGetInputTypeInfo(m.session, i, &typeInfo)
                                    : m.api->SessionGetOutputTypeInfo(m.session, i, &typeInfo);
        if (!ok_status(m.api, infoSt, "SessionGetIOTypeInfo") ||
            !ok_status(m.api, m.api->CastTypeInfoToTensorInfo(typeInfo, &tensorInfo), "CastTypeInfoToTensorInfo") ||
            tensorInfo == nullptr ||
            !ok_status(m.api, m.api->GetTensorElementType(tensorInfo, &type), "GetTensorElementType")) {
            if (typeInfo) {
                m.api->ReleaseTypeInfo(typeInfo);
            }
            return false;
        }
        types[i] = type;
        if (!ok_status(m.api, m.api->GetDimensionsCount(tensorInfo, &dimensions), "GetDimensionsCount")) {
            m.api->ReleaseTypeInfo(typeInfo);
            return false;
        }
        dims[i].resize(dimensions);
        if (!ok_status(m.api, m.api->GetDimensions(tensorInfo, dims[i].data(), dimensions), "GetDimensions")) {
            m.api->ReleaseTypeInfo(typeInfo);
            return false;
        }
        m.api->ReleaseTypeInfo(typeInfo);
    }
    return true;
}

inline bool extractIO(ModelRuntime& m, int epType) {
    if (!ok_status(m.api, m.api->GetAllocatorWithDefaultOptions(&m.allocator), "GetAllocatorWithDefaultOptions")) {
        return false;
    }
    OrtMemoryInfo* cpuMemInfo = nullptr;
    if (!ok_status(m.api, m.api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpuMemInfo), "CreateCpuMemoryInfo") || cpuMemInfo == nullptr) {
        return false;
    }
    m.memoryInfo = cpuMemInfo;
    m.ioMemoryInfo = m.memoryInfo;
    m.ioAllocator = m.allocator;

    // QNN HTP shared memory keeps KV feedback on-device when available.
    if (epType == EP_QNN) {
        OrtMemoryInfo* shared_info = nullptr;
        if (ok_status(m.api, m.api->CreateMemoryInfo("QnnHtpShared", OrtDeviceAllocator, 0, OrtMemTypeDefault, &shared_info),
                      "CreateMemoryInfo(QnnHtpShared)") && shared_info != nullptr) {
            OrtAllocator* dev_alloc = nullptr;
            if (ok_status(m.api, m.api->CreateAllocator(m.session, shared_info, &dev_alloc),
                          "CreateAllocator(QnnHtpShared)") && dev_alloc != nullptr) {
                m.ioAllocator = dev_alloc;
                if (ok_status(m.api, m.api->AllocatorGetInfo(dev_alloc, &m.ioMemoryInfo),
                              "AllocatorGetInfo(QnnHtpShared)")) {
                    LOGI("Model %d: QNN HTP shared-memory allocator active: KV cache is zero-copy on-device",
                         static_cast<int>(m.id));
                } else {
                    m.ioAllocator = m.allocator;
                    m.ioMemoryInfo = m.memoryInfo;
                }
            }
            m.api->ReleaseMemoryInfo(shared_info);
        }
    }

    if (!enumerateSessionIO(m, true) || !enumerateSessionIO(m, false)) {
        return false;
    }
    return true;
}

inline bool ortLoadModelSession(ModelRuntime& m, const char* onnxName, const char* externalName,
                                JNIEnv* env, jobject assetManager, int epType, bool lowMemoryMode) {
    m.fileName = (onnxName != nullptr) ? onnxName : "";
    m.externalFileName = (externalName != nullptr && externalName[0] != '\0') ? externalName : "";
    m.api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (m.session != nullptr) {
        LOGI("Model already loaded, skipping reload: %s", m.fileName.c_str());
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

        if (gSharedEnvUsesGlobalPools &&
            !ok_status(m.api, m.api->DisablePerSessionThreads(session_options), "DisablePerSessionThreads")) {
            m.api->ReleaseSessionOptions(session_options);
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

        for (const OrtConfigEntry& e : kOrtRunConfigs) {
            addRunConfig(e.key, e.value);
        }
        if (epType == EP_QNN) {
            for (const OrtConfigEntry& e : kQnnRunConfigs) {
                addRunConfig(e.key, e.value);
            }
        }

        check(m.api->DisableProfiling(session_options), "DisableProfiling");
        check(m.api->EnableCpuMemArena(session_options), "EnableCpuMemArena");
        check(m.api->EnableMemPattern(session_options), "EnableMemPattern");
        check(m.api->SetSessionExecutionMode(session_options, ORT_SEQUENTIAL), "SetSessionExecutionMode");

        check(m.api->SetInterOpNumThreads(session_options, kSessionInterOpThreads), "SetInterOpNumThreads");
        const std::string sessionDynamicBlockBase = std::to_string(kSessionDynamicBlockBase);
        addSessionConfig("session.dynamic_block_base", sessionDynamicBlockBase.c_str());
        if (epType != EP_XNNPACK) {
            addSessionConfig("session.intra_op_thread_affinities", kSessionIntraAffinities);
        }
        check(m.api->SetIntraOpNumThreads(session_options, kSessionIntraOpThreads), "SetIntraOpNumThreads");

        for (const OrtConfigEntry& e : kOrtSessionConfigs) {
            addSessionConfig(e.key, e.value);
        }
        check(m.api->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL), "SetSessionGraphOptimizationLevel");
        // Match optimizer exclusions to exported activation precision.
        addSessionConfig("optimization.disable_specified_optimizers",
                         ORT_FP16 ? "CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer" : "");
        if (epType == EP_QNN) {
            addSessionConfig("ep.dynamic.qnn_htp_performance_mode", "burst");
        }

        std::vector<const char*> option_keys;
        std::vector<const char*> option_values;
        switch (epType) {
            case EP_XNNPACK:
                for (const OrtConfigEntry& e : kXnnpackProviderOptions) {
                    option_keys.push_back(e.key);
                    option_values.push_back(e.value);
                }
                check(m.api->SetInterOpNumThreads(session_options, 4), "SetInterOpNumThreads(XNNPACK)");
                addSessionConfig("session.intra_op.allow_spinning", "0");
                addSessionConfig("session.dynamic_block_base", "1");
                check(m.api->SetIntraOpNumThreads(session_options, 1), "SetIntraOpNumThreads(XNNPACK)");
                check(m.api->SessionOptionsAppendExecutionProvider(session_options, "XNNPACK",
                    option_keys.data(), option_values.data(), option_keys.size()), "SessionOptionsAppendExecutionProvider(XNNPACK)");
                break;

            case EP_QNN:
                for (const OrtConfigEntry& e : kQnnProviderOptions) {
                    option_keys.push_back(e.key);
                    option_values.push_back(e.value);
                }
                check(m.api->SessionOptionsAppendExecutionProvider(session_options, "QNN",
                    option_keys.data(), option_values.data(), option_keys.size()), "SessionOptionsAppendExecutionProvider(QNN)");
                break;

            case EP_CPU:
            default:
                break;
        }
        option_keys.clear();
        option_values.clear();

        if (!configOk) {
            m.api->ReleaseSessionOptions(session_options);
            return false;
        }

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
