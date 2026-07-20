#pragma once

// Model-agnostic ONNX Runtime helpers: session loading, I/O binding, tensors, and UTF-8 text assembly.

#include <android/asset_manager_jni.h>
#include <android/log.h>
#include <jni.h>

#include "onnxruntime_cxx_api.h"
#include "onnxruntime_float16.h"
#include "user_settings.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <sched.h>
#include <string>
#include <sys/auxv.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
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

struct OrtThreadAffinityContext {
    std::vector<int> workerCpus;
    std::vector<pid_t> workerTids;
    std::vector<uint8_t> workerPinned;
    std::atomic<size_t> nextWorker{0};
    std::mutex stateMutex;

    void configure(const std::vector<int>& cpus) {
        workerCpus.clear();
        if (cpus.size() > 1) {
            workerCpus.assign(cpus.begin() + 1, cpus.end());
        }
        workerTids.assign(workerCpus.size(), -1);
        workerPinned.assign(workerCpus.size(), 0);
        nextWorker.store(0, std::memory_order_relaxed);
    }
};

struct ModelRuntime {
    int id = 0;
    std::string fileName;
    std::string externalFileName;
    bool mergedGraph = false;
    bool decodeGraph = false;
    bool prefillWorkload = false;
    const OrtApi* api = nullptr;
    OrtEnv* env = nullptr;
    OrtSession* session = nullptr;
    OrtRunOptions* runOptions = nullptr;
    OrtIoBinding* binding = nullptr;
    OrtIoBinding* alternateBinding = nullptr;
    OrtAllocator* allocator = nullptr;
    const OrtMemoryInfo* memoryInfo = nullptr;

    const OrtMemoryInfo* ioMemoryInfo = nullptr;
    OrtAllocator* ioAllocator = nullptr;
    std::unique_ptr<OrtThreadAffinityContext> threadAffinityContext;

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

inline void releaseModelRuntime(ModelRuntime& m) {
    if (m.api == nullptr) {
        m = ModelRuntime{};
        return;
    }

    if (m.binding != nullptr) {
        m.api->ClearBoundInputs(m.binding);
        m.api->ClearBoundOutputs(m.binding);
        m.api->ReleaseIoBinding(m.binding);
        m.binding = nullptr;
    }
    if (m.alternateBinding != nullptr) {
        m.api->ClearBoundInputs(m.alternateBinding);
        m.api->ClearBoundOutputs(m.alternateBinding);
        m.api->ReleaseIoBinding(m.alternateBinding);
        m.alternateBinding = nullptr;
    }

    if (m.allocator != nullptr) {
        for (const char* name : m.inputNames) {
            if (name != nullptr) {
                m.allocator->Free(m.allocator, const_cast<char*>(name));
            }
        }
        for (const char* name : m.outputNames) {
            if (name != nullptr) {
                m.allocator->Free(m.allocator, const_cast<char*>(name));
            }
        }
    }
    m.inputNames.clear();
    m.inputDims.clear();
    m.inputTypes.clear();
    m.outputNames.clear();
    m.outputDims.clear();
    m.outputTypes.clear();

    if (m.ioAllocator != nullptr && m.ioAllocator != m.allocator) {
        m.api->ReleaseAllocator(m.ioAllocator);
    }
    m.ioAllocator = nullptr;
    m.ioMemoryInfo = nullptr;

    if (m.memoryInfo != nullptr) {
        m.api->ReleaseMemoryInfo(const_cast<OrtMemoryInfo*>(m.memoryInfo));
        m.memoryInfo = nullptr;
    }
    if (m.session != nullptr) {
        m.api->ReleaseSession(m.session);
        m.session = nullptr;
    }
    if (m.runOptions != nullptr) {
        m.api->ReleaseRunOptions(m.runOptions);
        m.runOptions = nullptr;
    }
    m.threadAffinityContext.reset();

    m.allocator = nullptr;
    m.env = nullptr;
    m.api = nullptr;
    m.fileName.clear();
    m.externalFileName.clear();
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

ALWAYS_INLINE bool ensureAlternateBinding(ModelRuntime& m) {
    if (!m.alternateBinding && m.session) {
        return logOrtStatus(m.api, m.api->CreateIoBinding(m.session, &m.alternateBinding),
                            "CreateIoBinding(alternate)");
    }
    return m.alternateBinding != nullptr;
}

ALWAYS_INLINE void clearRuntimeBindings(ModelRuntime& m) {
    if (m.api == nullptr) {
        return;
    }
    for (OrtIoBinding* binding : {m.binding, m.alternateBinding}) {
        if (binding != nullptr) {
            m.api->ClearBoundInputs(binding);
            m.api->ClearBoundOutputs(binding);
        }
    }
}

ALWAYS_INLINE void resetBinding(ModelRuntime& m) {
    m.api->ClearBoundInputs(m.binding);
    m.api->ClearBoundOutputs(m.binding);
}

ALWAYS_INLINE bool bindIn(ModelRuntime& m, int idx, OrtValue* v) {
    return logOrtStatus(m.api, m.api->BindInput(m.binding, m.inputNames[idx], v), "BindInput");
}

// Name-based I/O lookup for the merged strategy graphs. Their leading state block is positional, but the
// tail inputs/outputs (embed_input_ids, phase counters, per-strategy scalars, token/kv_seq/save_id) vary
// by strategy, so those are located by name (mirrors Inference_Qwen_ONNX.py plan_merged_io). Returns -1
// when absent so callers can treat an input as optional.
inline int findInputIdx(const ModelRuntime& m, const char* name) {
    for (size_t i = 0; i < m.inputNames.size(); ++i) {
        if (m.inputNames[i] != nullptr && std::strcmp(m.inputNames[i], name) == 0) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

inline int findInputIdxPrefix(const ModelRuntime& m, const char* prefix) {
    const size_t prefixLen = std::strlen(prefix);
    for (size_t i = 0; i < m.inputNames.size(); ++i) {
        if (m.inputNames[i] != nullptr && std::strncmp(m.inputNames[i], prefix, prefixLen) == 0) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

inline int findOutputIdx(const ModelRuntime& m, const char* name) {
    for (size_t i = 0; i < m.outputNames.size(); ++i) {
        if (m.outputNames[i] != nullptr && std::strcmp(m.outputNames[i], name) == 0) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

// First present input whose name matches one of the candidates (for names that shift across strategies).
inline int findInputIdxAny(const ModelRuntime& m, std::initializer_list<const char*> names) {
    for (const char* name : names) {
        const int idx = findInputIdx(m, name);
        if (idx >= 0) {
            return idx;
        }
    }
    return -1;
}

// First present output whose name matches one of the candidates.
inline int findOutputIdxAny(const ModelRuntime& m, std::initializer_list<const char*> names) {
    for (const char* name : names) {
        const int idx = findOutputIdx(m, name);
        if (idx >= 0) {
            return idx;
        }
    }
    return -1;
}

ALWAYS_INLINE bool bindOutDevice(ModelRuntime& m, int idx) {
    return logOrtStatus(m.api, m.api->BindOutputToDevice(m.binding, m.outputNames[idx], m.ioMemoryInfo),
                        "BindOutputToDevice");
}

inline void applyCurrentThreadAffinityForModel(const ModelRuntime& model);

ALWAYS_INLINE bool runBinding(ModelRuntime& m) {
    applyCurrentThreadAffinityForModel(m);
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
            return 0;
    }
}

inline bool isIntegerTensorType(ONNXTensorElementDataType type) {
    return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
           type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

inline bool isFloatingTensorType(ONNXTensorElementDataType type) {
    return type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
           type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
           type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
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

inline bool readScalarInteger(const OrtApi* api, OrtValue* tensor,
                              ONNXTensorElementDataType type, int64_t& value) {
    if (tensor == nullptr || !isIntegerTensorType(type)) {
        LOGE("ORT: scalar tensor has unsupported integer type %d", static_cast<int>(type));
        return false;
    }
    void* data = nullptr;
    if (!logOrtStatus(api, api->GetTensorMutableData(tensor, &data), "GetTensorMutableData(integer)") ||
        data == nullptr) {
        return false;
    }
    value = type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
            ? *static_cast<const int32_t*>(data)
            : *static_cast<const int64_t*>(data);
    return true;
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
    const size_t elementSize = tensorElementSize(type);
    if (elementSize == 0) {
        LOGE("ORT: cannot allocate unsupported tensor element type %d", static_cast<int>(type));
        return buf;
    }
    size_t count = 1;
    bool empty = false;
    for (int64_t d : dims) {
        if (d <= 0) { empty = true; break; }
        if (count > std::numeric_limits<size_t>::max() / static_cast<size_t>(d)) {
            LOGE("ORT: tensor element count overflow");
            return buf;
        }
        count *= static_cast<size_t>(d);
    }
    if (!empty && count > std::numeric_limits<size_t>::max() / elementSize) {
        LOGE("ORT: tensor byte size overflow");
        return buf;
    }
    const size_t bytes = empty ? 0 : count * elementSize;
    const size_t allocBytes = (bytes == 0) ? 1u : bytes;

    const OrtMemoryInfo* mem = m.memoryInfo;
    void* backing = nullptr;
    if (m.ioAllocator && m.ioAllocator != m.allocator) {
        if (!logOrtStatus(api, api->AllocatorAlloc(m.ioAllocator, allocBytes, &backing),
                          "AllocatorAlloc") || backing == nullptr) {
            return buf;
        }
        std::memset(backing, 0, allocBytes);
        buf.devData = backing;
        buf.devAllocator = m.ioAllocator;
        mem = m.ioMemoryInfo;
    } else {
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

ALWAYS_INLINE bool writeScalarFloat(OrtBuffer& buf, ONNXTensorElementDataType type, float v) {
    if (buf.value == nullptr) {
        return false;
    }
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        *reinterpret_cast<float*>(buf.data()) = v;
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        *reinterpret_cast<uint16_t*>(buf.data()) = floatToHalf(v);
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
        uint32_t x;
        std::memcpy(&x, &v, sizeof(x));
        *reinterpret_cast<uint16_t*>(buf.data()) = static_cast<uint16_t>(x >> 16);
    } else {
        LOGE("ORT: unsupported floating scalar type %d", static_cast<int>(type));
        return false;
    }
    return true;
}

ALWAYS_INLINE bool writeScalarInt64(OrtBuffer& buf, ONNXTensorElementDataType type, int64_t v) {
    if (buf.value == nullptr) {
        return false;
    }
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        *reinterpret_cast<int64_t*>(buf.data()) = v;
    } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        if (v < std::numeric_limits<int32_t>::min() || v > std::numeric_limits<int32_t>::max()) {
            LOGE("ORT: integer scalar %lld is outside int32 range", static_cast<long long>(v));
            return false;
        }
        *reinterpret_cast<int32_t*>(buf.data()) = static_cast<int32_t>(v);
    } else {
        LOGE("ORT: unsupported integer scalar type %d", static_cast<int>(type));
        return false;
    }
    return true;
}

inline bool writeIntegerValues(OrtBuffer& buf, ONNXTensorElementDataType type,
                               const int* values, size_t count) {
    if (buf.value == nullptr || values == nullptr || !isIntegerTensorType(type)) {
        LOGE("ORT: cannot write integer tensor with type %d", static_cast<int>(type));
        return false;
    }
    if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        int32_t* destination = reinterpret_cast<int32_t*>(buf.data());
        for (size_t i = 0; i < count; ++i) {
            destination[i] = static_cast<int32_t>(values[i]);
        }
    } else {
        int64_t* destination = reinterpret_cast<int64_t*>(buf.data());
        for (size_t i = 0; i < count; ++i) {
            destination[i] = values[i];
        }
    }
    return true;
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

// JNI NewStringUTF consumes Modified UTF-8, not standard UTF-8. Decode model output explicitly so
// supplementary code points (emoji, historic scripts) arrive in Java as valid UTF-16 surrogate pairs.
inline jstring newJStringFromUtf8(JNIEnv* env, const std::string& utf8) {
    if (env == nullptr) {
        return nullptr;
    }
    static thread_local std::vector<jchar> utf16;
    utf16.clear();
    utf16.reserve(utf8.size());
    size_t offset = 0;
    while (offset < utf8.size()) {
        const unsigned char lead = static_cast<unsigned char>(utf8[offset]);
        uint32_t codePoint = 0xFFFD;
        size_t width = 1;
        if (lead < 0x80) {
            codePoint = lead;
        } else if (lead >= 0xC2 && lead <= 0xDF && offset + 1 < utf8.size()) {
            const unsigned char b1 = static_cast<unsigned char>(utf8[offset + 1]);
            if ((b1 & 0xC0) == 0x80) {
                codePoint = ((lead & 0x1F) << 6) | (b1 & 0x3F);
                width = 2;
            }
        } else if (lead >= 0xE0 && lead <= 0xEF && offset + 2 < utf8.size()) {
            const unsigned char b1 = static_cast<unsigned char>(utf8[offset + 1]);
            const unsigned char b2 = static_cast<unsigned char>(utf8[offset + 2]);
            const bool valid = (b1 & 0xC0) == 0x80 && (b2 & 0xC0) == 0x80 &&
                    (lead != 0xE0 || b1 >= 0xA0) && (lead != 0xED || b1 <= 0x9F);
            if (valid) {
                codePoint = ((lead & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F);
                width = 3;
            }
        } else if (lead >= 0xF0 && lead <= 0xF4 && offset + 3 < utf8.size()) {
            const unsigned char b1 = static_cast<unsigned char>(utf8[offset + 1]);
            const unsigned char b2 = static_cast<unsigned char>(utf8[offset + 2]);
            const unsigned char b3 = static_cast<unsigned char>(utf8[offset + 3]);
            const bool valid = (b1 & 0xC0) == 0x80 && (b2 & 0xC0) == 0x80 &&
                    (b3 & 0xC0) == 0x80 && (lead != 0xF0 || b1 >= 0x90) &&
                    (lead != 0xF4 || b1 <= 0x8F);
            if (valid) {
                codePoint = ((lead & 0x07) << 18) | ((b1 & 0x3F) << 12) |
                            ((b2 & 0x3F) << 6) | (b3 & 0x3F);
                width = 4;
            }
        }
        offset += width;
        if (codePoint <= 0xFFFF) {
            utf16.push_back(static_cast<jchar>(codePoint));
        } else {
            codePoint -= 0x10000;
            utf16.push_back(static_cast<jchar>(0xD800 + (codePoint >> 10)));
            utf16.push_back(static_cast<jchar>(0xDC00 + (codePoint & 0x3FF)));
        }
    }
    return env->NewString(utf16.data(), static_cast<jsize>(utf16.size()));
}

// GetStringUTFChars returns Modified UTF-8 (supplementary code points become CESU-8 surrogate
// sequences). Encode directly from Java UTF-16 so tokenizer/path inputs always receive standard UTF-8.
inline bool jStringToUtf8(JNIEnv* env, jstring value, std::string& utf8) {
    utf8.clear();
    if (env == nullptr || value == nullptr) {
        return false;
    }
    const jsize length = env->GetStringLength(value);
    const jchar* chars = env->GetStringChars(value, nullptr);
    if (chars == nullptr) {
        return false;
    }
    utf8.reserve(static_cast<size_t>(length) * 3);
    for (jsize index = 0; index < length; ++index) {
        uint32_t codePoint = chars[index];
        if (codePoint >= 0xD800 && codePoint <= 0xDBFF && index + 1 < length) {
            const uint32_t low = chars[index + 1];
            if (low >= 0xDC00 && low <= 0xDFFF) {
                codePoint = 0x10000 + ((codePoint - 0xD800) << 10) + (low - 0xDC00);
                ++index;
            } else {
                codePoint = 0xFFFD;
            }
        } else if (codePoint >= 0xD800 && codePoint <= 0xDFFF) {
            codePoint = 0xFFFD;
        }

        if (codePoint <= 0x7F) {
            utf8.push_back(static_cast<char>(codePoint));
        } else if (codePoint <= 0x7FF) {
            utf8.push_back(static_cast<char>(0xC0 | (codePoint >> 6)));
            utf8.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
        } else if (codePoint <= 0xFFFF) {
            utf8.push_back(static_cast<char>(0xE0 | (codePoint >> 12)));
            utf8.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
            utf8.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
        } else {
            utf8.push_back(static_cast<char>(0xF0 | (codePoint >> 18)));
            utf8.push_back(static_cast<char>(0x80 | ((codePoint >> 12) & 0x3F)));
            utf8.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3F)));
            utf8.push_back(static_cast<char>(0x80 | (codePoint & 0x3F)));
        }
    }
    env->ReleaseStringChars(value, chars);
    return true;
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

inline std::string assetPath(const char* assetDirectory, const char* fileName) {
    if (assetDirectory == nullptr || assetDirectory[0] == '\0') {
        return fileName != nullptr ? fileName : "";
    }
    std::string path(assetDirectory);
    if (!path.empty() && path.back() != '/') {
        path.push_back('/');
    }
    path.append(fileName != nullptr ? fileName : "");
    return path;
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

// Case-insensitive test for a tokenizer vocab file name: "vocab_*.txt" (any model-specific middle, e.g.
// vocab_Qwen3.5-0.8B.txt / VOCAB_qwen.TXT). Lets the runtime accept whatever the exporter named the file.
inline bool isVocabFileName(const char* name) {
    if (name == nullptr) {
        return false;
    }
    const size_t len = std::strlen(name);
    if (len < 10) {   // "vocab_" (6) + ".txt" (4)
        return false;
    }
    auto ciEq = [](char a, char b) {
        return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
    };
    static const char kPrefix[] = "vocab_";
    static const char kSuffix[] = ".txt";
    for (int i = 0; i < 6; ++i) {
        if (!ciEq(name[i], kPrefix[i])) {
            return false;
        }
    }
    for (int i = 0; i < 4; ++i) {
        if (!ciEq(name[len - 4 + i], kSuffix[i])) {
            return false;
        }
    }
    return true;
}

// First existing "vocab_*.txt" (case-insensitive) in `dir`, or "" if the dir is missing / has none.
inline std::string findVocabInDir(const std::string& dir) {
    DIR* handle = opendir(dir.c_str());
    if (handle == nullptr) {
        return "";
    }
    std::string match;
    while (const dirent* entry = readdir(handle)) {
        if (isVocabFileName(entry->d_name)) {
            const std::string full = dir + entry->d_name;
            if (cacheFileExists(full)) {
                match = full;
                break;
            }
        }
    }
    closedir(handle);
    return match;
}

// Resolve the tokenizer vocab path. Prefer the configured vocab_path when present, else auto-discover any
// vocab_*.txt (case-insensitive) staged in the cache/storage dir, so the exported vocab can keep its
// model-specific name without the runtime constant having to match. Falls back to vocab_path when nothing
// is found (createTokenizer then fails loudly rather than silently).
inline std::string resolveVocabPath() {
    if (cacheFileExists(vocab_path)) {
        return vocab_path;
    }
    std::string found = findVocabInDir(cache_path);
    if (!found.empty()) {
        return found;
    }
    found = findVocabInDir(storage_path);
    if (!found.empty()) {
        return found;
    }
    return vocab_path;
}

// Exported metadata carries model geometry, KV layout, chat-template IDs, and file names.
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

inline bool deviceIsArm64V82aOrAbove() {
#if defined(__aarch64__) && defined(AT_HWCAP)
    constexpr unsigned long kHwcapsFp16Arithmetic = (1UL << 9) | (1UL << 10); // FPHP + ASIMDHP
    const unsigned long hwcap = getauxval(AT_HWCAP);
    return (hwcap & kHwcapsFp16Arithmetic) == kHwcapsFp16Arithmetic;
#else
    return false;
#endif
}

struct CpuThreadPlan {
    std::vector<int> prefillCpus;
    std::vector<int> prefillCallerCpus;
    std::vector<int> decodeCpus;
    std::vector<int> decodeCallerCpus;
    int decodeIntraThreads = 1;
    std::string prefillAffinity;
    std::string decodeAffinity;
};

inline long cpuMaxFrequencyKHz(int cpu) {
    char path[128];
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
    std::FILE* file = std::fopen(path, "r");
    if (file == nullptr) {
        return -1;
    }
    long frequency = -1;
    if (std::fscanf(file, "%ld", &frequency) != 1) {
        frequency = -1;
    }
    std::fclose(file);
    return frequency;
}

inline long cpuCapacity(int cpu) {
    char path[128];
    std::snprintf(path, sizeof(path),
                  "/sys/devices/system/cpu/cpu%d/cpu_capacity", cpu);
    std::FILE* file = std::fopen(path, "r");
    if (file == nullptr) {
        return -1;
    }
    long capacity = -1;
    if (std::fscanf(file, "%ld", &capacity) != 1) {
        capacity = -1;
    }
    std::fclose(file);
    return capacity;
}

inline std::string workerAffinity(const std::vector<int>& cpus) {
    std::string affinity;
    for (size_t index = 1; index < cpus.size(); ++index) {
        if (!affinity.empty()) {
            affinity.push_back(';');
        }
        // ORT affinity processor IDs are 1-based; Linux sched affinity CPU IDs are 0-based.
        affinity.append(std::to_string(cpus[index] + 1));
    }
    return affinity;
}

inline std::string cpuIdList(const std::vector<int>& cpus) {
    std::string list;
    for (int cpu : cpus) {
        if (!list.empty()) {
            list.push_back(',');
        }
        list.append(std::to_string(cpu));
    }
    return list;
}

inline const CpuThreadPlan& cpuThreadPlan() {
    static const CpuThreadPlan plan = [] {
        CpuThreadPlan result;
        std::vector<int> availableCpus;
        cpu_set_t allowedSet;
        CPU_ZERO(&allowedSet);
        const bool haveAffinity = sched_getaffinity(0, sizeof(allowedSet), &allowedSet) == 0;
        if (haveAffinity) {
            for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
                if (CPU_ISSET(cpu, &allowedSet)) {
                    availableCpus.push_back(cpu);
                }
            }
        } else {
            const long configured = std::max<long>(1, sysconf(_SC_NPROCESSORS_CONF));
            for (int cpu = 0; cpu < configured && cpu < CPU_SETSIZE; ++cpu) {
                availableCpus.push_back(cpu);
            }
        }

        if (availableCpus.empty()) {
            availableCpus.push_back(0);
        }

        struct RankedCpu {
            int id;
            long capacity;
            long maxFrequencyKHz;
        };
        std::vector<RankedCpu> ranked;
        ranked.reserve(availableCpus.size());
        for (int cpu : availableCpus) {
            ranked.push_back({cpu, cpuCapacity(cpu), cpuMaxFrequencyKHz(cpu)});
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto& left, const auto& right) {
            if (left.capacity != right.capacity) {
                return left.capacity > right.capacity;
            }
            if (left.maxFrequencyKHz != right.maxFrequencyKHz) {
                return left.maxFrequencyKHz > right.maxFrequencyKHz;
            }
            return left.id > right.id;
        });

        auto selectCpus = [&](int requestedCount, std::vector<int>& destination) {
            const size_t count = std::min(ranked.size(), static_cast<size_t>(requestedCount));
            destination.reserve(count);
            for (size_t index = 0; index < count; ++index) {
                destination.push_back(ranked[index].id);
            }
        };
        selectCpus(kPrefillCpuCoreCount, result.prefillCpus);
        selectCpus(kDecodeCpuCoreCount, result.decodeCpus);
        result.prefillCallerCpus = result.prefillCpus;
        if (!result.decodeCpus.empty()) {
            result.decodeCallerCpus.push_back(result.decodeCpus.front());
        }
        result.decodeIntraThreads = std::max(1, static_cast<int>(result.decodeCpus.size()));
        result.prefillAffinity = workerAffinity(result.prefillCpus);
        result.decodeAffinity = workerAffinity(result.decodeCpus);
        LOGI("CPU plan: available=[%s] prefill=[%s] decode=[%s]",
             cpuIdList(availableCpus).c_str(), cpuIdList(result.prefillCpus).c_str(),
             cpuIdList(result.decodeCpus).c_str());
        return result;
    }();
    return plan;
}

struct OrtPinnedThreadHandle {
    std::thread thread;
};

inline bool threadHasExactCpuAffinity(pid_t tid, int cpu) {
    cpu_set_t effective;
    CPU_ZERO(&effective);
    if (sched_getaffinity(tid, sizeof(effective), &effective) != 0 ||
        !CPU_ISSET(cpu, &effective)) {
        return false;
    }
    for (int candidate = 0; candidate < CPU_SETSIZE; ++candidate) {
        if (candidate != cpu && CPU_ISSET(candidate, &effective)) {
            return false;
        }
    }
    return true;
}

inline bool pinThreadToCpu(pid_t tid, int cpu) {
    cpu_set_t requested;
    CPU_ZERO(&requested);
    CPU_SET(cpu, &requested);
    return sched_setaffinity(tid, sizeof(requested), &requested) == 0 &&
           threadHasExactCpuAffinity(tid, cpu);
}

inline bool threadAffinityWithinCpus(pid_t tid, const std::vector<int>& cpus) {
    cpu_set_t effective;
    CPU_ZERO(&effective);
    if (sched_getaffinity(tid, sizeof(effective), &effective) != 0) {
        return false;
    }
    bool foundEffectiveCpu = false;
    for (int candidate = 0; candidate < CPU_SETSIZE; ++candidate) {
        const bool present = CPU_ISSET(candidate, &effective) != 0;
        if (!present) {
            continue;
        }
        foundEffectiveCpu = true;
        if (std::find(cpus.begin(), cpus.end(), candidate) == cpus.end()) {
            return false;
        }
    }
    return foundEffectiveCpu;
}

inline bool pinThreadToCpus(pid_t tid, const std::vector<int>& cpus) {
    if (cpus.empty()) {
        return false;
    }
    cpu_set_t requested;
    CPU_ZERO(&requested);
    for (int cpu : cpus) {
        CPU_SET(cpu, &requested);
    }
    return sched_setaffinity(tid, sizeof(requested), &requested) == 0 &&
           threadAffinityWithinCpus(tid, cpus);
}

inline void retryPinnedOrtWorkers(OrtThreadAffinityContext& context) {
    std::lock_guard<std::mutex> lock(context.stateMutex);
    for (size_t index = 0; index < context.workerCpus.size(); ++index) {
        if (context.workerTids[index] <= 0) {
            continue;
        }
        const int cpu = context.workerCpus[index];
        if (context.workerPinned[index] &&
            threadHasExactCpuAffinity(context.workerTids[index], cpu)) {
            continue;
        }
        if (pinThreadToCpu(context.workerTids[index], cpu)) {
            context.workerPinned[index] = 1;
            LOGI("ORT: deferred worker pin succeeded on Android CPU %d", cpu);
        } else {
            context.workerPinned[index] = 0;
        }
    }
}

inline OrtCustomThreadHandle createPinnedOrtThread(void* options,
                                                    OrtThreadWorkerFn workerFn,
                                                    void* workerParam) {
    auto* context = static_cast<OrtThreadAffinityContext*>(options);
    if (context == nullptr || context->workerCpus.empty() || workerFn == nullptr) {
        LOGE("ORT: pinned worker creation received an invalid affinity context");
        return nullptr;
    }
    const size_t index = context->nextWorker.fetch_add(1, std::memory_order_relaxed);
    const size_t slot = index % context->workerCpus.size();
    const int cpu = context->workerCpus[slot];
    auto* handle = new (std::nothrow) OrtPinnedThreadHandle;
    if (handle == nullptr) {
        LOGE("ORT: could not allocate pinned worker handle");
        return nullptr;
    }
    handle->thread = std::thread([context, slot, cpu, workerFn, workerParam] {
        const pid_t tid = gettid();
        const bool pinned = pinThreadToCpu(tid, cpu);
        {
            std::lock_guard<std::mutex> lock(context->stateMutex);
            context->workerTids[slot] = tid;
            context->workerPinned[slot] = pinned ? 1 : 0;
        }
        if (pinned) {
            LOGI("ORT: worker pinned to Android CPU %d", cpu);
        } else {
            LOGI("ORT: Android CPU %d is temporarily paused; worker pin deferred", cpu);
        }
        workerFn(workerParam);
    });
    return reinterpret_cast<OrtCustomThreadHandle>(handle);
}

inline void joinPinnedOrtThread(OrtCustomThreadHandle customHandle) {
    auto* handle = reinterpret_cast<OrtPinnedThreadHandle*>(
            const_cast<void*>(reinterpret_cast<const void*>(customHandle)));
    if (handle != nullptr) {
        if (handle->thread.joinable()) {
            handle->thread.join();
        }
        delete handle;
    }
}

inline OrtThreadAffinityContext gPrefillThreadAffinityContext;
inline OrtThreadAffinityContext gDecodeThreadAffinityContext;

inline void applyCurrentThreadAffinityForModel(const ModelRuntime& model) {
    if (!model.decodeGraph && !model.prefillWorkload) {
        return;
    }
    const bool decode = model.decodeGraph;
    const int phase = decode ? 1 : 0;
    static thread_local int verifiedPhase = -1;
    if (verifiedPhase == phase) {
        return;
    }
    verifiedPhase = phase;
    const std::vector<int>& cpus = decode
        ? cpuThreadPlan().decodeCallerCpus
        : cpuThreadPlan().prefillCallerCpus;
    if (!cpus.empty()) {
        if (pinThreadToCpus(0, cpus)) {
            LOGI("ORT: %s caller affinity set to Android CPUs [%s]",
                 decode ? "decode" : "prefill", cpuIdList(cpus).c_str());
        } else {
            LOGE("Could not apply %s caller affinity to Android CPUs [%s]: %s",
                 decode ? "decode" : "prefill", cpuIdList(cpus).c_str(),
                 std::strerror(errno));
        }
    }
        OrtThreadAffinityContext* context = decode
            ? &gDecodeThreadAffinityContext
            : &gPrefillThreadAffinityContext;
    if (context != nullptr) {
        retryPinnedOrtWorkers(*context);
    }
}

// Inline globals share one OrtEnv across translation units.
inline OrtEnv* gSharedEnv = nullptr;
inline OrtEnv* gDecodeEnv = nullptr;
inline bool    gSharedEnvUsesGlobalPools = false;
inline bool    gDecodeEnvUsesGlobalPools = false;
inline bool    gSharedEnvArenaRegistered = false;
inline bool    gDecodeEnvArenaRegistered = false;
inline OrtPrepackedWeightsContainer* gSharedPrepackedWeightsContainer = nullptr;

inline OrtPrepackedWeightsContainer* getSharedPrepackedWeightsContainer(const OrtApi* api) {
    if (!kUseSharedPrepackedWeightsContainer) {
        return nullptr;
    }
    if (gSharedPrepackedWeightsContainer != nullptr) {
        return gSharedPrepackedWeightsContainer;
    }
    if (!logOrtStatus(api, api->CreatePrepackedWeightsContainer(&gSharedPrepackedWeightsContainer),
                      "CreatePrepackedWeightsContainer") || gSharedPrepackedWeightsContainer == nullptr) {
        gSharedPrepackedWeightsContainer = nullptr;
        return nullptr;
    }
    LOGI("ORT: shared prepacked-weights container enabled for session creation");
    return gSharedPrepackedWeightsContainer;
}

// Registers one CPU arena on the shared env so session.use_env_allocators="1" shares it across sessions.
// Must run before any session is created.
inline bool registerSharedEnvArena(const OrtApi* api, OrtEnv* env, bool& arenaRegistered) {
    if (!kRegisterSharedEnvAllocator) {
        return true;
    }
    if (env == nullptr) {
        return false;
    }
    if (arenaRegistered) {
        return true;
    }
    OrtMemoryInfo* cpuArenaInfo = nullptr;
    if (!logOrtStatus(api, api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpuArenaInfo),
                      "CreateCpuMemoryInfo(shared env arena)") || cpuArenaInfo == nullptr) {
        return false;
    }
    OrtArenaCfg* arenaCfg = nullptr;
    const bool wantCustomCfg = kEnvArenaExtendStrategy >= 0 || kEnvArenaInitialChunkBytes >= 0 ||
                               kEnvArenaMaxDeadBytesPerChunk >= 0 || kEnvArenaMaxMemBytes > 0;
    if (wantCustomCfg &&
        !logOrtStatus(api, api->CreateArenaCfg(kEnvArenaMaxMemBytes, kEnvArenaExtendStrategy,
                                               kEnvArenaInitialChunkBytes, kEnvArenaMaxDeadBytesPerChunk,
                                               &arenaCfg),
                      "CreateArenaCfg(shared env arena)")) {
                            api->ReleaseMemoryInfo(cpuArenaInfo);
                            return false;
    }
                            const bool registered = logOrtStatus(
                                api, api->CreateAndRegisterAllocator(env, cpuArenaInfo, arenaCfg),
                                "CreateAndRegisterAllocator(shared env cpu arena)");
                            if (registered) {
        arenaRegistered = true;
        LOGI("ORT: shared env CPU arena registered \u2014 use_env_allocators active (extend_strategy=%d, max_mem=%zu)",
             kEnvArenaExtendStrategy, static_cast<size_t>(kEnvArenaMaxMemBytes));
    }
    if (arenaCfg != nullptr) {
        api->ReleaseArenaCfg(arenaCfg);
    }
    api->ReleaseMemoryInfo(cpuArenaInfo);
    return registered;
}

inline OrtEnv* getSharedEnv(const OrtApi* api) {
    if (gSharedEnv != nullptr) {
        return gSharedEnv;
    }

    if (kUseGlobalThreadPool) {
        OrtThreadingOptions* threadingOptions = nullptr;
        if (!logOrtStatus(api, api->CreateThreadingOptions(&threadingOptions),
                          "CreateThreadingOptions") || threadingOptions == nullptr) {
            return nullptr;
        }
        bool threadingOk = true;
        const CpuThreadPlan& plan = cpuThreadPlan();
        const int intraThreads = std::max(1, static_cast<int>(plan.prefillCpus.size()));
        gPrefillThreadAffinityContext.configure(plan.prefillCpus);
        threadingOk = logOrtStatus(api, api->SetGlobalIntraOpNumThreads(threadingOptions, intraThreads),
                                   "SetGlobalIntraOpNumThreads") && threadingOk;
        threadingOk = logOrtStatus(api, api->SetGlobalInterOpNumThreads(threadingOptions, 1),
                                   "SetGlobalInterOpNumThreads") && threadingOk;
        // Prefill pool blocks when idle so its worker threads sleep (not spin) during the long decode phase.
        threadingOk = logOrtStatus(api, api->SetGlobalSpinControl(threadingOptions, kPrefillAllowSpinning),
                                   "SetGlobalSpinControl") && threadingOk;
        threadingOk = logOrtStatus(api, api->SetGlobalDenormalAsZero(threadingOptions),
                                   "SetGlobalDenormalAsZero") && threadingOk;
        threadingOk = logOrtStatus(
            api, api->SetGlobalCustomCreateThreadFn(threadingOptions, createPinnedOrtThread),
            "SetGlobalCustomCreateThreadFn") && threadingOk;
        threadingOk = logOrtStatus(
            api, api->SetGlobalCustomThreadCreationOptions(
                threadingOptions, &gPrefillThreadAffinityContext),
            "SetGlobalCustomThreadCreationOptions") && threadingOk;
        threadingOk = logOrtStatus(
            api, api->SetGlobalCustomJoinThreadFn(threadingOptions, joinPinnedOrtThread),
            "SetGlobalCustomJoinThreadFn") && threadingOk;
        if (!plan.prefillAffinity.empty()) {
            threadingOk = logOrtStatus(
                api, api->SetGlobalIntraOpThreadAffinity(
                    threadingOptions, plan.prefillAffinity.c_str()),
                    "SetGlobalIntraOpThreadAffinity") && threadingOk;
        }
        if (!threadingOk) {
            api->ReleaseThreadingOptions(threadingOptions);
            return nullptr;
        }
        const bool envCreated = logOrtStatus(
                api, api->CreateEnvWithGlobalThreadPools(
                        ORT_LOGGING_LEVEL_ERROR, "myapplication", threadingOptions, &gSharedEnv),
                "CreateEnvWithGlobalThreadPools");
        api->ReleaseThreadingOptions(threadingOptions);
        if (!envCreated || gSharedEnv == nullptr) {
            gSharedEnv = nullptr;
            return nullptr;
        }
        gSharedEnvUsesGlobalPools = true;
        if (!registerSharedEnvArena(api, gSharedEnv, gSharedEnvArenaRegistered)) {
            api->ReleaseEnv(gSharedEnv);
            gSharedEnv = nullptr;
            gSharedEnvUsesGlobalPools = false;
            return nullptr;
        }
                LOGI("ORT: prefill pool ready (threads=%d cpus=%s); decode plan (threads=%d cpus=%s)",
                         intraThreads, cpuIdList(plan.prefillCpus).c_str(),
                         plan.decodeIntraThreads, cpuIdList(plan.decodeCpus).c_str());
        return gSharedEnv;
    }

    if (!logOrtStatus(api, api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "myapplication", &gSharedEnv),
                      "CreateEnv") || gSharedEnv == nullptr) {
        gSharedEnv = nullptr;
        return nullptr;
    }
    gSharedEnvUsesGlobalPools = false;
    if (!registerSharedEnvArena(api, gSharedEnv, gSharedEnvArenaRegistered)) {
        api->ReleaseEnv(gSharedEnv);
        gSharedEnv = nullptr;
        return nullptr;
    }
    return gSharedEnv;
}

inline OrtEnv* getDecodeEnv(const OrtApi* api) {
    if (gDecodeEnv != nullptr) {
        return gDecodeEnv;
    }

    OrtThreadingOptions* threadingOptions = nullptr;
    if (!logOrtStatus(api, api->CreateThreadingOptions(&threadingOptions),
              "CreateThreadingOptions(decode)") || threadingOptions == nullptr) {
    return nullptr;
    }
    const CpuThreadPlan& plan = cpuThreadPlan();
    const int intraThreads = std::max(1, plan.decodeIntraThreads);
    gDecodeThreadAffinityContext.configure(plan.decodeCpus);
    bool threadingOk = true;
    threadingOk = logOrtStatus(api, api->SetGlobalIntraOpNumThreads(threadingOptions, intraThreads),
                   "SetGlobalIntraOpNumThreads(decode)") && threadingOk;
    threadingOk = logOrtStatus(api, api->SetGlobalInterOpNumThreads(threadingOptions, 1),
                   "SetGlobalInterOpNumThreads(decode)") && threadingOk;
    // Decode pool keeps spinning so its Top-N big cores respond with low latency to each per-token run.
    threadingOk = logOrtStatus(api, api->SetGlobalSpinControl(threadingOptions, kDecodeAllowSpinning),
                   "SetGlobalSpinControl(decode)") && threadingOk;
    threadingOk = logOrtStatus(api, api->SetGlobalDenormalAsZero(threadingOptions),
                   "SetGlobalDenormalAsZero(decode)") && threadingOk;
    threadingOk = logOrtStatus(
        api, api->SetGlobalCustomCreateThreadFn(threadingOptions, createPinnedOrtThread),
        "SetGlobalCustomCreateThreadFn(decode)") && threadingOk;
    threadingOk = logOrtStatus(
        api, api->SetGlobalCustomThreadCreationOptions(
            threadingOptions, &gDecodeThreadAffinityContext),
        "SetGlobalCustomThreadCreationOptions(decode)") && threadingOk;
    threadingOk = logOrtStatus(
        api, api->SetGlobalCustomJoinThreadFn(threadingOptions, joinPinnedOrtThread),
        "SetGlobalCustomJoinThreadFn(decode)") && threadingOk;
    if (!plan.decodeAffinity.empty()) {
    threadingOk = logOrtStatus(
        api, api->SetGlobalIntraOpThreadAffinity(
            threadingOptions, plan.decodeAffinity.c_str()),
        "SetGlobalIntraOpThreadAffinity(decode)") && threadingOk;
    }
    if (!threadingOk) {
    api->ReleaseThreadingOptions(threadingOptions);
    return nullptr;
    }
    const bool envCreated = logOrtStatus(
        api, api->CreateEnvWithGlobalThreadPools(
            ORT_LOGGING_LEVEL_ERROR, "myapplication-decode", threadingOptions, &gDecodeEnv),
        "CreateEnvWithGlobalThreadPools(decode)");
    api->ReleaseThreadingOptions(threadingOptions);
    if (!envCreated || gDecodeEnv == nullptr) {
        gDecodeEnv = nullptr;
        return nullptr;
    }
    gDecodeEnvUsesGlobalPools = true;
    if (!registerSharedEnvArena(api, gDecodeEnv, gDecodeEnvArenaRegistered)) {
        api->ReleaseEnv(gDecodeEnv);
        gDecodeEnv = nullptr;
    gDecodeEnvUsesGlobalPools = false;
        return nullptr;
    }
    LOGI("ORT: decode pool ready (threads=%d cpus=%s)",
     intraThreads, cpuIdList(plan.decodeCpus).c_str());
    return gDecodeEnv;
}

// ── Shared initializers ────────────────────────────────────────────────────────────────────────
// The merged strategy graphs (LLM_{Text,Image,Video}{Prefill,Decode}{Greedy,...}) all reference ONE
// external weight blob (LLM_SharedInitializers.onnx.data). Deserializing it per session would cost
// ~450MB x N graphs, so instead we map the blob ONCE and hand every merged session the same set of
// CPU OrtValues via OrtApi::AddInitializer (ORT then reuses that instance instead of reading the file).
// The OrtValues and their backing are process-global and outlive all sessions (AddInitializer contract).
// The offsets/lengths come from the small companion .onnx, parsed with the minimal protobuf reader
// below (the NDK build links no protobuf runtime). Mirrors Inference_Qwen_ONNX.py attach_shared_initializers.
namespace nativellm_pb {

// Minimal protobuf wire reader. Verified byte-identical to onnx.load over the real 334-initializer
// LLM_SharedInitializers.onnx (name, data_type, dims, external location/offset/length all matched).
struct Reader {
    const uint8_t* p;
    const uint8_t* end;
    bool ok = true;
    Reader(const uint8_t* d, size_t n) : p(d), end(d + n) {}
    bool eof() const { return p >= end; }
    uint64_t varint() {
        uint64_t v = 0; int shift = 0;
        while (p < end && shift < 64) {
            const uint8_t b = *p++;
            v |= uint64_t(b & 0x7F) << shift;
            if (!(b & 0x80)) return v;
            shift += 7;
        }
        ok = false; return v;
    }
    std::pair<const uint8_t*, size_t> lendelim() {
        const uint64_t len = varint();
        const uint8_t* s = p;
        if (p + len > end) { ok = false; return {s, 0}; }
        p += len;
        return {s, size_t(len)};
    }
    void skip(uint32_t wire) {
        switch (wire) {
            case 0: varint(); break;
            case 1: p += 8; break;
            case 5: p += 4; break;
            case 2: { (void)lendelim(); break; }
            default: ok = false; break;
        }
        if (p > end) ok = false;
    }
};

struct SharedInit {
    std::string name;
    int32_t     dataType = 0;
    std::vector<int64_t> dims;
    std::string location;
    int64_t     offset = 0;
    int64_t     length = 0;
};

// StringStringEntryProto: key = field 1 (string), value = field 2 (string).
inline void parseEntry(const uint8_t* d, size_t n, std::string& key, std::string& val) {
    Reader r(d, n);
    while (!r.eof() && r.ok) {
        const uint64_t tag = r.varint();
        const uint32_t field = uint32_t(tag >> 3), wire = uint32_t(tag & 7);
        if (wire == 2) {
            auto s = r.lendelim();
            if (field == 1) key.assign((const char*)s.first, s.second);
            else if (field == 2) val.assign((const char*)s.first, s.second);
        } else {
            r.skip(wire);
        }
    }
}

// TensorProto: dims=1 (repeated int64), data_type=2 (int32), name=8 (string), external_data=13 (repeated).
inline SharedInit parseTensor(const uint8_t* d, size_t n) {
    SharedInit init;
    Reader r(d, n);
    while (!r.eof() && r.ok) {
        const uint64_t tag = r.varint();
        const uint32_t field = uint32_t(tag >> 3), wire = uint32_t(tag & 7);
        if (field == 1 && wire == 0) {                 // dims (unpacked)
            init.dims.push_back((int64_t)r.varint());
        } else if (field == 1 && wire == 2) {          // dims (packed)
            auto s = r.lendelim();
            Reader pr(s.first, s.second);
            while (!pr.eof() && pr.ok) init.dims.push_back((int64_t)pr.varint());
        } else if (field == 2 && wire == 0) {          // data_type
            init.dataType = (int32_t)r.varint();
        } else if (field == 8 && wire == 2) {          // name
            auto s = r.lendelim();
            init.name.assign((const char*)s.first, s.second);
        } else if (field == 13 && wire == 2) {         // external_data entry
            auto s = r.lendelim();
            std::string k, v;
            parseEntry(s.first, s.second, k, v);
            if (k == "location") init.location = v;
            else if (k == "offset") init.offset = v.empty() ? 0 : std::strtoll(v.c_str(), nullptr, 10);
            else if (k == "length") init.length = v.empty() ? 0 : std::strtoll(v.c_str(), nullptr, 10);
        } else {
            r.skip(wire);
        }
    }
    return init;
}

// ModelProto(graph=7) -> GraphProto(initializer=5, repeated) -> TensorProto.
inline std::vector<SharedInit> parseSharedInitializers(const uint8_t* data, size_t len) {
    std::vector<SharedInit> out;
    Reader model(data, len);
    while (!model.eof() && model.ok) {
        const uint64_t tag = model.varint();
        const uint32_t field = uint32_t(tag >> 3), wire = uint32_t(tag & 7);
        if (field == 7 && wire == 2) {                 // graph
            auto g = model.lendelim();
            Reader graph(g.first, g.second);
            while (!graph.eof() && graph.ok) {
                const uint64_t gtag = graph.varint();
                const uint32_t gfield = uint32_t(gtag >> 3), gwire = uint32_t(gtag & 7);
                if (gfield == 5 && gwire == 2) {        // initializer
                    auto t = graph.lendelim();
                    out.push_back(parseTensor(t.first, t.second));
                } else {
                    graph.skip(gwire);
                }
            }
        } else {
            model.skip(wire);
        }
    }
    return out;
}

inline void appendVarint(std::vector<char>& output, uint64_t value) {
    while (value >= 0x80) {
        output.push_back(static_cast<char>((value & 0x7F) | 0x80));
        value >>= 7;
    }
    output.push_back(static_cast<char>(value));
}

inline void appendBytes(std::vector<char>& output, const uint8_t* data, size_t size) {
    output.insert(output.end(), reinterpret_cast<const char*>(data),
                  reinterpret_cast<const char*>(data + size));
}

inline bool isInjectedInitializer(const std::vector<std::string>& names,
                                  const std::string& name) {
    return std::find(names.begin(), names.end(), name) != names.end();
}

// AddInitializer replaces tensor contents, but ORT validates external-data paths before applying that
// replacement when a model is loaded from bytes. Retarget only the location/offset values of injected
// initializers to the uncompressed asset's real APK backing range. All other protobuf fields are copied
// byte-for-byte, including inline and graph-local initializers.
inline bool rewriteExternalDataEntry(const uint8_t* data, size_t size,
                                     const std::string& fileName, int64_t offsetBase,
                                     std::vector<char>& output,
                                     bool& rewroteLocation, bool& rewroteOffset) {
    output.clear();
    std::string key;
    std::string oldValue;
    parseEntry(data, size, key, oldValue);
    std::string newValue;
    if (key == "location") {
        newValue = fileName;
        rewroteLocation = true;
    } else if (key == "offset") {
        const int64_t oldOffset = oldValue.empty()
                ? 0 : std::strtoll(oldValue.c_str(), nullptr, 10);
        if (oldOffset < 0 || offsetBase > std::numeric_limits<int64_t>::max() - oldOffset) {
            return false;
        }
        newValue = std::to_string(offsetBase + oldOffset);
        rewroteOffset = true;
    } else {
        appendBytes(output, data, size);
        return true;
    }

    output.reserve(size + newValue.size());
    Reader entry(data, size);
    while (!entry.eof() && entry.ok) {
        const uint8_t* fieldStart = entry.p;
        const uint64_t tag = entry.varint();
        const uint32_t field = static_cast<uint32_t>(tag >> 3);
        const uint32_t wire = static_cast<uint32_t>(tag & 7);
        if (field == 2 && wire == 2) {
            (void)entry.lendelim();
            if (!entry.ok) {
                return false;
            }
            appendVarint(output, tag);
            appendVarint(output, newValue.size());
            output.insert(output.end(), newValue.begin(), newValue.end());
        } else {
            entry.skip(wire);
            if (!entry.ok) {
                return false;
            }
            appendBytes(output, fieldStart, static_cast<size_t>(entry.p - fieldStart));
        }
    }
    return entry.ok;
}

inline bool retargetInjectedTensorExternalData(const uint8_t* data, size_t size,
                                               const std::string& fileName, int64_t offsetBase,
                                               std::vector<char>& output) {
    output.clear();
    output.reserve(size + fileName.size());
    bool rewroteLocation = false;
    bool rewroteOffset = false;
    Reader tensor(data, size);
    while (!tensor.eof() && tensor.ok) {
        const uint8_t* fieldStart = tensor.p;
        const uint64_t tag = tensor.varint();
        const uint32_t field = static_cast<uint32_t>(tag >> 3);
        const uint32_t wire = static_cast<uint32_t>(tag & 7);
        if (field == 13 && wire == 2) {  // TensorProto.external_data
            const auto entryBytes = tensor.lendelim();
            if (!tensor.ok) {
                return false;
            }
            std::vector<char> rewrittenEntry;
            if (!rewriteExternalDataEntry(entryBytes.first, entryBytes.second,
                                          fileName, offsetBase, rewrittenEntry,
                                          rewroteLocation, rewroteOffset)) {
                return false;
            }
            appendVarint(output, tag);
            appendVarint(output, rewrittenEntry.size());
            output.insert(output.end(), rewrittenEntry.begin(), rewrittenEntry.end());
        } else {
            tensor.skip(wire);
            if (!tensor.ok) {
                return false;
            }
            appendBytes(output, fieldStart, static_cast<size_t>(tensor.p - fieldStart));
        }
    }
    return tensor.ok && rewroteLocation && rewroteOffset;
}

inline bool rewriteGraphInjectedInitializers(const uint8_t* data, size_t size,
                                             const std::vector<std::string>& injectedNames,
                                             const std::string& fileName, int64_t offsetBase,
                                             std::vector<char>& output, size_t& rewrittenCount) {
    output.clear();
    output.reserve(size);
    Reader graph(data, size);
    while (!graph.eof() && graph.ok) {
        const uint8_t* fieldStart = graph.p;
        const uint64_t tag = graph.varint();
        const uint32_t field = static_cast<uint32_t>(tag >> 3);
        const uint32_t wire = static_cast<uint32_t>(tag & 7);
        if (field == 5 && wire == 2) {  // GraphProto.initializer
            const auto tensorBytes = graph.lendelim();
            if (!graph.ok) {
                return false;
            }
            const SharedInit initializer = parseTensor(tensorBytes.first, tensorBytes.second);
            if (!initializer.location.empty() &&
                isInjectedInitializer(injectedNames, initializer.name)) {
                std::vector<char> rewrittenTensor;
                if (!retargetInjectedTensorExternalData(
                        tensorBytes.first, tensorBytes.second, fileName, offsetBase,
                        rewrittenTensor)) {
                    return false;
                }
                appendVarint(output, tag);
                appendVarint(output, rewrittenTensor.size());
                output.insert(output.end(), rewrittenTensor.begin(), rewrittenTensor.end());
                ++rewrittenCount;
            } else {
                appendBytes(output, fieldStart, static_cast<size_t>(graph.p - fieldStart));
            }
        } else {
            graph.skip(wire);
            if (!graph.ok) {
                return false;
            }
            appendBytes(output, fieldStart, static_cast<size_t>(graph.p - fieldStart));
        }
    }
    return graph.ok;
}

inline bool rewriteModelInjectedInitializers(const void* modelData, size_t modelSize,
                                             const std::vector<std::string>& injectedNames,
                                             const std::string& fileName, int64_t offsetBase,
                                             std::vector<char>& output, size_t& rewrittenCount) {
    output.clear();
    output.reserve(modelSize);
    rewrittenCount = 0;
    Reader model(static_cast<const uint8_t*>(modelData), modelSize);
    while (!model.eof() && model.ok) {
        const uint8_t* fieldStart = model.p;
        const uint64_t tag = model.varint();
        const uint32_t field = static_cast<uint32_t>(tag >> 3);
        const uint32_t wire = static_cast<uint32_t>(tag & 7);
        if (field == 7 && wire == 2) {  // ModelProto.graph
            const auto graphBytes = model.lendelim();
            if (!model.ok) {
                return false;
            }
            std::vector<char> rewrittenGraph;
            if (!rewriteGraphInjectedInitializers(
                    graphBytes.first, graphBytes.second, injectedNames,
                    fileName, offsetBase, rewrittenGraph, rewrittenCount)) {
                return false;
            }
            appendVarint(output, tag);
            appendVarint(output, rewrittenGraph.size());
            output.insert(output.end(), rewrittenGraph.begin(), rewrittenGraph.end());
        } else {
            model.skip(wire);
            if (!model.ok) {
                return false;
            }
            appendBytes(output, fieldStart, static_cast<size_t>(model.p - fieldStart));
        }
    }
    return model.ok && rewrittenCount > 0;
}

}  // namespace nativellm_pb

struct SharedInitializerStore {
    const OrtApi*   api = nullptr;
    OrtMemoryInfo*  memInfo = nullptr;
    // Blob backing (exactly one is active):
    void*   mmapBase = nullptr;   // mmap path (file staged in cache/storage)
    size_t  mmapLen  = 0;
    AAsset* asset    = nullptr;   // asset path (zero-copy AAsset_getBuffer, kept open)
    std::vector<char> blob;       // read-into-memory fallback
    const uint8_t* base = nullptr;
    size_t         baseLen = 0;
    std::string externalFileDirectory;
    std::string externalFileName;
    int64_t externalFileOffset = 0;
    std::vector<std::string> names;
    std::vector<OrtValue*>   values;   // one per shared initializer; outlive every session
    bool ready = false;
};

inline SharedInitializerStore gSharedInitStore;

inline bool readFileFully(const std::string& path, std::vector<char>& out) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (size <= 0) {
        std::fclose(f);
        return false;
    }
    out.resize(static_cast<size_t>(size));
    const size_t got = std::fread(out.data(), 1, out.size(), f);
    std::fclose(f);
    if (got != out.size()) {
        out.clear();
        return false;
    }
    return true;
}

inline void releaseSharedInitializers() {
    SharedInitializerStore& s = gSharedInitStore;
    if (s.api != nullptr) {
        for (OrtValue*& v : s.values) {
            if (v != nullptr) { s.api->ReleaseValue(v); v = nullptr; }
        }
        if (s.memInfo != nullptr) { s.api->ReleaseMemoryInfo(s.memInfo); s.memInfo = nullptr; }
    }
    s.values.clear();
    s.names.clear();
    if (s.mmapBase != nullptr && s.mmapLen > 0) { munmap(s.mmapBase, s.mmapLen); }
    s.mmapBase = nullptr; s.mmapLen = 0;
    if (s.asset != nullptr) { AAsset_close(s.asset); s.asset = nullptr; }
    s.blob.clear();
    s.base = nullptr; s.baseLen = 0;
    s.externalFileDirectory.clear();
    s.externalFileName.clear();
    s.externalFileOffset = 0;
    s.ready = false;
}

// Map the required shared weight blob and build one CPU OrtValue per initializer. Idempotent.
inline bool loadSharedInitializers(const OrtApi* api, JNIEnv* env, jobject assetManager, bool lowMemoryMode,
                                    const char* onnxName, const char* dataName,
                                    const char* assetDirectory = nullptr) {
    SharedInitializerStore& s = gSharedInitStore;
    if (s.ready) {
        return true;
    }
    releaseSharedInitializers();
    s.api = api;

    // 1. Structure file bytes (small): prefer a staged cache/storage file, else the APK asset.
    std::vector<char> onnxBytes;
    const std::string cacheOnnx   = cache_path + onnxName;
    const std::string storageOnnx = storage_path + onnxName;
    if (cacheFileExists(cacheOnnx)) {
        readFileFully(cacheOnnx, onnxBytes);
    } else if (cacheFileExists(storageOnnx)) {
        readFileFully(storageOnnx, onnxBytes);
    }
    AAssetManager* mgr = (!lowMemoryMode && assetManager != nullptr)
            ? AAssetManager_fromJava(env, assetManager) : nullptr;
    if (onnxBytes.empty() && mgr != nullptr) {
        const std::string onnxAssetPath = assetPath(assetDirectory, onnxName);
        AAsset* a = AAssetManager_open(mgr, onnxAssetPath.c_str(), AASSET_MODE_BUFFER);
        if (a != nullptr) {
            readAssetFully(a, onnxBytes, onnxAssetPath);
            AAsset_close(a);
        }
    }
    if (onnxBytes.empty()) {
        LOGI("shared initializers: %s not found; merged sessions will resolve weights individually", onnxName);
        return false;
    }

    // 2. Weight blob: mmap a staged file (zero copy) or zero-copy the (uncompressed) APK asset.
    const std::string cacheData   = cache_path + dataName;
    const std::string storageData = storage_path + dataName;
    std::string dataPath;
    if (cacheFileExists(cacheData))        dataPath = cacheData;
    else if (cacheFileExists(storageData)) dataPath = storageData;
    if (!dataPath.empty()) {
        const int fd = open(dataPath.c_str(), O_RDONLY);
        if (fd >= 0) {
            struct stat st{};
            if (fstat(fd, &st) == 0 && st.st_size > 0) {
                void* m = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
                if (m != MAP_FAILED) {
                    s.mmapBase = m;
                    s.mmapLen  = static_cast<size_t>(st.st_size);
                    s.base     = static_cast<const uint8_t*>(m);
                    s.baseLen  = s.mmapLen;
                    const size_t slash = dataPath.find_last_of('/');
                    s.externalFileDirectory = slash == std::string::npos
                            ? "." : dataPath.substr(0, slash);
                    s.externalFileName = slash == std::string::npos
                            ? dataPath : dataPath.substr(slash + 1);
                }
            }
            close(fd);
        }
    }
    if (s.base == nullptr && mgr != nullptr) {
        const std::string dataAssetPath = assetPath(assetDirectory, dataName);
        s.asset = AAssetManager_open(mgr, dataAssetPath.c_str(), AASSET_MODE_BUFFER);
        if (s.asset != nullptr) {
            const void* buf = AAsset_getBuffer(s.asset);
            const off_t len = AAsset_getLength(s.asset);
            if (buf != nullptr && len > 0) {
                s.base    = static_cast<const uint8_t*>(buf);
                s.baseLen = static_cast<size_t>(len);
                off64_t assetStart = 0;
                off64_t assetLength = 0;
                const int assetFd = AAsset_openFileDescriptor64(
                        s.asset, &assetStart, &assetLength);
                if (assetFd >= 0) {
                    char fdPath[64];
                    std::snprintf(fdPath, sizeof(fdPath), "/proc/self/fd/%d", assetFd);
                    std::vector<char> resolvedPath(4096, '\0');
                    const ssize_t pathLength = readlink(
                            fdPath, resolvedPath.data(), resolvedPath.size() - 1);
                    close(assetFd);
                    if (pathLength > 0 && assetLength == len) {
                        resolvedPath[static_cast<size_t>(pathLength)] = '\0';
                        const std::string apkPath(resolvedPath.data(),
                                                  static_cast<size_t>(pathLength));
                        const size_t slash = apkPath.find_last_of('/');
                        if (slash != std::string::npos) {
                            s.externalFileDirectory = apkPath.substr(0, slash);
                            s.externalFileName = apkPath.substr(slash + 1);
                            s.externalFileOffset = static_cast<int64_t>(assetStart);
                        }
                    }
                }
            } else if (readAssetFully(s.asset, s.blob, dataAssetPath)) {
                s.base    = reinterpret_cast<const uint8_t*>(s.blob.data());
                s.baseLen = s.blob.size();
            }
        }
    }
    if (s.base == nullptr) {
        LOGE("shared initializers: weight blob %s not found; cannot attach shared weights", dataName);
        releaseSharedInitializers();
        return false;
    }
    if (s.externalFileDirectory.empty() || s.externalFileName.empty()) {
        LOGE("shared initializers: no filesystem backing for zero-copy external validation");
        releaseSharedInitializers();
        return false;
    }

    // 3. CPU memory info + one OrtValue per initializer over the mapped blob.
    if (!logOrtStatus(api, api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &s.memInfo),
                      "CreateCpuMemoryInfo(shared_init)") || s.memInfo == nullptr) {
        releaseSharedInitializers();
        return false;
    }
    const std::vector<nativellm_pb::SharedInit> inits =
            nativellm_pb::parseSharedInitializers(reinterpret_cast<const uint8_t*>(onnxBytes.data()), onnxBytes.size());
    if (inits.empty()) {
        LOGE("shared initializers: parsed 0 initializers from %s", onnxName);
        releaseSharedInitializers();
        return false;
    }
    s.names.reserve(inits.size());
    s.values.reserve(inits.size());
    for (const nativellm_pb::SharedInit& init : inits) {
        if (init.name.empty() || init.location != dataName || init.offset < 0 || init.length < 0) {
            LOGE("shared initializers: invalid external metadata for '%s' "
                 "(location='%s', offset=%lld, length=%lld; expected location='%s')",
                 init.name.c_str(), init.location.c_str(), static_cast<long long>(init.offset),
                 static_cast<long long>(init.length), dataName);
            releaseSharedInitializers();
            return false;
        }
        const ONNXTensorElementDataType type = static_cast<ONNXTensorElementDataType>(init.dataType);
        // Sub-byte packed tensors cannot be represented by CreateTensorWithDataAsOrtValue.
        if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4 || type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4) {
            LOGE("shared initializers: unsupported sub-byte type %d for %s",
                 static_cast<int>(type), init.name.c_str());
            releaseSharedInitializers();
            return false;
        }
        int64_t elems = 1;
        for (int64_t d : init.dims) {
            if (d < 0 || (d > 0 && elems > std::numeric_limits<int64_t>::max() / d)) {
                LOGE("shared initializers: invalid/overflowing dimensions for %s", init.name.c_str());
                releaseSharedInitializers();
                return false;
            }
            elems *= (d > 0 ? d : 1);
        }
        const size_t elementSize = tensorElementSize(type);
        if (elementSize == 0) {
            LOGE("shared initializers: unsupported element type %d for %s",
                 static_cast<int>(type), init.name.c_str());
            releaseSharedInitializers();
            return false;
        }
        if (init.length == 0 && static_cast<uint64_t>(elems) >
                std::numeric_limits<size_t>::max() / elementSize) {
            LOGE("shared initializers: byte size overflow for %s", init.name.c_str());
            releaseSharedInitializers();
            return false;
        }
        size_t bytes = (init.length > 0)
                ? static_cast<size_t>(init.length)
                : static_cast<size_t>(elems) * elementSize;
        const size_t offset = static_cast<size_t>(init.offset);
        if (offset > s.baseLen || bytes > s.baseLen - offset) {
            LOGE("shared initializers: %s offset+len exceeds blob (%lld+%zu > %zu)",
                 init.name.c_str(), static_cast<long long>(init.offset), bytes, s.baseLen);
            releaseSharedInitializers();
            return false;
        }
        void* dataPtr = const_cast<uint8_t*>(s.base + offset);
        OrtValue* value = nullptr;
        if (!logOrtStatus(api, api->CreateTensorWithDataAsOrtValue(
                s.memInfo, dataPtr, bytes, init.dims.data(), init.dims.size(), type, &value),
                "CreateTensorWithDataAsOrtValue(shared_init)") || value == nullptr) {
            releaseSharedInitializers();
            return false;
        }
        s.names.push_back(init.name);
        s.values.push_back(value);
    }
    s.ready = true;
    LOGI("shared initializers ready: %zu tensors attached from %s (%zu MB blob)",
         s.values.size(), dataName, s.baseLen >> 20);
    return true;
}

// Inject the required shared initializers into a merged session before creation.
inline bool attachSharedInitializers(const OrtApi* api, OrtSessionOptions* options) {
    SharedInitializerStore& s = gSharedInitStore;
    if (!s.ready || options == nullptr) {
        LOGE("shared initializers: attach requested before the store was ready");
        return false;
    }
    for (size_t i = 0; i < s.names.size(); ++i) {
        if (!logOrtStatus(api, api->AddInitializer(options, s.names[i].c_str(), s.values[i]),
                          "AddInitializer(shared)")) {
            return false;
        }
    }
    return true;
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
        if (!ok_status(m.api, m.api->CreateMemoryInfo(
                "QnnHtpShared", OrtDeviceAllocator, 0, OrtMemTypeDefault, &shared_info),
                "CreateMemoryInfo(QnnHtpShared)") || shared_info == nullptr) {
            return false;
        }
        OrtAllocator* dev_alloc = nullptr;
        if (!ok_status(m.api, m.api->CreateAllocator(m.session, shared_info, &dev_alloc),
                       "CreateAllocator(QnnHtpShared)") || dev_alloc == nullptr) {
            m.api->ReleaseMemoryInfo(shared_info);
            return false;
        }
        m.api->ReleaseMemoryInfo(shared_info);
        m.ioAllocator = dev_alloc;
        if (!ok_status(m.api, m.api->AllocatorGetInfo(dev_alloc, &m.ioMemoryInfo),
                       "AllocatorGetInfo(QnnHtpShared)")) {
            m.api->ReleaseAllocator(dev_alloc);
            m.ioAllocator = nullptr;
            m.ioMemoryInfo = nullptr;
            return false;
        }
        LOGI("Model %d: QNN HTP shared-memory allocator active: KV cache is zero-copy on-device",
             static_cast<int>(m.id));
    }

    if (!enumerateSessionIO(m, true) || !enumerateSessionIO(m, false)) {
        return false;
    }
    return true;
}

inline bool ortLoadModelSession(ModelRuntime& m, const char* onnxName, const char* externalName,
                                JNIEnv* env, jobject assetManager, int epType, bool lowMemoryMode,
                                bool attachShared = false, const char* assetDirectory = nullptr,
                                bool forcePrefillWorkload = false, bool forceDecodeWorkload = false) {
    m.fileName = (onnxName != nullptr) ? onnxName : "";
    m.externalFileName = (externalName != nullptr && externalName[0] != '\0') ? externalName : "";
    m.mergedGraph = attachShared;
    // Decode-latency workloads run on the small Top-N big-core decode pool: the merged *Decode* strategy
    // graphs plus the tiny per-token KV/RoPE control graphs (forceDecodeWorkload). Everything else — merged
    // prefill strategy graphs, LLM_Vision, and the preprocess graphs — stays on the wide prefill pool.
    m.decodeGraph = forceDecodeWorkload || (attachShared && m.fileName.find("Decode") != std::string::npos);
    m.prefillWorkload = !m.decodeGraph && (forcePrefillWorkload || attachShared);
    m.api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (m.session != nullptr) {
        LOGI("Model already loaded, skipping reload: %s", m.fileName.c_str());
        return true;
    }
    const bool useDecodeThreadPool = epType == EP_CPU && m.decodeGraph;
    m.env = useDecodeThreadPool ? getDecodeEnv(m.api) : getSharedEnv(m.api);
    if (m.env == nullptr) {
        return false;
    }

    OrtStatus* status = nullptr;
    OrtSessionOptions* session_options = nullptr;
    std::unique_ptr<AAsset, decltype(&AAsset_close)> modelAsset(nullptr, AAsset_close);
    std::unique_ptr<AAsset, decltype(&AAsset_close)> externalAsset(nullptr, AAsset_close);
    const void* modelAssetBuffer = nullptr;
    const void* externalAssetBuffer = nullptr;
    size_t externalAssetSize = 0;
    {
        std::vector<char> fileBuffer;
        std::vector<char> fileBuffer_external;
        std::vector<char> normalizedModelBuffer;
        off_t fileSize = 0;
        bool use_storage_path = false;
        if (!lowMemoryMode) {
            if (assetManager != nullptr) {
                AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
                if (mgr == nullptr) {
                    LOGE("AAssetManager_fromJava failed");
                    return false;
                }
                const std::string modelAssetPath = assetPath(assetDirectory, m.fileName.c_str());
                modelAsset.reset(AAssetManager_open(mgr, modelAssetPath.c_str(), AASSET_MODE_BUFFER));
                if (modelAsset != nullptr) {
                    modelAssetBuffer = AAsset_getBuffer(modelAsset.get());
                    fileSize = AAsset_getLength(modelAsset.get());
                }
                if (modelAssetBuffer == nullptr || fileSize <= 0) {
                    if (!readAssetFully(modelAsset.get(), fileBuffer, modelAssetPath)) {
                        return false;
                    }
                    modelAssetBuffer = fileBuffer.data();
                    fileSize = static_cast<off_t>(fileBuffer.size());
                }
                if (!m.externalFileName.empty()) {
                    const std::string externalAssetPath = assetPath(
                            assetDirectory, m.externalFileName.c_str());
                    externalAsset.reset(AAssetManager_open(
                            mgr, externalAssetPath.c_str(), AASSET_MODE_BUFFER));
                    if (externalAsset == nullptr) {
                        LOGE("External data asset '%s' not found", m.externalFileName.c_str());
                        return false;
                    }
                    externalAssetBuffer = AAsset_getBuffer(externalAsset.get());
                    const off_t externalLength = AAsset_getLength(externalAsset.get());
                    if (externalAssetBuffer != nullptr && externalLength > 0) {
                        externalAssetSize = static_cast<size_t>(externalLength);
                    } else {
                        if (!readAssetFully(externalAsset.get(), fileBuffer_external,
                                            externalAssetPath)) {
                            return false;
                        }
                        externalAssetBuffer = fileBuffer_external.data();
                        externalAssetSize = fileBuffer_external.size();
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

        const bool sessionUsesGlobalPools = useDecodeThreadPool
            ? gDecodeEnvUsesGlobalPools
            : gSharedEnvUsesGlobalPools;
        if (sessionUsesGlobalPools &&
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
        if (epType == EP_CPU && attachShared) {
            addRunConfig("memory.enable_memory_arena_shrinkage", "cpu:0");
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

        if (!sessionUsesGlobalPools) {
            const CpuThreadPlan& threadPlan = cpuThreadPlan();
            const std::vector<int>& sessionCpus = useDecodeThreadPool
                ? threadPlan.decodeCpus : threadPlan.prefillCpus;
            const std::string& sessionAffinity = useDecodeThreadPool
                ? threadPlan.decodeAffinity : threadPlan.prefillAffinity;
            const int intraThreads = useDecodeThreadPool
                ? threadPlan.decodeIntraThreads
                : std::max(1, static_cast<int>(sessionCpus.size()));
            if (epType == EP_CPU && attachShared) {
                m.threadAffinityContext = std::make_unique<OrtThreadAffinityContext>();
                m.threadAffinityContext->configure(sessionCpus);
                check(m.api->SessionOptionsSetCustomCreateThreadFn(
                              session_options, createPinnedOrtThread),
                      "SessionOptionsSetCustomCreateThreadFn");
                check(m.api->SessionOptionsSetCustomThreadCreationOptions(
                              session_options, m.threadAffinityContext.get()),
                      "SessionOptionsSetCustomThreadCreationOptions");
                check(m.api->SessionOptionsSetCustomJoinThreadFn(
                              session_options, joinPinnedOrtThread),
                      "SessionOptionsSetCustomJoinThreadFn");
            }
            check(m.api->SetInterOpNumThreads(session_options, 1), "SetInterOpNumThreads");
            const std::string sessionDynamicBlockBase = std::to_string(
                std::max(1, intraThreads - 1));
            addSessionConfig("session.dynamic_block_base", sessionDynamicBlockBase.c_str());
            if (epType != EP_XNNPACK && !sessionAffinity.empty()) {
            addSessionConfig("session.intra_op_thread_affinities", sessionAffinity.c_str());
            }
            check(m.api->SetIntraOpNumThreads(session_options, intraThreads), "SetIntraOpNumThreads");
            LOGI("Model %d thread pool: phase=%s threads=%d cpus=%s affinity=%s",
             static_cast<int>(m.id), useDecodeThreadPool ? "decode" : "prefill",
             intraThreads, cpuIdList(sessionCpus).c_str(), sessionAffinity.c_str());
        }

        for (const OrtConfigEntry& e : kOrtSessionConfigs) {
            addSessionConfig(e.key, e.value);
        }
        const bool usesDedicatedProvider = epType == EP_XNNPACK || epType == EP_QNN;
        addSessionConfig("session.disable_cpu_ep_fallback", usesDedicatedProvider ? "1" : "0");
          check(m.api->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL),
              "SetSessionGraphOptimizationLevel");
        // Native FP16-capable ARM64 devices should retain the graph's declared precision.
        const bool disableFp16Optimizers = deviceIsArm64V82aOrAbove();
        std::string disabledOptimizers;
        auto appendDisabledOptimizers = [&](const char* optimizerNames) {
            if (optimizerNames == nullptr || optimizerNames[0] == '\0') {
                return;
            }
            if (!disabledOptimizers.empty()) {
                disabledOptimizers.push_back(';');
            }
            disabledOptimizers.append(optimizerNames);
        };
        if (disableFp16Optimizers) {
            appendDisabledOptimizers("CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer");
        }
        if (epType == EP_CPU && attachShared && kUseCpuMergedOptimizerExceptions) {
            appendDisabledOptimizers(kCpuMergedDisabledOptimizers);
        }
        addSessionConfig("optimization.disable_specified_optimizers", disabledOptimizers.c_str());
        if (!disabledOptimizers.empty()) {
            LOGI("Model %d: disabled ORT optimizers: %s", static_cast<int>(m.id), disabledOptimizers.c_str());
        }
        if (epType == EP_QNN) {
            addSessionConfig("ep.dynamic.qnn_htp_performance_mode", "burst");
        }
        if (attachShared) {
            addSessionConfig("session.model_external_initializers_file_folder_path",
                             gSharedInitStore.externalFileDirectory.c_str());
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

        // Merged strategy graphs share one weight blob injected as user-managed initializers; this makes
        // ORT reuse the single mapped copy instead of deserializing ~450MB per session.
        if (attachShared && !attachSharedInitializers(m.api, session_options)) {
            m.api->ReleaseSessionOptions(session_options);
            return false;
        }

        if (attachShared && !lowMemoryMode) {
            size_t normalizedInitializers = 0;
            if (!nativellm_pb::rewriteModelInjectedInitializers(
                    modelAssetBuffer, static_cast<size_t>(fileSize), gSharedInitStore.names,
                    gSharedInitStore.externalFileName, gSharedInitStore.externalFileOffset,
                    normalizedModelBuffer, normalizedInitializers)) {
                LOGE("Model %d: could not retarget injected external initializers",
                     static_cast<int>(m.id));
                m.api->ReleaseSessionOptions(session_options);
                return false;
            }
            modelAssetBuffer = normalizedModelBuffer.data();
            fileSize = static_cast<off_t>(normalizedModelBuffer.size());
            LOGI("Model %d: retargeted %zu injected initializers to %s + %lld",
                 static_cast<int>(m.id), normalizedInitializers,
                 gSharedInitStore.externalFileName.c_str(),
                 static_cast<long long>(gSharedInitStore.externalFileOffset));
        }

        OrtPrepackedWeightsContainer* prepackedWeights = getSharedPrepackedWeightsContainer(m.api);
        if (kUseSharedPrepackedWeightsContainer && prepackedWeights == nullptr) {
            m.api->ReleaseSessionOptions(session_options);
            return false;
        }
        if (lowMemoryMode) {
            const std::string path = use_storage_path ? (storage_path + m.fileName) : resolveModelPath(m);
            if (prepackedWeights != nullptr) {
                status = m.api->CreateSessionWithPrepackedWeightsContainer(
                        m.env, path.c_str(), session_options, prepackedWeights, &m.session);
            } else {
                status = m.api->CreateSession(m.env, path.c_str(), session_options, &m.session);
            }
        } else {
            if (externalAssetBuffer != nullptr && externalAssetSize > 0) {
                const char* external_file_names[]   = {m.externalFileName.c_str()};
                char*       external_file_buffers[] = {
                        const_cast<char*>(static_cast<const char*>(externalAssetBuffer))};
                size_t      external_file_sizes[]   = {externalAssetSize};
                if (!ok_status(m.api, m.api->AddExternalInitializersFromFilesInMemory(
                        session_options, external_file_names, external_file_buffers, external_file_sizes, 1),
                        "AddExternalInitializersFromFilesInMemory")) {
                    m.api->ReleaseSessionOptions(session_options);
                    return false;
                }
            }
            if (prepackedWeights != nullptr) {
                status = m.api->CreateSessionFromArrayWithPrepackedWeightsContainer(
                    m.env, modelAssetBuffer, fileSize, session_options, prepackedWeights, &m.session);
            } else {
                status = m.api->CreateSessionFromArray(
                    m.env, modelAssetBuffer, fileSize, session_options, &m.session);
            }
        }
    }
    m.api->ReleaseSessionOptions(session_options);
    if (!ok_status(m.api, status, "CreateSession")) {
        return false;
    }
    return extractIO(m, epType);
}
