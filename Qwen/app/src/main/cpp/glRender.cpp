#include "glRender.h"
#include <limits>

#if defined(__GNUC__) || defined(__clang__)
#define HOT_FN __attribute__((hot))
#define ALWAYS_INLINE inline __attribute__((always_inline))
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define PREFETCH(addr) __builtin_prefetch(addr, 0, 3)
#define CACHE_ALIGNED alignas(64)
#else
#define HOT_FN
#define ALWAYS_INLINE inline
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define PREFETCH(addr)
#define CACHE_ALIGNED
#endif

GLRender::GLRender(int camW, int camH)
        : sourceW_(camW > 0 ? camW : kDefaultCamSize),
            sourceH_(camH > 0 ? camH : kDefaultCamSize),
            camW_(sourceW_),
            camH_(sourceH_),
      groupsX_((((camW_ + 3) / 4) + 15) / 16),   // 4 px/invocation in x -> ceil(W/4) invocation columns
      groupsY_((camH_ + 15) / 16),
      sizeBytes_(camW_ * camH_ * BYTES_PER_PIXEL),
      chwBytes_(camW_ * camH_ * 3),
      invOutW_(1.0f / static_cast<float>(camW_)),
            invOutH_(1.0f / static_cast<float>(camH_)),
            requestedW_(camW_),
            requestedH_(camH_) {
    for (int i = 0; i < BUFFER_COUNT; ++i) {
        fences_[i] = nullptr;
    }
}

GLRender::~GLRender() {
    destroy();
}

HOT_FN void GLRender::onSurfaceCreated(std::function<std::string(const std::string&)> assetLoader) {
    // Compute-only pipeline: the visible preview is a separate TextureView camera output, so this
    // GL surface never draws on screen. We only create the external OES texture (sampled by the
    // compute shader) and the compute/SSBO capture pipeline -- no vertex/fragment program, no VBOs.
    initExternalTexture();
    initComputePipeline(assetLoader);
}

HOT_FN void GLRender::onDrawFrame(const float* vMatrix) {
    if (UNLIKELY(cameraTexId_ == 0)) return;

    // No on-screen draw. This GL surface is an invisible 1x1 that only owns the EGL context and
    // drives the compute capture; the live preview comes from a separate TextureView camera output.
    // So every idle frame does nothing but poll delivery, and we only touch the GPU (and only copy
    // the transform matrix) when a one-shot capture is actually pending this frame.
    if (vMatrix != nullptr && wantOneShot_.exchange(false, std::memory_order_acq_rel)) {
        memcpy(vMatrix_, vMatrix, vMatrix_size);
        if (configureOutputGeometry(requestedW_, requestedH_)) {
            dispatchCompute();
        } else {
            InferenceFrameCallback cb = takePendingCallback();
            if (cb) {
                cb(nullptr, nullptr, 0, 0, 0, 0);
            }
        }
    }
    tryDeliverCompletedBuffer();
}

void GLRender::destroy() {
    wantOneShot_.store(false, std::memory_order_release);
    InferenceFrameCallback pendingCb = takePendingCallback();
    if (pendingCb) {
        pendingCb(nullptr, nullptr, 0, 0, 0, 0);
    }

    for (int i = 0; i < BUFFER_COUNT; i++) {
        if (fences_[i]) {
            glDeleteSync(fences_[i]);
            fences_[i] = nullptr;
        }
    }

    // Delete SSBOs (previously missing)
    if (ssboIds_[0] != 0) {
        glDeleteBuffers(BUFFER_COUNT, ssboIds_);
        for (unsigned int & ssboId : ssboIds_) ssboId = 0;
    }
    if (chwSsboIds_[0] != 0) {
        glDeleteBuffers(BUFFER_COUNT, chwSsboIds_);
        for (unsigned int & id : chwSsboIds_) id = 0;
    }

    if (computeProgram_) {
        glDeleteProgram(computeProgram_);
        computeProgram_ = 0;
    }
    if (cameraTexId_) {
        glDeleteTextures(1, &cameraTexId_);
        cameraTexId_ = 0;
    }
}

GLuint GLRender::getCameraTextureId() const { return cameraTexId_; }

HOT_FN void GLRender::captureInferenceRGBAsync(int width, int height, InferenceFrameCallback cb) {
    if (UNLIKELY(!cb)) return;
    if (UNLIKELY(!computeReady_ || width <= 0 || height <= 0 || (width & 3) != 0)) {
        cb(nullptr, nullptr, 0, 0, 0, 0);
        return;
    }

    bool accepted = false;
    {
        std::lock_guard<std::mutex> lock(callbackMutex_);
        if (!oneShotCb_ && !wantOneShot_.load(std::memory_order_acquire)) {
            oneShotCb_ = std::move(cb);
            requestedW_ = width;
            requestedH_ = height;
            wantOneShot_.store(true, std::memory_order_release);
            accepted = true;
        }
    }
    if (UNLIKELY(!accepted)) {
        cb(nullptr, nullptr, 0, 0, 0, 0);
    }
}

HOT_FN void GLRender::initExternalTexture() {
    glGenTextures(1, &cameraTexId_);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, cameraTexId_);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, 0);
}

HOT_FN void GLRender::initComputePipeline(std::function<std::string(const std::string&)>& assetLoader) {
    std::string src = assetLoader("camera_compute_shader.glsl");
    if (UNLIKELY(src.empty())) { return; }

    const char* exts = (const char*)glGetString(GL_EXTENSIONS);
    if (UNLIKELY(!exts) || UNLIKELY(!strstr(exts, "GL_OES_EGL_image_external_essl3"))) {
        return;
    }

    computeProgram_ = createComputeProgram(src);
    if (UNLIKELY(computeProgram_ == 0)) return;

    glUseProgram(computeProgram_);
    locM2_ = glGetUniformLocation(computeProgram_, "uM2");
    locT_  = glGetUniformLocation(computeProgram_, "uT");
    locFit_ = glGetUniformLocation(computeProgram_, "uFit");
    locOutSize_ = glGetUniformLocation(computeProgram_, "uOutSize");
    locInvOut_ = glGetUniformLocation(computeProgram_, "uInvOut");
    GLint locCam     = glGetUniformLocation(computeProgram_, "uCamera");
    glUniform1i(locCam, 0);
    glUniform2i(locOutSize_, camW_, camH_);
    glUniform2f(locInvOut_, invOutW_, invOutH_);
    glUniform2f(locFit_, fitX_, fitY_);

    glGenBuffers(BUFFER_COUNT, ssboIds_);
    glGenBuffers(BUFFER_COUNT, chwSsboIds_);
    allocateSSBOs();
    computeReady_ = true;
}

HOT_FN void GLRender::allocateSSBOs() {
    for (int i = 0; i < BUFFER_COUNT; i++) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboIds_[i]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeBytes_, nullptr, GL_DYNAMIC_READ);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, chwSsboIds_[i]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, chwBytes_, nullptr, GL_DYNAMIC_READ);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        fences_[i] = nullptr;
    }
    curWrite_ = 0;
}

bool GLRender::configureOutputGeometry(int width, int height) {
    if (width == camW_ && height == camH_) {
        return true;
    }
    const int64_t pixels = static_cast<int64_t>(width) * static_cast<int64_t>(height);
    if (width <= 0 || height <= 0 || (width & 3) != 0 ||
        pixels > std::numeric_limits<int>::max() / BYTES_PER_PIXEL) {
        return false;
    }
    for (GLsync& fence : fences_) {
        if (fence == nullptr) {
            continue;
        }
        const GLenum status = glClientWaitSync(
                fence, GL_SYNC_FLUSH_COMMANDS_BIT, GLuint64(1000000000));
        if (status == GL_TIMEOUT_EXPIRED || status == GL_WAIT_FAILED) {
            return false;
        }
        glDeleteSync(fence);
        fence = nullptr;
    }

    camW_ = width;
    camH_ = height;
    groupsX_ = ((((camW_ + 3) / 4) + 15) / 16);
    groupsY_ = ((camH_ + 15) / 16);
    sizeBytes_ = static_cast<int>(pixels * BYTES_PER_PIXEL);
    chwBytes_ = static_cast<int>(pixels * 3);
    invOutW_ = 1.0f / static_cast<float>(camW_);
    invOutH_ = 1.0f / static_cast<float>(camH_);
    const float sourceAspect = static_cast<float>(sourceW_) / static_cast<float>(sourceH_);
    const float outputAspect = static_cast<float>(camW_) / static_cast<float>(camH_);
    if (sourceAspect > outputAspect) {
        fitX_ = 1.0f;
        fitY_ = outputAspect / sourceAspect;
    } else {
        fitX_ = sourceAspect / outputAspect;
        fitY_ = 1.0f;
    }

    glUseProgram(computeProgram_);
    glUniform2i(locOutSize_, camW_, camH_);
    glUniform2f(locInvOut_, invOutW_, invOutH_);
    allocateSSBOs();
    __android_log_print(ANDROID_LOG_INFO, "GLRender",
                        "Capture geometry source=%dx%d output=%dx%d fit=%.4fx%.4f (RGBA=%d, CHW=%d)",
                        sourceW_, sourceH_, camW_, camH_, fitX_, fitY_, sizeBytes_, chwBytes_);
    return true;
}

HOT_FN void GLRender::dispatchCompute() {
    if (UNLIKELY(!computeReady_)) return;

    if (fences_[curWrite_]) {
        GLenum status = glClientWaitSync(fences_[curWrite_], GL_SYNC_FLUSH_COMMANDS_BIT, GLuint64(1000000000));
        if (UNLIKELY(status == GL_TIMEOUT_EXPIRED || status == GL_WAIT_FAILED)) {
            return;
        }
        glDeleteSync(fences_[curWrite_]);
        fences_[curWrite_] = nullptr;
    }

    glUseProgram(computeProgram_);
    extractAffine2x2(vMatrix_, m2_, t_);
    glUniformMatrix2fv(locM2_, 1, GL_FALSE, m2_);
    glUniform2fv(locT_, 1, t_);
    glUniform2f(locFit_, fitX_, fitY_);

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssboIds_[curWrite_]);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, chwSsboIds_[curWrite_]);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, cameraTexId_);
    glDispatchCompute(groupsX_, groupsY_, 1);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

    fences_[curWrite_] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    curWrite_ = (curWrite_ + 1) % BUFFER_COUNT;
}

HOT_FN void GLRender::tryDeliverCompletedBuffer() {
    int idx = (curWrite_ + BUFFER_COUNT - 1) % BUFFER_COUNT;
    GLsync f = fences_[idx];
    if (UNLIKELY(!f)) return;

    // Non-blocking poll (timeout 0): if the GPU hasn't finished this buffer yet, just return and
    // let a later redraw pick it up. This keeps the GL thread free and the triple buffer pipelined.
    GLenum status = glClientWaitSync(f, 0, 0);
    if (LIKELY(status == GL_ALREADY_SIGNALED || status == GL_CONDITION_SATISFIED)) {
        glDeleteSync(f);
        fences_[idx] = nullptr;
        InferenceFrameCallback cb = takePendingCallback();
        if (cb) {
            // Map BOTH GPU outputs (interleaved RGBA for the preview + planar CHW for inference). A
            // single fence gated the one dispatch that wrote both, so both are ready here. The pointers
            // wrap CPU-coherent GPU memory (zero copy) and are valid only until we unmap below.
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboIds_[idx]);
            void* rgbaMapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, sizeBytes_, GL_MAP_READ_BIT);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, chwSsboIds_[idx]);
            void* chwMapped = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, chwBytes_, GL_MAP_READ_BIT);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

            cb(rgbaMapped, chwMapped,
               rgbaMapped ? camW_ : 0, rgbaMapped ? camH_ : 0,
               rgbaMapped ? static_cast<jlong>(sizeBytes_) : 0,
               chwMapped ? static_cast<jlong>(chwBytes_) : 0);

            if (rgbaMapped) {
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboIds_[idx]);
                if (UNLIKELY(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER) != GL_TRUE)) {
                    __android_log_print(ANDROID_LOG_ERROR, "GLRender", "RGBA SSBO unmap reported corrupted contents");
                }
            }
            if (chwMapped) {
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, chwSsboIds_[idx]);
                if (UNLIKELY(glUnmapBuffer(GL_SHADER_STORAGE_BUFFER) != GL_TRUE)) {
                    __android_log_print(ANDROID_LOG_ERROR, "GLRender", "CHW SSBO unmap reported corrupted contents");
                }
            }
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        }
    }
}

InferenceFrameCallback GLRender::takePendingCallback() {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    InferenceFrameCallback cb = std::move(oneShotCb_);
    oneShotCb_ = nullptr; // libc++ SBO move leaves the source non-empty; clear it so later captures aren't rejected
    return cb;
}

HOT_FN GLuint GLRender::createComputeProgram(const std::string& csSrc) {
    GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
    const char* c = csSrc.c_str();
    glShaderSource(cs, 1, &c, nullptr);
    glCompileShader(cs);
    GLint ok; glGetShaderiv(cs, GL_COMPILE_STATUS, &ok);
    if (UNLIKELY(!ok)) {
        glDeleteShader(cs);
        return 0;
    }
    GLuint prog = glCreateProgram();
    glAttachShader(prog, cs);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (UNLIKELY(!ok)) {
        glDeleteProgram(prog);
        prog = 0;
    }
    glDeleteShader(cs);
    return prog;
}

ALWAYS_INLINE void GLRender::extractAffine2x2(const float* m4, float* m2, float* t) {
    m2[0] = m4[0]; m2[1] = m4[1];
    m2[2] = m4[4]; m2[3] = m4[5];
    t[0] = m4[12]; t[1] = m4[13];
}

// =======================================================================================
// JNI Bridge
// =======================================================================================

// #2: cache the CameraService.nativeGLRender field id on first use. A jfieldID stays valid for the
// lifetime of the (never-unloaded) class, so the per-frame nativeOnDrawFrame path no longer pays a
// GetObjectClass + string-based GetFieldID lookup on every camera frame. A first-call race merely
// recomputes the same value, so no synchronization is needed.
static jfieldID nativeHandleFieldId(JNIEnv* env, jobject thiz) {
    static jfieldID cached = nullptr;
    if (UNLIKELY(cached == nullptr)) {
        jclass cls = env->GetObjectClass(thiz);
        cached = env->GetFieldID(cls, "nativeGLRender", "J");
        env->DeleteLocalRef(cls);
    }
    return cached;
}
#define GET_RENDERER(env, thiz) reinterpret_cast<GLRender*>(env->GetLongField(thiz, nativeHandleFieldId(env, thiz)))

static std::string loadAsset(AAssetManager* mgr, const std::string& path) {
    AAsset* asset = AAssetManager_open(mgr, path.c_str(), AASSET_MODE_BUFFER);
    if (UNLIKELY(!asset)) { return ""; }
    size_t size = AAsset_getLength(asset);
    std::string content(size, '\0');
    AAsset_read(asset, content.data(), size);
    AAsset_close(asset);
    return content;
}

namespace {

// Scoped JNIEnv that attaches the current thread only if needed and detaches on scope exit.
struct ScopedJniEnv {
    JavaVM* vm;
    JNIEnv* env = nullptr;
    bool attached = false;
    explicit ScopedJniEnv(JavaVM* vm_) : vm(vm_) {
        if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
            if (vm->AttachCurrentThread(&env, nullptr) == JNI_OK) {
                attached = true;
            } else {
                env = nullptr;
            }
        }
    }
    ~ScopedJniEnv() { if (attached) vm->DetachCurrentThread(); }
    ScopedJniEnv(const ScopedJniEnv&) = delete;
    ScopedJniEnv& operator=(const ScopedJniEnv&) = delete;
};

// Owns the Java callback's JNI global ref for a single capture. The ref is released EXACTLY once
// (when the last shared owner is destroyed) and delivered to Java AT MOST once. Because the
// std::function handed to the renderer only holds a shared_ptr to this, copying/moving that
// std::function and invoking it from multiple places (delivery, teardown) can never double-free
// the global ref or use it after it was freed — which previously aborted the VM with a stale ref
// (e.g. "onFrame on ViewRootImpl$W") when the activity was destroyed with a capture in flight.
class OneShotJavaCallback {
public:
    OneShotJavaCallback(JavaVM* vm, jobject globalRef) : vm_(vm), globalRef_(globalRef) {}
    ~OneShotJavaCallback() {
        if (globalRef_) {
            ScopedJniEnv scoped(vm_);
            if (scoped.env) scoped.env->DeleteGlobalRef(globalRef_);
        }
    }
    OneShotJavaCallback(const OneShotJavaCallback&) = delete;
    OneShotJavaCallback& operator=(const OneShotJavaCallback&) = delete;

    void deliver(void* rgba, void* chw, int width, int height, jlong rgbaLen, jlong chwLen) {
        if (delivered_.exchange(true, std::memory_order_acq_rel)) return; // fire once
        ScopedJniEnv scoped(vm_);
        JNIEnv* env = scoped.env;
        if (UNLIKELY(!env)) return;

        jobject rgbaBuf = (rgba && rgbaLen > 0) ? env->NewDirectByteBuffer(rgba, rgbaLen) : nullptr;
        jobject chwBuf  = (chw  && chwLen  > 0) ? env->NewDirectByteBuffer(chw,  chwLen)  : nullptr;
        jclass cbClass = env->GetObjectClass(globalRef_);
        jmethodID onFrameMethod = env->GetMethodID(cbClass, "onFrame", "(Ljava/nio/ByteBuffer;Ljava/nio/ByteBuffer;II)V");
        if (LIKELY(onFrameMethod)) {
            env->CallVoidMethod(globalRef_, onFrameMethod, rgbaBuf, chwBuf, width, height);
        }
        if (env->ExceptionCheck()) env->ExceptionClear(); // never leave a pending exception behind
        if (cbClass) env->DeleteLocalRef(cbClass);
        if (rgbaBuf) env->DeleteLocalRef(rgbaBuf);
        if (chwBuf) env->DeleteLocalRef(chwBuf);
    }

private:
    JavaVM* vm_;
    jobject globalRef_;
    std::atomic<bool> delivered_{false};
};

} // namespace

extern "C" {

JNIEXPORT void JNICALL
Java_com_example_myapplication_CameraService_nativeCreate(JNIEnv *env, jobject thiz, jobject assetManager, jint camWidth, jint camHeight) {
    auto* renderer = new GLRender(static_cast<int>(camWidth), static_cast<int>(camHeight));
    env->SetLongField(thiz, nativeHandleFieldId(env, thiz), (jlong)renderer);

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    auto assetLoader = [mgr](const std::string& path) { return loadAsset(mgr, path); };
    renderer->onSurfaceCreated(assetLoader);
}

JNIEXPORT void JNICALL
Java_com_example_myapplication_CameraService_nativeDestroy(JNIEnv *env, jobject thiz) {
    auto* renderer = GET_RENDERER(env, thiz);
    if (LIKELY(renderer)) {
        delete renderer;
    }
    env->SetLongField(thiz, nativeHandleFieldId(env, thiz), 0L);
}

JNIEXPORT void JNICALL
Java_com_example_myapplication_CameraService_nativeOnDrawFrame(JNIEnv *env, jobject thiz, jfloatArray vMatrix) {
    auto* renderer = GET_RENDERER(env, thiz);
    if (UNLIKELY(!renderer)) return;
    // #3: the common idle frame has no capture pending, so skip the GetFloatArrayElements copy of
    // the 4x4 transform entirely and just poll delivery. Only when a capture will be dispatched this
    // frame do we marshal the matrix the compute shader needs.
    if (renderer->captureRequested()) {
        jfloat* matrix = env->GetFloatArrayElements(vMatrix, nullptr);
        renderer->onDrawFrame(matrix);
        env->ReleaseFloatArrayElements(vMatrix, matrix, JNI_ABORT);
    } else {
        renderer->onDrawFrame(nullptr);
    }
}

JNIEXPORT jint JNICALL
Java_com_example_myapplication_CameraService_nativeGetCameraTextureId(JNIEnv *env, jobject thiz) {
    auto* renderer = GET_RENDERER(env, thiz);
    return renderer ? static_cast<jint>(renderer -> getCameraTextureId()) : 0;
}

JNIEXPORT void JNICALL
Java_com_example_myapplication_CameraService_nativeCaptureInferenceRGBAsync(
    JNIEnv *env, jobject thiz, jobject callback, jint width, jint height) {
    auto* renderer = GET_RENDERER(env, thiz);
    if (UNLIKELY(!renderer || !callback)) return;

    JavaVM* vm;
    env->GetJavaVM(&vm);
    jobject globalCallback = env->NewGlobalRef(callback);
    if (UNLIKELY(!globalCallback)) return;

    // Shared ownership so the global ref is freed exactly once and delivered at most once,
    // no matter how many times / from which thread the wrapped std::function is invoked.
    auto oneShot = std::make_shared<OneShotJavaCallback>(vm, globalCallback);
    renderer->captureInferenceRGBAsync(static_cast<int>(width), static_cast<int>(height),
        [oneShot](void* rgba, void* chw, int width, int height, jlong rgbaLen, jlong chwLen) {
            oneShot->deliver(rgba, chw, width, height, rgbaLen, chwLen);
        });
}

} // extern "C"
