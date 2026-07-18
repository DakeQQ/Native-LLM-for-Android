#pragma once
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>
#include <jni.h>
#include <string>
#include <vector>
#include <functional>
#include <atomic>
#include <memory>
#include <mutex>
#include <cstring>
#include <android/log.h>
#include <android/asset_manager_jni.h>


// Delivered once per capture. rgba = interleaved RGBA8 (W*H*4) for the preview; chw = planar CHW
// uint8 (W*H*3, rrr..ggg..bbb..) for inference pixel_values. Both pointers wrap mapped GPU memory and
// are valid only for the duration of the call; either may be null if that output was unavailable.
using InferenceFrameCallback = std::function<void(void* rgba, void* chw, int width, int height, jlong rgbaLen, jlong chwLen)>;

class GLRender {
public:
    // Camera producer size. One-shot image/video outputs may use different target geometries.
    GLRender(int camW, int camH);
    ~GLRender();

    void onSurfaceCreated(std::function<std::string(const std::string&)> assetLoader);
    void onDrawFrame(const float* vMatrix);

    GLuint getCameraTextureId() const;
    void captureInferenceRGBAsync(int width, int height, InferenceFrameCallback cb);
    // Peek (without consuming) whether a one-shot capture is pending, so the JNI layer can skip
    // marshalling the transform matrix on idle preview frames and only copy it when a capture will
    // actually be dispatched. Consumed authoritatively by onDrawFrame's atomic exchange.
    bool captureRequested() const { return wantOneShot_.load(std::memory_order_acquire); }

private:
    void destroy();
    void initExternalTexture();
    void initComputePipeline(std::function<std::string(const std::string&)>& assetLoader);
    void allocateSSBOs();
    bool configureOutputGeometry(int width, int height);

    void dispatchCompute();
    // Non-blocking: hands back whichever earlier triple-buffer entry the GPU has already finished,
    // keeping consecutive captures pipelined instead of stalling the GL thread on a fence wait.
    //
    // NOTE: an AHardwareBuffer zero-readback variant (imageStore into an AHB + AHardwareBuffer_lock)
    // was implemented and benchmarked against this SSBO path on two devices (ZTE nubia Z50 / Adreno,
    // and OPPO PKU110 / Snapdragon 8 Elite / Adreno 830). It was consistently ~9-10% SLOWER for
    // GPU->CPU delivery: glMapBufferRange here is effectively zero-cost (Adreno backs GL_DYNAMIC_READ
    // SSBOs with CPU-coherent memory, so there is no implicit readback copy to eliminate), whereas
    // AHardwareBuffer_lock adds a few microseconds of bookkeeping. The AHB path was therefore removed
    // and this SSBO readback is kept as the sole, faster implementation.
    void tryDeliverCompletedBuffer();
    // Atomically moves out the pending one-shot callback AND clears oneShotCb_. A bare
    // std::move is not enough: libc++'s small-buffer std::function leaves the moved-from
    // source non-empty, which would make every later capture look "busy" and get rejected.
    InferenceFrameCallback takePendingCallback();

    static GLuint createComputeProgram(const std::string& csSrc);
    static void extractAffine2x2(const float* m4, float* m2, float* t);

    static constexpr int kDefaultCamSize = 960; // fallback until model metadata is applied
    static constexpr int BYTES_PER_PIXEL = 4;   // 打包 uint (RGB24 + 1字节填充)
    static constexpr int BUFFER_COUNT = 3;

    const int sourceW_;
    const int sourceH_;

    // Compute output geometry. Image and video exports may use different aspect ratios, so one-shot
    // requests switch this geometry on the GL thread; repeated frames in one mode stay allocation-free.
    int camW_;
    int camH_;
    int groupsX_;
    int groupsY_;
    int sizeBytes_;     // interleaved RGBA8 buffer bytes (W*H*4), preview
    int chwBytes_;      // planar CHW uint8 buffer bytes (W*H*3), inference pixel_values
    float invOutW_;
    float invOutH_;
    int requestedW_;
    int requestedH_;

    GLuint cameraTexId_ = 0;
    float vMatrix_[16]{};
    size_t vMatrix_size = sizeof(float) * 16;

    GLuint computeProgram_ = 0;
    GLint locM2_ = -1;
    GLint locT_ = -1;
    GLint locFit_ = -1;
    GLint locOutSize_ = -1;
    GLint locInvOut_ = -1;

    float m2_[4]{};
    float t_[2]{};
    // Fraction of the target occupied by the source after aspect-preserving fit. Updated only on GL.
    float fitX_ = 1.0f;
    float fitY_ = 1.0f;

    GLuint ssboIds_[BUFFER_COUNT]{};     // interleaved RGBA8 (compute binding 0)
    GLuint chwSsboIds_[BUFFER_COUNT]{};  // planar CHW uint8   (compute binding 1)
    GLsync fences_[BUFFER_COUNT]{};
    int curWrite_ = 0;
    bool computeReady_ = false;

    InferenceFrameCallback oneShotCb_;
    std::mutex callbackMutex_;
    std::atomic<bool> wantOneShot_{false};
};
