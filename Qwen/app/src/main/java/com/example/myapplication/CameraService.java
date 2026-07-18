package com.example.myapplication;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraMetadata;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.OutputConfiguration;
import android.hardware.camera2.params.SessionConfiguration;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.opengl.GLSurfaceView;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.util.Log;
import android.util.Range;
import android.util.Size;
import android.view.Surface;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * Minimal live-camera pipeline that hands back, per capture, <b>both</b> an interleaved uint8 RGBA
 * frame (for the on-screen preview) and a planar uint8 CHW frame (for inference pixel_values),
 * through a zero-copy chain:
 * <pre>
 *   Camera2 → SurfaceTexture (external OES texture, GPU)
 *           → compute shader packs, on the GPU in one pass, interleaved RGBA8 (preview) AND planar
 *             CHW uint8 (inference pixel_values) into two fenced SSBOs
 *           → the signaled SSBOs are mapped and wrapped directly as direct ByteBuffers (no copy)
 *           → delivered to {@link InferenceFrameCallback}.
 * </pre>
 * The heavy lifting (GL, compute dispatch, triple-buffered fences, transient map) lives in
 * {@code glRender.cpp}; this class only drives Camera2 and the {@link GLSurfaceView} render loop.
 */
public class CameraService implements GLSurfaceView.Renderer {

    static { System.loadLibrary("myapplication"); }

    /**
     * Receives the zero-copy frame. {@code rgba} is interleaved RGBA8 (W*H*4) for the preview Bitmap;
     * {@code chw} is planar CHW uint8 (3*W*H, rrr..ggg..bbb..) for the inference pixel_values. Both
     * directly wrap mapped GPU memory and are valid only during this call; copy out anything you keep.
     */
    public interface InferenceFrameCallback {
        void onFrame(ByteBuffer rgba, ByteBuffer chw, int width, int height);
    }

    /** Receives display-only RGBA frames sampled while the user is holding the capture button. */
    public interface VideoFrameCallback {
        void onFrame(ByteBuffer rgba, int width, int height, int sampledFrames,
                     float sampledSeconds, boolean timedOut);
    }

    public interface CameraStatusCallback {
        void onCameraReady();
        void onCameraError(int messageRes);
    }

    private static final String TAG = "CameraService";
    // Fallback capture size (matches DEFAULT_INPUT_IMAGE_SIZE in native) used for text-only models or
    // before a model's metadata has been applied.
    private static final int DEFAULT_CAM_SIZE = 960;
    // Camera producer geometry. When video is available, use its widescreen geometry as the shared
    // source; GLRender letterboxes that source into the square image target without stretching it.
    private final int camW;
    private final int camH;
    private final String cameraId;
    private final Range<Integer> targetFpsRange;
    // One uint per pixel (RGB + padding) — must match sizeBytes_ (camW_*camH_*BYTES_PER_PIXEL) in glRender.
    // #5: let AE pick anywhere in [15,30]: still hits 30fps in good light but can drop to 15 in low
    // light (longer exposure, less noise) and cuts camera + GL wakeups/power. On-demand snapshots
    // always grab the latest frame regardless of the live rate, so capture quality is unaffected.
    private static final int TARGET_FPS_MIN = 15;
    private static final int TARGET_FPS_MAX = 30;
    private static final Object CAMERA_SELECTION_LOCK = new Object();
        private static final Handler MAIN_HANDLER = new Handler(Looper.getMainLooper());
        private static final ExecutorService VISION_MAINTENANCE_EXECUTOR =
                Executors.newSingleThreadExecutor(runnable -> {
                    Thread thread = new Thread(runnable, "vision-maintenance");
                    thread.setDaemon(true);
                    return thread;
                });
    private static CameraSelection cachedCameraSelection;
    private static int cachedSelectionWidth;
    private static int cachedSelectionHeight;

    private final Context context;
    private final GLSurfaceView glView;

    // Native GLRender handle; assigned from C++ via SetLongField in nativeCreate / cleared in nativeDestroy.
    private long nativeGLRender = 0;
    private SurfaceTexture surfaceTexture;
    private final float[] vMatrix = new float[16];

    private HandlerThread bgThread;
    private volatile Handler bgHandler;
    private CameraDevice cameraDevice;
    private CameraCaptureSession captureSession;
    private Surface cameraInputSurface;
    private Surface floatingPreviewSurface;
    private boolean glResumed;
    private volatile boolean cameraOpenRequested;
    private final AtomicInteger cameraGeneration = new AtomicInteger();
    private final AtomicInteger glSurfaceGeneration = new AtomicInteger();

    // Offscreen capture path uses the model geometry so the chat snapshot and inference frame match
    // the visible square preview.
    private final int computeBufW;
    private final int computeBufH;
    private final int videoComputeBufW;
    private final int videoComputeBufH;
    // Visible floating preview: bind to the model input geometry (typically 960x960) so the window and
    // captured frame use the same aspect. Sensor orientation + lens facing still drive upright rotation.
    private final int previewBufW;
    private final int previewBufH;
    private final int sensorOrientation;
    private final boolean lensFacingFront;

    // Sampling is decoupled from preview FPS; native gates ring writes during inference.
    private final long videoSampleIntervalMs;
    private final float videoFps;
    private final int videoFrameCapacity;
    private final boolean videoFixedLength;
    private volatile boolean videoSampling = false;
    private Runnable videoSampler;
    private volatile VideoFrameCallback videoFrameCallback;
    private volatile int videoSampledFrames = 0;
    private volatile CameraStatusCallback cameraStatusCallback;

    public CameraService(Context context, GLSurfaceView glView) {
        this.context = context;
        this.glView = glView;
        // Resolve the capture size from the exported model's metadata (input_image_size) so the
        // pipeline auto-fits the model. Falls back to DEFAULT_CAM_SIZE when no vision metadata is
        // present (text-only models, or before a model has been loaded).
        int resolvedW = DEFAULT_CAM_SIZE;
        int resolvedH = DEFAULT_CAM_SIZE;
        int[] modelSize = nativeGetInputImageSize();
        if (modelSize != null && modelSize.length == 2) {
            if (modelSize[0] > 0) resolvedW = modelSize[0];
            if (modelSize[1] > 0) resolvedH = modelSize[1];
        }
        this.computeBufW = resolvedW;
        this.computeBufH = resolvedH;
        int resolvedVideoW = resolvedW;
        int resolvedVideoH = resolvedH;
        try {
            int[] videoSize = nativeGetInputVideoSize();
            if (videoSize != null && videoSize.length == 2) {
                if (videoSize[0] > 0) resolvedVideoW = videoSize[0];
                if (videoSize[1] > 0) resolvedVideoH = videoSize[1];
            }
        } catch (Throwable ignored) {
        }
        this.videoComputeBufW = resolvedVideoW;
        this.videoComputeBufH = resolvedVideoH;

        // Video sampling interval from the model's video_fps (decoupled from the preview frame rate).
        float vfps = 2.0f;
        try {
            float f = nativeGetVideoFps();
            if (f > 0.0f) vfps = f;
        } catch (Throwable ignored) {
        }
        this.videoFps = vfps;
        this.videoSampleIntervalMs = Math.max(1L, (long) (1000.0f / vfps));
        int frameCapacity = 0;
        try {
            frameCapacity = nativeGetVideoNumFrames();
        } catch (Throwable ignored) {
        }
        this.videoFrameCapacity = Math.max(0, frameCapacity);
        int desiredProducerW = this.videoFrameCapacity > 0 ? resolvedVideoW : resolvedW;
        int desiredProducerH = this.videoFrameCapacity > 0 ? resolvedVideoH : resolvedH;
        CameraSelection selection = preparedCameraSelection(
            context, desiredProducerW, desiredProducerH);
        this.cameraId = selection.cameraId;
        this.camW = selection.outputSize.getWidth();
        this.camH = selection.outputSize.getHeight();
        this.previewBufW = this.camW;
        this.previewBufH = this.camH;
        this.targetFpsRange = selection.fpsRange;
        this.sensorOrientation = ((selection.sensorOrientation % 360) + 360) % 360;
        this.lensFacingFront = selection.lensFacingFront;
        boolean fixedLength = true;
        try {
            fixedLength = nativeIsVideoFixedLength();
        } catch (Throwable ignored) {
        }
        this.videoFixedLength = fixedLength;
        glView.setEGLContextClientVersion(3);
        glView.setPreserveEGLContextOnPause(true);   // reuse the GL context across start/stop cycles
        glView.setRenderer(this);
        glView.setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
        ViewGroup.LayoutParams params = glView.getLayoutParams();
        if (params != null) {
            // Keep the on-screen surface to a single (invisible) pixel: we only need a valid GL
            // context; the RGB data never travels through the framebuffer.
            params.width = 1;
            params.height = 1;
            glView.setLayoutParams(params);
        }
    }

    /** Resolves Camera2 capabilities off the UI thread after the vision model geometry is known. */
    public static void preloadCameraSelection(Context context) {
        int imageWidth = DEFAULT_CAM_SIZE;
        int imageHeight = DEFAULT_CAM_SIZE;
        int[] imageSize = nativeGetInputImageSize();
        if (imageSize != null && imageSize.length == 2) {
            if (imageSize[0] > 0) imageWidth = imageSize[0];
            if (imageSize[1] > 0) imageHeight = imageSize[1];
        }
        int producerWidth = imageWidth;
        int producerHeight = imageHeight;
        if (nativeGetVideoNumFrames() > 0) {
            int[] videoSize = nativeGetInputVideoSize();
            if (videoSize != null && videoSize.length == 2) {
                if (videoSize[0] > 0) producerWidth = videoSize[0];
                if (videoSize[1] > 0) producerHeight = videoSize[1];
            }
        }
        CameraSelection selection = selectCamera(
                context.getApplicationContext(), producerWidth, producerHeight);
        synchronized (CAMERA_SELECTION_LOCK) {
            cachedCameraSelection = selection;
            cachedSelectionWidth = producerWidth;
            cachedSelectionHeight = producerHeight;
        }
    }

    public static boolean hasPreparedCameraSelection() {
        synchronized (CAMERA_SELECTION_LOCK) {
            return cachedCameraSelection != null && cachedCameraSelection.cameraId != null;
        }
    }

    private static CameraSelection preparedCameraSelection(
            Context context, int desiredWidth, int desiredHeight) {
        synchronized (CAMERA_SELECTION_LOCK) {
            if (cachedCameraSelection != null && cachedSelectionWidth == desiredWidth &&
                    cachedSelectionHeight == desiredHeight) {
                return cachedCameraSelection;
            }
        }
        // This fallback is used only when vision loading could not prepare a selection.
        return new CameraSelection(null,
                new Size(Math.max(1, desiredWidth), Math.max(1, desiredHeight)), null, 90, false);
    }

    private static CameraSelection selectCamera(Context context, int desiredWidth, int desiredHeight) {
        CameraSelection fallback = new CameraSelection(null,
                new Size(Math.max(1, desiredWidth), Math.max(1, desiredHeight)), null, 90, false);
        CameraManager manager = (CameraManager) context.getSystemService(Context.CAMERA_SERVICE);
        if (manager == null) {
            return fallback;
        }
        CameraSelection best = null;
        int bestFacingRank = Integer.MAX_VALUE;
        double bestSizeScore = Double.POSITIVE_INFINITY;
        try {
            for (String id : manager.getCameraIdList()) {
                CameraCharacteristics characteristics = manager.getCameraCharacteristics(id);
                StreamConfigurationMap map = characteristics.get(
                        CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
                Size[] sizes = map != null ? map.getOutputSizes(SurfaceTexture.class) : null;
                if (sizes == null || sizes.length == 0) {
                    continue;
                }
                Integer facing = characteristics.get(CameraCharacteristics.LENS_FACING);
                int facingRank = facing != null && facing == CameraMetadata.LENS_FACING_BACK ? 0
                        : (facing != null && facing == CameraMetadata.LENS_FACING_FRONT ? 1 : 2);
                Size output = chooseOutputSize(sizes, desiredWidth, desiredHeight);
                double sizeScore = outputSizeScore(output, desiredWidth, desiredHeight);
                if (best == null || facingRank < bestFacingRank ||
                        (facingRank == bestFacingRank && sizeScore < bestSizeScore)) {
                    Integer orientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION);
                    best = new CameraSelection(id, output,
                            chooseFpsRange(characteristics.get(
                                    CameraCharacteristics.CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES)),
                            orientation != null ? orientation : 90,
                            facing != null && facing == CameraMetadata.LENS_FACING_FRONT);
                    bestFacingRank = facingRank;
                    bestSizeScore = sizeScore;
                }
            }
        } catch (CameraAccessException | SecurityException error) {
            Log.e(TAG, "Could not enumerate camera capabilities", error);
        }
        if (best != null) {
            Log.i(TAG, "Selected camera=" + best.cameraId + " output=" + best.outputSize +
                    " fps=" + best.fpsRange);
        }
        return best != null ? best : fallback;
    }

    private static Size chooseOutputSize(Size[] sizes, int desiredWidth, int desiredHeight) {
        Size best = sizes[0];
        double bestScore = outputSizeScore(best, desiredWidth, desiredHeight);
        for (int index = 1; index < sizes.length; ++index) {
            double score = outputSizeScore(sizes[index], desiredWidth, desiredHeight);
            if (score < bestScore) {
                best = sizes[index];
                bestScore = score;
            }
        }
        return best;
    }

    private static double outputSizeScore(Size size, int desiredWidth, int desiredHeight) {
        double outputLong = Math.max(size.getWidth(), size.getHeight());
        double outputShort = Math.max(1, Math.min(size.getWidth(), size.getHeight()));
        double desiredLong = Math.max(1, Math.max(desiredWidth, desiredHeight));
        double desiredShort = Math.max(1, Math.min(desiredWidth, desiredHeight));
        double aspectError = Math.abs(Math.log((outputLong / outputShort) /
                (desiredLong / desiredShort)));
        double outputPixels = Math.max(1.0, (double) size.getWidth() * size.getHeight());
        double desiredPixels = Math.max(1.0, (double) desiredWidth * desiredHeight);
        double areaError = Math.abs(Math.log(outputPixels / desiredPixels));
        return aspectError * 4.0 + areaError;
    }

    private static Range<Integer> chooseFpsRange(Range<Integer>[] ranges) {
        if (ranges == null || ranges.length == 0) {
            return null;
        }
        Range<Integer> best = ranges[0];
        int bestScore = Integer.MAX_VALUE;
        for (Range<Integer> range : ranges) {
            int lower = range.getLower();
            int upper = range.getUpper();
            int score = Math.abs(upper - TARGET_FPS_MAX) * 4 +
                    Math.abs(lower - TARGET_FPS_MIN);
            if (upper < TARGET_FPS_MIN) {
                score += 10_000;
            }
            if (lower <= TARGET_FPS_MIN && upper >= TARGET_FPS_MAX) {
                score -= 1_000;
            }
            if (score < bestScore) {
                best = range;
                bestScore = score;
            }
        }
        return best;
    }

    private static final class CameraSelection {
        final String cameraId;
        final Size outputSize;
        final Range<Integer> fpsRange;
        final int sensorOrientation;
        final boolean lensFacingFront;

        CameraSelection(String cameraId, Size outputSize, Range<Integer> fpsRange,
                        int sensorOrientation, boolean lensFacingFront) {
            this.cameraId = cameraId;
            this.outputSize = outputSize;
            this.fpsRange = fpsRange;
            this.sensorOrientation = sensorOrientation;
            this.lensFacingFront = lensFacingFront;
        }
    }

    /** Bring the pipeline up: make the 1×1 compute surface live, resume the GL loop, open the camera. */
    public void start() {
        if (cameraId == null) {
            reportCameraError(R.string.camera_error_unavailable);
            return;
        }
        startBgThread();
        cameraOpenRequested = true;
        cameraGeneration.incrementAndGet();
        glView.setVisibility(View.VISIBLE);
        if (!glResumed) {
            glView.onResume();
            glResumed = true;
        }
        requestOpenCamera();   // no-op until SurfaceTexture exists; onSurfaceCreated retries it
    }

    public void setCameraStatusCallback(CameraStatusCallback callback) {
        cameraStatusCallback = callback;
        if (callback != null && cameraId == null) {
            reportCameraError(R.string.camera_error_unavailable);
        }
    }

    public void retryOpen() {
        if (cameraId == null) {
            reportCameraError(R.string.camera_error_unavailable);
            return;
        }
        startBgThread();
        cameraOpenRequested = true;
        final int generation = cameraGeneration.incrementAndGet();
        postToCameraThread(() -> {
            closeCaptureSession();
            releaseCameraInputSurface();
            CameraDevice device = cameraDevice;
            if (device != null) {
                createPreviewSession(generation, device);
            } else {
                openCameraOnCameraThread(generation);
            }
        });
    }

    private void reportCameraReady() {
        CameraStatusCallback callback = cameraStatusCallback;
        if (callback != null) {
            MAIN_HANDLER.post(callback::onCameraReady);
        }
    }

    private void reportCameraError(int messageRes) {
        CameraStatusCallback callback = cameraStatusCallback;
        if (callback != null) {
            MAIN_HANDLER.post(() -> callback.onCameraError(messageRes));
        }
    }

    /** Close the camera stream but keep the hidden GL compute surface warm for a later {@link #start()}. */
    public void stop() {
        stopVideoSampling();
        cameraOpenRequested = false;
        cameraGeneration.incrementAndGet();
        postToCameraThread(this::closeCameraOnCameraThread);
    }

    /** Full Activity/background pause: close camera and pause the hidden GL compute surface. */
    public void pause() {
        stopVideoSampling();
        cameraOpenRequested = false;
        cameraGeneration.incrementAndGet();
        postToCameraThread(this::closeCameraOnCameraThread);
        if (glResumed) {
            glView.onPause();
            glResumed = false;
        }
        glView.setVisibility(View.GONE);
    }

    /** Permanently free the native renderer (call once, when the owner is destroyed). */
    public void release() {
        stopVideoSampling();
        cameraOpenRequested = false;
        cameraGeneration.incrementAndGet();
        cameraStatusCallback = null;
        Handler handler = bgHandler;
        bgHandler = null;
        bgThread = null;
        Runnable finishRelease = () -> {
            closeCameraOnCameraThread();
            glView.queueEvent(() -> {
                if (surfaceTexture != null) {
                    surfaceTexture.setOnFrameAvailableListener(null);
                    surfaceTexture.release();
                    surfaceTexture = null;
                }
                if (nativeGLRender != 0) {
                    nativeDestroy();
                    nativeGLRender = 0;
                }
                glView.post(() -> {
                    if (glResumed) {
                        glView.onPause();
                        glResumed = false;
                    }
                    glView.setVisibility(View.GONE);
                });
            });
            if (handler != null) {
                handler.getLooper().quitSafely();
            }
        };
        if (handler != null) {
            handler.post(finishRelease);
        } else {
            finishRelease.run();
        }
    }

    /** Attach/detach the visible floating-window preview surface. The GL compute path remains separate. */
    public void setFloatingPreviewSurfaceTexture(SurfaceTexture texture, int width, int height) {
        Surface replacement = null;
        if (texture != null) {
            // TextureView applies the aspect-preserving display transform.
            texture.setDefaultBufferSize(camW, camH);
            replacement = new Surface(texture);
        }
        final Surface nextSurface = replacement;
        Handler handler = bgHandler;
        if (handler == null) {
            if (nextSurface != null) {
                nextSurface.release();
            }
            return;
        }
        final int generation = cameraGeneration.get();
        handler.post(() -> {
            releaseFloatingPreviewSurface();
            floatingPreviewSurface = nextSurface;
            if (cameraDevice != null && cameraOpenRequested &&
                    generation == cameraGeneration.get()) {
                createPreviewSession(generation, cameraDevice);
            }
        });
    }

    /** Preview buffer width used by the floating TextureView aspect transform. */
    public int getPreviewBufferWidth() {
        return previewBufW;
    }

    /** Preview buffer height used by the floating TextureView aspect transform. */
    public int getPreviewBufferHeight() {
        return previewBufH;
    }

    /** Degrees to rotate the preview buffer so it appears upright for the given display rotation. */
    public int getPreviewContentRotation(int displayRotationDegrees) {
        // Empirically this device family needs an extra 90 deg CCW vs the textbook formula (the landscape
        // preview buffer + TextureView transform pipeline), so subtract 90.
        if (lensFacingFront) {
            return (sensorOrientation + displayRotationDegrees + 90) % 360;
        }
        return (sensorOrientation - displayRotationDegrees - 90 + 360) % 360;
    }

    /** Exported model input size {width, height} (input_image_size); {0,0} for a text-only model. */
    public static int[] getInputImageSize() {
        try {
            return nativeGetInputImageSize();
        } catch (Throwable t) {
            return new int[]{0, 0};
        }
    }

    // Callback runs on the GL thread; mapped buffers are valid only until it returns.
    private void captureInferenceRGBAsync(int width, int height, InferenceFrameCallback cb) {
        if (nativeGLRender != 0) {
            nativeCaptureInferenceRGBAsync(cb, width, height);
            glView.requestRender();
        } else if (cb != null) {
            cb.onFrame(null, null, 0, 0);
        }
    }

    // ================= GLSurfaceView.Renderer (runs on the GL thread) =================

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        final int surfaceGeneration = glSurfaceGeneration.incrementAndGet();
        cameraGeneration.incrementAndGet();
        Handler handler = bgHandler;
        if (handler != null) {
            handler.post(() -> {
                // Preserve the floating preview binding: only the camera's GL texture is being
                // recreated here. The on-screen TextureView surface is independent and still valid, so
                // releasing it would drop it from the first reopened session and leave the window black.
                closeCameraOnCameraThread(false);
                glView.queueEvent(() -> createRendererOnGlThread(surfaceGeneration));
            });
        } else {
            createRendererOnGlThread(surfaceGeneration);
        }
    }

    private void createRendererOnGlThread(int surfaceGeneration) {
        if (surfaceGeneration != glSurfaceGeneration.get()) {
            return;
        }
        if (surfaceTexture != null) {
            surfaceTexture.setOnFrameAvailableListener(null);
            surfaceTexture.release();
            surfaceTexture = null;
        }
        if (nativeGLRender != 0) {
            nativeDestroy();
            nativeGLRender = 0;
        }
        nativeCreate(context.getAssets(), camW, camH);
        int textureId = nativeGetCameraTextureId();
        if (textureId > 0) {
            surfaceTexture = new SurfaceTexture(textureId);
            // The producer stays at the video geometry when video is available. GLRender derives a
            // per-request fit for the square image output or the widescreen video output.
            surfaceTexture.setDefaultBufferSize(camW, camH);
            surfaceTexture.setOnFrameAvailableListener(st -> glView.requestRender());
            requestOpenCamera();
        }
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        // No-op: this GL surface never renders on screen (compute-only capture path), so the viewport
        // size is irrelevant. Kept only to satisfy the GLSurfaceView.Renderer contract.
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        if (nativeGLRender == 0 || surfaceTexture == null) {
            return;
        }
        surfaceTexture.updateTexImage();
        surfaceTexture.getTransformMatrix(vMatrix);
        nativeOnDrawFrame(vMatrix);
    }

    // ================= Camera2 (runs on the background handler thread) =================

    private boolean hasCameraPermission() {
        return ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED;
    }

    private void requestOpenCamera() {
        Handler handler = bgHandler;
        if (handler == null) {
            return;
        }
        final int generation = cameraGeneration.get();
        handler.post(() -> openCameraOnCameraThread(generation));
    }

    private void openCameraOnCameraThread(int generation) {
        Handler handler = bgHandler;
        if (!cameraOpenRequested || generation != cameraGeneration.get() ||
                !hasCameraPermission() || cameraId == null || surfaceTexture == null ||
                cameraDevice != null || handler == null) {
            return;
        }
        CameraManager manager = (CameraManager) context.getSystemService(Context.CAMERA_SERVICE);
        if (manager == null) {
            reportCameraError(R.string.camera_error_unavailable);
            return;
        }
        try {
            manager.openCamera(cameraId, stateCallback(generation), handler);
        } catch (CameraAccessException | SecurityException e) {
            Log.e(TAG, "openCamera failed", e);
            reportCameraError(R.string.camera_error_open);
        }
    }

    private CameraDevice.StateCallback stateCallback(int generation) {
        return new CameraDevice.StateCallback() {
            @Override public void onOpened(@NonNull CameraDevice camera) {
                if (!cameraOpenRequested || generation != cameraGeneration.get() ||
                        bgHandler == null) {
                    camera.close();
                    return;
                }
                cameraDevice = camera;
                createPreviewSession(generation, camera);
            }
            @Override public void onDisconnected(@NonNull CameraDevice camera) {
                camera.close();
                if (cameraDevice == camera) {
                    cameraDevice = null;
                }
                if (cameraOpenRequested) {
                    reportCameraError(R.string.camera_error_open);
                }
            }
            @Override public void onError(@NonNull CameraDevice camera, int error) {
                camera.close();
                if (cameraDevice == camera) {
                    cameraDevice = null;
                }
                reportCameraError(R.string.camera_error_open);
            }
        };
    }

    private void createPreviewSession(int generation, CameraDevice device) {
        Handler handler = bgHandler;
        if (!cameraOpenRequested || generation != cameraGeneration.get() ||
                cameraDevice != device || surfaceTexture == null || handler == null) {
            return;
        }
        Handler sessionHandler = handler;
        try {
            closeCaptureSession();
            releaseCameraInputSurface();
            cameraInputSurface = new Surface(surfaceTexture);
            CaptureRequest.Builder builder = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            builder.addTarget(cameraInputSurface);
            List<OutputConfiguration> outputs = new ArrayList<>(2);
            outputs.add(new OutputConfiguration(cameraInputSurface));
            if (floatingPreviewSurface != null) {
                builder.addTarget(floatingPreviewSurface);
                outputs.add(new OutputConfiguration(floatingPreviewSurface));
            }
            builder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
            builder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON);
            if (targetFpsRange != null) {
                builder.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, targetFpsRange);
            }
            SessionConfiguration sessionConfig = new SessionConfiguration(
                    SessionConfiguration.SESSION_REGULAR,
                    outputs,
                    command -> sessionHandler.post(command),
                    new CameraCaptureSession.StateCallback() {
                        @Override public void onConfigured(@NonNull CameraCaptureSession session) {
                            // The session can finish configuring after the user already closed the
                            // camera (device/thread torn down). Bail out and swallow the state-machine
                            // exceptions that would otherwise crash this background thread.
                            if (!cameraOpenRequested || generation != cameraGeneration.get() ||
                                    cameraDevice != device || bgHandler == null) {
                                session.close();
                                return;
                            }
                            captureSession = session;
                            try {
                                session.setRepeatingRequest(builder.build(), null, bgHandler);
                                reportCameraReady();
                            } catch (CameraAccessException | IllegalStateException e) {
                                Log.e(TAG, "setRepeatingRequest failed", e);
                                reportCameraError(R.string.camera_error_session);
                            }
                        }
                        @Override public void onConfigureFailed(@NonNull CameraCaptureSession session) {
                            session.close();
                            Log.e(TAG, "camera session configure failed");
                            reportCameraError(R.string.camera_error_session);
                        }
                    });
            device.createCaptureSession(sessionConfig);
        } catch (CameraAccessException | IllegalStateException e) {
            Log.e(TAG, "createPreviewSession failed", e);
            releaseCameraInputSurface();
            reportCameraError(R.string.camera_error_session);
        }
    }

    private void closeCameraOnCameraThread() {
        closeCameraOnCameraThread(true);
    }

    // releaseFloatingPreview=false keeps the visible TextureView binding alive across an internal GL
    // renderer re-init (onSurfaceCreated). The floating preview surface wraps the TextureView's own
    // SurfaceTexture, which is independent of the camera's GL texture; tearing it down there would drop
    // it from the very first capture session and leave the preview window black until the next open.
    private void closeCameraOnCameraThread(boolean releaseFloatingPreview) {
        try {
            closeCaptureSession();
            if (cameraDevice != null) {
                cameraDevice.close();
                cameraDevice = null;
            }
            releaseCameraInputSurface();
            if (releaseFloatingPreview) {
                releaseFloatingPreviewSurface();
            }
        } catch (Exception e) {
            Log.e(TAG, "closeCamera error", e);
        }
    }

    private void closeCaptureSession() {
        if (captureSession != null) {
            captureSession.close();
            captureSession = null;
        }
    }

    private void releaseCameraInputSurface() {
        if (cameraInputSurface != null) {
            cameraInputSurface.release();
            cameraInputSurface = null;
        }
    }

    private void releaseFloatingPreviewSurface() {
        if (floatingPreviewSurface != null) {
            floatingPreviewSurface.release();
            floatingPreviewSurface = null;
        }
    }

    private void startBgThread() {
        if (bgThread == null) {
            bgThread = new HandlerThread("CameraBackground");
            bgThread.start();
            bgHandler = new Handler(bgThread.getLooper());
        }
    }

    private void postToCameraThread(Runnable action) {
        Handler handler = bgHandler;
        if (handler != null) {
            handler.post(action);
        } else {
            action.run();
        }
    }

    // ================= Video sampling =================
    private void startVideoSampling(VideoFrameCallback callback) {
        if (videoSampling || bgHandler == null) {
            return;
        }
        videoFrameCallback = callback;
        videoSampledFrames = 0;
        videoSampling = true;
        videoSampler = new Runnable() {
            @Override public void run() {
                if (!videoSampling) {
                    return;
                }
                // The GL callback pushes model-sized CHW directly into the native ring.
                captureInferenceRGBAsync(videoComputeBufW, videoComputeBufH,
                        (rgba, chw, width, height) -> {
                    if (chw != null && width == videoComputeBufW && height == videoComputeBufH) {
                        nativePushVideoFrameCHW(chw, width, height);
                        videoSampledFrames += 1;
                        VideoFrameCallback cb = videoFrameCallback;
                        if (cb != null) {
                            float sampledSeconds = videoFps > 0f ? videoSampledFrames / videoFps : 0f;
                            boolean timedOut = videoFrameCapacity > 0 && videoSampledFrames >= videoFrameCapacity;
                            cb.onFrame(rgba, width, height, videoSampledFrames, sampledSeconds, timedOut);
                        }
                    }
                });
                if (!videoFixedLength && videoFrameCapacity > 0 && videoSampledFrames >= videoFrameCapacity) {
                    videoSampling = false;
                    return;
                }
                if (videoSampling && bgHandler != null) {
                    bgHandler.postDelayed(this, videoSampleIntervalMs);
                }
            }
        };
        bgHandler.post(videoSampler);
    }

    private void stopVideoSampling() {
        videoSampling = false;
        if (bgHandler != null && videoSampler != null) {
            bgHandler.removeCallbacks(videoSampler);
        }
        videoSampler = null;
        videoFrameCallback = null;
    }

    // ================= native (JNI bridge lives in glRender.cpp / processLLM.cpp) =================
    private native void nativeCreate(android.content.res.AssetManager assetManager, int camWidth, int camHeight);
    private native void nativeDestroy();
    private native void nativeOnDrawFrame(float[] vMatrix);
    private native int nativeGetCameraTextureId();
    private native void nativeCaptureInferenceRGBAsync(InferenceFrameCallback callback,
                                                       int width, int height);

    // Exported vision input geometry {width, height} from the model's metadata (input_image_size /
    // input_video_size), implemented in processLLM.cpp. Returns the native default until a vision model
    // has been loaded.
    private static native int[] nativeGetInputImageSize();
    private static native int[] nativeGetInputVideoSize();
    // Vision capability bitmask (bit0 image, bit1 video); 0 for a text-only model.
    private static native int nativeGetVisionCaps();
    private static native int nativeGetVideoNumFrames();
    private static native float nativeGetVideoFps();
    private static native boolean nativeIsVideoFixedLength();
    // Hand a captured frame to the native runtime. Image: one snapshot -> pixel_values slot 0. Video:
    // one sampled frame -> the next ring slot. submit freezes the video window for the next Run_LLM.
    private static native boolean nativePushImageCHW(java.nio.ByteBuffer chw, int width, int height);
    private static native void nativeSetVideoStream(boolean on);
    private static native void nativePushVideoFrameCHW(java.nio.ByteBuffer chw, int width, int height);
    private static native boolean nativeSubmitVideoQuery();
    private static native long nativeRequestCancelPendingVision();
    private static native void nativeFinishCancelPendingVision(long generation);

    // ================= vision capability + capture API (used by the UI) =================

    /** Vision capability bitmask (bit0 = image, bit1 = video); 0 for a text-only model. */
    public static int getVisionCaps() {
        try {
            return nativeGetVisionCaps();
        } catch (Throwable t) {
            return 0;
        }
    }

    public static boolean isVideoCapable() { return (getVisionCaps() & 0x2) != 0; }

    /** Frames the video ring holds (VIDEO_NUM_FRAMES); 0 when video is unavailable. */
    public static int getVideoNumFrames() {
        try {
            return nativeGetVideoNumFrames();
        } catch (Throwable t) {
            return 0;
        }
    }

    /** Target video sampling FPS (metadata video_fps); 0 when unavailable. */
    public static float getVideoFps() {
        try {
            return nativeGetVideoFps();
        } catch (Throwable t) {
            return 0f;
        }
    }

    /** Model video memory span in seconds, derived from frame capacity and sampling FPS. */
    public static float getVideoWindowSeconds() {
        float fps = getVideoFps();
        int frames = getVideoNumFrames();
        return (fps > 0f && frames > 0) ? frames / fps : 0f;
    }

    /** Publishes cancellation immediately; slow resource release runs on the serial maintenance worker. */
    public static Future<?> cancelPendingVision(Runnable onComplete) {
        final long generation;
        try {
            generation = nativeRequestCancelPendingVision();
        } catch (Throwable ignored) {
            if (onComplete != null) {
                MAIN_HANDLER.post(onComplete);
            }
            return null;
        }
        return VISION_MAINTENANCE_EXECUTOR.submit(() -> {
            try {
                nativeFinishCancelPendingVision(generation);
            } finally {
                if (onComplete != null) {
                    MAIN_HANDLER.post(onComplete);
                }
            }
        });
    }

    public static void awaitPendingVisionCancel(Future<?> task) {
        if (task == null) {
            return;
        }
        try {
            task.get();
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
        } catch (java.util.concurrent.ExecutionException ignored) {
        }
    }

    /**
     * Capture one snapshot for image inference: runs the compute shader once and, on the GL thread,
     * pushes the letterboxed CHW into the native pixel_values (single memcpy). {@code onReady} is invoked
     * only when native accepted the frame; the RGBA buffer is display-only and valid only in the callback.
     */
    public void captureImageForInference(InferenceFrameCallback onReady) {
        captureInferenceRGBAsync(computeBufW, computeBufH, (rgba, chw, width, height) -> {
            boolean pushed = chw != null && width == computeBufW && height == computeBufH &&
                    nativePushImageCHW(chw, width, height);
            if (onReady != null) {
                onReady.onFrame(pushed ? rgba : null, pushed ? chw : null,
                        pushed ? width : 0, pushed ? height : 0);
            }
        });
    }

    /** Start recording only while the capture button is held. */
    public void beginVideoRecording(VideoFrameCallback callback) {
        nativeSetVideoStream(true);
        startVideoSampling(callback);
    }

    /** Stop recording, freeze the sampled video for the next Run_LLM, and leave sampling disabled. */
    public boolean finishVideoRecording() {
        stopVideoSampling();
        boolean ok = nativeSubmitVideoQuery();
        nativeSetVideoStream(false);
        return ok;
    }

    /** Drop an in-progress long-press recording without submitting it. */
    public void cancelVideoRecording() {
        stopVideoSampling();
        nativeSetVideoStream(false);
    }
}
