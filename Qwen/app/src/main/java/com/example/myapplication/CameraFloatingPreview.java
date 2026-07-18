package com.example.myapplication;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.SurfaceTexture;
import android.graphics.Typeface;
import android.graphics.drawable.ColorDrawable;
import android.graphics.drawable.GradientDrawable;
import android.opengl.GLSurfaceView;
import android.os.Handler;
import android.os.Looper;
import android.os.SystemClock;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.PopupWindow;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.DefaultLifecycleObserver;
import androidx.lifecycle.LifecycleOwner;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Owns the live-camera session ({@link CameraService}) plus a draggable floating window that
 * reviews the RGB screenshots produced by the zero-copy chain. A single capture copies the
 * mapped GPU buffer straight into a {@link Bitmap} before the native callback returns (the only copy,
 * and only because it is being shown on screen).
 */
public class CameraFloatingPreview implements DefaultLifecycleObserver {

    /**
     * Notified when the user captures a vision input from the preview. The thumbnail (display-only RGBA
     * of the snapshot / latest video frame) is handed to the chat; the CHW pixel_values were already
     * pushed to the native runtime, so the listener only has to add the bubble and trigger Run_LLM.
     */
    public interface VisionCaptureListener {
        void onImageCaptured(Bitmap thumbnail);
        // A long-press video clip was frozen for the next query. ok=false means no frame was sampled.
        void onVideoCaptured(List<Bitmap> frames, float seconds, boolean ok);
    }

    private static final int MAX_VIDEO_BUBBLE_FRAMES = 8;
    private static final int IMAGE_THUMB_MAX_SIDE_PX = 512;
    private static final int VIDEO_BUBBLE_MAX_SIDE_PX = 320;
    private static final long VIDEO_PROGRESS_TICK_MS = 32L;

    private final AppCompatActivity activity;
    private final GLSurfaceView glView;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    private CameraService cameraService;
    private PopupWindow popupWindow;
    private TextureView livePreview;
    private TextView statusText;
    private Button captureButton;
    private ProgressBar videoProgress;
    private TextView videoProgressText;
    private VisionCaptureListener captureListener;
    private boolean recordingVideo;
    private boolean cameraError;
    private boolean cameraRetryRunning;
    private boolean videoTimeoutShown;
    private boolean videoProgressTimedOut;
    private boolean videoFramesOwned;
    private long videoRecordStartMs;
    private volatile int videoSampledFrames;
    private float videoRecordFps;
    private float videoWindowSeconds;
    private int videoBubbleFrameCap;
    private int videoBubbleMaxSidePx;
    private int[] videoBubblePixels;
    private final ArrayList<Bitmap> videoFrames = new ArrayList<>();
    private final Runnable videoProgressTicker = new Runnable() {
        @Override public void run() {
            if (!recordingVideo) {
                return;
            }
            float seconds = videoProgressTimedOut && videoWindowSeconds > 0f
                    ? videoWindowSeconds
                    : currentVideoProgressSeconds();
            setVideoProgress(seconds, videoProgressTimedOut);
            if (!videoProgressTimedOut) {
                mainHandler.postDelayed(this, VIDEO_PROGRESS_TICK_MS);
            }
        }
    };

    private boolean active;
    private int offsetX;
    private int offsetY;

    public CameraFloatingPreview(AppCompatActivity activity, GLSurfaceView glView) {
        this.activity = activity;
        this.glView = glView;
        activity.getLifecycle().addObserver(this);
    }

    /** Register the listener that receives captured vision inputs (MainActivity wires this to Run_LLM). */
    public void setVisionCaptureListener(VisionCaptureListener listener) {
        this.captureListener = listener;
    }

    /** Tap-to-open / tap-to-close entry point for the header camera button. */
    public void toggle() {
        if (active) {
            close();
        } else {
            open();
        }
    }

    /** Stop capture and pause GL while the app releases optional vision runtime memory. */
    public void pauseForResourcePressure() {
        close(true);
    }

    private void open() {
        if (active) {
            return;
        }
        active = true;
        showPopup();
        if (!CameraService.hasPreparedCameraSelection()) {
            showCameraError(R.string.camera_error_unavailable);
            return;
        }
        ensureCameraService();
        showCameraConnecting();
        cameraService.start();
        bindFloatingPreviewSurface();
    }

    private void ensureCameraService() {
        if (cameraService != null) {
            return;
        }
        cameraService = new CameraService(activity, glView);
        cameraService.setCameraStatusCallback(new CameraService.CameraStatusCallback() {
            @Override public void onCameraReady() {
                if (!active) {
                    return;
                }
                cameraError = false;
                cameraRetryRunning = false;
                if (statusText != null) {
                    statusText.setText(R.string.camera_status_idle);
                }
                configureCaptureButton();
            }

            @Override public void onCameraError(int messageRes) {
                if (active) {
                    showCameraError(messageRes);
                }
            }
        });
    }

    private void showCameraConnecting() {
        cameraError = false;
        if (statusText != null) {
            statusText.setText(R.string.camera_status_opening);
        }
        if (captureButton != null) {
            captureButton.setEnabled(false);
        }
    }

    private void showCameraError(int messageRes) {
        cameraError = true;
        cameraRetryRunning = false;
        if (statusText != null) {
            statusText.setText(messageRes);
        }
        if (captureButton != null) {
            captureButton.setEnabled(true);
            captureButton.setText(R.string.camera_retry);
            captureButton.setOnClickListener(v -> retryCamera());
        }
    }

    private void retryCamera() {
        if (!active || cameraRetryRunning) {
            return;
        }
        cameraRetryRunning = true;
        showCameraConnecting();
        if (cameraService != null) {
            cameraService.retryOpen();
            return;
        }
        new Thread(() -> {
            CameraService.preloadCameraSelection(activity.getApplicationContext());
            activity.runOnUiThread(() -> {
                if (!active) {
                    return;
                }
                if (!CameraService.hasPreparedCameraSelection()) {
                    showCameraError(R.string.camera_error_unavailable);
                    return;
                }
                ensureCameraService();
                cameraService.start();
                bindFloatingPreviewSurface();
            });
        }, "camera-selection-retry").start();
    }

    private void configureCaptureButton() {
        if (captureButton == null) {
            return;
        }
        captureButton.setEnabled(true);
        captureButton.setText(R.string.camera_capture);
        captureButton.setOnClickListener(v -> capture());
    }

    private void close() {
        close(false);
    }

    private void close(boolean pauseGl) {
        if (!active) {
            return;
        }
        if (recordingVideo) {
            if (cameraService != null) {
                cameraService.cancelVideoRecording();
            }
            recordingVideo = false;
        }
        stopVideoProgressTicker();
        active = false;
        cameraError = false;
        cameraRetryRunning = false;
        if (cameraService != null) {
            if (pauseGl) {
                cameraService.pause();
            } else {
                cameraService.stop();
            }
        }
        synchronized (videoFrames) {
            clearVideoFramesLocked();
            videoFrames.clear();
        }
        dismissPopup();
    }

    // Snapshot for image inference. Holding the button starts video instead.
    private void capture() {
        if (!active || cameraService == null || recordingVideo || cameraError) {
            return;
        }
        if (statusText != null) {
            statusText.setText(activity.getString(R.string.camera_status_capturing));
        }
        // Image: push the CHW pixel_values to native (single memcpy) and build a chat thumbnail.
        cameraService.captureImageForInference((rgba, chw, w, h) -> {
            final Bitmap thumb = toThumbnail(rgba, w, h);
            activity.runOnUiThread(() -> {
                if (thumb == null) {
                    if (statusText != null) {
                        statusText.setText(activity.getString(R.string.camera_status_no_frame));
                    }
                    return;
                }
                if (captureListener != null) {
                    captureListener.onImageCaptured(thumb);
                }
                afterCapture();
            });
        });
    }

    private void startVideoRecording() {
        if (!active || cameraService == null || recordingVideo || cameraError ||
            !CameraService.isVideoCapable()) {
            return;
        }
        recordingVideo = true;
        videoTimeoutShown = false;
        videoProgressTimedOut = false;
        videoFramesOwned = true;
        videoRecordStartMs = SystemClock.elapsedRealtime();
        videoSampledFrames = 0;
        videoRecordFps = CameraService.getVideoFps();
        videoWindowSeconds = CameraService.getVideoWindowSeconds();
        videoBubbleFrameCap = Math.max(1, Math.min(CameraService.getVideoNumFrames(), MAX_VIDEO_BUBBLE_FRAMES));
        videoBubbleMaxSidePx = VIDEO_BUBBLE_MAX_SIDE_PX;
        synchronized (videoFrames) {
            clearVideoFramesLocked();
            videoFrames.clear();
        }
        setVideoProgress(0f, false);
        if (videoProgress != null) {
            videoProgress.setVisibility(View.VISIBLE);
        }
        if (videoProgressText != null) {
            videoProgressText.setVisibility(View.VISIBLE);
        }
        if (statusText != null) {
            statusText.setText(activity.getString(R.string.camera_status_video_recording));
        }
        startVideoProgressTicker();
        cameraService.beginVideoRecording((rgba, width, height, sampledFrames, sampledSeconds, timedOut) -> {
            synchronized (videoFrames) {
                Bitmap reusable = videoFrames.size() >= videoBubbleFrameCap
                        ? videoFrames.remove(0) : null;
                Bitmap frame = toVideoBubbleFrame(rgba, width, height, reusable);
                if (frame != null) {
                    videoFrames.add(frame);
                } else if (reusable != null && !reusable.isRecycled()) {
                    reusable.recycle();
                }
            }
            videoSampledFrames = sampledFrames;
            activity.runOnUiThread(() -> {
                if (timedOut && !videoTimeoutShown) {
                    videoProgressTimedOut = true;
                    stopVideoProgressTicker();
                    setVideoProgress(videoWindowSeconds > 0f ? videoWindowSeconds : currentVideoProgressSeconds(), true);
                    videoTimeoutShown = true;
                    String msg = activity.getString(R.string.camera_video_timeout);
                    if (statusText != null) {
                        statusText.setText(msg);
                    }
                    Toast.makeText(activity, msg, Toast.LENGTH_SHORT).show();
                }
            });
        });
    }

    private void finishVideoRecording() {
        if (!recordingVideo || cameraService == null) {
            return;
        }
        recordingVideo = false;
        stopVideoProgressTicker();
        final boolean ok = cameraService.finishVideoRecording();
        final ArrayList<Bitmap> frames;
        synchronized (videoFrames) {
            frames = new ArrayList<>(videoFrames);
            if (ok) {
                videoFramesOwned = false;
            } else {
                clearVideoFramesLocked();
                videoFrames.clear();
                frames.clear();
            }
        }
        final float fps = videoRecordFps;
        final float sampledSeconds = fps > 0f ? videoSampledFrames / fps
                : (SystemClock.elapsedRealtime() - videoRecordStartMs) / 1000f;
        final float seconds = videoWindowSeconds > 0f ? Math.min(sampledSeconds, videoWindowSeconds)
                : sampledSeconds;
        afterCapture();   // close the preview immediately on release, matching image capture.
        if (captureListener != null) {
            captureListener.onVideoCaptured(ok ? frames : null, seconds, ok);
        } else {
            recycleFrames(frames);
        }
    }

    private void startVideoProgressTicker() {
        stopVideoProgressTicker();
        if (videoWindowSeconds > 0f) {
            mainHandler.post(videoProgressTicker);
        }
    }

    private void stopVideoProgressTicker() {
        mainHandler.removeCallbacks(videoProgressTicker);
    }

    private float currentVideoProgressSeconds() {
        long elapsedMs = Math.max(0L, SystemClock.elapsedRealtime() - videoRecordStartMs);
        if (videoWindowSeconds > 0f) {
            long windowMs = Math.max(1L, Math.round(videoWindowSeconds * 1000f));
            elapsedMs = Math.min(elapsedMs, windowMs);
        }
        return elapsedMs / 1000f;
    }

    private void setVideoProgress(float seconds, boolean timedOut) {
        float window = videoWindowSeconds;
        if (videoProgress != null) {
            if (window > 0f) {
                int windowMs = Math.max(1, Math.round(window * 1000f));
                int progressMs = Math.min(windowMs, Math.max(0, Math.round(seconds * 1000f)));
                if (timedOut) {
                    progressMs = windowMs;
                    seconds = window;
                }
                videoProgress.setIndeterminate(false);
                videoProgress.setMax(windowMs);
                videoProgress.setProgress(progressMs);
            } else {
                videoProgress.setIndeterminate(true);
            }
        }
        if (videoProgressText != null) {
            float displaySeconds = window > 0f ? Math.min(seconds, window) : seconds;
            String secondsText = window > 0f
                ? String.format(Locale.US, "%.2f / %.2f", displaySeconds, window)
                : String.format(Locale.US, "%.2f", displaySeconds);
            videoProgressText.setText(activity.getString(
                    timedOut ? R.string.camera_video_progress_full : R.string.camera_video_progress,
                    secondsText));
        }
    }

    // Build a fresh ARGB_8888 bitmap from the mapped RGBA buffer (valid only during the callback) so the
    // chat can retain it. This is display-only; the model path consumes the separate CHW buffer directly.
    private Bitmap toThumbnail(java.nio.ByteBuffer rgba, int w, int h) {
        if (rgba == null || w <= 0 || h <= 0) {
            return null;
        }
        int maxSide = IMAGE_THUMB_MAX_SIDE_PX;
        int largest = Math.max(w, h);
        if (largest > maxSide) {
            return toScaledRgbaBitmap(rgba, w, h, maxSide);
        }
        Bitmap bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888);
        rgba.position(0);
        bmp.copyPixelsFromBuffer(rgba);
        return bmp;
    }

    private Bitmap toVideoBubbleFrame(java.nio.ByteBuffer rgba, int w, int h, Bitmap reusable) {
        if (rgba == null || w <= 0 || h <= 0) {
            return null;
        }
        return toScaledRgbaBitmap(
                rgba, w, h, Math.max(1, videoBubbleMaxSidePx), reusable);
    }

    private Bitmap toScaledRgbaBitmap(java.nio.ByteBuffer rgba, int w, int h, int maxSide) {
        return toScaledRgbaBitmap(rgba, w, h, maxSide, null);
    }

    private Bitmap toScaledRgbaBitmap(java.nio.ByteBuffer rgba, int w, int h, int maxSide,
                                      Bitmap reusable) {
        int largest = Math.max(w, h);
        int outW = w;
        int outH = h;
        if (largest > maxSide) {
            float scale = maxSide / (float) largest;
            outW = Math.max(1, Math.round(w * scale));
            outH = Math.max(1, Math.round(h * scale));
        }
        int pixelCount = outW * outH;
        if (videoBubblePixels == null || videoBubblePixels.length < pixelCount) {
            videoBubblePixels = new int[pixelCount];
        }
        for (int y = 0; y < outH; y++) {
            int srcY = Math.min(h - 1, (int) ((long) y * h / outH));
            for (int x = 0; x < outW; x++) {
                int srcX = Math.min(w - 1, (int) ((long) x * w / outW));
                int src = ((srcY * w) + srcX) * 4;
                int r = rgba.get(src) & 0xFF;
                int g = rgba.get(src + 1) & 0xFF;
                int b = rgba.get(src + 2) & 0xFF;
                int a = rgba.get(src + 3) & 0xFF;
                videoBubblePixels[y * outW + x] = (a << 24) | (r << 16) | (g << 8) | b;
            }
        }
        Bitmap frame = reusable;
        if (frame == null || frame.isRecycled() || frame.getWidth() != outW ||
                frame.getHeight() != outH || frame.getConfig() != Bitmap.Config.ARGB_8888) {
            if (frame != null && !frame.isRecycled()) {
                frame.recycle();
            }
            frame = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888);
        }
        frame.setPixels(videoBubblePixels, 0, outW, 0, 0, outW, outH);
        return frame;
    }

    private void clearVideoFramesLocked() {
        if (videoFramesOwned) {
            recycleFrames(videoFrames);
        }
        videoFramesOwned = true;
    }

    private static void recycleFrames(List<Bitmap> frames) {
        if (frames == null) {
            return;
        }
        for (Bitmap frame : frames) {
            if (frame != null && !frame.isRecycled()) {
                frame.recycle();
            }
        }
    }

    // Image capture hands one frame to the chat, then closes the preview (video mode keeps it open).
    private void afterCapture() {
        close();
    }

    // ================= Floating window (built programmatically to avoid a dedicated layout) =================

    @SuppressLint("ClickableViewAccessibility")
    private void showPopup() {
        if (popupWindow != null) {
            return;
        }
        int pad = dp(14);
        LinearLayout root = new LinearLayout(activity);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(pad, dp(10), pad, pad);
        GradientDrawable chrome = new GradientDrawable();
        chrome.setColor(0xF21A1B22);
        chrome.setCornerRadius(dp(18));
        chrome.setStroke(dp(1), 0x33FFFFFF);
        root.setBackground(chrome);
        root.setElevation(dp(12));

        // Title bar doubles as the drag handle.
        LinearLayout header = new LinearLayout(activity);
        header.setOrientation(LinearLayout.HORIZONTAL);
        header.setGravity(Gravity.CENTER_VERTICAL);
        TextView title = new TextView(activity);
        title.setText(activity.getString(R.string.camera_window_title));
        title.setTextColor(Color.WHITE);
        title.setTextSize(14);
        title.setTypeface(title.getTypeface(), Typeface.BOLD);
        header.addView(title, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        root.addView(header, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        FrameLayout previewFrame = new FrameLayout(activity);
        // Neutral letterbox padding (represents the model's gray-fit border), not a harsh black edge.
        previewFrame.setBackgroundColor(0xFF23242C);
        livePreview = new TextureView(activity);
        livePreview.setOpaque(true);
        livePreview.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {
            @Override
            public void onSurfaceTextureAvailable(@NonNull SurfaceTexture surface, int width, int height) {
                bindFloatingPreviewSurface();
                configureTransform(width, height);
            }

            @Override
            public void onSurfaceTextureSizeChanged(@NonNull SurfaceTexture surface, int width, int height) {
                bindFloatingPreviewSurface();
                configureTransform(width, height);
            }

            @Override
            public boolean onSurfaceTextureDestroyed(@NonNull SurfaceTexture surface) {
                if (cameraService != null) {
                    cameraService.setFloatingPreviewSurfaceTexture(null, 0, 0);
                }
                return true;
            }

            @Override
            public void onSurfaceTextureUpdated(@NonNull SurfaceTexture surface) {
            }
        });
        previewFrame.addView(livePreview, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT));

        // Match the image model aspect ratio; the camera frame is fitted without distortion.
        int base = Math.min(dp(320),
                activity.getResources().getDisplayMetrics().widthPixels - dp(56));
        int frameW = base;
        int frameH = base;
        int[] modelSize = CameraService.getInputImageSize();
        if (modelSize != null && modelSize.length == 2 && modelSize[0] > 0 && modelSize[1] > 0) {
            int maxH = dp(360);
            frameH = Math.round(base * (modelSize[1] / (float) modelSize[0]));
            if (frameH > maxH) {
                frameH = maxH;
                frameW = Math.round(maxH * (modelSize[0] / (float) modelSize[1]));
            }
        }
        LinearLayout.LayoutParams previewLp = new LinearLayout.LayoutParams(frameW, frameH);
        previewLp.topMargin = dp(10);
        root.addView(previewFrame, previewLp);

        statusText = new TextView(activity);
        statusText.setText(activity.getString(R.string.camera_status_idle));
        statusText.setTextColor(Color.WHITE);
        statusText.setTextSize(11);
        LinearLayout.LayoutParams statusLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        statusLp.topMargin = dp(8);
        root.addView(statusText, statusLp);

        videoProgress = new ProgressBar(activity, null, android.R.attr.progressBarStyleHorizontal);
        videoProgress.setMax(1000);
        videoProgress.setProgress(0);
        videoProgress.setVisibility(View.GONE);
        LinearLayout.LayoutParams progressLp = new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, dp(6));
        progressLp.topMargin = dp(8);
        root.addView(videoProgress, progressLp);

        videoProgressText = new TextView(activity);
        videoProgressText.setTextColor(0xCCFFFFFF);
        videoProgressText.setTextSize(10);
        videoProgressText.setVisibility(View.GONE);
        LinearLayout.LayoutParams progressTextLp = new LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        progressTextLp.topMargin = dp(5);
        root.addView(videoProgressText, progressTextLp);

        // Flat action row: tap Capture for an image; long-press Capture for video when the model supports it.
        LinearLayout buttons = new LinearLayout(activity);
        buttons.setOrientation(LinearLayout.HORIZONTAL);

        captureButton = new Button(activity);
        captureButton.setText(activity.getString(R.string.camera_capture));
        captureButton.setAllCaps(false);
        captureButton.setTextSize(13);
        captureButton.setTextColor(ContextCompat.getColor(activity, R.color.textOnYellow));
        captureButton.setPadding(0, 0, 0, 0);
        captureButton.setMinHeight(0);
        captureButton.setMinimumHeight(0);
        GradientDrawable captureBg = new GradientDrawable();
        captureBg.setColor(ContextCompat.getColor(activity, R.color.energyYellow));
        captureBg.setCornerRadius(dp(10));
        captureButton.setBackground(captureBg);
        configureCaptureButton();
        captureButton.setOnLongClickListener(v -> {
            startVideoRecording();
            return CameraService.isVideoCapable();
        });
        captureButton.setOnTouchListener((v, event) -> {
            int action = event.getActionMasked();
            if ((action == MotionEvent.ACTION_UP || action == MotionEvent.ACTION_CANCEL) && recordingVideo) {
                finishVideoRecording();
                return true;
            }
            return false;
        });
        buttons.addView(captureButton, new LinearLayout.LayoutParams(0, dp(40), 1f));

        final int energyYellow = ContextCompat.getColor(activity, R.color.energyYellow);
        Button closeButton = new Button(activity);
        closeButton.setText(activity.getString(R.string.camera_close));
        closeButton.setAllCaps(false);
        closeButton.setTextSize(13);
        closeButton.setTextColor(Color.WHITE);
        closeButton.setPadding(0, 0, 0, 0);
        closeButton.setMinHeight(0);
        closeButton.setMinimumHeight(0);
        GradientDrawable closeBg = new GradientDrawable();
        closeBg.setColor(0x22FFFFFF);
        closeBg.setCornerRadius(dp(10));
        closeBg.setStroke(dp(1), energyYellow);
        closeButton.setBackground(closeBg);
        closeButton.setOnClickListener(v -> close());
        LinearLayout.LayoutParams closeLp = new LinearLayout.LayoutParams(0, dp(40), 1f);
        closeLp.leftMargin = dp(8);
        buttons.addView(closeButton, closeLp);

        LinearLayout.LayoutParams buttonsLp = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        buttonsLp.topMargin = dp(12);
        root.addView(buttons, buttonsLp);

        popupWindow = new PopupWindow(root,
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT, false);
        popupWindow.setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));
        popupWindow.setOutsideTouchable(false);
        popupWindow.setClippingEnabled(false);
        // Keep IME focus in the composer; popup buttons still receive touches.
        popupWindow.setFocusable(false);
        popupWindow.setOnDismissListener(() -> {
            if (active) {
                close();
            }
        });

        header.setOnTouchListener(new View.OnTouchListener() {
            private float downRawX;
            private float downRawY;
            private int startOffsetX;
            private int startOffsetY;

            @Override
            public boolean onTouch(View v, MotionEvent event) {
                switch (event.getActionMasked()) {
                    case MotionEvent.ACTION_DOWN:
                        downRawX = event.getRawX();
                        downRawY = event.getRawY();
                        startOffsetX = offsetX;
                        startOffsetY = offsetY;
                        return true;
                    case MotionEvent.ACTION_MOVE:
                        offsetX = startOffsetX + Math.round(event.getRawX() - downRawX);
                        offsetY = startOffsetY + Math.round(event.getRawY() - downRawY);
                        if (popupWindow != null) {
                            clampPopupOffsets();
                            popupWindow.update(offsetX, offsetY, -1, -1);
                        }
                        return true;
                    default:
                        return false;
                }
            }
        });

        offsetX = 0;
        offsetY = 0;
        popupWindow.showAtLocation(glView.getRootView(), Gravity.CENTER, offsetX, offsetY);
    }

    private void clampPopupOffsets() {
        if (popupWindow == null || popupWindow.getContentView() == null) {
            return;
        }
        View root = glView.getRootView();
        View content = popupWindow.getContentView();
        int maxOffsetX = Math.max(0, (root.getWidth() - content.getWidth()) / 2);
        int maxOffsetY = Math.max(0, (root.getHeight() - content.getHeight()) / 2);
        offsetX = Math.max(-maxOffsetX, Math.min(offsetX, maxOffsetX));
        offsetY = Math.max(-maxOffsetY, Math.min(offsetY, maxOffsetY));
    }

    private void dismissPopup() {
        if (popupWindow != null) {
            if (popupWindow.isShowing()) {
                popupWindow.dismiss();
            }
            popupWindow = null;
        }
        livePreview = null;
        statusText = null;
        captureButton = null;
        videoProgress = null;
        videoProgressText = null;
    }

    private void bindFloatingPreviewSurface() {
        if (cameraService != null && livePreview != null && livePreview.isAvailable()) {
            cameraService.setFloatingPreviewSurfaceTexture(
                    livePreview.getSurfaceTexture(), livePreview.getWidth(), livePreview.getHeight());
            configureTransform(livePreview.getWidth(), livePreview.getHeight());
        }
    }

    // Restore the camera aspect, rotate upright, then fit with letterboxing.
    private void configureTransform(int viewW, int viewH) {
        if (livePreview == null || cameraService == null || viewW <= 0 || viewH <= 0) {
            return;
        }
        int pW = cameraService.getPreviewBufferWidth();
        int pH = cameraService.getPreviewBufferHeight();
        if (pW <= 0 || pH <= 0) {
            return;
        }
        int rot = cameraService.getPreviewContentRotation(displayRotationDegrees());
        Matrix matrix = new Matrix();
        float cx = viewW / 2f;
        float cy = viewH / 2f;
        // Undo TextureView's implicit stretch.
        float fitScale = Math.min(viewW / (float) pW, viewH / (float) pH);
        float fittedW = pW * fitScale;
        float fittedH = pH * fitScale;
        matrix.postScale(fittedW / viewW, fittedH / viewH, cx, cy);
        matrix.postRotate(rot, cx, cy);
        // Uniformly fit the rotated frame without cropping.
        boolean swap = (rot % 180 != 0);
        float shownW = swap ? fittedH : fittedW;
        float shownH = swap ? fittedW : fittedH;
        float fitFinal = Math.min(viewW / shownW, viewH / shownH);
        matrix.postScale(fitFinal, fitFinal, cx, cy);
        livePreview.setTransform(matrix);
        livePreview.invalidate();
    }

    private int displayRotationDegrees() {
        android.view.Display display = null;
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.R) {
            display = activity.getDisplay();
        }
        if (display == null) {
            display = activity.getWindowManager().getDefaultDisplay();
        }
        switch (display.getRotation()) {
            case Surface.ROTATION_90:  return 90;
            case Surface.ROTATION_180: return 180;
            case Surface.ROTATION_270: return 270;
            default:                   return 0;
        }
    }

    private int dp(float value) {
        return Math.round(activity.getResources().getDisplayMetrics().density * value);
    }

    // ================= Lifecycle: never keep the camera running in the background =================

    @Override
    public void onPause(@NonNull LifecycleOwner owner) {
        if (active) {
            close(true);
        } else if (cameraService != null) {
            cameraService.pause();
        }
    }

    @Override
    public void onDestroy(@NonNull LifecycleOwner owner) {
        active = false;
        stopVideoProgressTicker();
        if (cameraService != null) {
            cameraService.release();
            cameraService = null;
        }
        dismissPopup();
        activity.getLifecycle().removeObserver(this);
    }
}
