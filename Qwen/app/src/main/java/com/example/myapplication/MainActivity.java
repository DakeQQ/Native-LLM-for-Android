package com.example.myapplication;

import android.Manifest;
import android.animation.ObjectAnimator;
import android.animation.PropertyValuesHolder;
import android.animation.ValueAnimator;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.os.Debug;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.os.PowerManager;
import android.os.SystemClock;
import android.text.Editable;
import android.text.InputFilter;
import android.text.TextWatcher;
import android.transition.ChangeBounds;
import android.transition.Fade;
import android.transition.TransitionManager;
import android.transition.TransitionSet;
import android.view.Choreographer;
import android.view.View;
import android.view.ViewGroup;
import android.view.ViewStub;
import android.view.Window;
import android.view.WindowManager;
import android.view.animation.AccelerateDecelerateInterpolator;
import android.view.animation.LinearInterpolator;
import android.view.accessibility.AccessibilityNodeInfo;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.ToggleButton;
import android.widget.Toast;
import android.util.Base64;
import android.util.Log;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.app.AppCompatDelegate;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.google.android.material.button.MaterialButtonToggleGroup;
import com.google.android.material.materialswitch.MaterialSwitch;
import com.google.android.material.slider.RangeSlider;
import com.google.android.material.slider.Slider;
import com.google.android.material.textfield.TextInputEditText;

import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.ref.WeakReference;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

public class MainActivity extends AppCompatActivity {
    private static final String FLOW_TAG = "QwenFlow";
    public static final int font_size = 18;
    private ToggleButton thinkButton;
    private Button sendButton;
    // Shown only while generating; tapping halts decode and native tags the reply "user manually stopped".
    private Button stopButton;
    // Clear = panic reset (always live); Compute launches the benchmark (greyed while it would contend with inference).
    private Button clearButton;
    private ImageButton computeButton;
    private ImageButton themeToggle;
    private TextInputEditText inputBox;
    // Decode strategy is a required single choice. Only the selected mode's parameter group is visible.
    private static final int DECODE_MODE_GREEDY = 0;
    private static final int DECODE_MODE_BEAM = 1;
    private static final int DECODE_MODE_SAMPLING = 2;
    private int decodeMode = DECODE_MODE_GREEDY;
    private DecodePreferences decodePreferences;
    private MaterialButtonToggleGroup decodeStrategyGroup;
    private View beamControls;
    private View directPenaltyControls;
    private View samplerControls;
    private Slider beamSlider;
    private Slider topKSlider;
    private Slider penaltySlider;
    private Slider penaltyRangeSlider;
    private Slider samplingTemperatureSlider;
    private Slider samplingTopKSlider;
    private Slider samplingTopPSlider;
    private Slider samplingPenaltySlider;
    private Slider memorySlider;
    private Slider prefillSlider;
    private View prefillControls;
    private Slider decodeLimitSlider;
    private TextView beamValue;
    private TextView topKValue;
    private TextView penaltyValue;
    private TextView penaltyRangeValue;
    private TextView samplingTemperatureValue;
    private TextView samplingTopKValue;
    private TextView samplingTopPValue;
    private TextView samplingPenaltyValue;
    private TextView memoryValue;
    // Header headline (free %% + capacity) for the retained cross-turn memory window.
    private TextView memoryRemainingValue;
    // memory_remaining_bar shows the USED fraction (fills from the left, green->red as it fills); glow pulses on climb.
    private View memoryRemainingBar;
    private View memoryRemainingGlow;
    // Draggable hysteresis-band markers: green = post-reset target, red = rebuild trigger. Bounds MUST match
    // user_settings.h MIN/MAX_MEMORY_BAND_PERCENT (native re-clamps), else caption + persisted value disagree.
    private static final int MIN_MEMORY_BAND_PERCENT = 15;
    private static final int MAX_MEMORY_BAND_PERCENT = 95;
    private static final int MIN_MEMORY_BAND_GAP = 1;
    private static final int ACTION_GREEN_DECREASE = 0x01020001;
    private static final int ACTION_GREEN_INCREASE = 0x01020002;
    private static final int ACTION_RED_DECREASE = 0x01020003;
    private static final int ACTION_RED_INCREASE = 0x01020004;
    private View memoryRemainingTrack;
    private View memoryThresholdGreen;
    private View memoryThresholdRed;
    private RangeSlider memoryThresholdSlider;
    private TextView memoryBandValue;
    private int memoryGreenPercent = 60;
    private int memoryRedPercent = 95;
    private GradientDrawable memoryBarDrawable;
    private GradientDrawable memoryGlowDrawable;
    private TextView prefillValue;
    private TextView decodeLimitValue;
    // Performance-panel readouts (Prefill / Decode tokens/s), updated from the native perf payload.
    private TextView perfPrefillValue;
    private TextView perfDecodeValue;
    private View postProcessingStatus;
    private ObjectAnimator postProcessingPulse;
    // True while native is organising cross-turn memory (整理记忆). Greys + tap-guards Send/Test/Theme;
    // Clear stays live. Native publishes this before posting UI work so Activity recreation sees it.
    private static volatile boolean postProcessingActive = false;
    // True while a reply is generating (mirrors the Send<->Stop swap); greys + tap-guards Compute. UI-thread only.
    private static boolean generatingActive = false;
    private static boolean stopRequested = false;
    // True while Clear_Cache or Rollback_LLM owns native maintenance. Survives Activity recreation.
    private static boolean maintenanceActive = false;
    // Live system monitor: app CPU% + resident RAM, sampled ~1 Hz on a background thread via /proc/self/{stat,statm}.
    // CPU% = jiffy delta (utime+stime) over wall-clock, normalised by core count, clamped 0..100.
    private TextView cpuValue;
    private TextView memValue;
    private HandlerThread monitorThread;
    private Handler monitorHandler;
    private volatile boolean monitorActive = false;
    private final ProcessStatsSampler processStatsSampler = new ProcessStatsSampler();
    private static final long MONITOR_INTERVAL_MS = 1000L;
    // Collapsible decode-strategy slider container, toggled by the top-bar gear; collapsed by default.
    private View decodeContent;
    private ImageButton decodeSettingsButton;
    private boolean decodeControlsInitialized;
    private boolean modelReadyUiInitialized;
    // Panel edits are drafts; native receives one final snapshot only when the panel is collapsed.
    private boolean decodeConfigDirty = false;
    private View jumpToBottomButton;
    private boolean autoScrollLocked = false;
    // Header ambient-glow views + "breathing" animators (decorative; safe to be null). Started onResume, cancelled onPause.
    private View avatarGlow;
    private ObjectAnimator avatarPulse;
    private ImageButton systemPromptButton;
    private ImageButton cameraButton;
    // Vision timing is rendered in the snapshot bubble; video also shows real-time factor.
    private TextView modelInfoValue;
    // Set at a vision Send; consumed when the first reply token lands so we can time the processing window.
    private static boolean visionTimingPending = false;
    private static boolean visionTimingIsVideo = false;
    private static int visionTimingTurnId = ChatMessage.NO_TURN;
    private static int visionPreprocessedTurnId = ChatMessage.NO_TURN;
    private static int visionCaptureSerial = 0;
    private static long visionSendElapsedMs = 0L;
    private static int visionFrameCount = 0;
    private static float visionFps = 0f;
    private static float visionVideoSeconds = 0f;
    private static int pendingVideoFrameCount = 0;
    private static float pendingVideoSeconds = 0f;
    // Mirrors the latest beam toggle. Beam decodes silently so it needs the "typing" placeholder.
    private static boolean useBeamSearch = false;
    private RecyclerView answerView;
    private ChatAdapter chatAdapter;
    private static List<ChatMessage> messages;
    private static final int TRANSCRIPT_RESIDENT_MESSAGES = 128;
    private static final AtomicBoolean transcriptPagingRunning = new AtomicBoolean();
    private static final AtomicLong transcriptPagingSerial = new AtomicLong();
    private static volatile int transcriptPagedThrough = -1;
    private static String usrInputText = "";
    private static WeakReference<MainActivity> activeActivity = new WeakReference<>(null);
    // Shared main-thread handler for native callbacks + UI posts. Generation IDs invalidate only stale
    // decode callbacks; unrelated model, memory and vision tasks remain queued.
    private static final Handler mainHandler = new Handler(Looper.getMainLooper());
    private static final Object STREAM_UI_LOCK = new Object();
    private static final StringBuilder pendingStreamUiText = new StringBuilder(512);
    private static boolean streamUiFrameScheduled = false;
    private static long pendingStreamGenerationId = -1L;
    private static long pendingStreamEventTimeMillis = 0L;
    private static final Choreographer.FrameCallback STREAM_UI_FRAME_CALLBACK =
            frameTimeNanos -> flushPendingStreamUi();
    // Current generation worker; joined before a new one starts (native is a single non-reentrant context).
    private static volatile LLMThread llmThread;
    // Monotonic per-turn id (never reused); passed to Run_LLM to checkpoint the KV cache, used by Rollback_LLM.
    private static int nextTurnId = 0;
    private static int usrTurnId = ChatMessage.NO_TURN;
    private static final AtomicLong generationSerial = new AtomicLong();
    private static volatile long activeGenerationId = 0L;
    private static volatile PendingGenerationCompletion pendingGenerationCompletion;
    private static volatile PendingRollbackCompletion pendingRollbackCompletion;
    // Captures are shown and pre-encoded immediately; the next Send consumes the pending vision input.
    private static boolean pendingVisionCapture = false;
    private static boolean pendingVisionIsVideo = false;
    private static int pendingVisionTurnId = ChatMessage.NO_TURN;
    // Fallback vocab asset name if none is auto-detected. Staging prefers whichever vocab_*.txt asset is
    // present (isVocabFileName, case-insensitive) and native Pre_Process auto-discovers it, so an exact
    // name match is no longer required.
    private static final String VOCAB_FILE_NAME = "vocab_Qwen3.5-0.8B.txt";
    private static final String ASSET_MANIFEST_NAME = "model_bundle.sha256";
    // Editor "Reset" default. MUST match native DEFAULT_SYSTEM_PROMPT (user_settings.h).
    private static final String SYSTEM_PROMPT_DEFAULT = "You are a helpful assistant.";
        private static final int MAX_SYSTEM_PROMPT_CHARS = 16_384;
    private static final String RUN_ERROR_PREFIX = "LLM_ERROR|";
    private static final String RUN_STATUS_CANCELLED = "LLM_STATUS|CANCELLED";
    // Multi-turn chat: history persists across turns; Clear sets this so the NEXT Run_LLM resets native context once.
    private static boolean clear_flag = false;
    private static volatile boolean chatting = false;
    // Models + tokenizer load ONCE per process; this guard stops a theme-switch recreate from reloading them
    // (native global state + the static messages list survive the recreate).
    private static boolean modelsReady = false;
    // True while the background model-load thread runs; stops a theme-switch recreate from starting a 2nd load.
    private static volatile boolean modelsLoading = false;
    private static volatile boolean modelLoadFailed = false;
    private static volatile int modelLoadingStageRes = R.string.model_loading_verify;
    private static String modelLoadErrorText;
    private static volatile double modelAssetStageMs = 0.0;
    private static volatile double nativeOrtLoadMs = 0.0;
    private static volatile double vocabAssetStageMs = 0.0;
    private static volatile double tokenizerSetupMs = 0.0;
    private static volatile double modelReadyMs = 0.0;
    private static volatile String loadedModelDisplayName = "Qwen";
    private static volatile long loadedModelAssetBytes = 0L;
    // Text becomes ready first; capable devices load vision later while idle or immediately on camera tap.
    private static volatile boolean visionModelsReady = false;
    private static volatile boolean visionModelsLoading = false;
    private static boolean runtimeTrimRunning = false;
    private static boolean runtimeTrimPending = false;
    private static boolean runtimeTrimPendingVisionRelease = false;
    private static int runtimePressureLevel = 0;
    private static int thermalPressureLevel = 0;
    private static int memoryPressureLevel = 0;
    private static int memoryPressureSerial = 0;
    private PowerManager.OnThermalStatusChangedListener thermalStatusListener;
    // Coalesce decode-setting updates on one worker. Only a major mode change creates/releases ORT sessions;
    // scalar slider updates are published without putting the UI into the strategy-loading state.
    private static final Object DECODE_CONFIG_LOCK = new Object();
    private static final ExecutorService DECODE_CONFIG_EXECUTOR = Executors.newSingleThreadExecutor(runnable -> {
        Thread thread = new Thread(runnable, "decode-config");
        thread.setDaemon(true);
        return thread;
    });
    private static final ExecutorService CHAT_MEDIA_EXECUTOR = Executors.newSingleThreadExecutor(runnable -> {
        Thread thread = new Thread(runnable, "chat-media-encode");
        thread.setDaemon(true);
        return thread;
    });
    // Shared pool for one-shot background maintenance (memory stats, KV rollback, runtime trim, vision
    // load/prewarm, etc.). A CACHED pool reuses idle threads to cut per-call thread-creation churn but
    // still spawns on demand, so tasks never serialize -- timing/concurrency semantics are unchanged.
    private static final ExecutorService BACKGROUND_EXECUTOR = Executors.newCachedThreadPool(runnable -> {
        Thread thread = new Thread(runnable);
        thread.setDaemon(true);
        return thread;
    });
    private static final CountDownLatch STARTUP_RUNTIME_CONFIG_READY = new CountDownLatch(1);
    private static DecodeConfig pendingDecodeConfig;
    private static boolean decodeConfigWorkerRunning = false;
    private static volatile boolean decodeConfigLoading = false;
    private static volatile boolean decodeConfigReady = true;
    private static volatile int configuredDecodeMode = DECODE_MODE_GREEDY;
    private static final AtomicInteger pendingSystemPromptConfigs = new AtomicInteger();
    private static final AtomicLong systemPromptConfigSerial = new AtomicLong();
    private static volatile boolean systemPromptConfigLoading = false;
    private static volatile boolean systemPromptConfigReady = true;
    private static volatile String appliedSystemPrompt;

    // ADB-only benchmark mode. Normal launches have no benchmark_workload extra and never enter this path.
    private static final String BENCHMARK_TAG = "QwenV35Bench";
    private static final Object BENCHMARK_CAPTURE_LOCK = new Object();
    private static final StringBuilder benchmarkOutput = new StringBuilder(4096);
    private static String benchmarkPerfPayload = "";
    private static boolean benchmarkCaptureActive = false;
    private static final AtomicBoolean benchmarkRunning = new AtomicBoolean();
    private static volatile String claimedBenchmarkRunId;

    private static final int EP_CPU = 0;
    private static final int DEFAULT_MEMORY_TOKENS_FALLBACK = 2048;
    private static final int DEFAULT_PREFILL_TOKENS_FALLBACK = 256;
    private static final int DEFAULT_DECODE_TOKENS_FALLBACK = 4095;
    private static final int MIN_MEMORY_TOKENS_FALLBACK = 256;
    private static final int MAX_MEMORY_TOKENS_FALLBACK = 8160;
    private static final int MIN_PREFILL_TOKENS_FALLBACK = 0;
    private static final int MAX_PREFILL_TOKENS_FALLBACK = 8160;
    private static final int MIN_DECODE_TOKENS_FALLBACK = 2;
    private static final int MAX_DECODE_TOKENS_FALLBACK = 8192;
    private static final int PREFERRED_MEMORY_TOKENS_STEP = 512;
    private static final int PREFERRED_PREFILL_TOKENS_STEP = 128;
    private static final int PREFERRED_DECODE_TOKENS_STEP = 1;
    private int minMemoryTokens = MIN_MEMORY_TOKENS_FALLBACK;
    private int maxMemoryTokens = MAX_MEMORY_TOKENS_FALLBACK;
    private int defaultMemoryTokens = DEFAULT_MEMORY_TOKENS_FALLBACK;
    private int minPrefillTokens = MIN_PREFILL_TOKENS_FALLBACK;
    private int maxPrefillTokens = MAX_PREFILL_TOKENS_FALLBACK;
    private int defaultPrefillTokens = DEFAULT_PREFILL_TOKENS_FALLBACK;
    private int minDecodeTokens = MIN_DECODE_TOKENS_FALLBACK;
    private int maxDecodeTokens = MAX_DECODE_TOKENS_FALLBACK;
    private int defaultDecodeTokens = DEFAULT_DECODE_TOKENS_FALLBACK;
    private int memoryTokensStep = PREFERRED_MEMORY_TOKENS_STEP;
    private int prefillTokensStep = PREFERRED_PREFILL_TOKENS_STEP;
    private int decodeTokensStep = PREFERRED_DECODE_TOKENS_STEP;
    private int memoryUsedTokens = 0;
    private int memoryCapacityTokens = DEFAULT_MEMORY_TOKENS_FALLBACK;
    private double memoryRemainingPercent = 100.0;
    private int memoryStatsRequestId = 0;
    private float memoryBarFraction = 0.0f;   // used fraction; starts empty (fresh memory)
    private int lastMemoryBarColor = 0;
    private int lastDisplayedMemoryUsedTokens = -1;
    private int lastDisplayedMemoryCapacityTokens = -1;
    private ValueAnimator memoryBarAnimator;
    private ObjectAnimator memoryGlowAnimator;

    // Live-camera zero-copy RGB preview (created lazily on the first camera-button tap).
    private CameraFloatingPreview cameraPreview;
    private final ActivityResultLauncher<String> cameraPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), granted -> {
                if (granted) {
                    ensureVisionModelsAndOpenCamera();
                } else {
                    showToast(this, getString(R.string.camera_permission_denied), false);
                }
            });

    // ==================== Activity lifecycle ====================

    static {
        System.loadLibrary("myapplication");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        MaxHeightNestedScrollView header = findViewById(R.id.header);
        header.setMaxHeight(Math.max(dp(160),
            Math.round(getResources().getDisplayMetrics().heightPixels * 0.52f)));
        activeActivity = new WeakReference<>(this);
        ImageView set_photo = findViewById(R.id.role_image);
        set_photo.setImageResource(R.drawable.psyduck);
        clearButton = findViewById(R.id.clear);
        sendButton = findViewById(R.id.send);
        stopButton = findViewById(R.id.stop);
        stopButton.setOnClickListener(v -> requestStopGeneration(true));
        inputBox = findViewById(R.id.input_text);
        // The static messages list survives the Activity recreate a Light/Dark switch triggers; only allocate on cold start.
        if (messages == null) {
            File cacheDirectory = getCacheDir();
            CHAT_MEDIA_EXECUTOR.execute(() -> ChatMessage.clearStaleMedia(cacheDirectory));
            messages = new ArrayList<>();
        }
        chatAdapter = new ChatAdapter(messages);
        chatAdapter.setOnUserMessageClickListener(this::onEditUserMessage);
        answerView = findViewById(R.id.result_text);
        LinearLayoutManager chatLayoutManager = new LinearLayoutManager(this);
        // Anchor to the bottom so a growing reply keeps its newest line in view.
        chatLayoutManager.setStackFromEnd(true);
        answerView.setLayoutManager(chatLayoutManager);
        // Streaming rebinds the same bubble many times/sec; drop item animations so text swaps don't flicker.
        answerView.setItemAnimator(null);
        answerView.setAdapter(chatAdapter);
        setupAutoScrollLock();
        clearButton.setOnClickListener(v -> {
            // Panic button: NOT gated by guardPostProcessing() -- must stay actionable while native organises memory.
            clearHistory();
        });
        thinkButton = findViewById(R.id.think_mode);
        thinkButton.setChecked(false);
        perfPrefillValue = findViewById(R.id.perf_prefill_value);
        perfDecodeValue = findViewById(R.id.perf_decode_value);
        postProcessingStatus = findViewById(R.id.post_processing_status);
        renderPostProcessingState();
        cpuValue = findViewById(R.id.perf_cpu_value);
        memValue = findViewById(R.id.perf_mem_value);
        setupSystemMonitor();
        setupPerfChartCells();
        avatarGlow = findViewById(R.id.avatar_glow);
        setupThemeToggle();
        // Open the Compute / Memory benchmark page (its own libmembw.so, never touches the LLM); greyed +
        // tap-guarded while generating / organising memory.
        computeButton = findViewById(R.id.btn_compute);
        computeButton.setOnClickListener(v -> {
            if (guardPostProcessing() || guardInferenceBusy()) {
                return;
            }
            startActivity(new Intent(MainActivity.this, ComputeMemoryActivity.class));
        });
        // Open the slot system-prompt editor. Each live turn injects this text; Organize Memory ON strips it
        // from the rebuilt cache.
        systemPromptButton = findViewById(R.id.btn_system_prompt);
        systemPromptButton.setOnClickListener(v -> onEditSystemPrompt());
        // Live camera: open/close the floating zero-copy RGB capture window.
        cameraButton = findViewById(R.id.btn_camera);
        cameraButton.setOnClickListener(v -> onCameraButtonClicked());
        updateHeaderActionStates();
        sendButton.setEnabled(false);
        Choreographer.getInstance().postFrameCallback(frameTimeNanos -> mainHandler.post(() -> {
            MainActivity activity = currentActivity();
            if (activity != null && !activity.isFinishing() && !activity.isDestroyed()) {
                activity.setupDecodeControls();
                if (modelsReady) {
                    activity.initializeModelReadyUi();
                }
                activity.updateModelLoadingUi();
                STARTUP_RUNTIME_CONFIG_READY.countDown();
            }
        }));
        // Stage the selected merged-ONNX bundle and its external-data siblings into the flat cache namespace
        // consumed by native path loading. Source assets may be flat or grouped under one model directory.
        if (!modelsReady && !modelsLoading) {
            // Stage assets + load graphs + tokenizer OFF the UI thread (multi-second I/O would ANR).
            modelLoadFailed = false;
            modelLoadErrorText = null;
            modelLoadingStageRes = R.string.model_loading_verify;
            modelsLoading = true;
            final Context loadContext = getApplicationContext();
            GenerationService.startModelLoading(loadContext);
            updateModelLoadingUi();
            Intent launchIntent = getIntent();
            final boolean keepLoadingServiceForBenchmark = launchIntent != null &&
                    launchIntent.hasExtra("benchmark_workload");
            // Keep the transcript empty while loading; TYPE_LOADING is reserved for beam replies.
            final AssetManager bgMgr = loadContext.getAssets();
            new Thread(() -> {
                final long modelReadyStartNanos = SystemClock.elapsedRealtimeNanos();
                byte[] copyBuffer = new byte[102400];
                boolean ok = false;
                String error = null;
                try {
                    STARTUP_RUNTIME_CONFIG_READY.await();
                    AssetManifest assetManifest = loadAssetManifest(bgMgr);
                    SharedPreferences stagePrefs = loadContext.getSharedPreferences(
                            App.PREFS, Context.MODE_PRIVATE);
                    boolean restageVocab = !assetManifest.fingerprint.equals(
                            stagePrefs.getString(App.KEY_ASSET_BUNDLE_FINGERPRINT, ""));
                    String modelAssetDir = findModelAssetDirectory(bgMgr);
                    String[] files = bgMgr.list(modelAssetDir);
                    int count_file = 0;
                    String vocabAssetPath = null;   // auto-detected vocab_*.txt; source may be in a bundle directory
                    String vocabCacheName = null;
                    long modelAssetBytes = 0L;
                    // Models load directly from uncompressed APK assets. Remove legacy staged copies so
                    // they cannot shadow the signed bundle or consume a second model-sized disk footprint.
                    File cacheDirectory = loadContext.getCacheDir();
                    clearStagedModelAssets(cacheDirectory);
                    final long modelAssetStageStartNanos = SystemClock.elapsedRealtimeNanos();
                    if (files != null) {
                        for (String fileName : files) {
                            String assetPath = modelAssetDir.isEmpty()
                                    ? fileName
                                    : modelAssetDir + "/" + fileName;
                            String[] subEntries = bgMgr.list(assetPath);
                            if (subEntries == null || subEntries.length == 0) {
                                if (fileName.endsWith(".onnx")) {
                                    count_file += 1;
                                }
                                if (fileName.endsWith(".onnx") || fileName.endsWith(".data")) {
                                    try (android.content.res.AssetFileDescriptor descriptor =
                                                 bgMgr.openFd(assetPath)) {
                                        modelAssetBytes += Math.max(0L, descriptor.getLength());
                                    }
                                } else if (vocabAssetPath == null && isVocabFileName(fileName)) {
                                    vocabAssetPath = assetPath;
                                    vocabCacheName = fileName;
                                }
                            }
                        }
                    }
                    loadedModelAssetBytes = modelAssetBytes;
                    loadedModelDisplayName = vocabCacheName == null
                            ? "Qwen"
                            : vocabCacheName.substring("vocab_".length(),
                                    vocabCacheName.length() - ".txt".length());
                    modelAssetStageMs = elapsedMilliseconds(modelAssetStageStartNanos);
                    if (count_file == 0) {
                        error = loadContext.getString(R.string.model_load_error_missing_bundle);
                    } else {
                        File externalFilesDirectory = loadContext.getExternalFilesDir(null);
                        File storageDirectory = externalFilesDirectory != null &&
                                externalFilesDirectory.getParentFile() != null
                                ? externalFilesDirectory.getParentFile() : cacheDirectory;
                        publishModelLoadingStage(R.string.model_loading_runtime);
                        final long nativeOrtLoadStartNanos = SystemClock.elapsedRealtimeNanos();
                        final boolean modelsLoaded = Load_Models_A(
                            bgMgr, modelAssetDir, cacheDirectory.getAbsolutePath(),
                            storageDirectory.getAbsolutePath(), EP_CPU, false);
                        nativeOrtLoadMs = elapsedMilliseconds(nativeOrtLoadStartNanos);
                        if (!modelsLoaded) {
                            visionModelsReady = false;
                            error = loadContext.getString(R.string.model_load_error_runtime);
                        } else {
                        visionModelsReady = Vision_Models_Ready();
                        // Stage whichever vocab_*.txt asset is present (its name need not match
                        // VOCAB_FILE_NAME); native Pre_Process then auto-discovers it in the cache dir.
                        final long vocabAssetStageStartNanos = SystemClock.elapsedRealtimeNanos();
                        publishModelLoadingStage(R.string.model_loading_tokenizer);
                        if (!Copy_from_Asset_to_Cache(
                            vocabAssetPath != null ? vocabAssetPath : VOCAB_FILE_NAME,
                            vocabCacheName != null ? vocabCacheName : VOCAB_FILE_NAME,
                                cacheDirectory, bgMgr, copyBuffer, restageVocab,
                                assetManifest.hashes.get(
                                        vocabAssetPath != null ? vocabAssetPath : VOCAB_FILE_NAME))) {
                            throw new IOException("Failed to stage tokenizer vocab");
                        }
                        vocabAssetStageMs = elapsedMilliseconds(vocabAssetStageStartNanos);
                        // Pre_Process builds the tokenizer from the staged vocab. A false return (e.g. no
                        // vocab_*.txt asset at all) would leave the native tokenizer null, so every Run_LLM
                        // returns "" and the app looks ready but stays mute. Surface it as a load failure
                        // instead of silently marking the models ready.
                        final long tokenizerSetupStartNanos = SystemClock.elapsedRealtimeNanos();
                        ok = Pre_Process();
                        tokenizerSetupMs = elapsedMilliseconds(tokenizerSetupStartNanos);
                        if (ok) {
                            stagePrefs.edit().putString(App.KEY_ASSET_BUNDLE_FINGERPRINT,
                                    assetManifest.fingerprint).apply();
                            // Continuous startup: chain the vision graphs onto the SAME load sequence so the
                            // header never shows a "ready" gap between the tokenizer and the vision model.
                            // Best-effort -- a text-only model or a vision failure just leaves the camera off.
                            if (!keepLoadingServiceForBenchmark && CameraService.getVisionCaps() != 0) {
                                publishModelLoadingStage(R.string.model_loading_vision);
                                boolean visionLoaded = false;
                                try {
                                    visionLoaded = Ensure_Vision_Models() && Vision_Models_Ready();
                                } catch (Throwable visionError) {
                                    Log.e(BENCHMARK_TAG, "Vision model load failed during startup", visionError);
                                }
                                visionModelsReady = visionLoaded;
                                if (visionLoaded) {
                                    CameraService.preloadCameraSelection(loadContext);
                                }
                            }
                        } else {
                            error = loadContext.getString(R.string.model_load_error_tokenizer);
                        }
                        }
                    }
                } catch (OutOfMemoryError outOfMemory) {
                    Log.e(BENCHMARK_TAG, "Model loading ran out of memory", outOfMemory);
                    error = loadContext.getString(R.string.model_load_error_memory);
                } catch (IOException | SecurityException assetError) {
                    Log.e(BENCHMARK_TAG, "Model asset validation/staging failed", assetError);
                    error = loadContext.getString(R.string.model_load_error_assets,
                            assetError.getMessage() == null
                                    ? assetError.getClass().getSimpleName() : assetError.getMessage());
                } catch (Throwable unexpected) {
                    Log.e(BENCHMARK_TAG, "Unexpected model loading failure", unexpected);
                    error = loadContext.getString(R.string.model_load_error_unknown,
                            unexpected.getClass().getSimpleName());
                }
                modelReadyMs = elapsedMilliseconds(modelReadyStartNanos);
                final boolean loaded = ok;
                if (!loaded || !keepLoadingServiceForBenchmark) {
                    GenerationService.stop(loadContext);
                }
                final String err = error;
                mainHandler.post(() -> {
                    modelsLoading = false;
                    if (loaded) {
                        modelsReady = true;
                        modelLoadFailed = false;
                        modelLoadErrorText = null;
                    } else {
                        modelLoadFailed = true;
                        modelLoadErrorText = err;
                    }
                    MainActivity activity = currentActivity();
                    if (activity == null) {
                        if (keepLoadingServiceForBenchmark) {
                            GenerationService.stop(loadContext);
                        }
                        return;
                    }
                    if (loaded) {
                        activity.setupDecodeControls();
                        activity.initializeModelReadyUi();
                    } else {
                        activity.addHistory(ChatMessage.TYPE_SERVER, err);
                        activity.reportIntentBenchmarkLoadFailure(activity.getIntent(), err);
                    }
                    activity.updateModelLoadingUi();
                });
            }, "model-load").start();
        }
        updateModelLoadingUi();
        resumePendingRollbackCompletion();
        resumePendingGenerationCompletion();
        schedulePendingStreamUiFrame();
        setGenerating(generatingActive);
        registerThermalListener();
        maybeStartRuntimeTrim();
    }

    private void initializeModelReadyUi() {
        if (modelReadyUiInitialized || !modelsReady) {
            return;
        }
        modelReadyUiInitialized = true;
        boolean thinkingSupported = Supports_Thinking();
        thinkButton.setChecked(false);
        thinkButton.setEnabled(thinkingSupported);
        thinkButton.setAlpha(thinkingSupported ? 1.0f : 0.45f);
        applySavedSystemPrompt();
        refreshMemoryControlsFromNative();
        Start_Chat();
        updateHeaderActionStates();
        updateModelInfo();
        refreshMemoryStatsAsync();
        maybeRunIntentBenchmark(getIntent());
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        if (modelsReady) {
            maybeRunIntentBenchmark(intent);
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        startAmbientAnimations();
        renderPostProcessingState();
        updateMemoryRemainingDisplay(false);
        startMonitoring();       // CPU/RAM are always shown while this screen is visible
    }

    @Override
    protected void onPause() {
        stopMonitoring();        // never sample while backgrounded
        stopAmbientAnimations();
        stopTransientAnimations();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        unregisterThermalListener();
        stopAmbientAnimations();
        stopTransientAnimations();
        monitorActive = false;
        if (monitorHandler != null) {
            monitorHandler.removeCallbacks(sampleTask);
            monitorHandler.removeCallbacks(resetAndSampleTask);
        }
        if (monitorThread != null) {
            monitorThread.quitSafely();
            monitorThread = null;
        }
        processStatsSampler.close();
        if (activeActivity.get() == this) {
            activeActivity = new WeakReference<>(null);
        }
        super.onDestroy();
    }

    private void stopTransientAnimations() {
        if (postProcessingPulse != null) {
            postProcessingPulse.cancel();
            postProcessingPulse = null;
        }
        if (memoryBarAnimator != null) {
            memoryBarAnimator.cancel();
            memoryBarAnimator = null;
        }
        if (memoryGlowAnimator != null) {
            memoryGlowAnimator.cancel();
            memoryGlowAnimator = null;
        }
    }

    @Override
    public void onTrimMemory(int level) {
        super.onTrimMemory(level);
        if (level >= TRIM_MEMORY_RUNNING_CRITICAL) {
            noteMemoryPressure(2);
            requestRuntimeTrim(true);
        } else if (level >= TRIM_MEMORY_RUNNING_LOW) {
            noteMemoryPressure(1);
            requestRuntimeTrim(false);
        } else if (level == TRIM_MEMORY_UI_HIDDEN) {
            requestRuntimeTrim(false);
        }
    }

    @Override
    public void onLowMemory() {
        super.onLowMemory();
        noteMemoryPressure(2);
        requestRuntimeTrim(true);
    }

    private void registerThermalListener() {
        if (thermalStatusListener != null) {
            return;
        }
        PowerManager powerManager = getSystemService(PowerManager.class);
        if (powerManager == null) {
            return;
        }
        thermalStatusListener = this::onThermalStatusChanged;
        powerManager.addThermalStatusListener(getMainExecutor(), thermalStatusListener);
        onThermalStatusChanged(powerManager.getCurrentThermalStatus());
    }

    private void unregisterThermalListener() {
        if (thermalStatusListener == null) {
            return;
        }
        PowerManager powerManager = getSystemService(PowerManager.class);
        if (powerManager != null) {
            powerManager.removeThermalStatusListener(thermalStatusListener);
        }
        thermalStatusListener = null;
    }

    private void onThermalStatusChanged(int status) {
        thermalPressureLevel = status >= PowerManager.THERMAL_STATUS_CRITICAL
                ? 2
                : (status >= PowerManager.THERMAL_STATUS_SEVERE ? 1 : 0);
        updateRuntimePressureLevel();
        if (thermalPressureLevel >= 2) {
            requestRuntimeTrim(true);
        } else if (thermalPressureLevel == 1) {
            requestRuntimeTrim(false);
        }
    }

    private static void noteMemoryPressure(int pressure) {
        memoryPressureLevel = Math.max(memoryPressureLevel, clamp(pressure, 0, 2));
        memoryPressureSerial++;
        updateRuntimePressureLevel();
    }

    private static void updateRuntimePressureLevel() {
        int pressure = Math.max(thermalPressureLevel, memoryPressureLevel);
        if (pressure != runtimePressureLevel) {
            runtimePressureLevel = pressure;
            Configure_Runtime_Pressure(runtimePressureLevel);
        }
    }

    private void requestRuntimeTrim(boolean releaseVision) {
        if (!modelsReady) {
            return;
        }
        runtimeTrimPending = true;
        runtimeTrimPendingVisionRelease |= releaseVision;
        maybeStartRuntimeTrim();
    }

    private void maybeStartRuntimeTrim() {
        if (!modelsReady || !runtimeTrimPending || runtimeTrimRunning || generatingActive ||
                postProcessingActive || maintenanceActive) {
            return;
        }
        runtimeTrimPending = false;
        if (runtimeTrimRunning || generatingActive || postProcessingActive || maintenanceActive) {
            return;
        }
        runtimeTrimRunning = true;
        final boolean releaseVisionNow = runtimeTrimPendingVisionRelease;
        final int pressureSerialAtStart = memoryPressureSerial;
        runtimeTrimPendingVisionRelease = false;
        setMaintenanceActive(true);
        Future<?> visionCancelTask = null;
        if (releaseVisionNow) {
            if (cameraPreview != null) {
                cameraPreview.pauseForResourcePressure();
            }
            if (pendingVisionCapture) {
                removePendingVisionBubble();
                pendingVisionCapture = false;
                pendingVisionIsVideo = false;
                pendingVisionTurnId = ChatMessage.NO_TURN;
                pendingVideoFrameCount = 0;
                pendingVideoSeconds = 0f;
                visionCaptureSerial++;
                visionPreprocessedTurnId = ChatMessage.NO_TURN;
                visionCancelTask = CameraService.cancelPendingVision(null);
            }
        }
        final Future<?> pendingVisionCancelTask = visionCancelTask;
        runBackground("runtime-trim", () -> {
            CameraService.awaitPendingVisionCancel(pendingVisionCancelTask);
            final boolean trimmed = Trim_Runtime(releaseVisionNow);
            mainHandler.post(() -> {
                runtimeTrimRunning = false;
                if (!trimmed) {
                    runtimeTrimPending = true;
                    runtimeTrimPendingVisionRelease |= releaseVisionNow;
                } else if (pressureSerialAtStart == memoryPressureSerial) {
                    memoryPressureLevel = 0;
                    updateRuntimePressureLevel();
                }
                visionModelsReady = Vision_Models_Ready();
                MainActivity activity = currentActivity();
                if (activity != null) {
                    activity.setMaintenanceActive(false);
                    activity.refreshMemoryStatsAsync();
                } else {
                    maintenanceActive = false;
                }
            });
        });
    }

    private static MainActivity currentActivity() {
        return activeActivity.get();
    }

    // Run a one-shot task on the shared background pool, tagging the worker with a debug name for the
    // duration (restored afterward since pool threads are reused). Replaces per-call `new Thread(...)`.
    private static void runBackground(String name, Runnable task) {
        BACKGROUND_EXECUTOR.execute(() -> {
            Thread current = Thread.currentThread();
            String previousName = current.getName();
            current.setName(name);
            try {
                task.run();
            } finally {
                current.setName(previousName);
            }
        });
    }

    // ==================== Ambient header animations ====================

    // Always-on "breathing" glow for the avatar aura + status dot; cancelled in onPause.
    private void startAmbientAnimations() {
        if (avatarPulse == null) {
            avatarPulse = buildPulse(avatarGlow, 0.40f, 0.95f, 0.90f, 1.12f, 2200L);
        }
    }

    private void stopAmbientAnimations() {
        if (avatarPulse != null) {
            avatarPulse.cancel();
            avatarPulse = null;
        }
    }

    private static ObjectAnimator buildPulse(View view, float alphaFrom, float alphaTo,
                                             float scaleFrom, float scaleTo, long duration) {
        if (view == null) {
            return null;
        }
        ObjectAnimator anim = ObjectAnimator.ofPropertyValuesHolder(view,
                PropertyValuesHolder.ofFloat(View.ALPHA, alphaFrom, alphaTo),
                PropertyValuesHolder.ofFloat(View.SCALE_X, scaleFrom, scaleTo),
                PropertyValuesHolder.ofFloat(View.SCALE_Y, scaleFrom, scaleTo));
        anim.setDuration(duration);
        anim.setRepeatCount(ValueAnimator.INFINITE);
        anim.setRepeatMode(ValueAnimator.REVERSE);
        anim.setInterpolator(new AccelerateDecelerateInterpolator());
        anim.start();
        return anim;
    }

    // ==================== Theme (light / dark) ====================

    // Wire the header sun/moon button to the Light/Dark switch (glyph shows the CURRENT mode). AppCompatDelegate
    // recreates the Activity to repaint, but the native model + KV cache + history survive (modelsReady).
    private void setupThemeToggle() {
        themeToggle = findViewById(R.id.theme_toggle);
        themeToggle.setImageResource(isNightMode() ? R.drawable.ic_light_mode : R.drawable.ic_dark_mode);
        themeToggle.setOnClickListener(v -> {
            if (guardPostProcessing()) {
                return;
            }
            toggleTheme();
        });
    }

    private boolean isNightMode() {
        return (getResources().getConfiguration().uiMode & Configuration.UI_MODE_NIGHT_MASK)
                == Configuration.UI_MODE_NIGHT_YES;
    }

    // Flip Light <-> Dark: persist the new mode (so App re-applies it on cold start), then hand it to
    // AppCompatDelegate, which recreates the Activity.
    private void toggleTheme() {
        int newMode = isNightMode()
                ? AppCompatDelegate.MODE_NIGHT_NO
                : AppCompatDelegate.MODE_NIGHT_YES;
        getSharedPreferences(App.PREFS, MODE_PRIVATE)
                .edit()
                .putInt(App.KEY_NIGHT_MODE, newMode)
                .apply();
        AppCompatDelegate.setDefaultNightMode(newMode);
    }

    // ==================== System monitor (CPU / RAM) ====================

    // Spins up the background sampling thread. _SC_CLK_TCK / _SC_PAGESIZE resolved once so the CPU%/RAM math
    // hard-codes no kernel constants.
    private void setupSystemMonitor() {
        monitorThread = new HandlerThread("sys-monitor");
        monitorThread.start();
        monitorHandler = new Handler(monitorThread.getLooper());
    }

    // Periodic sampler (monitor thread): one CPU + one RAM sample, push to the UI, then re-arm while resumed.
    private final Runnable sampleTask = new Runnable() {
        @Override
        public void run() {
            sampleSystemUsage();
            if (monitorActive && monitorHandler != null) {
                monitorHandler.postDelayed(this, MONITOR_INTERVAL_MS);
            }
        }
    };

    private final Runnable resetAndSampleTask = () -> {
        if (!monitorActive) {
            return;
        }
        processStatsSampler.resetCpuBaseline();
        sampleTask.run();
    };

    // (Re)start sampling idempotently without clearing unrelated work from the monitor looper.
    private void startMonitoring() {
        if (monitorHandler == null) {
            return;
        }
        monitorActive = true;
        monitorHandler.removeCallbacks(sampleTask);
        monitorHandler.removeCallbacks(resetAndSampleTask);
        monitorHandler.post(resetAndSampleTask);
    }

    private void stopMonitoring() {
        monitorActive = false;
        if (monitorHandler != null) {
            monitorHandler.removeCallbacks(sampleTask);
            monitorHandler.removeCallbacks(resetAndSampleTask);
        }
    }

    // One monitor tick: CPU% from the jiffies delta over wall-clock (norm by cores, clamped 0..100) + RAM MB.
    // First tick only sets the CPU baseline (shows "measuring"). lastCpu* are thread-confined.
    private void sampleSystemUsage() {
        ProcessStatsSampler.Sample sample = processStatsSampler.sample();
        final String cpuText;
        if (sample.hasCpuPercent()) {
            cpuText = getString(R.string.perf_cpu_value, (int) Math.round(sample.cpuPercent));
            MetricsHistory.get().addCpu(sample.timestampMs, (float) sample.cpuPercent);
        } else {
            cpuText = getString(R.string.perf_measuring);
        }
        final String memText = sample.rssBytes >= 0L
                ? getString(R.string.perf_mem_value, sample.rssBytes / (1024L * 1024L))
                : getString(R.string.perf_pending);
        if (sample.rssBytes >= 0L) {
            MetricsHistory.get().addMemory(
                    sample.timestampMs, sample.rssBytes / (1024.0f * 1024.0f));
        }
        mainHandler.post(() -> {
            if (monitorActive) {
                cpuValue.setText(cpuText);
                memValue.setText(memText);
            }
        });
    }

    // ==================== Performance panel (throughput) ====================

    // Make the four telemetry cells tap-to-chart: CPU/RAM -> system time-series, Prefill/Decode -> throughput,
    // each focused on the tapped metric.
    private void setupPerfChartCells() {
        View cpuCell = findViewById(R.id.perf_cpu_cell);
        View memCell = findViewById(R.id.perf_mem_cell);
        View prefillCell = findViewById(R.id.perf_prefill_cell);
        View decodeCell = findViewById(R.id.perf_decode_cell);
        if (cpuCell != null) {
            cpuCell.setOnClickListener(v ->
                    MetricChartActivity.launch(this, MetricChartActivity.MODE_SYSTEM, MetricChartView.SERIES_A));
        }
        if (memCell != null) {
            memCell.setOnClickListener(v ->
                    MetricChartActivity.launch(this, MetricChartActivity.MODE_SYSTEM, MetricChartView.SERIES_B));
        }
        if (prefillCell != null) {
            prefillCell.setOnClickListener(v ->
                    MetricChartActivity.launch(this, MetricChartActivity.MODE_THROUGHPUT, MetricChartView.SERIES_A));
        }
        if (decodeCell != null) {
            decodeCell.setOnClickListener(v ->
                    MetricChartActivity.launch(this, MetricChartActivity.MODE_THROUGHPUT, MetricChartView.SERIES_B));
        }
    }

    private void updatePerfStats(float prefillRate, float decodeRate, int remaining, int capacity,
                                 int prefillTokens, int decodeTokens, boolean finalSample) {
        if (perfPrefillValue != null && prefillRate >= 0f) {
            perfPrefillValue.setText(getString(
                    R.string.perf_token_rate, String.format(Locale.US, "%.2f", prefillRate)));
        }
        if (perfDecodeValue != null && decodeRate >= 0f) {
            perfDecodeValue.setText(getString(
                    R.string.perf_token_rate, String.format(Locale.US, "%.2f", decodeRate)));
        }
        if (capacity > 0) {
            applyMemoryStatsFromValues(capacity - remaining, remaining, capacity, false);
        }
        if (!finalSample && prefillRate > 0f && prefillTokens > 0) {
            MetricsHistory.get().addPrefillPoint(prefillTokens, prefillRate);
        }
        if (!finalSample && decodeRate > 0f && decodeTokens > 0) {
            MetricsHistory.get().addDecodePoint(decodeTokens, decodeRate);
        }
    }

    // Reset the perf panel to a placeholder; native fills Prefill on completion and Decode live until the final rate.
    private void resetPerfStats() {
        String pending = getString(R.string.perf_pending);
        if (perfPrefillValue != null) {
            perfPrefillValue.setText(pending);
        }
        if (perfDecodeValue != null) {
            perfDecodeValue.setText(pending);
        }
    }

    // ==================== Native streaming callbacks ====================

    // C++ -> Java streaming entry point (batched from the native decode loop). Marshalled to the main Looper;
    // the chatting gate drops batches arriving after Clear/cancel.
    private static void onTokenStream(String text) {
        final long tokenTimeMillis = System.currentTimeMillis();
        final long generationId = callbackGenerationId();
        final boolean benchmarkCapture;
        synchronized (BENCHMARK_CAPTURE_LOCK) {
            benchmarkCapture = benchmarkCaptureActive;
            if (benchmarkCapture && text != null) {
                benchmarkOutput.append(text);
            }
        }
        if (benchmarkCapture) {
            return;
        }
        if (text == null || text.isEmpty()) {
            Log.w(FLOW_TAG, "token_empty generation=" + generationId);
            return;
        }
        boolean scheduleFrame = false;
        int pendingChars;
        synchronized (STREAM_UI_LOCK) {
            if (pendingStreamGenerationId != generationId) {
                pendingStreamUiText.setLength(0);
                pendingStreamGenerationId = generationId;
            }
            pendingStreamUiText.append(text);
            pendingStreamEventTimeMillis = tokenTimeMillis;
            if (!streamUiFrameScheduled) {
                streamUiFrameScheduled = true;
                scheduleFrame = true;
            }
            pendingChars = pendingStreamUiText.length();
        }
        Log.i(FLOW_TAG, "token_callback generation=" + generationId + " chars=" + text.length() +
                " pendingChars=" + pendingChars + " scheduleFrame=" + scheduleFrame +
                " active=" + isActiveGeneration(generationId));
        if (scheduleFrame) {
            mainHandler.post(() -> Choreographer.getInstance()
                    .postFrameCallback(STREAM_UI_FRAME_CALLBACK));
        }
    }

    private static void flushPendingStreamUi() {
        final String text;
        final long generationId;
        final long eventTimeMillis;
        MainActivity activity = currentActivity();
        synchronized (STREAM_UI_LOCK) {
            streamUiFrameScheduled = false;
            if (pendingStreamUiText.length() == 0) {
                return;
            }
            generationId = pendingStreamGenerationId;
            if (!isActiveGeneration(generationId)) {
                Log.w(FLOW_TAG, "ui_flush_drop generation=" + generationId + " active=" +
                        activeGenerationId + " chatting=" + chatting + " chars=" +
                        pendingStreamUiText.length());
                pendingStreamUiText.setLength(0);
                return;
            }
            if (activity == null) {
                Log.w(FLOW_TAG, "ui_flush_defer generation=" + generationId + " reason=no_activity chars=" +
                        pendingStreamUiText.length());
                return;
            }
            text = pendingStreamUiText.toString();
            eventTimeMillis = pendingStreamEventTimeMillis;
            pendingStreamUiText.setLength(0);
        }
        if (isActiveGeneration(generationId)) {
            Log.i(FLOW_TAG, "ui_flush_apply generation=" + generationId + " chars=" + text.length());
            activity.removeLoadingBubble();
            activity.captureVisionTimingIfPending();
            activity.addStreamingServerText(text, eventTimeMillis);
        }
    }

    private static void schedulePendingStreamUiFrame() {
        boolean scheduleFrame = false;
        synchronized (STREAM_UI_LOCK) {
            if (pendingStreamUiText.length() > 0 && !streamUiFrameScheduled) {
                streamUiFrameScheduled = true;
                scheduleFrame = true;
            }
        }
        if (scheduleFrame) {
            mainHandler.post(() -> Choreographer.getInstance()
                    .postFrameCallback(STREAM_UI_FRAME_CALLBACK));
        }
    }

    private static void clearPendingStreamUi() {
        synchronized (STREAM_UI_LOCK) {
            pendingStreamUiText.setLength(0);
            pendingStreamGenerationId = -1L;
            pendingStreamEventTimeMillis = 0L;
            streamUiFrameScheduled = false;
        }
    }

    // C++ -> Java typed throughput stats. Negative rates mean that phase is absent from this update.
    private static void onPerfStats(float prefillRate, float decodeRate, int remaining, int capacity,
                                    int prefillTokens, int decodeTokens, boolean finalSample) {
        final long generationId = callbackGenerationId();
        if (finalSample) {
            Log.i(FLOW_TAG, "perf_final generation=" + generationId + " prefillRate=" + prefillRate +
                    " decodeRate=" + decodeRate + " prefillTokens=" + prefillTokens +
                    " decodeTokens=" + decodeTokens + " remaining=" + remaining +
                    " capacity=" + capacity);
        }
        final boolean benchmarkCapture;
        synchronized (BENCHMARK_CAPTURE_LOCK) {
            benchmarkCapture = benchmarkCaptureActive;
            if (benchmarkCapture && finalSample) {
                benchmarkPerfPayload = String.format(Locale.US, "%.2f|%.2f|%d|%d|%d|%d",
                        prefillRate, decodeRate, remaining, capacity, prefillTokens, decodeTokens);
            }
        }
        if (benchmarkCapture) {
            return;
        }
        mainHandler.post(() -> {
            MainActivity activity = currentActivity();
            if (!isActiveGeneration(generationId) || activity == null ||
                    activity.perfPrefillValue == null || activity.perfDecodeValue == null) {
                return;
            }
            activity.updatePerfStats(prefillRate, decodeRate, remaining, capacity,
                    prefillTokens, decodeTokens, finalSample);
        });
    }

    private static long callbackGenerationId() {
        Thread callbackThread = Thread.currentThread();
        return callbackThread instanceof LLMThread
                ? ((LLMThread) callbackThread).generationId
                : activeGenerationId;
    }

    private static boolean isActiveGeneration(long generationId) {
        return chatting && generationId == activeGenerationId;
    }

    private static void onPostProcessingState(boolean active) {
        postProcessingActive = active;
        mainHandler.post(() -> {
            MainActivity current = currentActivity();
            if (current != null) {
                current.renderPostProcessingState();
            }
        });
    }

    // ==================== Chat history / RecyclerView ====================

    private void addHistory(int messageType, String result) {
        addHistory(messageType, result, ChatMessage.NO_TURN);
    }

    private void addHistory(int messageType, String result, int turnId) {
        addHistory(messageType, result, turnId, true);
    }

    private void addStreamingServerText(String result, long tokenTimeMillis) {
        addHistory(ChatMessage.TYPE_SERVER, result, ChatMessage.NO_TURN, false, tokenTimeMillis);
    }

    private void addHistory(int messageType, String result, int turnId, boolean markdownReady) {
        addHistory(messageType, result, turnId, markdownReady, System.currentTimeMillis());
    }

    private void addHistory(int messageType, String result, int turnId, boolean markdownReady, long tokenTimeMillis) {
        long eventTimeMillis = tokenTimeMillis > 0L ? tokenTimeMillis : System.currentTimeMillis();
        int lastMessageIndex = messages.size() - 1;
        if (lastMessageIndex >= 0 && messages.get(lastMessageIndex).type() == messageType) {
            // Append to the current bubble and rebind ONLY that row, so streaming doesn't re-bind the whole list.
            ChatMessage previous = messages.get(lastMessageIndex);
            if (messageType == ChatMessage.TYPE_SERVER && !markdownReady) {
                previous.appendStreaming(result, eventTimeMillis);
                if (chatAdapter != null) {
                    chatAdapter.notifyStreamingAppend(lastMessageIndex, result, eventTimeMillis);
                }
            } else {
                boolean ready = previous.markdownReady() && markdownReady;
                messages.set(lastMessageIndex, previous.withStreamingUpdate(
                        previous.content() + result, ready, eventTimeMillis));
                if (chatAdapter != null) {
                    chatAdapter.notifyItemChanged(lastMessageIndex);
                }
            }
            scrollToBottom();
        } else {
                messages.add(new ChatMessage(messageType, result, turnId, markdownReady, null, null,
                    eventTimeMillis, eventTimeMillis));
            int newIndex = messages.size() - 1;
            if (chatAdapter != null) {
                chatAdapter.notifyItemInserted(newIndex);
            }
            scrollToBottom();
        }
        scheduleTranscriptPaging();
    }

    private void scheduleTranscriptPaging() {
        final int target = messages.size() - TRANSCRIPT_RESIDENT_MESSAGES - 1;
        if (target <= transcriptPagedThrough || !transcriptPagingRunning.compareAndSet(false, true)) {
            return;
        }
        final int start = Math.max(0, transcriptPagedThrough + 1);
        final List<ChatMessage> batch = new ArrayList<>(messages.subList(start, target + 1));
        final File cacheDirectory = getCacheDir();
        final long serial = transcriptPagingSerial.get();
        CHAT_MEDIA_EXECUTOR.execute(() -> {
            for (ChatMessage message : batch) {
                message.pageContentToDisk(cacheDirectory);
            }
            if (serial == transcriptPagingSerial.get()) {
                transcriptPagedThrough = Math.max(transcriptPagedThrough, target);
            }
            transcriptPagingRunning.set(false);
            mainHandler.post(() -> {
                MainActivity activity = currentActivity();
                if (activity != null && serial == transcriptPagingSerial.get()) {
                    activity.scheduleTranscriptPaging();
                }
            });
        });
    }

    private void finalizeStreamingMarkdown() {
        for (int i = messages.size() - 1; i >= 0; --i) {
            ChatMessage message = messages.get(i);
            if (message.type() == ChatMessage.TYPE_SERVER) {
                if (!message.markdownReady()) {
                    message.finishStreaming();
                    if (chatAdapter != null) {
                        chatAdapter.notifyItemChanged(i);
                    }
                }
                return;
            }
            if (message.type() == ChatMessage.TYPE_USER) {
                return;
            }
        }
    }

    // ==================== Auto-scroll ====================

    private void setupAutoScrollLock() {
        jumpToBottomButton = findViewById(R.id.jump_to_bottom);
        jumpToBottomButton.setOnClickListener(v -> {
            autoScrollLocked = false;
            scrollToBottomNow();
        });
        answerView.addOnScrollListener(new RecyclerView.OnScrollListener() {
            @Override
            public void onScrollStateChanged(RecyclerView recyclerView, int newState) {
                if (newState == RecyclerView.SCROLL_STATE_DRAGGING && recyclerView.canScrollVertically(1)) {
                    autoScrollLocked = true;
                    updateJumpToBottomVisibility();
                }
            }

            @Override
            public void onScrolled(RecyclerView recyclerView, int dx, int dy) {
                if (!recyclerView.canScrollVertically(1)) {
                    autoScrollLocked = false;
                } else if (dy < 0) {
                    autoScrollLocked = true;
                }
                updateJumpToBottomVisibility();
            }
        });
    }

    // Pin the view to the newest line unless the user scrolled up. INSTANT scroll (not smooth) because
    // streaming fires many updates per second.
    private void scrollToBottom() {
        if (answerView == null) {
            return;
        }
        int lastMessageIndex = messages.size() - 1;
        if (lastMessageIndex >= 0) {
            if (autoScrollLocked && answerView.canScrollVertically(1)) {
                updateJumpToBottomVisibility();
                return;
            }
            answerView.scrollToPosition(lastMessageIndex);
            autoScrollLocked = false;
            updateJumpToBottomVisibility();
        }
    }

    private void scrollToBottomNow() {
        if (answerView == null) {
            return;
        }
        int lastMessageIndex = messages.size() - 1;
        if (lastMessageIndex >= 0) {
            answerView.scrollToPosition(lastMessageIndex);
        }
        updateJumpToBottomVisibility();
    }

    private void updateJumpToBottomVisibility() {
        if (jumpToBottomButton != null) {
            jumpToBottomButton.setVisibility(autoScrollLocked && !messages.isEmpty() ? View.VISIBLE : View.GONE);
        }
    }

    // ==================== Loading placeholder bubble ====================

    // Append the "typing" placeholder row (beam decodes silently). Idempotent: never stacked on another placeholder.
    private void showLoadingBubble() {
        int lastIndex = messages.size() - 1;
        if (lastIndex >= 0 && messages.get(lastIndex).type() == ChatMessage.TYPE_LOADING) {
            return;
        }
        messages.add(new ChatMessage(ChatMessage.TYPE_LOADING, ""));
        if (chatAdapter != null) {
            chatAdapter.notifyItemInserted(messages.size() - 1);
        }
        scrollToBottom();
    }

    // Remove the trailing "typing" placeholder once real content is ready. No-op if the last row isn't one.
    private void removeLoadingBubble() {
        int lastIndex = messages.size() - 1;
        if (lastIndex >= 0 && messages.get(lastIndex).type() == ChatMessage.TYPE_LOADING) {
            messages.remove(lastIndex);
            if (chatAdapter != null) {
                chatAdapter.notifyItemRemoved(lastIndex);
            }
        }
    }

    // ==================== Control locking (post-processing / busy) ====================

    private void setPostProcessingActive(boolean active) {
        postProcessingActive = active;
        renderPostProcessingState();
    }

    private void renderPostProcessingState() {
        boolean active = postProcessingActive;
        // Grey + tap-guard the whole top bar while native organises memory; Clear stays live (panic escape).
        updateModelLoadingUi();
        if (!active) {
            maybeStartRuntimeTrim();
        }
        if (postProcessingStatus == null) {
            return;
        }
        if (active) {
            postProcessingStatus.setVisibility(View.VISIBLE);
            if (postProcessingPulse == null) {
                postProcessingPulse = buildPulse(postProcessingStatus, 0.42f, 1.0f, 0.94f, 1.07f, 860L);
            }
            return;
        }
        if (postProcessingPulse != null) {
            postProcessingPulse.cancel();
            postProcessingPulse = null;
        }
        postProcessingStatus.setAlpha(1.0f);
        postProcessingStatus.setScaleX(1.0f);
        postProcessingStatus.setScaleY(1.0f);
        postProcessingStatus.setVisibility(View.GONE);
    }

    // Keep model availability visible in the composer instead of relying on a transient Toast. Send and Clear
    // remain disabled and grey until both startup and the latest strategy-session swap have completed.
    private void updateModelLoadingUi() {
        updateHeaderActionStates();
        if (decodeStrategyGroup != null) {
            setDecodeControlsEnabled(modelsReady && !modelsLoading && !generatingActive &&
                    !decodeConfigLoading && !systemPromptConfigLoading && !maintenanceActive);
        }
        boolean modelAvailable = isModelAvailable();
        boolean retryAvailable = modelLoadFailed && !modelsLoading;
        if (clearButton != null) {
            clearButton.setEnabled(modelAvailable);
            clearButton.setAlpha(modelAvailable ? 1.0f : 0.45f);
        }
        if (sendButton == null) {
            return;
        }
        sendButton.setEnabled(modelAvailable || retryAvailable);
        sendButton.setAlpha(((modelAvailable && !postProcessingActive) || retryAvailable) ? 1.0f : 0.45f);
        if (modelsLoading) {
            sendButton.setText(modelLoadingStageRes);
        } else if (decodeConfigLoading) {
            sendButton.setText(R.string.decode_loading_action);
        } else if (systemPromptConfigLoading) {
            sendButton.setText(R.string.system_prompt_applying_action);
        } else if (maintenanceActive) {
            sendButton.setText(R.string.maintenance_loading_action);
        } else if (retryAvailable) {
            sendButton.setText(R.string.model_retry_action);
            sendButton.setOnClickListener(v -> retryModelLoading());
        } else if (!modelAvailable) {
            sendButton.setText(R.string.model_unavailable_action);
        } else {
            sendButton.setText(R.string.send);
        }
    }

    private void retryModelLoading() {
        if (!modelLoadFailed || modelsLoading) {
            return;
        }
        if (modelLoadErrorText != null) {
            for (int index = messages.size() - 1; index >= 0; --index) {
                ChatMessage message = messages.get(index);
                if (message.type() == ChatMessage.TYPE_SERVER &&
                        modelLoadErrorText.equals(message.content())) {
                    messages.remove(index);
                    if (chatAdapter != null) {
                        chatAdapter.notifyItemRemoved(index);
                    }
                    break;
                }
            }
        }
        modelLoadFailed = false;
        modelLoadErrorText = null;
        recreate();
    }

    private static void publishModelLoadingStage(int stageRes) {
        modelLoadingStageRes = stageRes;
        notifyModelLoadingStateChanged();
    }

    private static boolean isModelAvailable() {
        return modelsReady && decodeConfigReady && !modelsLoading && !decodeConfigLoading &&
            systemPromptConfigReady && !systemPromptConfigLoading && !maintenanceActive;
    }

    private static void notifyModelLoadingStateChanged() {
        mainHandler.post(() -> {
            MainActivity activity = currentActivity();
            if (activity != null) {
                activity.updateModelLoadingUi();
            }
        });
    }

    // Single authority for ALL five top-bar icons (gear, system-prompt, theme, compute, camera). They stay
    // greyed + tap-guarded from cold start through the WHOLE model load -- text, tokenizer AND the inline
    // vision graphs, since vision now loads before modelsReady flips -- and during any decode-strategy swap,
    // system-prompt apply, maintenance, active generation or memory organisation. Camera additionally needs a
    // vision-capable model. Every widget is null-checked (icons resolve at different points in onCreate).
    private void updateHeaderActionStates() {
        boolean idle = isModelAvailable() && !visionModelsLoading && !generatingActive &&
                !postProcessingActive;
        applyHeaderIconState(decodeSettingsButton, idle, 0.45f);
        applyHeaderIconState(systemPromptButton, idle, 0.35f);
        applyHeaderIconState(themeToggle, idle, 0.45f);
        applyHeaderIconState(computeButton, idle, 0.45f);
        applyHeaderIconState(cameraButton, idle && CameraService.getVisionCaps() != 0, 0.35f);
    }

    private static void applyHeaderIconState(View icon, boolean enabled, float disabledAlpha) {
        if (icon != null) {
            icon.setEnabled(enabled);
            icon.setAlpha(enabled ? 1.0f : disabledAlpha);
        }
    }

    // Swallow a tap during memory organisation, surfacing a hint. Returns true when the tap should be ignored.
    private boolean guardPostProcessing() {
        if (postProcessingActive) {
            showToast(this, getString(R.string.post_processing_lock_hint), false);
            return true;
        }
        return false;
    }

    // Swallow a tap on the test button while generating, surfacing a hint. Returns true when the tap is ignored.
    private boolean guardInferenceBusy() {
        if (generatingActive || decodeConfigLoading || maintenanceActive) {
            showToast(this, getString(R.string.inference_lock_hint), false);
            return true;
        }
        return false;
    }

    private void setMaintenanceActive(boolean active) {
        maintenanceActive = active;
        // updateModelLoadingUi() already re-renders every header icon via updateHeaderActionStates().
        updateModelLoadingUi();
        if (!active) {
            maybeStartRuntimeTrim();
        }
    }

    private void cancelPendingVisionWithMaintenance() {
        setMaintenanceActive(true);
        CameraService.cancelPendingVision(() -> {
            MainActivity activity = currentActivity();
            if (activity != null) {
                activity.setMaintenanceActive(false);
            } else {
                maintenanceActive = false;
            }
        });
    }

    private static void recycleBitmap(Bitmap bitmap) {
        if (bitmap != null && !bitmap.isRecycled()) {
            bitmap.recycle();
        }
    }

    private static void recycleBitmapList(List<Bitmap> frames) {
        if (frames == null) {
            return;
        }
        for (Bitmap frame : frames) {
            recycleBitmap(frame);
        }
    }

    // ==================== Clear / reset ====================

    @SuppressLint("NotifyDataSetChanged")
    private void clearHistory(){
        if (!isModelAvailable()) {
            return;
        }
        // PANIC RESET: forcibly stop the in-flight reply AND any 整理记忆 commit, then reinitialise. UI reset
        // is synchronous; the native teardown runs on a bg thread (it can briefly block on a running ORT op).
        requestStopGeneration(false);    // cancel the decode loop (the worker bails within a token or two)
        invalidateGenerationCallbacks();
        pendingRollbackCompletion = null;
        setMaintenanceActive(true);
        setGenerating(false);            // restore Send; the partial reply (if any) is abandoned
        setPostProcessingActive(false);  // drop the "整理记忆" chip + un-grey the locked controls immediately
        chatting = false;                // gate out the cancelled worker's remaining streamed-token UI posts
        // Drop any queued-but-unsent vision capture (and its background pre-encode) so it can't leak into
        // the fresh conversation as a stale vision turn.
        pendingVisionCapture = false;
        final Future<?> visionCancelTask = CameraService.cancelPendingVision(null);
        visionTimingPending = false;
        visionTimingTurnId = ChatMessage.NO_TURN;
        visionPreprocessedTurnId = ChatMessage.NO_TURN;
        visionCaptureSerial++;
        inputBox.setText("");
        usrInputText = "";
        if (chatAdapter != null) {
            chatAdapter.releaseVisibleVisionMediaFrom(answerView, 0);
            chatAdapter.clearMediaCache();
        }
        // Snapshot the resident messages and delete all their media in ONE background task; then report the
        // removal as a range so the adapter avoids a full-transcript invalidation.
        final List<ChatMessage> clearedMessages = new ArrayList<>(messages);
        deleteMessagesMediaAsync(clearedMessages);
        messages.clear();
        transcriptPagingSerial.incrementAndGet();
        transcriptPagedThrough = -1;
        autoScrollLocked = false;
        updateJumpToBottomVisibility();
        if (!clearedMessages.isEmpty()) {
            chatAdapter.notifyItemRangeRemoved(0, clearedMessages.size());
        }
        chatAdapter.resetEntranceAnimation();
        chatAdapter.clearThinkExpandState();
        resetPerfStats();
        // Throughput stats are intentionally NOT wiped here -- they persist, cleared only from the chart screen.
        memoryStatsRequestId++;   // invalidate any in-flight async memory-stat read
        memoryUsedTokens = 0;
        updateMemoryRemainingFromKnownUsage();
        // Backstop: the NEXT Run_LLM must still reset the native context if the eager reset below is skipped
        // (a new turn grabbed the inference lock first).
        clear_flag = true;
        // Reset the native runtime OFF the UI thread: join the cancelled worker, then Clear_Cache aborts any
        // running 整理记忆 commit + drops the KV cache. Refresh the memory readout once native is empty.
        final LLMThread workerToReap = llmThread;
        llmThread = null;
        runBackground("clear-reset", () -> {
            if (workerToReap != null) {
                try {
                    workerToReap.join();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            CameraService.awaitPendingVisionCancel(visionCancelTask);
            Clear_Cache();
            mainHandler.post(() -> {
                maintenanceActive = false;
                MainActivity activity = currentActivity();
                if (activity != null) {
                    activity.updateModelLoadingUi();
                    activity.refreshMemoryStatsAsync();
                }
            });
        });
        showToast(MainActivity.this, getString(R.string.history_cleared), false);
    }

    // ==================== Generation orchestration ====================

    private void Start_Chat() {
        sendButton.setOnClickListener(view -> {
            if (guardPostProcessing() || guardInferenceBusy()) {
                return;
            }
            usrInputText = String.valueOf(inputBox.getText());
            if (pendingVisionCapture) {
                // Reuse the capture turn ID so media, query and reply roll back together.
                String query = usrInputText;
                inputBox.setText("");
                dispatchVisionSend(query, pendingVisionTurnId, pendingVisionIsVideo);
            } else if (usrInputText.isEmpty()){
                showToast(MainActivity.this, getString(R.string.empty_input_prompt), false);
            } else {
                String query = usrInputText;
                inputBox.setText("");
                dispatchGeneration(query, nextTurnId++);
            }
        });
    }

    // Shared send path (fresh message + edited-bubble resend): snapshot query + turn id, launch the worker,
    // append the user bubble, show the typing placeholder for beam.
    private void dispatchGeneration(String query, int turnId) {
        usrInputText = query;
        usrTurnId = turnId;
        autoScrollLocked = false;
        updateJumpToBottomVisibility();
        startLLM();
        addHistory(ChatMessage.TYPE_USER, query, turnId);
        if (useBeamSearch) {
            // Beam streams nothing until done; show a placeholder so the wait isn't a frozen screen.
            showLoadingBubble();
        }
    }

    private void startLLM() {
        // Cancel + reap any in-flight generation so native state is free (else the re-entrancy guard drops
        // this query) and stale streamed posts can't leak in.
        cancelOngoingGeneration();
        final long generationId = generationSerial.incrementAndGet();
        activeGenerationId = generationId;
        chatting = true;
        stopRequested = false;
        Log.i(FLOW_TAG, "start generation=" + generationId + " turn=" + usrTurnId +
            " clear=" + clear_flag + " think=" + thinkButton.isChecked() +
            " queryChars=" + (usrInputText == null ? -1 : usrInputText.length()));
        setGenerating(true);           // swap Send -> Stop so the user can halt this reply
        resetPerfStats();              // show the "measuring" placeholder until this turn's numbers land
        llmThread = new LLMThread(usrInputText, thinkButton.isChecked(), clear_flag,
            usrTurnId, generationId, getApplicationContext());
        clear_flag = false;            // the pending clear (if any) is now owned by this generation
        llmThread.start();
    }

    // Swap the composer button between Send (idle) and Stop (generating), so a long reply can be halted
    // without Clearing the whole chat.
    private void setGenerating(boolean generating) {
        generatingActive = generating;
        if (!generating) {
            stopRequested = false;
        }
        if (generating) {
            GenerationService.startGeneration(this);
        } else if (!modelsLoading && !visionModelsLoading) {
            GenerationService.stop(this);
        }
        if (sendButton != null) {
            sendButton.setVisibility(generating ? View.GONE : View.VISIBLE);
        }
        if (stopButton != null) {
            stopButton.setVisibility(generating ? View.VISIBLE : View.GONE);
            stopButton.setEnabled(!stopRequested);
            stopButton.setText(stopRequested ? R.string.stopping : R.string.stop);
        }
        updateModelLoadingUi();
        setDecodeControlsEnabled(!generating);
        if (!generating) {
            maybeStartRuntimeTrim();
        }
    }

    private void requestStopGeneration(boolean manual) {
        LLMThread worker = llmThread;
        Log.i(FLOW_TAG, "stop_request manual=" + manual + " worker=" +
            (worker == null ? "null" : worker.generationId) + " forward=" +
            (worker != null && worker.shouldForwardStop()) + " alreadyRequested=" + stopRequested);
        if (worker == null || !worker.shouldForwardStop() || (manual && stopRequested)) {
            return;
        }
        if (manual) {
            stopRequested = true;
            if (stopButton != null) {
                stopButton.setEnabled(false);
                stopButton.setText(R.string.stopping);
            }
        }
        Stop_LLM(manual);
    }

    // Stop + join the active worker and drop its queued UI posts, leaving native idle so the caller can
    // start a new generation or mutate the KV cache (e.g. edit rollback).
    private void cancelOngoingGeneration() {
        LLMThread worker = llmThread;
        Log.i(FLOW_TAG, "cancel_previous worker=" + (worker == null ? "null" : worker.generationId) +
            " alive=" + (worker != null && worker.isAlive()) + " nativeFinished=" +
            (worker != null && worker.nativeFinished));
        invalidateGenerationCallbacks();
        if (worker != null && worker.shouldForwardStop()) {
            Stop_LLM(false);
        }
        if (worker != null) {
            try {
                worker.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        if (llmThread == worker) {
            llmThread = null;
        }
    }

    private static void invalidateGenerationCallbacks() {
        chatting = false;
        activeGenerationId = generationSerial.incrementAndGet();
        pendingGenerationCompletion = null;
        clearPendingStreamUi();
    }

    private static final class PendingGenerationCompletion {
        final long generationId;
        final String result;

        PendingGenerationCompletion(long generationId, String result) {
            this.generationId = generationId;
            this.result = result;
        }
    }

    private static final class PendingRollbackCompletion {
        final String editedText;
        final int newTurnId;
        final boolean rolledBack;
        final int[] memoryStats;

        PendingRollbackCompletion(String editedText, int newTurnId, boolean rolledBack,
                                  int[] memoryStats) {
            this.editedText = editedText;
            this.newTurnId = newTurnId;
            this.rolledBack = rolledBack;
            this.memoryStats = memoryStats;
        }
    }

    private static final class LLMThread extends Thread {
        // Snapshot per-generation inputs on the UI thread at construction so the worker never races the UI.
        private final String query;
        private final boolean thinkMode;
        private final boolean clear;
        private final int turnId;
        private final long generationId;
        private final Context appContext;
        private volatile boolean nativeFinished = false;

        LLMThread(String query, boolean thinkMode, boolean clear, int turnId, long generationId,
                  Context appContext) {
            super("llm-generation-" + generationId);
            this.query = query;
            this.thinkMode = thinkMode;
            this.clear = clear;
            this.turnId = turnId;
            this.generationId = generationId;
            this.appContext = appContext;
        }

        boolean shouldForwardStop() {
            return isAlive() && !nativeFinished;
        }

        @Override
        public void run() {
            // ONE native call runs prefill + the whole decode loop, streaming via onTokenStream(). Returns
            // a structured status/error or a "<prefill>|<decode>" tok/s payload.
            String runResult = "";
                final long startedMs = SystemClock.elapsedRealtime();
            try {
                boolean configReady = awaitModelConfigBarrier();
                Log.i(FLOW_TAG, "native_call generation=" + generationId + " turn=" + turnId +
                    " configReady=" + configReady + " clear=" + clear + " think=" + thinkMode);
                runResult = configReady
                    ? Run_LLM(query, clear, thinkMode, turnId)
                    : RUN_ERROR_PREFIX + "NOT_READY";
                } catch (Throwable error) {
                Log.e(FLOW_TAG, "native_exception generation=" + generationId + " turn=" + turnId,
                    error);
                runResult = RUN_ERROR_PREFIX + "NATIVE_EXCEPTION";
            } finally {
                nativeFinished = true;
                GenerationService.stop(appContext);
            }
                Log.i(FLOW_TAG, "native_return generation=" + generationId + " turn=" + turnId +
                    " elapsedMs=" + (SystemClock.elapsedRealtime() - startedMs) +
                    " result=" + runResult);
            final String result = runResult;
            final LLMThread finishedThread = this;
            mainHandler.post(() -> {
                // A theme switch mid-generation recreates the Activity, so target the CURRENT active one
                // (weak ref), not the destroyed launcher.
                MainActivity activity = currentActivity();
                if (isActiveGeneration(generationId)) {
                    if (activity != null) {
                        activity.completeGenerationOnMain(generationId, result);
                    } else {
                        // Keep the generation gate open so its buffered tail remains valid. onCreate()
                        // consumes this completion after the replacement Activity has rebound its views.
                        pendingGenerationCompletion =
                                new PendingGenerationCompletion(generationId, result);
                    }
                }
                if (llmThread == finishedThread) {
                    llmThread = null;
                }
            });
        }
    }

    private void resumePendingGenerationCompletion() {
        PendingGenerationCompletion completion = pendingGenerationCompletion;
        if (completion == null) {
            return;
        }
        if (!isActiveGeneration(completion.generationId)) {
            pendingGenerationCompletion = null;
            return;
        }
        completeGenerationOnMain(completion.generationId, completion.result);
    }

    private void resumePendingRollbackCompletion() {
        PendingRollbackCompletion completion = pendingRollbackCompletion;
        if (completion != null) {
            completeRollbackOnMain(completion);
        }
    }

    private void completeRollbackOnMain(PendingRollbackCompletion completion) {
        pendingRollbackCompletion = null;
        applyMemoryStats(completion.memoryStats);
        if (!completion.rolledBack) {
            clear_flag = true;
            showToast(this, getString(R.string.rollback_context_reset), true);
        }
        usrInputText = completion.editedText;
        usrTurnId = completion.newTurnId;
        setMaintenanceActive(false);
        startLLM();
    }

    private void completeGenerationOnMain(long generationId, String result) {
        if (!isActiveGeneration(generationId)) {
            Log.w(FLOW_TAG, "complete_drop generation=" + generationId + " active=" +
                    activeGenerationId + " chatting=" + chatting + " result=" + result);
            return;
        }
        Log.i(FLOW_TAG, "complete_main generation=" + generationId + " result=" + result);
        // The worker can finish before the next VSync; commit its coalesced tail before closing the
        // generation gate so short replies and manual stops never lose final text.
        flushPendingStreamUi();
        pendingGenerationCompletion = null;
        removeLoadingBubble();
        if (result != null && result.startsWith(RUN_ERROR_PREFIX)) {
            String code = result.substring(RUN_ERROR_PREFIX.length());
            addHistory(ChatMessage.TYPE_SERVER, generationErrorMessage(code));
        } else if (RUN_STATUS_CANCELLED.equals(result)) {
            // A cancelled generation intentionally adds no error bubble.
        } else if (result != null && result.indexOf('|') >= 0) {
            // Perf stats for a completed turn arrive via the typed onPerfStats(finalSample=true) callback,
            // which is posted before this completion runnable; here we only finalize the streamed markdown.
            finalizeStreamingMarkdown();
        }
        chatting = false;
        setGenerating(false);
        MetricsHistory.get().flushThroughput();
        captureVisionTimingIfPending();
        refreshMemoryStatsAsync();
    }

    private String generationErrorMessage(String code) {
        switch (code) {
            case "INPUT_TOO_LONG":
                return getString(R.string.input_too_long);
            case "BUSY":
                return getString(R.string.generation_error_busy);
            case "NOT_READY":
            case "RUNTIME_UNAVAILABLE":
            case "STRATEGY_UNAVAILABLE":
                return getString(R.string.generation_error_not_ready);
            case "VISION_ENCODE":
                return getString(R.string.generation_error_vision);
            case "ALLOCATION":
                return getString(R.string.generation_error_memory);
            case "INVALID_INPUT":
            case "INPUT_DECODE":
                return getString(R.string.generation_error_input);
            case "MODEL_CONTRACT":
                return getString(R.string.generation_error_model);
            case "PREFILL_STATE":
            case "PREFILL_RUN":
            case "PREFILL_OUTPUT":
                clear_flag = true;
                return getString(R.string.generation_error_retry_reset);
            default:
                return getString(R.string.generation_error_generic);
        }
    }

    // ==================== Edit & resend ====================

    // User bubble tapped: open an editor seeded with its text. "Resend" rolls back this turn + regenerates.
    private void onEditUserMessage(int position) {
        if (position < 0 || position >= messages.size()) {
            return;
        }
        ChatMessage message = messages.get(position);
        if (message.type() != ChatMessage.TYPE_USER) {
            return;
        }
        final EditText editText = new EditText(this);
        editText.setText(message.content());
        editText.setSelection(editText.getText().length());
        editText.setInputType(android.text.InputType.TYPE_CLASS_TEXT
                | android.text.InputType.TYPE_TEXT_FLAG_MULTI_LINE
                | android.text.InputType.TYPE_TEXT_FLAG_CAP_SENTENCES);
        editText.setMaxLines(6);
        int pad = Math.round(getResources().getDisplayMetrics().density * 20f);
        FrameLayout container = new FrameLayout(this);
        container.setPadding(pad, pad / 2, pad, 0);
        container.addView(editText);
        new AlertDialog.Builder(this)
                .setTitle(R.string.edit_message_title)
                .setView(container)
                .setPositiveButton(R.string.edit_message_resend, (dialog, which) -> {
                    String edited = editText.getText().toString();
                    if (!edited.trim().isEmpty()) {
                        resendEditedMessage(position, edited);
                    }
                })
                .setNegativeButton(R.string.edit_message_cancel, null)
                .show();
    }

    // Drop this turn + everything after it, show the edited bubble + placeholder immediately, then cancel the
    // in-flight generation + roll the KV cache back on a BACKGROUND thread (may re-prefill) so the UI never
    // freezes; launch generation once rollback finishes, keeping earlier turns as context.
    private void resendEditedMessage(int position, String editedText) {
        if (position < 0 || position >= messages.size()) {
            return;
        }
        ChatMessage edited = messages.get(position);
        if (edited.type() != ChatMessage.TYPE_USER) {
            return;
        }
        final int turnId = edited.turnId();
        int removeFrom = position;
        if (position > 0) {
            ChatMessage previous = messages.get(position - 1);
            if (previous.turnId() == turnId &&
                    (previous.type() == ChatMessage.TYPE_USER_IMAGE ||
                     previous.type() == ChatMessage.TYPE_USER_VIDEO)) {
                removeFrom = position - 1;
            }
        }
        // An edit/resend is a text turn: drop any queued-but-unsent vision capture so the resend's Run_LLM
        // isn't misread as a vision turn (and its background pre-encode is released).
        setMaintenanceActive(true);
        Future<?> visionCancelTask = null;
        if (pendingVisionCapture) {
            pendingVisionCapture = false;
            visionCancelTask = CameraService.cancelPendingVision(null);
        }
        // UI first (instant feedback): gate stale posts, drop this turn + later, reseed, show edited bubble + placeholder.
        invalidateGenerationCallbacks();
        int removeCount = messages.size() - removeFrom;
        if (chatAdapter != null) {
            chatAdapter.releaseVisibleVisionMediaFrom(answerView, removeFrom);
            chatAdapter.clearMediaCache();
        }
        // Snapshot the removed tail, delete its media in ONE background task, then drop the range at once.
        final List<ChatMessage> removedMessages =
                new ArrayList<>(messages.subList(removeFrom, messages.size()));
        deleteMessagesMediaAsync(removedMessages);
        messages.subList(removeFrom, messages.size()).clear();
        transcriptPagingSerial.incrementAndGet();
        transcriptPagedThrough = Math.min(transcriptPagedThrough, removeFrom - 1);
        chatAdapter.notifyItemRangeRemoved(removeFrom, removeCount);
        chatAdapter.clearThinkExpandState();
        autoScrollLocked = false;
        updateJumpToBottomVisibility();
        inputBox.setText("");
        final int newTurnId = nextTurnId++;
        addHistory(ChatMessage.TYPE_USER, editedText, newTurnId);
        showLoadingBubble();   // placeholder until the rollback completes and the reply starts streaming
        // Stop+join the worker and roll back the KV cache OFF the UI thread (both can block), then launch the
        // new generation once the context is idle + rewound.
        final Future<?> pendingVisionCancelTask = visionCancelTask;
        runBackground("edit-rollback", () -> {
            cancelOngoingGeneration();   // Stop_LLM + join + drop stale posts: native context now free
            CameraService.awaitPendingVisionCancel(pendingVisionCancelTask);
            final boolean rolledBack = Rollback_LLM(turnId);
            final int[] memoryStats = Get_Memory_Stats();
            mainHandler.post(() -> {
                PendingRollbackCompletion completion = new PendingRollbackCompletion(
                        editedText, newTurnId, rolledBack, memoryStats);
                MainActivity activity = currentActivity();
                if (activity == null) {
                    pendingRollbackCompletion = completion;
                    return;
                }
                activity.completeRollbackOnMain(completion);
            });
        });
    }

    // ==================== System prompt editor ====================

    // Slot system-prompt editor (multiline AlertDialog). Save persists+applies (empty disables), Reset restores
    // the default. Takes effect NEXT turn (no retro-edit of cached history).
    private void onEditSystemPrompt() {
        if (!modelsReady || modelsLoading || systemPromptConfigLoading || generatingActive ||
            postProcessingActive || maintenanceActive) {
            return;   // tokenizer not ready yet (Pre_Process runs during the model load)
        }
        View content = getLayoutInflater().inflate(R.layout.dialog_system_prompt, null, false);
        TextInputEditText editText = content.findViewById(R.id.systemPromptInput);
        TextView counter = content.findViewById(R.id.systemPromptCounter);
        editText.setFilters(new InputFilter[]{new InputFilter.LengthFilter(MAX_SYSTEM_PROMPT_CHARS)});
        String current = getSharedPreferences(App.PREFS, MODE_PRIVATE)
                .getString(App.KEY_SYSTEM_PROMPT, SYSTEM_PROMPT_DEFAULT);
        editText.setText(current);
        editText.setSelection(editText.getText().length());
        updateSystemPromptCounter(counter, editText.getText());
        editText.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence s, int start, int count, int after) {
            }

            @Override
            public void onTextChanged(CharSequence s, int start, int before, int count) {
                updateSystemPromptCounter(counter, s);
            }

            @Override
            public void afterTextChanged(Editable s) {
            }
        });

        AlertDialog dialog = new AlertDialog.Builder(this)
                .setView(content)
                .create();
        content.findViewById(R.id.systemPromptClear).setOnClickListener(v -> editText.setText(""));
        content.findViewById(R.id.systemPromptReset).setOnClickListener(v -> {
            editText.setText(SYSTEM_PROMPT_DEFAULT);
            editText.setSelection(editText.getText().length());
        });
        content.findViewById(R.id.systemPromptCancel).setOnClickListener(v -> dialog.dismiss());
        content.findViewById(R.id.systemPromptSave).setOnClickListener(v -> {
            Editable text = editText.getText();
            saveSystemPrompt(text == null ? "" : text.toString());
            dialog.dismiss();
        });
        dialog.setOnShowListener(d -> {
            Window window = dialog.getWindow();
            if (window != null) {
                window.setBackgroundDrawable(new ColorDrawable(Color.TRANSPARENT));
                window.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE
                        | WindowManager.LayoutParams.SOFT_INPUT_STATE_VISIBLE);
                int width = Math.min(getResources().getDisplayMetrics().widthPixels - dp(24), dp(640));
                window.setLayout(width, ViewGroup.LayoutParams.WRAP_CONTENT);
            }
            editText.requestFocus();
        });
        dialog.show();
    }

    private void updateSystemPromptCounter(TextView counter, CharSequence text) {
        counter.setText(getString(R.string.system_prompt_counter,
                text == null ? 0 : text.length(), MAX_SYSTEM_PROMPT_CHARS));
    }

    // Persist immediately, then tokenize/apply on the serialized model-config executor. Send stays gated
    // until completion, and the success toast is emitted only after native accepted the prompt.
    private void saveSystemPrompt(String prompt) {
        getSharedPreferences(App.PREFS, MODE_PRIVATE).edit()
                .putString(App.KEY_SYSTEM_PROMPT, prompt).apply();
        if (prompt.equals(appliedSystemPrompt) && !systemPromptConfigLoading) {
            showToast(this, getString(R.string.system_prompt_saved), false);
            return;
        }
        enqueueSystemPromptConfig(prompt, true);
    }

    // Push a saved slot prompt to native after the tokenizer is ready (once after Pre_Process); else the
    // native default stands.
    private void applySavedSystemPrompt() {
        SharedPreferences prefs = getSharedPreferences(App.PREFS, MODE_PRIVATE);
        String prompt = prefs.contains(App.KEY_SYSTEM_PROMPT)
                ? prefs.getString(App.KEY_SYSTEM_PROMPT, SYSTEM_PROMPT_DEFAULT)
                : SYSTEM_PROMPT_DEFAULT;
        if (!prompt.equals(appliedSystemPrompt)) {
            enqueueSystemPromptConfig(prompt, false);
        }
    }

    private static void enqueueSystemPromptConfig(String prompt, boolean notifyUser) {
        final long serial = systemPromptConfigSerial.incrementAndGet();
        pendingSystemPromptConfigs.incrementAndGet();
        systemPromptConfigLoading = true;
        systemPromptConfigReady = false;
        notifyModelLoadingStateChanged();
        DECODE_CONFIG_EXECUTOR.execute(() -> {
            boolean applied = Set_System_Prompt(prompt);
            if (applied) {
                appliedSystemPrompt = prompt;
            }
            boolean latest = serial == systemPromptConfigSerial.get();
            if (latest) {
                systemPromptConfigReady = applied;
            }
            systemPromptConfigLoading = pendingSystemPromptConfigs.decrementAndGet() > 0;
            mainHandler.post(() -> {
                MainActivity activity = currentActivity();
                if (activity != null) {
                    activity.updateModelLoadingUi();
                    if (latest && notifyUser) {
                        showToast(activity, activity.getString(applied
                                ? R.string.system_prompt_saved
                                : R.string.system_prompt_save_failed), !applied);
                    }
                }
            });
        });
    }

    private static boolean awaitModelConfigBarrier() {
        CountDownLatch barrier = new CountDownLatch(1);
        DECODE_CONFIG_EXECUTOR.execute(barrier::countDown);
        try {
            barrier.await();
            return decodeConfigReady && systemPromptConfigReady;
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    // ==================== Decode-strategy controls ====================

    // Wire the required Greedy / Beam / Sampler choice. Values and guards mirror the Python inference path:
    // Beam >= 2, Beam Top-K >= Beam, sampling Top-K >= 1, and 1.0 disables either penalty formulation.
    private void setupDecodeControls() {
        if (decodeControlsInitialized) {
            return;
        }
        ViewStub panelStub = findViewById(R.id.decode_panel_stub);
        if (panelStub != null) {
            panelStub.inflate();
        }
        decodeControlsInitialized = true;
        decodeStrategyGroup = findViewById(R.id.decode_strategy_group);
        memorySlider = findViewById(R.id.memory_slider);
        prefillSlider = findViewById(R.id.prefill_slider);
        prefillControls = findViewById(R.id.prefill_controls);
        decodeLimitSlider = findViewById(R.id.decode_limit_slider);
        setCompactSliderRowHeights(memorySlider, prefillSlider, decodeLimitSlider);
        beamValue = findViewById(R.id.beam_value);
        topKValue = findViewById(R.id.top_k_value);
        penaltyValue = findViewById(R.id.penalty_value);
        penaltyRangeValue = findViewById(R.id.penalty_range_value);
        samplingTemperatureValue = findViewById(R.id.sampling_temperature_value);
        samplingTopKValue = findViewById(R.id.sampling_top_k_value);
        samplingTopPValue = findViewById(R.id.sampling_top_p_value);
        samplingPenaltyValue = findViewById(R.id.sampling_penalty_value);
        memoryValue = findViewById(R.id.memory_value);
        memoryRemainingValue = findViewById(R.id.memory_remaining_value);
        memoryRemainingBar = findViewById(R.id.memory_remaining_bar);
        memoryRemainingGlow = findViewById(R.id.memory_remaining_glow);
        if (memoryRemainingBar != null) {
            memoryRemainingBar.setPivotX(0f);
        }
        if (memoryRemainingGlow != null) {
            memoryRemainingGlow.setPivotX(0f);
        }
        memoryRemainingTrack = findViewById(R.id.memory_remaining_track);
        memoryThresholdGreen = findViewById(R.id.memory_threshold_green);
        memoryThresholdRed = findViewById(R.id.memory_threshold_red);
        memoryThresholdSlider = findViewById(R.id.memory_threshold_slider);
        memoryBandValue = findViewById(R.id.memory_band_value);
        prefillValue = findViewById(R.id.prefill_value);
        decodeLimitValue = findViewById(R.id.decode_limit_value);
        decodeContent = findViewById(R.id.decode_content);
        modelInfoValue = findViewById(R.id.model_info_value);
        decodeSettingsButton = findViewById(R.id.btn_decode_settings);
        MaterialSwitch organizeMemorySwitch = findViewById(R.id.organize_memory_switch);
        decodeSettingsButton.setOnClickListener(v -> toggleDecodePanel());
        updateHeaderActionStates();
        SharedPreferences prefs = getSharedPreferences(App.PREFS, MODE_PRIVATE);
        decodePreferences = readDecodePreferences(prefs);
        decodeMode = decodePreferences.mode;
        int checkedId = decodeMode == DECODE_MODE_BEAM
            ? R.id.decode_mode_beam
            : (decodeMode == DECODE_MODE_SAMPLING
            ? R.id.decode_mode_sampler : R.id.decode_mode_greedy);
        decodeStrategyGroup.check(checkedId);
        decodeStrategyGroup.addOnButtonCheckedListener((group, selectedId, isChecked) -> {
            if (!isChecked) {
                return;
            }
            if (selectedId == R.id.decode_mode_beam) {
                decodeMode = DECODE_MODE_BEAM;
            } else if (selectedId == R.id.decode_mode_sampler) {
                decodeMode = DECODE_MODE_SAMPLING;
            } else {
                decodeMode = DECODE_MODE_GREEDY;
            }
            useBeamSearch = decodeMode == DECODE_MODE_BEAM;
            ensureDecodeStrategyControlsInflated();
            updateDecodeModeVisibility();
            applyDecodeConfig(null);
        });
        loadMemoryLimits();
        restoreMemoryControls();
        Slider.OnChangeListener memoryListener = (slider, value, fromUser) -> {
            applyMemoryConfig(slider, false);
            if (fromUser && !slider.isPressed()) {
                applyMemoryConfig(null, true);   // keyboard/accessibility step
            }
        };
        Slider.OnSliderTouchListener memoryTouchListener = new Slider.OnSliderTouchListener() {
            @Override public void onStartTrackingTouch(@NonNull Slider slider) {
            }

            @Override public void onStopTrackingTouch(@NonNull Slider slider) {
                applyMemoryConfig(null, true);
            }
        };
        for (Slider slider : new Slider[]{memorySlider, prefillSlider, decodeLimitSlider}) {
            slider.addOnChangeListener(memoryListener);
            slider.addOnSliderTouchListener(memoryTouchListener);
        }
        boolean organizeMemoryEnabled = getSharedPreferences(App.PREFS, MODE_PRIVATE)
                .getBoolean(App.KEY_ORGANIZE_MEMORY, false);
        if (organizeMemorySwitch != null) {
            organizeMemorySwitch.setChecked(organizeMemoryEnabled);
            organizeMemorySwitch.setOnCheckedChangeListener(
                    (button, checked) -> applyOrganizeMemoryConfig(checked, true));
        }
        applyMemoryConfig(null);   // push saved/default memory profile to native before the first run
        applySavedDecodeConfig();  // strategy controls stay uninflated until the gear is first opened
        applyOrganizeMemoryConfig(organizeMemoryEnabled, false);   // push saved/default organize-memory state
        setupMemoryThresholds();   // load + wire the two draggable hysteresis-band markers
    }

    private void ensureDecodeStrategyControlsInflated() {
        if (beamSlider != null) {
            return;
        }
        View controls = findViewById(R.id.decode_strategy_controls);
        if (controls == null) {
            ViewStub stub = findViewById(R.id.decode_strategy_controls_stub);
            if (stub == null) {
                throw new IllegalStateException("Decode strategy controls stub is unavailable");
            }
            controls = stub.inflate();
        }
        beamControls = controls.findViewById(R.id.beam_controls);
        directPenaltyControls = controls.findViewById(R.id.direct_penalty_controls);
        samplerControls = controls.findViewById(R.id.sampler_controls);
        beamSlider = controls.findViewById(R.id.beam_slider);
        topKSlider = controls.findViewById(R.id.top_k_slider);
        penaltySlider = controls.findViewById(R.id.penalty_slider);
        penaltyRangeSlider = controls.findViewById(R.id.penalty_range_slider);
        samplingTemperatureSlider = controls.findViewById(R.id.sampling_temperature_slider);
        samplingTopKSlider = controls.findViewById(R.id.sampling_top_k_slider);
        samplingTopPSlider = controls.findViewById(R.id.sampling_top_p_slider);
        samplingPenaltySlider = controls.findViewById(R.id.sampling_penalty_slider);
        setCompactSliderRowHeights(beamSlider, topKSlider, penaltySlider, penaltyRangeSlider,
            samplingTemperatureSlider, samplingTopKSlider, samplingTopPSlider,
            samplingPenaltySlider);
        beamValue = controls.findViewById(R.id.beam_value);
        topKValue = controls.findViewById(R.id.top_k_value);
        penaltyValue = controls.findViewById(R.id.penalty_value);
        penaltyRangeValue = controls.findViewById(R.id.penalty_range_value);
        samplingTemperatureValue = controls.findViewById(R.id.sampling_temperature_value);
        samplingTopKValue = controls.findViewById(R.id.sampling_top_k_value);
        samplingTopPValue = controls.findViewById(R.id.sampling_top_p_value);
        samplingPenaltyValue = controls.findViewById(R.id.sampling_penalty_value);
        restoreDecodeControls();
        Slider.OnChangeListener decodeListener = (slider, value, fromUser) -> applyDecodeConfig(slider);
        for (Slider slider : new Slider[]{beamSlider, topKSlider, penaltySlider, penaltyRangeSlider,
                samplingTemperatureSlider, samplingTopKSlider, samplingTopPSlider,
                samplingPenaltySlider}) {
            slider.addOnChangeListener(decodeListener);
        }
        updateDecodeModeVisibility();
        setDecodeControlsEnabled(!generatingActive);
    }

    private void setCompactSliderRowHeights(Slider... sliders) {
        for (Slider slider : sliders) {
            if (slider == null || !(slider.getParent() instanceof View row)) {
                continue;
            }
            ViewGroup.LayoutParams params = row.getLayoutParams();
            params.height = dp(36);
            row.setLayoutParams(params);
        }
    }

    // Expand the compact settings deck to the composer. Edits stay local while expanded; collapse
    // commits one final snapshot.
    private void toggleDecodePanel() {
        if (generatingActive || decodeConfigLoading) {
            return;
        }
        ensureDecodeStrategyControlsInflated();
        boolean expand = decodeContent.getVisibility() != View.VISIBLE;
        MaxHeightNestedScrollView header = findViewById(R.id.header);
        View root = findViewById(R.id.use_think);
        if (expand) {
            View inputBar = findViewById(R.id.input_bar);
            int availableHeight = inputBar.getTop() > header.getTop()
                ? inputBar.getTop() - header.getTop()
                : root.getHeight() - inputBar.getHeight();
            int expandedHeaderHeight = Math.max(dp(160), availableHeight);
            ViewGroup.LayoutParams params = decodeContent.getLayoutParams();
            params.height = ViewGroup.LayoutParams.WRAP_CONTENT;
            decodeContent.setLayoutParams(params);
            header.setMaxHeight(expandedHeaderHeight);
        } else {
            header.setMaxHeight(Math.max(dp(160),
                    Math.round(getResources().getDisplayMetrics().heightPixels * 0.52f)));
            ViewGroup.LayoutParams params = decodeContent.getLayoutParams();
            params.height = ViewGroup.LayoutParams.WRAP_CONTENT;
            decodeContent.setLayoutParams(params);
            header.scrollTo(0, 0);
        }
        if (root instanceof ViewGroup) {
            // Scope the transition to the header, the decode panel, and the chat surface instead of capturing
            // start/end bounds across the whole Activity root (every RecyclerView row + the composer).
            TransitionSet decodeTransition = new TransitionSet();
            decodeTransition.setOrdering(TransitionSet.ORDERING_TOGETHER);
            decodeTransition.addTransition(new ChangeBounds());
            decodeTransition.addTransition(new Fade());
            decodeTransition.setDuration(220);
            decodeTransition.addTarget(header);
            decodeTransition.addTarget(decodeContent);
            View chatSurface = findViewById(R.id.result_text);
            if (chatSurface != null) {
                decodeTransition.addTarget(chatSurface);
            }
            TransitionManager.beginDelayedTransition((ViewGroup) root, decodeTransition);
        }
        decodeContent.setVisibility(expand ? View.VISIBLE : View.GONE);
        if (!expand && decodeConfigDirty) {
            applyDecodeConfig(null);
            // A strategy swap marks decodeConfigLoading synchronously; lock the gear before another tap lands.
            updateHeaderActionStates();
        }
    }

        private void restoreDecodeControls() {
        DecodePreferences preferences = decodePreferences;
        decodeMode = preferences.mode;
        beamSlider.setValue(preferences.beamSize);
        topKSlider.setValue(preferences.beamTopK);
        penaltySlider.setValue(preferences.directPenalty);
        penaltyRangeSlider.setValue(preferences.penaltyRange);
        samplingTemperatureSlider.setValue(preferences.samplingTemperature);
        samplingTopKSlider.setValue(preferences.samplingTopK);
        samplingTopPSlider.setValue(preferences.samplingTopP);
        samplingPenaltySlider.setValue(preferences.samplingPenalty);
        int checkedId = decodeMode == DECODE_MODE_BEAM
            ? R.id.decode_mode_beam
            : (decodeMode == DECODE_MODE_SAMPLING
                ? R.id.decode_mode_sampler : R.id.decode_mode_greedy);
        decodeStrategyGroup.check(checkedId);
        }

        private void updateDecodeModeVisibility() {
        beamControls.setVisibility(decodeMode == DECODE_MODE_BEAM ? View.VISIBLE : View.GONE);
        directPenaltyControls.setVisibility(
            decodeMode == DECODE_MODE_SAMPLING ? View.GONE : View.VISIBLE);
        samplerControls.setVisibility(decodeMode == DECODE_MODE_SAMPLING ? View.VISIBLE : View.GONE);
        }

        private void setDecodeControlsEnabled(boolean enabled) {
            if (decodeStrategyGroup != null) {
                decodeStrategyGroup.setEnabled(enabled);
            }
            for (int id : new int[]{R.id.decode_mode_greedy, R.id.decode_mode_beam,
                                    R.id.decode_mode_sampler}) {
                View control = findViewById(id);
                if (control != null) {
                    control.setEnabled(enabled);
                }
            }
            for (Slider slider : new Slider[]{beamSlider, topKSlider, penaltySlider, penaltyRangeSlider,
                                              samplingTemperatureSlider, samplingTopKSlider,
                                              samplingTopPSlider, samplingPenaltySlider}) {
                if (slider != null) {
                    slider.setEnabled(enabled);
                }
            }
            updateHeaderActionStates();
        }

        private static final class DecodeConfig {
            final int mode;
            final int topK;
            final int beamSize;
            final float repeatPenalty;
            final int penaltyRange;
            final float temperature;
            final float topP;

            DecodeConfig(int mode, int topK, int beamSize, float repeatPenalty, int penaltyRange,
                         float temperature, float topP) {
                this.mode = mode;
                this.topK = topK;
                this.beamSize = beamSize;
                this.repeatPenalty = repeatPenalty;
                this.penaltyRange = penaltyRange;
                this.temperature = temperature;
                this.topP = topP;
            }
        }

        private static final class DecodePreferences {
            final int mode;
            final int beamSize;
            final int beamTopK;
            final float directPenalty;
            final int penaltyRange;
            final float samplingTemperature;
            final int samplingTopK;
            final float samplingTopP;
            final float samplingPenalty;

            DecodePreferences(int mode, int beamSize, int beamTopK, float directPenalty,
                              int penaltyRange, float samplingTemperature, int samplingTopK,
                              float samplingTopP, float samplingPenalty) {
                this.mode = mode;
                this.beamSize = beamSize;
                this.beamTopK = beamTopK;
                this.directPenalty = directPenalty;
                this.penaltyRange = penaltyRange;
                this.samplingTemperature = samplingTemperature;
                this.samplingTopK = samplingTopK;
                this.samplingTopP = samplingTopP;
                this.samplingPenalty = samplingPenalty;
            }
        }

        private static DecodePreferences readDecodePreferences(SharedPreferences prefs) {
            int mode = clamp(prefs.getInt(App.KEY_DECODE_MODE, DECODE_MODE_GREEDY),
                    DECODE_MODE_GREEDY, DECODE_MODE_SAMPLING);
            int beam = clamp(prefs.getInt(App.KEY_BEAM_SIZE, 3), 2, 10);
            int beamTopK = clamp(prefs.getInt(App.KEY_BEAM_TOP_K, 3), beam, 10);
            return new DecodePreferences(
                    mode,
                    beam,
                    beamTopK,
                    snapToStep(prefs.getFloat(App.KEY_DIRECT_REPEAT_PENALTY, 0.8f),
                            0.0f, 1.0f, 0.05f),
                    clamp(prefs.getInt(App.KEY_PENALTY_RANGE, 20), 1, 256),
                    snapToStep(prefs.getFloat(App.KEY_SAMPLING_TEMPERATURE, 0.8f),
                            0.1f, 2.0f, 0.05f),
                    clamp(prefs.getInt(App.KEY_SAMPLING_TOP_K, 3), 1, 50),
                    snapToStep(prefs.getFloat(App.KEY_SAMPLING_TOP_P, 0.95f),
                            0.05f, 1.0f, 0.05f),
                    snapToStep(prefs.getFloat(App.KEY_SAMPLING_REPEAT_PENALTY, 1.2f),
                            1.0f, 2.0f, 0.05f));
        }

            private void applySavedDecodeConfig() {
            DecodePreferences preferences = decodePreferences;
            decodeMode = preferences.mode;
            useBeamSearch = decodeMode == DECODE_MODE_BEAM;
            int nativeTopK = decodeMode == DECODE_MODE_SAMPLING
                ? preferences.samplingTopK : (useBeamSearch ? preferences.beamTopK : 1);
            int nativeBeam = useBeamSearch ? preferences.beamSize : 1;
            float nativePenalty = decodeMode == DECODE_MODE_SAMPLING
                ? preferences.samplingPenalty : preferences.directPenalty;
            boolean applied = Configure_LLM(decodeMode, nativeTopK, nativeBeam, nativePenalty,
                preferences.penaltyRange, preferences.samplingTemperature, preferences.samplingTopP);
            if (applied) {
                configuredDecodeMode = decodeMode;
                visionModelsReady = Vision_Models_Ready();
            }
            if (modelsReady) {
                decodeConfigReady = applied;
                notifyModelLoadingStateChanged();
            }
            }

        private static void enqueueDecodeConfig(DecodeConfig config) {
            final boolean strategySwap = config.mode != configuredDecodeMode;
            final boolean loadingStarted;
            final boolean startWorker;
            synchronized (DECODE_CONFIG_LOCK) {
                pendingDecodeConfig = config;
                loadingStarted = strategySwap && !decodeConfigLoading;
                if (strategySwap) {
                    decodeConfigLoading = true;
                }
                startWorker = !decodeConfigWorkerRunning;
                if (startWorker) {
                    decodeConfigWorkerRunning = true;
                }
            }
            if (loadingStarted) {
                notifyModelLoadingStateChanged();
            }
            if (startWorker) {
                DECODE_CONFIG_EXECUTOR.execute(MainActivity::drainDecodeConfigQueue);
            }
        }

        private static void drainDecodeConfigQueue() {
            boolean latestApplied = true;
            while (true) {
                final DecodeConfig config;
                synchronized (DECODE_CONFIG_LOCK) {
                    config = pendingDecodeConfig;
                    pendingDecodeConfig = null;
                    if (config == null) {
                        decodeConfigWorkerRunning = false;
                        decodeConfigReady = latestApplied;
                        decodeConfigLoading = false;
                        break;
                    }
                }
                latestApplied = Configure_LLM(config.mode, config.topK, config.beamSize,
                        config.repeatPenalty, config.penaltyRange, config.temperature, config.topP);
                if (latestApplied) {
                    configuredDecodeMode = config.mode;
                    visionModelsReady = Vision_Models_Ready();
                }
            }
            notifyModelLoadingStateChanged();
            if (!latestApplied) {
                Log.e("NativeLLM", "Could not apply the latest decode configuration");
                mainHandler.post(() -> {
                    MainActivity activity = currentActivity();
                    if (activity != null) {
                        showToast(activity, activity.getString(R.string.decode_config_load_failed), true);
                    }
                });
            }
        }

        // Read all mode-specific controls, enforce Beam Top-K >= Beam, refresh labels, persist, then push one
        // immutable configuration snapshot to native. Hidden modes retain their independent tuning values.
    private void applyDecodeConfig(Slider changed) {
        int beam = Math.round(beamSlider.getValue());
        int topk = Math.round(topKSlider.getValue());
        if (changed == beamSlider && topk < beam) {
            topKSlider.setValue(beam);
            return;
        }
        if (changed == topKSlider && topk < beam) {
            beamSlider.setValue(topk);
            return;
        }
        float penalty = penaltySlider.getValue();
        int penaltyRange = Math.round(penaltyRangeSlider.getValue());
        float samplingTemperature = samplingTemperatureSlider.getValue();
        int samplingTopK = Math.round(samplingTopKSlider.getValue());
        float samplingTopP = samplingTopPSlider.getValue();
        float samplingPenalty = samplingPenaltySlider.getValue();
        beamValue.setText(String.valueOf(beam));
        topKValue.setText(String.valueOf(topk));
        String penaltyText = penalty >= 1.0f
                ? getString(R.string.penalty_off)
                : String.format(Locale.US, "%.2f", penalty);
        penaltyValue.setText(penaltyText);
            penaltyRangeValue.setText(String.valueOf(penaltyRange));
            samplingTemperatureValue.setText(String.format(Locale.US, "%.2f", samplingTemperature));
            samplingTopKValue.setText(String.valueOf(samplingTopK));
            samplingTopPValue.setText(String.format(Locale.US, "%.2f", samplingTopP));
            samplingPenaltyValue.setText(samplingPenalty >= 1.0f && samplingPenalty < 1.001f
                ? getString(R.string.penalty_off)
                : String.format(Locale.US, "%.2f", samplingPenalty));
            if (decodeContent != null && decodeContent.getVisibility() == View.VISIBLE) {
                decodeConfigDirty = true;
                return;
            }
            decodeConfigDirty = false;
                decodePreferences = new DecodePreferences(decodeMode, beam, topk, penalty,
                    penaltyRange, samplingTemperature, samplingTopK, samplingTopP,
                    samplingPenalty);
            getSharedPreferences(App.PREFS, MODE_PRIVATE).edit()
                .putInt(App.KEY_DECODE_MODE, decodeMode)
                .putInt(App.KEY_BEAM_SIZE, beam)
                .putInt(App.KEY_BEAM_TOP_K, topk)
                .putFloat(App.KEY_DIRECT_REPEAT_PENALTY, penalty)
                .putInt(App.KEY_PENALTY_RANGE, penaltyRange)
                .putFloat(App.KEY_SAMPLING_TEMPERATURE, samplingTemperature)
                .putInt(App.KEY_SAMPLING_TOP_K, samplingTopK)
                .putFloat(App.KEY_SAMPLING_TOP_P, samplingTopP)
                .putFloat(App.KEY_SAMPLING_REPEAT_PENALTY, samplingPenalty)
                .apply();
            useBeamSearch = decodeMode == DECODE_MODE_BEAM;
            int nativeTopK = decodeMode == DECODE_MODE_SAMPLING ? samplingTopK
                : (decodeMode == DECODE_MODE_BEAM ? topk : 1);
            int nativeBeam = decodeMode == DECODE_MODE_BEAM ? beam : 1;
            float nativePenalty = decodeMode == DECODE_MODE_SAMPLING ? samplingPenalty : penalty;
            DecodeConfig config = new DecodeConfig(decodeMode, nativeTopK, nativeBeam,
                nativePenalty, penaltyRange, samplingTemperature, samplingTopP);
            boolean applyImmediately = !modelsLoading && !systemPromptConfigLoading &&
                (!modelsReady || (!decodeConfigLoading && config.mode == configuredDecodeMode));
            if (applyImmediately) {
                boolean applied = Configure_LLM(config.mode, config.topK, config.beamSize,
                    config.repeatPenalty, config.penaltyRange, config.temperature, config.topP);
                if (applied) {
                    configuredDecodeMode = config.mode;
                    visionModelsReady = Vision_Models_Ready();
                }
                if (modelsReady) {
                    decodeConfigReady = applied;
                    notifyModelLoadingStateChanged();
                    if (!applied) {
                        Log.e("NativeLLM", "Could not apply decode parameters");
                        showToast(this, getString(R.string.decode_config_load_failed), true);
                    }
                }
            } else {
                enqueueDecodeConfig(config);
            }
            updateHeaderActionStates();
    }

    // ==================== Memory profile controls ====================

    private void restoreMemoryControls() {
        SharedPreferences prefs = getSharedPreferences(App.PREFS, MODE_PRIVATE);
        int memory = snapToStep(prefs.getInt(App.KEY_MEMORY_TOKENS, defaultMemoryTokens),
            minMemoryTokens, maxMemoryTokens, memoryTokensStep);
        int prefill = snapToStep(prefs.getInt(App.KEY_PREFILL_TOKENS, defaultPrefillTokens),
            minPrefillTokens, Math.min(maxPrefillTokens, memory), prefillTokensStep);
        int decode = snapToStep(prefs.getInt(App.KEY_DECODE_TOKENS, defaultDecodeTokens),
            minDecodeTokens, maxDecodeTokens, decodeTokensStep);
        memorySlider.setValue(memory);
        prefillSlider.setValue(prefill);
        decodeLimitSlider.setValue(decode);
    }

    private void refreshMemoryControlsFromNative() {
        loadMemoryLimits();
        restoreMemoryControls();
        boolean prefillSupported = Supports_Prefill_Lookback();
        if (prefillControls != null) {
            prefillControls.setVisibility(prefillSupported ? View.VISIBLE : View.GONE);
        }
        if (!prefillSupported) {
            prefillSlider.setValue(0f);
        }
        applyMemoryConfig(null);
    }

    private void loadMemoryLimits() {
        int[] limits = Get_Memory_Limits();
        // Adopt native limits ONLY once model metadata is loaded. setupDecodeControls() runs early in onCreate()
        // while the ONNX graphs still load on a bg thread, so Get_Memory_Limits() returns zeros (max_seq_len
        // unfilled) at cold start -- feeding those to a Material Slider makes valueFrom == valueTo, which throws
        // when the decode panel is first expanded. While degenerate, keep the fallback ranges.
        if (limits != null && limits.length >= 9
                && limits[1] > limits[0]      // memory:  max > min
                && limits[4] > limits[3]      // prefill: max > min
                && limits[7] > limits[6]) {   // decode:  max > min
            minMemoryTokens = limits[0];
            maxMemoryTokens = limits[1];
            defaultMemoryTokens = limits[2];
            minPrefillTokens = limits[3];
            maxPrefillTokens = limits[4];
            defaultPrefillTokens = limits[5];
            minDecodeTokens = limits[6];
            maxDecodeTokens = limits[7];
            defaultDecodeTokens = limits[8];
        }
        memoryTokensStep = configureDiscreteSlider(memorySlider, minMemoryTokens, maxMemoryTokens,
                PREFERRED_MEMORY_TOKENS_STEP);
        prefillTokensStep = configureDiscreteSlider(prefillSlider, minPrefillTokens, maxPrefillTokens,
                PREFERRED_PREFILL_TOKENS_STEP);
        decodeTokensStep = configureDiscreteSlider(decodeLimitSlider, minDecodeTokens, maxDecodeTokens,
                PREFERRED_DECODE_TOKENS_STEP);
    }

    private int configureDiscreteSlider(Slider slider, int minValue, int maxValue, int preferredStep) {
        int min = Math.min(minValue, maxValue);
        int max = Math.max(minValue, maxValue);
        int step = compatibleSliderStep(max - min, preferredStep);
        int current = snapToStep(Math.round(slider.getValue()), min, max, step);
        if (current >= slider.getValueFrom() && current <= slider.getValueTo()) {
            slider.setValue(current);
        }
        slider.setValueFrom(min);
        slider.setValueTo(max);
        slider.setStepSize(step);
        slider.setValue(current);
        return step;
    }

    private static int compatibleSliderStep(int range, int preferredStep) {
        int safeRange = Math.max(1, Math.abs(range));
        int safePreferred = Math.max(1, Math.abs(preferredStep));
        return Math.max(1, gcd(safeRange, safePreferred));
    }

    private static int gcd(int a, int b) {
        int x = Math.abs(a);
        int y = Math.abs(b);
        while (y != 0) {
            int next = x % y;
            x = y;
            y = next;
        }
        return Math.max(1, x);
    }

    private void applyMemoryConfig(Slider changed) {
        applyMemoryConfig(changed, true);
    }

    private void applyMemoryConfig(Slider changed, boolean commit) {
        int memory = Math.round(memorySlider.getValue());
        int prefill = prefillControls != null && prefillControls.getVisibility() == View.GONE
            ? 0 : Math.round(prefillSlider.getValue());
        if (prefill > memory) {
            prefillSlider.setValue(memory);
            return;
        }
        int decode = Math.round(decodeLimitSlider.getValue());
        memoryValue.setText(String.valueOf(memory));
        prefillValue.setText(String.valueOf(prefill));
        decodeLimitValue.setText(String.valueOf(decode));
        updateMemoryRemainingFromKnownUsage();
        if (!commit) {
            return;
        }
        getSharedPreferences(App.PREFS, MODE_PRIVATE).edit()
                .putInt(App.KEY_MEMORY_TOKENS, memory)
                .putInt(App.KEY_PREFILL_TOKENS, prefill)
                .putInt(App.KEY_DECODE_TOKENS, decode)
                .apply();
        Configure_Memory(memory, prefill, decode);
    }

    // Push the "organize memory" (整理记忆) toggle to native (next turn), optionally persisting it.
    private void applyOrganizeMemoryConfig(boolean enabled, boolean persist) {
        if (persist) {
            getSharedPreferences(App.PREFS, MODE_PRIVATE).edit()
                    .putBoolean(App.KEY_ORGANIZE_MEMORY, enabled).apply();
        }
        Configure_Organize_Memory(enabled);
    }

    // ==================== Memory stats ====================

    private void refreshMemoryStatsAsync() {
        if (chatting) {
            return;
        }
        final int requestId = ++memoryStatsRequestId;
        runBackground("memory-stats", () -> {
            final int[] memoryStats = Get_Memory_Stats();
            mainHandler.post(() -> {
                if (requestId == memoryStatsRequestId && !chatting) {
                    applyMemoryStats(memoryStats);
                }
            });
        });
    }

    private void applyMemoryStats(int[] stats) {
        if (stats == null || stats.length < 4) {
            updateMemoryRemainingFromKnownUsage();
            return;
        }
        int capacity = Math.max(1, stats[3]);
        int used = clamp(stats[0], 0, capacity);
        applyMemoryStatsFromValues(used, capacity - used, capacity);
    }

    private void applyMemoryStatsFromValues(int usedTokens, int remainingTokens, int capacityTokens) {
        applyMemoryStatsFromValues(usedTokens, remainingTokens, capacityTokens, true);
    }

    private void applyMemoryStatsFromValues(int usedTokens, int remainingTokens, int capacityTokens,
                                            boolean animate) {
        int capacity = Math.max(1, capacityTokens);
        memoryUsedTokens = clamp(usedTokens, 0, capacity);
        int remaining = clamp(remainingTokens, 0, capacity);
        memoryCapacityTokens = capacity;
        memoryRemainingPercent = remainingPercent(remaining, capacity);
        updateMemoryRemainingDisplay(animate);
    }

    private void updateMemoryRemainingFromKnownUsage() {
        int capacity = memorySlider != null ? Math.round(memorySlider.getValue()) : defaultMemoryTokens;
        capacity = Math.max(1, capacity);
        int used = clamp(memoryUsedTokens, 0, capacity);
        memoryCapacityTokens = capacity;
        memoryRemainingPercent = remainingPercent(capacity - used, capacity);
        updateMemoryRemainingDisplay();
    }

    // ==================== Memory threshold band (draggable) ====================

    // ---- Draggable hysteresis-band markers (green = reset target, red = rebuild trigger) ----
    private void setupMemoryThresholds() {
        // Seed from prefs, else native defaults; push to native so a fresh install matches the C++ defaults.
        int defaultRed = MAX_MEMORY_BAND_PERCENT;
        int defaultGreen = MIN_MEMORY_BAND_PERCENT;
        int[] nativeDefaults = Get_Memory_Thresholds();
        if (nativeDefaults != null && nativeDefaults.length >= 2) {
            defaultRed = nativeDefaults[0];
            defaultGreen = nativeDefaults[1];
        }
        SharedPreferences prefs = getSharedPreferences(App.PREFS, MODE_PRIVATE);
        memoryRedPercent = prefs.getInt(App.KEY_MEMORY_RED_PERCENT, defaultRed);
        memoryGreenPercent = prefs.getInt(App.KEY_MEMORY_GREEN_PERCENT, defaultGreen);
        clampThresholds();
        Configure_Memory_Thresholds(memoryRedPercent, memoryGreenPercent);
        updateMemoryBandCaption();
        if (memoryThresholdSlider != null) {
            memoryThresholdSlider.setMinSeparationValue(MIN_MEMORY_BAND_GAP);
            memoryThresholdSlider.setValues(
                    (float) memoryGreenPercent, (float) memoryRedPercent);
            memoryThresholdSlider.addOnChangeListener((slider, value, fromUser) -> {
                List<Float> values = slider.getValues();
                if (values.size() >= 2) {
                    memoryGreenPercent = Math.round(values.get(0));
                    memoryRedPercent = Math.round(values.get(1));
                    clampThresholds();
                    positionThresholdMarkers();
                    updateMemoryBandCaption();
                }
            });
            memoryThresholdSlider.addOnSliderTouchListener(
                    new RangeSlider.OnSliderTouchListener() {
                        @Override public void onStartTrackingTouch(@NonNull RangeSlider slider) {
                        }

                        @Override public void onStopTrackingTouch(@NonNull RangeSlider slider) {
                            commitThresholds();
                        }
                    });
        }
        if (memoryRemainingTrack != null) {
            memoryRemainingTrack.setAccessibilityDelegate(new View.AccessibilityDelegate() {
            @Override
            public void onInitializeAccessibilityNodeInfo(
                View host, AccessibilityNodeInfo info) {
                super.onInitializeAccessibilityNodeInfo(host, info);
                info.setClassName(RangeSlider.class.getName());
                info.setContentDescription(getString(R.string.memory_threshold_state,
                    memoryGreenPercent, memoryRedPercent));
                info.addAction(new AccessibilityNodeInfo.AccessibilityAction(
                    ACTION_GREEN_DECREASE, getString(R.string.memory_green_decrease)));
                info.addAction(new AccessibilityNodeInfo.AccessibilityAction(
                    ACTION_GREEN_INCREASE, getString(R.string.memory_green_increase)));
                info.addAction(new AccessibilityNodeInfo.AccessibilityAction(
                    ACTION_RED_DECREASE, getString(R.string.memory_red_decrease)));
                info.addAction(new AccessibilityNodeInfo.AccessibilityAction(
                    ACTION_RED_INCREASE, getString(R.string.memory_red_increase)));
            }

            @Override
            public boolean performAccessibilityAction(View host, int action, Bundle args) {
                if (adjustMemoryThresholdFromAccessibility(action)) {
                host.sendAccessibilityEvent(
                    android.view.accessibility.AccessibilityEvent.TYPE_VIEW_SELECTED);
                return true;
                }
                return super.performAccessibilityAction(host, action, args);
            }
            });
            // Marker position depends on the track width, only known after layout (and after a rotation).
            memoryRemainingTrack.addOnLayoutChangeListener(
                    (v, l, t, r, b, ol, ot, or, ob) -> positionThresholdMarkers());
            positionThresholdMarkers();
        }
    }

    private void clampThresholds() {
        memoryRedPercent = clamp(memoryRedPercent,
                MIN_MEMORY_BAND_PERCENT + MIN_MEMORY_BAND_GAP, MAX_MEMORY_BAND_PERCENT);
        memoryGreenPercent = clamp(memoryGreenPercent,
                MIN_MEMORY_BAND_PERCENT, memoryRedPercent - MIN_MEMORY_BAND_GAP);
    }

    private boolean adjustMemoryThresholdFromAccessibility(int action) {
        switch (action) {
            case ACTION_GREEN_DECREASE:
                memoryGreenPercent -= 1;
                break;
            case ACTION_GREEN_INCREASE:
                memoryGreenPercent += 1;
                break;
            case ACTION_RED_DECREASE:
                memoryRedPercent -= 1;
                break;
            case ACTION_RED_INCREASE:
                memoryRedPercent += 1;
                break;
            default:
                return false;
        }
        commitThresholds();
        return true;
    }

    private void positionThresholdMarkers() {
        if (memoryRemainingTrack == null) {
            return;
        }
        float width = memoryRemainingTrack.getWidth();
        if (width <= 0f) {
            return;   // not laid out yet; the layout listener re-runs this once a width is known
        }
        if (memoryThresholdGreen != null) {
            memoryThresholdGreen.setTranslationX(
                    memoryGreenPercent / 100f * width - memoryThresholdGreen.getWidth() / 2f);
        }
        if (memoryThresholdRed != null) {
            memoryThresholdRed.setTranslationX(
                    memoryRedPercent / 100f * width - memoryThresholdRed.getWidth() / 2f);
        }
    }

    // Persist + push to native once the drag settles (takes effect on the next turn's boundary trim).
    private void commitThresholds() {
        clampThresholds();
        if (memoryThresholdSlider != null) {
            memoryThresholdSlider.setValues(
                (float) memoryGreenPercent, (float) memoryRedPercent);
        }
        positionThresholdMarkers();
        updateMemoryBandCaption();
        getSharedPreferences(App.PREFS, MODE_PRIVATE).edit()
                .putInt(App.KEY_MEMORY_RED_PERCENT, memoryRedPercent)
                .putInt(App.KEY_MEMORY_GREEN_PERCENT, memoryGreenPercent)
                .apply();
        Configure_Memory_Thresholds(memoryRedPercent, memoryGreenPercent);
    }

    private void updateMemoryBandCaption() {
        if (memoryBandValue != null) {
            memoryBandValue.setText(getString(R.string.memory_band_value, memoryGreenPercent, memoryRedPercent));
        }
        if (memoryRemainingTrack != null) {
            memoryRemainingTrack.setContentDescription(getString(
                    R.string.memory_threshold_state, memoryGreenPercent, memoryRedPercent));
        }
    }

    // ==================== Memory gauge rendering ====================

    // Refresh the memory panel: free-headroom headline (health colour) + usage bar. Used = tokens held in the
    // window; the track behind the bar is free headroom.
    private void updateMemoryRemainingDisplay() {
        updateMemoryRemainingDisplay(true);
    }

    private void updateMemoryRemainingDisplay(boolean animate) {
        int healthColor = memorySemanticColor(memoryRemainingPercent);
        if (memoryRemainingValue != null) {
            memoryRemainingValue.setText(getString(R.string.memory_remaining_value,
                    (int) Math.round(memoryRemainingPercent), memoryCapacityTokens));
            memoryRemainingValue.setTextColor(healthColor);
        }
        updateMemoryRemainingBar(animate);
    }

    // Gauge fills from the left with the USED portion (shades toward red as it fills). Pulses only when usage
    // climbs, not when a reset frees it.
    private void updateMemoryRemainingBar(boolean animate) {
        if (memoryRemainingBar == null || memoryRemainingGlow == null) {
            return;
        }
        int targetColor = memorySemanticColor(memoryRemainingPercent);
        int capacity = Math.max(1, memoryCapacityTokens);
        float targetFraction = (float) clamp(memoryUsedTokens, 0, capacity) / (float) capacity;
        targetFraction = Math.max(0.0f, Math.min(1.0f, targetFraction));
        boolean pulse = lastDisplayedMemoryCapacityTokens == memoryCapacityTokens
                && lastDisplayedMemoryUsedTokens >= 0
                && memoryUsedTokens > lastDisplayedMemoryUsedTokens;
        if (animate) {
            animateMemoryBarTo(targetFraction, targetColor, pulse);
        } else {
            if (memoryBarAnimator != null) {
                memoryBarAnimator.cancel();
                memoryBarAnimator = null;
            }
            memoryBarFraction = targetFraction;
            memoryRemainingBar.setScaleX(targetFraction);
            memoryRemainingGlow.setScaleX(targetFraction);
            applyMemoryBarColor(targetColor);
        }
        lastDisplayedMemoryUsedTokens = memoryUsedTokens;
        lastDisplayedMemoryCapacityTokens = memoryCapacityTokens;
    }

    private void animateMemoryBarTo(float targetFraction, int targetColor, boolean pulse) {
        if (memoryBarAnimator != null) {
            memoryBarAnimator.cancel();
        }
        final float startFraction = memoryBarFraction;
        final int startColor = lastMemoryBarColor != 0 ? lastMemoryBarColor : targetColor;
        memoryBarAnimator = ValueAnimator.ofFloat(0.0f, 1.0f);
        memoryBarAnimator.setDuration(pulse ? 280L : 180L);
        memoryBarAnimator.setInterpolator(new LinearInterpolator());
        memoryBarAnimator.addUpdateListener(anim -> {
            float t = (float) anim.getAnimatedValue();
            memoryBarFraction = startFraction + (targetFraction - startFraction) * t;
            memoryRemainingBar.setScaleX(memoryBarFraction);
            memoryRemainingGlow.setScaleX(memoryBarFraction);
            applyMemoryBarColor(interpolateColor(startColor, targetColor, t));
        });
        memoryBarAnimator.start();
        if (pulse) {
            playMemoryGlow();
        }
    }

    private void playMemoryGlow() {
        if (memoryGlowAnimator != null) {
            memoryGlowAnimator.cancel();
        }
        memoryRemainingGlow.setAlpha(0.75f);
        memoryRemainingGlow.setScaleY(1.0f);
        memoryGlowAnimator = ObjectAnimator.ofPropertyValuesHolder(memoryRemainingGlow,
                PropertyValuesHolder.ofFloat(View.ALPHA, 0.75f, 0.0f),
                PropertyValuesHolder.ofFloat(View.SCALE_Y, 1.0f, 1.75f),
                PropertyValuesHolder.ofFloat(View.TRANSLATION_Z, dp(4), 0f));
        memoryGlowAnimator.setDuration(520L);
        memoryGlowAnimator.setInterpolator(new AccelerateDecelerateInterpolator());
        memoryGlowAnimator.start();
    }

    private int memorySemanticColor(double percent) {
        float normalized = (float) Math.max(0.0, Math.min(1.0, percent / 100.0));
        int wine = ContextCompat.getColor(this, R.color.memoryBarWine);
        int red = ContextCompat.getColor(this, R.color.memoryBarRed);
        int orange = ContextCompat.getColor(this, R.color.memoryBarOrange);
        int yellow = ContextCompat.getColor(this, R.color.memoryBarYellow);
        int yellowGreen = ContextCompat.getColor(this, R.color.memoryBarYellowGreen);
        int tealGreen = ContextCompat.getColor(this, R.color.memoryBarTealGreen);
        if (normalized <= 0.25f) {
            return interpolateColor(wine, red, normalized / 0.25f);
        }
        if (normalized <= 0.50f) {
            return interpolateColor(red, orange, (normalized - 0.25f) / 0.25f);
        }
        if (normalized <= 0.70f) {
            return interpolateColor(orange, yellow, (normalized - 0.50f) / 0.20f);
        }
        if (normalized <= 0.85f) {
            return interpolateColor(yellow, yellowGreen, (normalized - 0.70f) / 0.15f);
        }
        return interpolateColor(yellowGreen, tealGreen, (normalized - 0.85f) / 0.15f);
    }

    private void applyMemoryBarColor(int color) {
        int[] barColors = solidBarColors(color, 0xF0);
        int[] glowColors = solidBarColors(color, 0x36);
        if (memoryBarDrawable == null) {
            memoryBarDrawable = makeMemoryBarDrawable(barColors);
            memoryRemainingBar.setBackground(memoryBarDrawable);
        } else {
            memoryBarDrawable.setColors(barColors);
        }
        if (memoryGlowDrawable == null) {
            memoryGlowDrawable = makeMemoryBarDrawable(glowColors);
            memoryRemainingGlow.setBackground(memoryGlowDrawable);
        } else {
            memoryGlowDrawable.setColors(glowColors);
        }
        lastMemoryBarColor = color;
    }

    private static int[] solidBarColors(int color, int alpha) {
        int solidColor = withAlpha(color, alpha);
        return new int[]{solidColor, solidColor};
    }

    private GradientDrawable makeMemoryBarDrawable(int[] colors) {
        GradientDrawable drawable = new GradientDrawable(GradientDrawable.Orientation.LEFT_RIGHT,
                colors);
        drawable.setCornerRadius(getResources().getDimension(R.dimen.ui_corner_radius_tiny));
        return drawable;
    }

    // ==================== Shared helpers ====================

    private static int interpolateColor(int startColor, int endColor, float fraction) {
        float t = Math.max(0.0f, Math.min(1.0f, fraction));
        int a = Math.round(Color.alpha(startColor) + (Color.alpha(endColor) - Color.alpha(startColor)) * t);
        int r = Math.round(Color.red(startColor) + (Color.red(endColor) - Color.red(startColor)) * t);
        int g = Math.round(Color.green(startColor) + (Color.green(endColor) - Color.green(startColor)) * t);
        int b = Math.round(Color.blue(startColor) + (Color.blue(endColor) - Color.blue(startColor)) * t);
        return Color.argb(a, r, g, b);
    }

    private static int withAlpha(int color, int alpha) {
        return Color.argb(alpha, Color.red(color), Color.green(color), Color.blue(color));
    }

    private int dp(float value) {
        return Math.round(getResources().getDisplayMetrics().density * value);
    }

    private static int clamp(int value, int lo, int hi) {
        return Math.max(lo, Math.min(value, hi));
    }

    private static double remainingPercent(int remainingTokens, int capacityTokens) {
        int capacity = Math.max(1, capacityTokens);
        int remaining = clamp(remainingTokens, 0, capacity);
        return remaining * 100.0 / capacity;
    }

    private static int snapToStep(int value, int lo, int hi, int step) {
        int clamped = clamp(value, lo, hi);
        int steps = Math.round((clamped - lo) / (float) step);
        return clamp(lo + steps * step, lo, hi);
    }

    private static float snapToStep(float value, float lo, float hi, float step) {
        float clamped = Math.max(lo, Math.min(value, hi));
        int steps = Math.round((clamped - lo) / step);
        return Math.max(lo, Math.min(lo + steps * step, hi));
    }

    private static void showToast(final Activity context, final String content, boolean display_long){
        if (display_long) {
            context.runOnUiThread(() -> Toast.makeText(context, content, Toast.LENGTH_LONG).show());
        } else {
            context.runOnUiThread(() -> Toast.makeText(context, content, Toast.LENGTH_SHORT).show());
        }
    }

    // ==================== Asset staging ====================

    // True when fileName is a tokenizer vocab table (vocab_*.txt, case-insensitive), regardless of the
    // model-specific middle, so the exported vocab can keep whatever name Export_Vocab.py gave it.
    private static boolean isVocabFileName(String fileName) {
        if (fileName == null) {
            return false;
        }
        String lower = fileName.toLowerCase(java.util.Locale.ROOT);
        return lower.startsWith("vocab_") && lower.endsWith(".txt");
    }

    // Prefer a flat model bundle; otherwise select the first top-level directory carrying the metadata
    // model. Model files are flattened into cache by basename because native loading is basename-based.
    private static String findModelAssetDirectory(AssetManager mgr) throws IOException {
        String[] rootEntries = mgr.list("");
        if (rootEntries == null) {
            return "";
        }
        for (String entry : rootEntries) {
            if ("LLM_Metadata.onnx".equals(entry)) {
                return "";
            }
        }
        for (String entry : rootEntries) {
            String[] children = mgr.list(entry);
            if (children == null) {
                continue;
            }
            for (String child : children) {
                if ("LLM_Metadata.onnx".equals(child)) {
                    return entry;
                }
            }
        }
        return "";
    }

    private static AssetManifest loadAssetManifest(AssetManager manager)
            throws IOException, NoSuchAlgorithmException {
        byte[] bytes;
        try (InputStream input = manager.open(ASSET_MANIFEST_NAME);
             ByteArrayOutputStream output = new ByteArrayOutputStream(4096)) {
            byte[] buffer = new byte[8192];
            int read;
            while ((read = input.read(buffer)) != -1) {
                output.write(buffer, 0, read);
            }
            bytes = output.toByteArray();
        }
        Map<String, String> hashes = new HashMap<>();
        String text = new String(bytes, StandardCharsets.UTF_8);
        for (String line : text.split("\n")) {
            String trimmed = line.trim();
            if (trimmed.isEmpty()) {
                continue;
            }
            int separator = trimmed.indexOf("  ");
            if (separator != 64 || trimmed.length() <= separator + 2) {
                throw new IOException("Malformed model bundle manifest");
            }
            hashes.put(trimmed.substring(separator + 2), trimmed.substring(0, separator));
        }
        if (hashes.isEmpty()) {
            throw new IOException("Model bundle manifest is empty");
        }
        return new AssetManifest(hashes, sha256(bytes));
    }

    private static void clearStagedModelAssets(File cacheDirectory) {
        File[] staged = cacheDirectory.listFiles();
        if (staged == null) {
            return;
        }
        for (File file : staged) {
            String name = file.getName();
                if (file.isFile() &&
                    (name.endsWith(".onnx") || name.endsWith(".data")) &&
                    !file.delete()) {
                Log.w(BENCHMARK_TAG, "Could not remove stale model asset: " + file);
            }
        }
    }

    private static boolean Copy_from_Asset_to_Cache(String assetPath, String cacheFileName,
                                                    File cacheDir, AssetManager mgr,
                                                    byte[] buffer, boolean forceOverwrite,
                                                    String expectedSha256){
        File tempFile = null;
        try {
            if (!cacheDir.exists()) {
                if (!cacheDir.mkdirs()) {
                    return false;
                }
            }
            if (expectedSha256 == null || expectedSha256.length() != 64) {
                return false;
            }
            File outFile = new File(cacheDir, cacheFileName);
            if (outFile.exists() && !forceOverwrite && outFile.length() > 10) {
                MessageDigest cachedDigest = MessageDigest.getInstance("SHA-256");
                boolean cacheReadable = true;
                try (InputStream cached = new java.io.FileInputStream(outFile)) {
                    int len;
                    while ((len = cached.read(buffer)) != -1) {
                        cachedDigest.update(buffer, 0, len);
                    }
                } catch (IOException ignored) {
                    cacheReadable = false;
                }
                if (cacheReadable && expectedSha256.equals(toHex(cachedDigest.digest()))) {
                    return true;
                }
            }
            tempFile = new File(cacheDir, cacheFileName + ".staging");
            if (tempFile.exists() && !tempFile.delete()) {
                return false;
            }
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            try (InputStream is = mgr.open(assetPath);
                 FileOutputStream os = new FileOutputStream(tempFile)) {
                int len;
                while ((len = is.read(buffer)) != -1) {
                    os.write(buffer, 0, len);
                    digest.update(buffer, 0, len);
                }
                os.flush();
            }
            if (!expectedSha256.equals(toHex(digest.digest())) || tempFile.length() <= 10 ||
                    (outFile.exists() && !outFile.delete()) || !tempFile.renameTo(outFile)) {
                return false;
            }
            tempFile = null;
            return true;
        } catch (Throwable error) {
            Log.e(BENCHMARK_TAG, "Could not stage asset " + assetPath, error);
            return false;
        } finally {
            if (tempFile != null && tempFile.exists() && !tempFile.delete()) {
                Log.w(BENCHMARK_TAG, "Could not remove partial staged asset: " + tempFile);
            }
        }
    }

    // ==================== Live camera (zero-copy RGB) ====================

    // Camera button tap: capture needs the runtime CAMERA grant, so gate on it before opening.
    private void onCameraButtonClicked() {
        if (!modelsReady || guardPostProcessing() || guardInferenceBusy()) {
            return;
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            ensureVisionModelsAndOpenCamera();
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    private void ensureVisionModelsAndOpenCamera() {
        if (visionModelsReady) {
            openCameraPreview();
            return;
        }
        if (visionModelsLoading) {
            showToast(this, getString(R.string.vision_models_loading), false);
            return;
        }
        startVisionModelLoad(true);
    }

    private void startVisionModelLoad(boolean openCameraWhenReady) {
        visionModelsLoading = true;
        GenerationService.startModelLoading(this);
        setMaintenanceActive(true);
        if (openCameraWhenReady) {
            showToast(this, getString(R.string.vision_models_loading), false);
        }
        runBackground("vision-model-load", () -> {
            final boolean loaded = Ensure_Vision_Models();
            final boolean ready = loaded && Vision_Models_Ready();
            if (ready) {
                CameraService.preloadCameraSelection(getApplicationContext());
            }
            GenerationService.stop(getApplicationContext());
            mainHandler.post(() -> {
                visionModelsLoading = false;
                visionModelsReady = ready;
                MainActivity activity = currentActivity();
                if (activity == null) {
                    maintenanceActive = false;
                    return;
                }
                activity.setMaintenanceActive(false);
                if (visionModelsReady && openCameraWhenReady) {
                    activity.openCameraPreview();
                } else if (!visionModelsReady && openCameraWhenReady) {
                    showToast(activity, activity.getString(R.string.vision_models_load_failed), true);
                }
            });
        });
    }

    // Lazily build the preview controller (binds the offscreen GLSurfaceView) and toggle the window.
    private void openCameraPreview() {
        if (cameraPreview == null) {
            cameraPreview = new CameraFloatingPreview(this, findViewById(R.id.glSurfaceView));
            cameraPreview.setVisionCaptureListener(new CameraFloatingPreview.VisionCaptureListener() {
                @Override public void onImageCaptured(Bitmap thumbnail) { onVisionCaptured(thumbnail, false); }
                @Override public void onVideoCaptured(List<Bitmap> frames, float seconds, boolean ok) {
                    MainActivity.this.onVideoCaptured(frames, seconds, ok);
                }
            });
        }
        cameraPreview.toggle();
    }

    // Native already owns the pixels; show the capture and pre-encode while the user writes a query.
    private void onVisionCaptured(Bitmap thumbnail, boolean video) {
        if (guardPostProcessing() || guardInferenceBusy()) {
            // A generation is already running: drop the just-pushed pixel_values so the next turn is not
            // misread as a stale vision capture.
            cancelPendingVisionWithMaintenance();
            recycleBitmap(thumbnail);
            return;
        }
        if (!pendingVisionCapture) {
            pendingVisionTurnId = nextTurnId++;   // allocate once; reused if the user re-captures before sending
        } else {
            removePendingVisionBubble();
        }
        pendingVisionCapture = true;
        pendingVisionIsVideo = video;
        pendingVideoFrameCount = 0;
        pendingVideoSeconds = 0f;
        int captureSerial = ++visionCaptureSerial;
        visionPreprocessedTurnId = ChatMessage.NO_TURN;
        if (thumbnail != null) {
            addImageHistory(thumbnail, pendingVisionTurnId, captureSerial);
            prewarmPendingVisionAsync(pendingVisionTurnId, captureSerial, video);
        }
    }

    private void onVideoCaptured(List<Bitmap> frames, float seconds, boolean ok) {
        if (!ok || frames == null || frames.isEmpty()) {
            recycleBitmapList(frames);
            cancelPendingVisionWithMaintenance();
            showToast(this, getString(R.string.camera_status_no_frame), false);
            return;
        }
        if (guardPostProcessing() || guardInferenceBusy()) {
            recycleBitmapList(frames);
            cancelPendingVisionWithMaintenance();
            return;
        }
        if (!pendingVisionCapture) {
            pendingVisionTurnId = nextTurnId++;
        } else {
            removePendingVisionBubble();
        }
        pendingVisionCapture = true;
        pendingVisionIsVideo = true;
        pendingVideoFrameCount = Math.max(1, Math.round(seconds * Math.max(CameraService.getVideoFps(), 0f)));
        pendingVideoSeconds = seconds;
        int captureSerial = ++visionCaptureSerial;
        visionPreprocessedTurnId = ChatMessage.NO_TURN;
        addVideoHistory(frames, pendingVisionTurnId, captureSerial);
        prewarmPendingVisionAsync(pendingVisionTurnId, captureSerial, true);
    }

    private void removePendingVisionBubble() {
        for (int index = messages.size() - 1; index >= 0; --index) {
            ChatMessage message = messages.get(index);
            if (message.turnId() != pendingVisionTurnId) {
                continue;
            }
            if (message.type() != ChatMessage.TYPE_USER_IMAGE &&
                    message.type() != ChatMessage.TYPE_USER_VIDEO) {
                continue;
            }
            if (chatAdapter != null) {
                chatAdapter.releaseVisibleVisionMediaFrom(answerView, index);
            }
            deleteMessageMediaAsync(message);
            messages.remove(index);
            if (chatAdapter != null) {
                chatAdapter.notifyItemRemoved(index);
            }
            return;
        }
    }

    // Pre-encode the query-independent ViT features off the caller thread so vision_hidden is ready
    // before the user finishes typing. Best-effort: on any failure the vision turn encodes inline.
    private void prewarmPendingVisionAsync(int turnId, int captureSerial, boolean video) {
        final long startMs = SystemClock.elapsedRealtime();
        runBackground("vision-prewarm", () -> {
            boolean prewarmed = false;
            try {
                prewarmed = Prewarm_Vision();
            } catch (Throwable ignored) {
                // pre-encode is a pure optimization; ignore and let the turn encode inline
            }
            if (!prewarmed) {
                return;
            }
            final float seconds = (SystemClock.elapsedRealtime() - startMs) / 1000f;
            mainHandler.post(() -> {
                MainActivity activity = currentActivity();
                if (activity != null) {
                    activity.onVisionPreprocessFinished(turnId, captureSerial, seconds, video);
                }
            });
        });
    }

    private void onVisionPreprocessFinished(int turnId, int captureSerial, float seconds, boolean video) {
        if (captureSerial != visionCaptureSerial) {
            return;
        }
        visionPreprocessedTurnId = turnId;
        if (video && !pendingVisionCapture && !visionTimingPending) {
            return;
        }
        if (visionTimingPending && visionTimingTurnId == turnId) {
            visionTimingPending = false;
            visionTimingTurnId = ChatMessage.NO_TURN;
        }
        if (video) {
            float windowSec = 0f;
            if (visionTimingPending && visionTimingIsVideo && visionTimingTurnId == turnId) {
                windowSec = visionVideoSeconds > 0f
                        ? visionVideoSeconds
                        : ((visionFps > 0f) ? visionFrameCount / visionFps : 0f);
            } else {
                float fps = CameraService.getVideoFps();
                windowSec = pendingVideoSeconds > 0f
                        ? pendingVideoSeconds
                        : ((fps > 0f && pendingVideoFrameCount > 0) ? pendingVideoFrameCount / fps : 0f);
            }
            float rtf = windowSec > 0f ? seconds / windowSec : 0f;
            updateVisionBubbleCaption(turnId,
                    getString(R.string.vision_preprocess_time_video, seconds, rtf));
        } else {
            updateVisionBubbleCaption(turnId, getString(R.string.vision_preprocess_time_image, seconds));
        }
    }

    // Consume the pending capture, reusing its pre-encoded vision hidden state when ready.
    private void dispatchVisionSend(String query, int turnId, boolean video) {
        pendingVisionCapture = false;
        pendingVisionIsVideo = false;
        usrInputText = query;
        usrTurnId = turnId;
        autoScrollLocked = false;
        updateJumpToBottomVisibility();
        if (visionPreprocessedTurnId != turnId) {
            // Prewarm already reports the actual preprocess + ViT elapsed time. Only fall back to
            // send-to-first-token timing when no completed prewarm measurement exists for this capture.
            beginVisionTiming(turnId, video);
        }
        startLLM();
        if (query != null && !query.isEmpty()) {
            addHistory(ChatMessage.TYPE_USER, query, turnId);
        }
        showLoadingBubble();
    }

    private void addImageHistory(Bitmap thumbnail, int turnId, int mediaId) {
        ChatMessage placeholder = ChatMessage.visionPlaceholder(false, turnId, mediaId);
        messages.add(placeholder);
        int newIndex = messages.size() - 1;
        if (chatAdapter != null) {
            chatAdapter.notifyItemInserted(newIndex);
        }
        scrollToBottom();
        scheduleTranscriptPaging();
        File cacheDirectory = getCacheDir();
        CHAT_MEDIA_EXECUTOR.execute(() -> completeVisionMediaEncoding(
                ChatMessage.image(thumbnail, "", turnId, mediaId, cacheDirectory)));
    }

    private void addVideoHistory(List<Bitmap> frames, int turnId, int mediaId) {
        ChatMessage placeholder = ChatMessage.visionPlaceholder(true, turnId, mediaId);
        messages.add(placeholder);
        int newIndex = messages.size() - 1;
        if (chatAdapter != null) {
            chatAdapter.notifyItemInserted(newIndex);
        }
        scrollToBottom();
        scheduleTranscriptPaging();
        File cacheDirectory = getCacheDir();
        CHAT_MEDIA_EXECUTOR.execute(() -> completeVisionMediaEncoding(
                ChatMessage.video(frames, "", turnId, mediaId, cacheDirectory)));
    }

    private static void completeVisionMediaEncoding(ChatMessage encoded) {
        mainHandler.post(() -> {
            for (int index = messages.size() - 1; index >= 0; --index) {
                ChatMessage current = messages.get(index);
                if (current.mediaId() != encoded.mediaId() || current.type() != encoded.type()) {
                    continue;
                }
                messages.set(index, current.withMediaFrom(encoded));
                MainActivity activity = currentActivity();
                if (activity != null && activity.chatAdapter != null) {
                    activity.chatAdapter.notifyItemChanged(index);
                }
                return;
            }
            deleteMessageMediaAsync(encoded);
        });
    }

    private static void deleteMessageMediaAsync(ChatMessage message) {
        if (message != null) {
            CHAT_MEDIA_EXECUTOR.execute(message::deleteMedia);
        }
    }

    // Delete the media backing a batch of removed messages in ONE background task, instead of submitting one
    // executor Runnable per message (avoids main-thread queue pressure when clearing/editing a long chat).
    private static void deleteMessagesMediaAsync(List<ChatMessage> removed) {
        if (removed == null || removed.isEmpty()) {
            return;
        }
        CHAT_MEDIA_EXECUTOR.execute(() -> {
            for (ChatMessage message : removed) {
                if (message != null) {
                    message.deleteMedia();
                }
            }
        });
    }

    // ==================== Vision timing ====================

    // Begin timing a vision turn without showing an in-progress caption; the snapshot bubble is updated only
    // when preprocessing or first-token timing completes.
    private void beginVisionTiming(int turnId, boolean video) {
        visionTimingPending = true;
        visionTimingIsVideo = video;
        visionTimingTurnId = turnId;
        visionSendElapsedMs = SystemClock.elapsedRealtime();
        if (video) {
            visionFrameCount = pendingVideoFrameCount > 0 ? pendingVideoFrameCount
                    : CameraService.getVideoNumFrames();
            visionFps = CameraService.getVideoFps();
            visionVideoSeconds = pendingVideoSeconds > 0f
                    ? pendingVideoSeconds
                    : (visionFps > 0f ? visionFrameCount / visionFps : 0f);
        } else {
            visionFrameCount = 0;
            visionFps = 0f;
            visionVideoSeconds = 0f;
        }
    }

    // Stamp the elapsed processing time when the first reply token lands (or the turn ends). Image fallback =
    // elapsed to first token when prewarm could not finish first; video = elapsed + RTF over the sampled window.
    private void captureVisionTimingIfPending() {
        if (!visionTimingPending) {
            return;
        }
        visionTimingPending = false;
        int turnId = visionTimingTurnId;
        visionTimingTurnId = ChatMessage.NO_TURN;
        float sec = (SystemClock.elapsedRealtime() - visionSendElapsedMs) / 1000f;
        if (visionTimingIsVideo) {
            float windowSec = visionVideoSeconds > 0f
                    ? visionVideoSeconds
                    : ((visionFps > 0f) ? visionFrameCount / visionFps : 0f);
            float rtf = (windowSec > 0f) ? sec / windowSec : 0f;
            updateVisionBubbleCaption(turnId, getString(R.string.vision_time_video, sec, rtf));
        } else {
            visionPreprocessedTurnId = turnId;
            updateVisionBubbleCaption(turnId, getString(R.string.vision_time_image, sec));
        }
    }

    private void updateVisionBubbleCaption(int turnId, String caption) {
        for (int i = messages.size() - 1; i >= 0; --i) {
            ChatMessage message = messages.get(i);
            if (message.turnId() == turnId &&
                    (message.type() == ChatMessage.TYPE_USER_IMAGE ||
                     message.type() == ChatMessage.TYPE_USER_VIDEO)) {
                messages.set(i, message.withContent(caption));
                if (chatAdapter != null) {
                    chatAdapter.notifyItemChanged(i);
                }
                scrollToBottom();
                return;
            }
        }
    }

    // ==================== Loaded model identity ====================

    // Model identity and APK byte size are computed once by the model-load worker; Activity recreation
    // only formats the immutable cached values.
    private void updateModelInfo() {
        if (modelInfoValue == null) {
            return;
        }
        modelInfoValue.setText(getString(R.string.model_info_summary,
                loadedModelDisplayName, formatBytes(loadedModelAssetBytes)));
    }

    private static String formatBytes(long bytes) {
        if (bytes <= 0L) {
            return "\u2014";
        }
        double gb = bytes / (1024.0 * 1024.0 * 1024.0);
        if (gb >= 1.0) {
            return String.format(Locale.US, "%.2f GB", gb);
        }
        double mb = bytes / (1024.0 * 1024.0);
        return String.format(Locale.US, "%.1f MB", mb);
    }

    private static double elapsedMilliseconds(long startNanos) {
        return (SystemClock.elapsedRealtimeNanos() - startNanos) / 1_000_000.0;
    }

    private void maybeRunIntentBenchmark(Intent intent) {
        if (intent == null || benchmarkRunning.get() || !intent.hasExtra("benchmark_workload")) {
            return;
        }
        final String workload = intent.getStringExtra("benchmark_workload");
        String decodedPrompt = intent.getStringExtra("benchmark_prompt");
        final String encodedPrompt = intent.getStringExtra("benchmark_prompt_base64");
        if (encodedPrompt != null && !encodedPrompt.isEmpty()) {
            try {
                decodedPrompt = new String(Base64.decode(encodedPrompt, Base64.DEFAULT), StandardCharsets.UTF_8);
            } catch (IllegalArgumentException error) {
                Log.e(BENCHMARK_TAG, "Invalid benchmark_prompt_base64", error);
                return;
            }
        }
        final String prompt = decodedPrompt;
        if (workload == null || workload.isEmpty() || prompt == null || prompt.isEmpty()) {
            Log.e(BENCHMARK_TAG, "Missing benchmark_workload or benchmark_prompt");
            return;
        }
        final String variant = intent.getStringExtra("benchmark_variant") == null
                ? "unspecified" : intent.getStringExtra("benchmark_variant");
        final String runId = intent.getStringExtra("benchmark_run_id") == null
                ? Long.toString(System.currentTimeMillis()) : intent.getStringExtra("benchmark_run_id");
        synchronized (BENCHMARK_CAPTURE_LOCK) {
            if (runId.equals(claimedBenchmarkRunId) || !benchmarkRunning.compareAndSet(false, true)) {
                return;
            }
            claimedBenchmarkRunId = runId;
        }
        final String outputName = intent.getStringExtra("benchmark_output") == null
                ? "qwen_v35_benchmark.jsonl" : intent.getStringExtra("benchmark_output");
        final int warmups = Math.max(0, intent.getIntExtra("benchmark_warmups", 2));
        final int iterations = Math.max(1, intent.getIntExtra("benchmark_iterations", 10));
        final int memoryTokens = Math.max(256, intent.getIntExtra("benchmark_memory_tokens", 4064));
        final int prefillTokens = Math.max(0, intent.getIntExtra("benchmark_prefill_tokens", 4064));
        final int decodeTokens = Math.max(2, intent.getIntExtra("benchmark_decode_tokens", 32));
        final boolean profile = intent.getBooleanExtra("benchmark_profile", false);
        final boolean finishAfter = intent.getBooleanExtra("benchmark_finish", false);

        new Thread(() -> runIntentBenchmark(
                runId, variant, workload, prompt, outputName, warmups, iterations,
            memoryTokens, prefillTokens, decodeTokens, profile, finishAfter), "intent-benchmark").start();
    }

    private void reportIntentBenchmarkLoadFailure(Intent intent, String error) {
        if (intent == null || !intent.hasExtra("benchmark_workload")) {
            return;
        }
        final String runId = intent.getStringExtra("benchmark_run_id") == null
                ? Long.toString(System.currentTimeMillis()) : intent.getStringExtra("benchmark_run_id");
        final String variant = intent.getStringExtra("benchmark_variant") == null
                ? "unspecified" : intent.getStringExtra("benchmark_variant");
        final String workload = intent.getStringExtra("benchmark_workload");
        final String outputName = intent.getStringExtra("benchmark_output") == null
                ? "qwen_v35_benchmark.jsonl" : intent.getStringExtra("benchmark_output");
        try {
            JSONObject failure = new JSONObject();
            failure.put("schema_version", 1);
            failure.put("record_type", "failure");
            failure.put("run_id", runId);
            failure.put("variant", variant);
            failure.put("workload", workload);
                failure.put("error", error == null
                    ? getString(R.string.model_load_failed_fallback) : error);
            failure.put("model_asset_stage_ms", modelAssetStageMs);
            failure.put("native_ort_load_ms", nativeOrtLoadMs);
            failure.put("vocab_asset_stage_ms", vocabAssetStageMs);
            failure.put("tokenizer_setup_ms", tokenizerSetupMs);
            failure.put("model_ready_ms", modelReadyMs);
            appendBenchmarkRecord(outputName, failure);
        } catch (Throwable reportError) {
            Log.e(BENCHMARK_TAG, "Could not write model-load benchmark failure", reportError);
        }
    }

    private void runIntentBenchmark(String runId, String variant, String workload, String prompt,
                                    String outputName, int warmups, int iterations,
                                    int memoryTokens, int prefillTokens, int decodeTokens,
                        boolean profile, boolean finishAfter) {
        GenerationService.startGeneration(getApplicationContext());
        try {
            if (!awaitModelConfigBarrier()) {
                throw new IllegalStateException("Model configuration did not complete");
            }
            Configure_LLM(DECODE_MODE_GREEDY, 1, 1, 1.0f, 20, 0.8f, 0.95f);
            Configure_Organize_Memory(false);
            Configure_Memory(memoryTokens, prefillTokens, decodeTokens);
            Set_System_Prompt(SYSTEM_PROMPT_DEFAULT);

            Runtime.getRuntime().gc();
            System.runFinalization();
            for (int iteration = -warmups; iteration < iterations; ++iteration) {
                Clear_Cache();
                final boolean profileIteration = profile && iteration == 0;
                if (profileIteration) {
                    File profileDirectory = getExternalFilesDir(null);
                    if (profileDirectory == null) {
                        throw new IOException("Profiling output directory is unavailable");
                    }
                    Configure_Run_Profiling(
                            new File(profileDirectory, "ort_profile_" + runId).getAbsolutePath(), true);
                }
                beginBenchmarkCapture();
                Debug.MemoryInfo beforeMemory = new Debug.MemoryInfo();
                Debug.getMemoryInfo(beforeMemory);
                long startNanos = SystemClock.elapsedRealtimeNanos();
                String runResult = Run_LLM(prompt, true, false, 1_000_000 + iteration + warmups);
                long endNanos = SystemClock.elapsedRealtimeNanos();
                String nativeCorrectness = Get_Last_Benchmark_Correctness();
                if (profileIteration) {
                    Configure_Run_Profiling("", false);
                }
                Debug.MemoryInfo afterMemory = new Debug.MemoryInfo();
                Debug.getMemoryInfo(afterMemory);
                BenchmarkCapture capture = endBenchmarkCapture();

                JSONObject record = new JSONObject();
                record.put("schema_version", 1);
                record.put("record_type", iteration < 0 ? "warmup" : "measurement");
                record.put("run_id", runId);
                record.put("variant", variant);
                record.put("workload", workload);
                record.put("iteration", iteration);
                record.put("warmup_iterations", warmups);
                record.put("measured_iterations", iterations);
                record.put("decode_limit_tokens", decodeTokens);
                record.put("memory_tokens", memoryTokens);
                record.put("prefill_limit_tokens", prefillTokens);
                record.put("profiled", profileIteration);
                record.put("model_asset_stage_ms", modelAssetStageMs);
                record.put("native_ort_load_ms", nativeOrtLoadMs);
                record.put("vocab_asset_stage_ms", vocabAssetStageMs);
                record.put("tokenizer_setup_ms", tokenizerSetupMs);
                record.put("model_ready_ms", modelReadyMs);
                record.put("prompt_utf8_bytes", prompt.getBytes(StandardCharsets.UTF_8).length);
                record.put("prompt_sha256", sha256(prompt));
                record.put("response_utf8_bytes", capture.output.getBytes(StandardCharsets.UTF_8).length);
                record.put("response_sha256", sha256(capture.output));
                record.put("elapsed_ms", (endNanos - startNanos) / 1_000_000.0);
                record.put("pss_before_kb", beforeMemory.getTotalPss());
                record.put("pss_after_kb", afterMemory.getTotalPss());
                record.put("native_result", runResult == null ? JSONObject.NULL : runResult);
                record.put("native_perf", capture.perfPayload);
                if (nativeCorrectness != null && !nativeCorrectness.isEmpty()) {
                    JSONObject correctness = new JSONObject(nativeCorrectness);
                    record.put("native_correctness", correctness);
                    if (correctness.has("token_ids")) {
                        record.put("token_ids_sha256", sha256(correctness.getJSONArray("token_ids").toString()));
                    }
                }
                addNativePerfFields(record, capture.perfPayload);
                appendBenchmarkRecord(outputName, record);
                if (iteration < 0) {
                    Log.i(BENCHMARK_TAG, "warmup_complete workload=" + workload
                            + " index=" + (iteration + warmups));
                }
            }

            JSONObject completion = new JSONObject();
            completion.put("schema_version", 1);
            completion.put("record_type", "complete");
            completion.put("run_id", runId);
            completion.put("variant", variant);
            completion.put("workload", workload);
            completion.put("measured_iterations", iterations);
            appendBenchmarkRecord(outputName, completion);
        } catch (Throwable error) {
            Configure_Run_Profiling("", false);
            Log.e(BENCHMARK_TAG, "Benchmark failed", error);
            try {
                JSONObject failure = new JSONObject();
                failure.put("schema_version", 1);
                failure.put("record_type", "failure");
                failure.put("run_id", runId);
                failure.put("variant", variant);
                failure.put("workload", workload);
                failure.put("error", error.toString());
                appendBenchmarkRecord(outputName, failure);
            } catch (Throwable reportError) {
                Log.e(BENCHMARK_TAG, "Could not write benchmark failure", reportError);
            }
        } finally {
            GenerationService.stop(getApplicationContext());
            benchmarkRunning.set(false);
            if (finishAfter) {
                runOnUiThread(this::finishAndRemoveTask);
            }
        }
    }

    private static void beginBenchmarkCapture() {
        synchronized (BENCHMARK_CAPTURE_LOCK) {
            benchmarkOutput.setLength(0);
            benchmarkPerfPayload = "";
            benchmarkCaptureActive = true;
        }
    }

    private static BenchmarkCapture endBenchmarkCapture() {
        synchronized (BENCHMARK_CAPTURE_LOCK) {
            benchmarkCaptureActive = false;
            return new BenchmarkCapture(benchmarkOutput.toString(), benchmarkPerfPayload);
        }
    }

    private static void addNativePerfFields(JSONObject record, String payload) throws Exception {
        if (payload == null || payload.isEmpty()) {
            return;
        }
        String[] fields = payload.split("\\|", -1);
        if (fields.length > 0 && !fields[0].isEmpty()) record.put("prefill_tokens_per_second", Double.parseDouble(fields[0]));
        if (fields.length > 1 && !fields[1].isEmpty()) record.put("decode_tokens_per_second", Double.parseDouble(fields[1]));
        if (fields.length > 2 && !fields[2].isEmpty()) record.put("memory_remaining_tokens", Integer.parseInt(fields[2]));
        if (fields.length > 3 && !fields[3].isEmpty()) record.put("memory_capacity_tokens", Integer.parseInt(fields[3]));
        if (fields.length > 4 && !fields[4].isEmpty()) record.put("prefill_tokens", Integer.parseInt(fields[4]));
        if (fields.length > 5 && !fields[5].isEmpty()) record.put("generated_tokens", Integer.parseInt(fields[5]));
    }

    private void appendBenchmarkRecord(String outputName, JSONObject record) throws IOException {
        File outputDirectory = getExternalFilesDir(null);
        if (outputDirectory == null || (!outputDirectory.isDirectory() && !outputDirectory.mkdirs())) {
            throw new IOException("Benchmark output directory is unavailable");
        }
        File outputFile = new File(outputDirectory, new File(outputName).getName());
        byte[] line = (record.toString() + "\n").getBytes(StandardCharsets.UTF_8);
        try (FileOutputStream stream = new FileOutputStream(outputFile, true)) {
            stream.write(line);
            stream.getFD().sync();
        }
        Log.i(BENCHMARK_TAG, record.toString());
    }

    private static String sha256(String text) throws NoSuchAlgorithmException {
        return sha256(text.getBytes(StandardCharsets.UTF_8));
    }

    private static String sha256(byte[] bytes) throws NoSuchAlgorithmException {
        return toHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    }

    private static String toHex(byte[] digest) {
        StringBuilder hex = new StringBuilder(digest.length * 2);
        for (byte value : digest) {
            hex.append(String.format(Locale.ROOT, "%02x", value & 0xff));
        }
        return hex.toString();
    }

    private static final class AssetManifest {
        final Map<String, String> hashes;
        final String fingerprint;

        AssetManifest(Map<String, String> hashes, String fingerprint) {
            this.hashes = hashes;
            this.fingerprint = fingerprint;
        }
    }

    private static final class BenchmarkCapture {
        final String output;
        final String perfPayload;

        BenchmarkCapture(String output, String perfPayload) {
            this.output = output;
            this.perfPayload = perfPayload;
        }
    }

    // ==================== Native methods (JNI) ====================

    private static native boolean Pre_Process();
    // Load ONNX models. assetManager null = low-memory mode (external data format). Returns true on success.
    private static native boolean Load_Models_A(AssetManager assetManager, String assetDirectory,
                                                String cacheDirectory, String storageDirectory,
                                                int EP_TYPE, boolean LOW_MEMORY_MODE);
    private static native boolean Ensure_Vision_Models();
    private static native boolean Vision_Models_Ready();
    private static native boolean Configure_Runtime_Pressure(int PRESSURE_LEVEL);
    private static native boolean Trim_Runtime(boolean RELEASE_VISION);
    // Runs prefill + the full decode loop in C++, streaming via onTokenStream(); returns final perf stats or
    // a structured LLM_ERROR/LLM_STATUS result. CLEAR resets context; TURN_ID checkpoints KV.
    private static native String Run_LLM(String QUERY, boolean CLEAR, boolean USE_THINK, int TURN_ID);
    private static native String Get_Last_Benchmark_Correctness();
    // Rolls the KV cache back to before TURN_ID's prompt (dropping it + later turns) so an edited bubble can
    // regenerate. Returns false (and clears history) when no longer rollable.
    private static native boolean Rollback_LLM(int TURN_ID);
    // Runtime decode-strategy: beam when enabled with TOP_K/BEAM_SIZE >= 2, greedy for penalty, else argmax.
    // Values clamped natively to the exported graph limits.
    private static native boolean Configure_LLM(int DECODE_MODE, int TOP_K, int BEAM_SIZE,
                                                float REPEAT_PENALTY, int PENALTY_RANGE,
                                                float TEMPERATURE, float TOP_P);
    // Replaces the slot system prompt (next turn). Organize Memory ON strips its block during clean recompute.
    private static native boolean Set_System_Prompt(String SYSTEM_PROMPT);
    // Runtime memory profile (retained memory, prefill window, per-turn decode cap); snapshotted at turn start.
    private static native boolean Configure_Memory(int MEMORY_TOKENS, int PREFILL_TOKENS, int DECODE_TOKENS);
    // Organize Memory rebuilds clean cross-turn text and replaces visual state with note-backed history. Next turn.
    private static native boolean Configure_Organize_Memory(boolean ENABLE);
    // Enables ORT's run-level profiler on every loaded model. The profile prefix is made model-specific natively.
    private static native boolean Configure_Run_Profiling(String PROFILE_PREFIX, boolean ENABLE);
    // Returns min/max/default triples for Memory, Prefill, Decode Limit (derived natively from max_seq_len).
    private static native int[] Get_Memory_Limits();
    // True only when the loaded architecture can split/reattach KV without recurrent linear state.
    private static native boolean Supports_Prefill_Lookback();
    // True only when the model's official chat template supports an explicit thinking prefix.
    private static native boolean Supports_Thinking();
    // Live-tunable hysteresis band (% of cap): RED = rebuild trigger, GREEN = post-reset target (green < red
    // enforced natively). Get returns {red, green}.
    private static native boolean Configure_Memory_Thresholds(int RED_PERCENT, int GREEN_PERCENT);
    private static native int[] Get_Memory_Thresholds();
    // Returns {used, remaining, remainingPercent, capacity} for the retained cross-turn memory cache.
    private static native int[] Get_Memory_Stats();
    // Cooperatively cancels an in-flight Run_LLM. MANUAL=true (user Stop) also tags the reply with the
    // manual-stop notice; internal cancels pass false.
    private static native void Stop_LLM(boolean MANUAL);
    // Panic reset for Clear: aborts any in-flight 整理记忆 commit + drops the KV cache immediately. Call AFTER
    // Stop_LLM(false) + joining the worker, off the UI thread (it can briefly wait out a running ORT op).
    private static native void Clear_Cache();
    // Best-effort background ViT pre-encode for the pending capture.
    private static native boolean Prewarm_Vision();
}
