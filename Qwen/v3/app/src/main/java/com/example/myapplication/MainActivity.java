package com.example.myapplication;

import android.animation.ObjectAnimator;
import android.animation.PropertyValuesHolder;
import android.animation.ValueAnimator;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.res.AssetManager;
import android.content.res.Configuration;
import android.graphics.Color;
import android.graphics.drawable.ColorDrawable;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.os.SystemClock;
import android.system.Os;
import android.system.OsConstants;
import android.text.Editable;
import android.text.TextWatcher;
import android.transition.AutoTransition;
import android.transition.TransitionManager;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.view.WindowManager;
import android.view.animation.AccelerateDecelerateInterpolator;
import android.view.animation.LinearInterpolator;
import android.widget.Button;
import android.widget.EditText;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.ToggleButton;
import android.widget.Toast;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.app.AppCompatDelegate;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.example.myapplication.databinding.ActivityMainBinding;
import com.google.android.material.materialswitch.MaterialSwitch;
import com.google.android.material.slider.Slider;
import com.google.android.material.textfield.TextInputEditText;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

public class MainActivity extends AppCompatActivity {
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
    // Decode-strategy sliders wired to native Configure_LLM; defaults (1, 1, 1.00) = pure argmax.
    private Slider beamSlider;
    private Slider topKSlider;
    private Slider penaltySlider;
    private Slider memorySlider;
    private Slider prefillSlider;
    private Slider decodeLimitSlider;
    private MaterialSwitch organizeMemorySwitch;
    private TextView beamValue;
    private TextView topKValue;
    private TextView penaltyValue;
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
    private View memoryRemainingTrack;
    private View memoryThresholdGreen;
    private View memoryThresholdRed;
    private TextView memoryBandValue;
    private int memoryGreenPercent = 60;
    private int memoryRedPercent = 95;
    private int draggingThreshold = 0;   // 0 = none, 1 = green target, 2 = red trigger
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
    // Clear stays live. UI-thread only.
    private boolean postProcessingActive = false;
    // True while a reply is generating (mirrors the Send<->Stop swap); greys + tap-guards Compute. UI-thread only.
    private boolean generatingActive = false;
    // Live system monitor: app CPU% + resident RAM, sampled ~1 Hz on a background thread via /proc/self/{stat,statm}.
    // CPU% = jiffy delta (utime+stime) over wall-clock, normalised by core count, clamped 0..100.
    private TextView cpuValue;
    private TextView memValue;
    private HandlerThread monitorThread;
    private Handler monitorHandler;
    private volatile boolean monitorActive = false;
    private long lastCpuJiffies = -1L;   // previous (utime+stime); -1 == no baseline captured yet
    private long lastCpuWallMs = 0L;     // wall clock (ms) of the previous CPU sample
    private final int cpuCount = Math.max(1, Runtime.getRuntime().availableProcessors());
    private long clockTicksPerSec = 100L;   // _SC_CLK_TCK (USER_HZ); resolved from sysconf in onCreate
    private long pageSizeBytes = 4096L;     // _SC_PAGESIZE; resolved from sysconf in onCreate
    private static final long MONITOR_INTERVAL_MS = 1000L;
    // Pre-compiled /proc field splitter, reused by the 1 Hz sampler.
    private static final Pattern WHITESPACE = Pattern.compile("\\s+");
    // Reused slots for the native "prefill|decode|remaining|capacity[|prefillTokens|decodeTokens]" payload
    // (UI-thread only, so one buffer avoids a split + alloc per refresh).
    private final String[] perfFields = new String[6];
    // Collapsible decode-strategy slider container, toggled by the top-bar gear; collapsed by default.
    private View decodeContent;
    private View jumpToBottomButton;
    private boolean autoScrollLocked = false;
    // Header ambient-glow views + "breathing" animators (decorative; safe to be null). Started onResume, cancelled onPause.
    private View avatarGlow;
    private View statusDot;
    private ObjectAnimator avatarPulse;
    private ObjectAnimator statusPulse;
    // Mirrors the latest beam toggle. Beam decodes silently so it needs the "typing" placeholder.
    private boolean useBeamSearch = false;
    // Mirrors the "organize memory" (整理记忆) switch; default off. On = native strips the system prompt
    // from cross-turn memory.
    private boolean organizeMemoryEnabled = false;
    private RecyclerView answerView;
    private ChatAdapter chatAdapter;
    private static List<ChatMessage> messages;
    private static String usrInputText = "";
    private static WeakReference<MainActivity> activeActivity = new WeakReference<>(null);
    // Shared main-thread handler for native callbacks + UI posts; removeCallbacksAndMessages(null) drops
    // stale generation callbacks after a reset.
    private static final Handler mainHandler = new Handler(Looper.getMainLooper());
    // Reused accumulation buffer for the streaming reply bubble (appends in place, cutting GC churn).
    // streamIndex ties it to the mirrored row; a mismatch reseeds it. Main-thread confined.
    private static final StringBuilder streamBuilder = new StringBuilder(512);
    private static int streamIndex = -1;
    // Current generation worker; joined before a new one starts (native is a single non-reentrant context).
    private LLMThread llmThread;
    // Monotonic per-turn id (never reused); passed to Run_LLM to checkpoint the KV cache, used by Rollback_LLM.
    private int nextTurnId = 0;
    private int usrTurnId = ChatMessage.NO_TURN;
    private static final String VOCAB_FILE_NAME = "vocab_Qwen.txt";
    // Editor "Reset" default. MUST match native DEFAULT_SYSTEM_PROMPT (user_settings.h).
    private static final String SYSTEM_PROMPT_DEFAULT = "You are a helpful assistant.";
    private static final String first_talk = "请输入问题 Enter Questions";
    private static final String load_failed = "模型加载失败。\nModel loading failed.";
    private static final String over_inputs = "一次输入太多单词 \nInput too many words at once.";
    private static final String low_memory_mode_error = "未在 assets 中找到 .onnx 模型文件。\nNo .onnx model files found in assets.";
    // Multi-turn chat: history persists across turns; Clear sets this so the NEXT Run_LLM resets native context once.
    private static boolean clear_flag = false;
    private static boolean chatting = false;
    // Models + tokenizer load ONCE per process; this guard stops a theme-switch recreate from reloading them
    // (native global state + the static messages list survive the recreate).
    private static boolean modelsReady = false;
    // True while the background model-load thread runs; stops a theme-switch recreate from starting a 2nd load.
    private static volatile boolean modelsLoading = false;

    private static final int EP_CPU = 0;
    // Bump when a bundled .onnx/vocab asset changes to force-re-stage the cache once (else a stale cached graph shadows it).
    private static final int ASSET_STAGE_VERSION = 2;
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
    private int memoryRemainingTokens = DEFAULT_MEMORY_TOKENS_FALLBACK;
    private int memoryCapacityTokens = DEFAULT_MEMORY_TOKENS_FALLBACK;
    private double memoryRemainingPercent = 100.0;
    private int memoryStatsRequestId = 0;
    private float memoryBarFraction = 0.0f;   // used fraction; starts empty (fresh memory)
    private int lastMemoryBarColor = 0;
    private int lastDisplayedMemoryUsedTokens = -1;
    private int lastDisplayedMemoryCapacityTokens = -1;
    private ValueAnimator memoryBarAnimator;
    private ObjectAnimator memoryGlowAnimator;

    // ==================== Activity lifecycle ====================

    static {
        System.loadLibrary("myapplication");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(ActivityMainBinding.inflate(getLayoutInflater()).getRoot());
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
        setupDecodeControls();
        perfPrefillValue = findViewById(R.id.perf_prefill_value);
        perfDecodeValue = findViewById(R.id.perf_decode_value);
        postProcessingStatus = findViewById(R.id.post_processing_status);
        setPostProcessingActive(false);
        cpuValue = findViewById(R.id.perf_cpu_value);
        memValue = findViewById(R.id.perf_mem_value);
        setupSystemMonitor();
        setupPerfChartCells();
        avatarGlow = findViewById(R.id.avatar_glow);
        statusDot = findViewById(R.id.status_dot);
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
        findViewById(R.id.btn_system_prompt).setOnClickListener(v -> onEditSystemPrompt());
        // The decoder is a split pipeline of memory-mapped .onnx graphs, so stage every graph + external-data
        // sibling into the cache. Load_Models_A(null,...) resolves by path.
        if (!modelsReady && !modelsLoading) {
            // Stage assets + load graphs + tokenizer OFF the UI thread (multi-second I/O would ANR).
            modelsLoading = true;
            sendButton.setEnabled(false);
            // Keep the transcript empty while loading; TYPE_LOADING is reserved for beam replies.
            final AssetManager bgMgr = getAssets();
            new Thread(() -> {
                SharedPreferences stagePrefs = getSharedPreferences(App.PREFS, MODE_PRIVATE);
                boolean restageAssets = stagePrefs.getInt(App.KEY_ASSET_STAGE_VERSION, 0) < ASSET_STAGE_VERSION;
                byte[] copyBuffer = new byte[102400];
                boolean ok = false;
                String error = null;
                try {
                    String[] files = bgMgr.list("");
                    int count_file = 0;
                    if (files != null) {
                        for (String fileName : files) {
                            String[] subEntries = bgMgr.list(fileName);
                            if (subEntries == null || subEntries.length == 0) {
                                if (fileName.endsWith(".onnx") || fileName.endsWith(".data")) {
                                    Copy_from_Asset_to_Cache(fileName, bgMgr, copyBuffer, restageAssets);
                                    if (fileName.endsWith(".onnx")) {
                                        count_file += 1;
                                    }
                                }
                            }
                        }
                    }
                    if (count_file == 0) {
                        error = low_memory_mode_error;
                    } else if (Load_Models_A(null, EP_CPU, true)) {
                        Copy_from_Asset_to_Cache(VOCAB_FILE_NAME, bgMgr, copyBuffer, restageAssets);
                        Pre_Process();
                        ok = true;
                        if (restageAssets) {
                            stagePrefs.edit().putInt(App.KEY_ASSET_STAGE_VERSION, ASSET_STAGE_VERSION).apply();
                        }
                    } else {
                        error = load_failed;
                    }
                } catch (Throwable t) {
                    error = low_memory_mode_error;
                }
                final boolean loaded = ok;
                final String err = error;
                mainHandler.post(() -> {
                    modelsLoading = false;
                    if (loaded) {
                        modelsReady = true;
                        applySavedSystemPrompt();   // push any user-saved slot prompt (else native default stands)
                        refreshMemoryControlsFromNative();   // max_seq_len is known now; shrink slider ranges to it
                        Start_Chat();
                        sendButton.setEnabled(true);
                        refreshMemoryStatsAsync();
                    } else {
                        addHistory(ChatMessage.TYPE_SERVER, err);
                    }
                });
            }, "model-load").start();
        }
        // Re-wire the send button on every view-tree (re)build (e.g. theme switch), once models are loaded.
        if (modelsReady) {
            Start_Chat();
            refreshMemoryStatsAsync();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        startAmbientAnimations();
        startMonitoring();       // CPU/RAM are always shown while this screen is visible
    }

    @Override
    protected void onPause() {
        stopMonitoring();        // never sample while backgrounded
        stopAmbientAnimations();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        monitorActive = false;
        if (monitorHandler != null) {
            monitorHandler.removeCallbacksAndMessages(null);
        }
        if (monitorThread != null) {
            monitorThread.quitSafely();
            monitorThread = null;
        }
        if (activeActivity.get() == this) {
            activeActivity = new WeakReference<>(null);
        }
        super.onDestroy();
    }

    private static MainActivity currentActivity() {
        return activeActivity.get();
    }

    // ==================== Ambient header animations ====================

    // Always-on "breathing" glow for the avatar aura + status dot; cancelled in onPause.
    private void startAmbientAnimations() {
        if (avatarPulse == null) {
            avatarPulse = buildPulse(avatarGlow, 0.40f, 0.95f, 0.90f, 1.12f, 2200L);
        }
        if (statusPulse == null) {
            statusPulse = buildPulse(statusDot, 0.35f, 1.00f, 0.75f, 1.30f, 1500L);
        }
    }

    private void stopAmbientAnimations() {
        if (avatarPulse != null) {
            avatarPulse.cancel();
            avatarPulse = null;
        }
        if (statusPulse != null) {
            statusPulse.cancel();
            statusPulse = null;
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
        try {
            long tck = Os.sysconf(OsConstants._SC_CLK_TCK);
            if (tck > 0L) {
                clockTicksPerSec = tck;
            }
            long pageSize = Os.sysconf(OsConstants._SC_PAGESIZE);
            if (pageSize > 0L) {
                pageSizeBytes = pageSize;
            }
        } catch (Throwable ignored) {
            // Keep the safe defaults (100 Hz / 4096 B) if sysconf is unavailable on this device.
        }
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

    // (Re)start sampling: clear the monitor queue (idempotent vs resume), reset the CPU baseline, sample now.
    private void startMonitoring() {
        if (monitorHandler == null) {
            return;
        }
        monitorActive = true;
        monitorHandler.removeCallbacksAndMessages(null);
        monitorHandler.post(() -> {
            lastCpuJiffies = -1L;
            sampleTask.run();
        });
    }

    private void stopMonitoring() {
        monitorActive = false;
        if (monitorHandler != null) {
            monitorHandler.removeCallbacksAndMessages(null);
        }
    }

    // One monitor tick: CPU% from the jiffies delta over wall-clock (norm by cores, clamped 0..100) + RAM MB.
    // First tick only sets the CPU baseline (shows "measuring"). lastCpu* are thread-confined.
    private void sampleSystemUsage() {
        long nowMs = SystemClock.elapsedRealtime();
        long jiffies = readProcessCpuJiffies();
        long rssBytes = readProcessRssBytes();
        final String cpuText;
        if (jiffies >= 0L && lastCpuJiffies >= 0L && nowMs > lastCpuWallMs) {
            double procSeconds = (jiffies - lastCpuJiffies) / (double) clockTicksPerSec;
            double wallSeconds = (nowMs - lastCpuWallMs) / 1000.0;
            double percent = procSeconds / wallSeconds * 100.0 / cpuCount;
            if (percent < 0.0) {
                percent = 0.0;
            } else if (percent > 100.0) {
                percent = 100.0;
            }
            cpuText = getString(R.string.perf_cpu_value, (int) Math.round(percent));
            MetricsHistory.get().addCpu(nowMs, (float) percent);
        } else {
            cpuText = getString(R.string.perf_measuring);
        }
        if (jiffies >= 0L) {
            lastCpuJiffies = jiffies;
            lastCpuWallMs = nowMs;
        }
        final String memText = rssBytes >= 0L
                ? getString(R.string.perf_mem_value, rssBytes / (1024L * 1024L))
                : getString(R.string.perf_pending);
        if (rssBytes >= 0L) {
            MetricsHistory.get().addMemory(nowMs, rssBytes / (1024.0f * 1024.0f));
        }
        mainHandler.post(() -> {
            if (monitorActive) {
                cpuValue.setText(cpuText);
                memValue.setText(memText);
            }
        });
    }

    // User+system CPU jiffies (utime+stime from /proc/self/stat), or -1 on error. comm (field 2) can hold
    // spaces/parens, so parse AFTER the last ')': utime = token 11, stime = token 12.
    private long readProcessCpuJiffies() {
        try (BufferedReader reader = new BufferedReader(new FileReader("/proc/self/stat"))) {
            String line = reader.readLine();
            if (line == null) {
                return -1L;
            }
            int afterComm = line.lastIndexOf(')');
            if (afterComm < 0 || afterComm + 2 >= line.length()) {
                return -1L;
            }
            String[] fields = WHITESPACE.split(line.substring(afterComm + 2));
            if (fields.length < 13) {
                return -1L;
            }
            return Long.parseLong(fields[11]) + Long.parseLong(fields[12]);
        } catch (Throwable t) {
            return -1L;
        }
    }

    // This process's RSS in bytes (field 2 of /proc/self/statm, in pages), or -1 on error. RSS includes the
    // mmap'd ONNX weights, so it reflects the LLM's real on-device footprint.
    private long readProcessRssBytes() {
        try (BufferedReader reader = new BufferedReader(new FileReader("/proc/self/statm"))) {
            String line = reader.readLine();
            if (line == null) {
                return -1L;
            }
            String[] fields = WHITESPACE.split(line);
            if (fields.length < 2) {
                return -1L;
            }
            return Long.parseLong(fields[1]) * pageSizeBytes;
        } catch (Throwable t) {
            return -1L;
        }
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

    // Route the native "prefill|decode[|remaining|capacity[|prefillTokens|decodeTokens]]" payload into the
    // perf panel: tok/s text, optional live-memory fields, throughput pins. The final (both-rate) payload is
    // skipped for pins to avoid a double-record.
    private void updatePerfStats(String payload) {
        int fieldCount = splitPerfPayload(payload);
        if (fieldCount < 2) {
            return;
        }
        String prefill = perfFields[0];
        String decode = perfFields[1];
        if (perfPrefillValue != null && !prefill.isEmpty()) {
            perfPrefillValue.setText(getString(R.string.perf_token_rate, prefill));
        }
        if (perfDecodeValue != null && !decode.isEmpty()) {
            perfDecodeValue.setText(getString(R.string.perf_token_rate, decode));
        }
        if (fieldCount >= 4 && !perfFields[2].isEmpty() && !perfFields[3].isEmpty()) {
            try {
                int remaining = Integer.parseInt(perfFields[2]);
                int capacity = Integer.parseInt(perfFields[3]);
                if (capacity > 0) {
                    applyMemoryStatsFromValues(capacity - remaining, remaining, capacity);
                }
            } catch (NumberFormatException ignored) {
                // Keep any valid throughput update even if live-memory fields are malformed.
            }
        }
        if (fieldCount >= 6) {
            // Prefill pin (fill amount, tok/s): only the prefill-complete payload has a prefill rate + empty decode.
            if (!prefill.isEmpty() && decode.isEmpty() && !perfFields[4].isEmpty()) {
                recordThroughputPoint(true, prefill, perfFields[4]);
            }
            // Decode pin (decode amount, live tok/s): only a decode-live payload has a decode rate + empty prefill.
            if (!decode.isEmpty() && prefill.isEmpty() && !perfFields[5].isEmpty()) {
                recordThroughputPoint(false, decode, perfFields[5]);
            }
        }
    }

    private int splitPerfPayload(String payload) {
        int fieldCount = 0;
        int start = 0;
        int length = payload.length();
        while (fieldCount < perfFields.length) {
            int sep = payload.indexOf('|', start);
            int end = sep < 0 ? length : sep;
            perfFields[fieldCount++] = payload.substring(start, end);
            if (sep < 0) {
                break;
            }
            start = sep + 1;
        }
        for (int i = fieldCount; i < perfFields.length; i++) {
            perfFields[i] = "";
        }
        return fieldCount;
    }

    // Record ONE throughput scatter pin as measured -- X = token total, Y = tok/s -- no bucketing/fitting.
    private void recordThroughputPoint(boolean isPrefill, String rate, String tokens) {
        try {
            float r = Float.parseFloat(rate);
            float tokenTotal = Float.parseFloat(tokens);
            if (r <= 0f || tokenTotal <= 0f) {
                return;
            }
            if (isPrefill) {
                MetricsHistory.get().addPrefillPoint(tokenTotal, r);
            } else {
                MetricsHistory.get().addDecodePoint(tokenTotal, r);
            }
        } catch (NumberFormatException ignored) {
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
        mainHandler.post(() -> {
            MainActivity activity = currentActivity();
            if (chatting && activity != null) {
                activity.removeLoadingBubble();   // real output is arriving; drop the typing placeholder
                activity.addStreamingServerText(text);
            }
        });
    }

    // C++ -> Java live throughput stats. Empty fields allow independent updates: "<prefill>|" (prefill done),
    // "|<decode>" (live decode), "<prefill>|<decode>" (final). Gated like tokens.
    private static void onPerfStats(String payload) {
        if (payload == null || payload.indexOf('|') < 0) {
            return;
        }
        mainHandler.post(() -> {
            MainActivity activity = currentActivity();
            if (!chatting || activity == null || activity.perfPrefillValue == null || activity.perfDecodeValue == null) {
                return;
            }
            activity.updatePerfStats(payload);
        });
    }

    private static void onPostProcessingState(boolean active) {
        MainActivity activity = currentActivity();
        if (activity == null) {
            return;
        }
        activity.runOnUiThread(() -> {
            MainActivity current = currentActivity();
            if (current != null) {
                current.setPostProcessingActive(active);
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

    private void addStreamingServerText(String result) {
        addHistory(ChatMessage.TYPE_SERVER, result, ChatMessage.NO_TURN, false);
    }

    private void addHistory(int messageType, String result, int turnId, boolean markdownReady) {
        int lastMessageIndex = messages.size() - 1;
        if (lastMessageIndex >= 0 && messages.get(lastMessageIndex).type() == messageType) {
            // Append to the current bubble and rebind ONLY that row, so streaming doesn't re-bind the whole list.
            ChatMessage previous = messages.get(lastMessageIndex);
            if (streamIndex != lastMessageIndex) {
                // First append onto this row: reseed the buffer from the existing text (byte-identical to a concat).
                streamBuilder.setLength(0);
                streamBuilder.append(previous.content());
                streamIndex = lastMessageIndex;
            }
            streamBuilder.append(result);
            boolean ready = previous.markdownReady() && markdownReady;
            messages.set(lastMessageIndex, new ChatMessage(messageType, streamBuilder.toString(), previous.turnId(), ready));
            if (chatAdapter != null) {
                chatAdapter.notifyItemChanged(lastMessageIndex);
            }
            scrollToBottom();
        } else {
            messages.add(new ChatMessage(messageType, result, turnId, markdownReady));
            int newIndex = messages.size() - 1;
            // Seed the reuse buffer for this freshly-inserted row so later batches append in place.
            streamBuilder.setLength(0);
            streamBuilder.append(result);
            streamIndex = newIndex;
            if (chatAdapter != null) {
                chatAdapter.notifyItemInserted(newIndex);
            }
            scrollToBottom();
        }
    }

    private void finalizeStreamingMarkdown() {
        for (int i = messages.size() - 1; i >= 0; --i) {
            ChatMessage message = messages.get(i);
            if (message.type() == ChatMessage.TYPE_SERVER) {
                if (!message.markdownReady()) {
                    messages.set(i, message.withMarkdownReady(true));
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
        setControlsLockedForPostProcessing(active);
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

    // While native organises memory, grey + tap-guard Send/Test/Theme. Clear is EXCLUDED (panic escape).
    // Buttons stay clickable so a tap surfaces a hint; the action is gated by guardPostProcessing().
    private void setControlsLockedForPostProcessing(boolean locked) {
        postProcessingActive = locked;
        float alpha = locked ? 0.45f : 1.0f;
        if (sendButton != null) {
            sendButton.setAlpha(alpha);
        }
        updateComputeButtonLock();   // test button also stays locked while a reply is generating
        if (themeToggle != null) {
            themeToggle.setAlpha(alpha);
        }
    }

    // Grey the Compute button when EITHER lock is set (postProcessing or generating). Null-safe before assign.
    private void updateComputeButtonLock() {
        if (computeButton != null) {
            computeButton.setAlpha((postProcessingActive || generatingActive) ? 0.45f : 1.0f);
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
        if (generatingActive) {
            showToast(this, getString(R.string.inference_lock_hint), false);
            return true;
        }
        return false;
    }

    // ==================== Clear / reset ====================

    @SuppressLint("NotifyDataSetChanged")
    private void clearHistory(){
        // PANIC RESET: forcibly stop the in-flight reply AND any 整理记忆 commit, then reinitialise. UI reset
        // is synchronous; the native teardown runs on a bg thread (it can briefly block on a running ORT op).
        requestStopGeneration(false);    // cancel the decode loop (the worker bails within a token or two)
        setGenerating(false);            // restore Send; the partial reply (if any) is abandoned
        setPostProcessingActive(false);  // drop the "整理记忆" chip + un-grey the locked controls immediately
        chatting = false;                // gate out the cancelled worker's remaining streamed-token UI posts
        mainHandler.removeCallbacksAndMessages(null);
        inputBox.setText("");
        usrInputText = "";
        messages.clear();
        streamIndex = -1;   // invalidate the streaming reuse buffer; the list was structurally reset
        autoScrollLocked = false;
        updateJumpToBottomVisibility();
        chatAdapter.notifyDataSetChanged();
        chatAdapter.resetEntranceAnimation();
        chatAdapter.clearThinkExpandState();
        answerView.smoothScrollToPosition(0);
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
        new Thread(() -> {
            if (workerToReap != null) {
                try {
                    workerToReap.join();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
            Clear_Cache();
            MainActivity activity = currentActivity();
            if (activity != null) {
                mainHandler.post(activity::refreshMemoryStatsAsync);
            }
        }, "clear-reset").start();
        showToast(MainActivity.this, "已清除 Cleared",false);
    }

    // ==================== Generation orchestration ====================

    private void Start_Chat() {
        sendButton.setOnClickListener(view -> {
            if (guardPostProcessing()) {
                return;
            }
            usrInputText = String.valueOf(inputBox.getText());
            if (usrInputText.isEmpty()){
                showToast(MainActivity.this, first_talk,false);
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
        chatting = true;
        setGenerating(true);           // swap Send -> Stop so the user can halt this reply
        resetPerfStats();              // show the "measuring" placeholder until this turn's numbers land
        llmThread = new LLMThread();   // snapshots usrInputText / think_mode / clear_flag / usrTurnId
        clear_flag = false;            // the pending clear (if any) is now owned by this generation
        llmThread.start();
    }

    // Swap the composer button between Send (idle) and Stop (generating), so a long reply can be halted
    // without Clearing the whole chat.
    private void setGenerating(boolean generating) {
        generatingActive = generating;
        if (sendButton != null) {
            sendButton.setVisibility(generating ? View.GONE : View.VISIBLE);
        }
        if (stopButton != null) {
            stopButton.setVisibility(generating ? View.VISIBLE : View.GONE);
        }
        updateComputeButtonLock();   // grey + tap-guard the test button for the duration of the reply
    }

    private void requestStopGeneration(boolean manual) {
        LLMThread worker = llmThread;
        if (worker == null || !worker.shouldForwardStop()) {
            return;
        }
        Stop_LLM(manual);
    }

    // Stop + join the active worker and drop its queued UI posts, leaving native idle so the caller can
    // start a new generation or mutate the KV cache (e.g. edit rollback).
    private void cancelOngoingGeneration() {
        LLMThread worker = llmThread;
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
        mainHandler.removeCallbacksAndMessages(null);
    }

    private class LLMThread extends Thread {
        // Snapshot per-generation inputs on the UI thread at construction so the worker never races the UI.
        private final String query = usrInputText;
        private final boolean think_mode = thinkButton.isChecked();
        private final boolean clear = clear_flag;
        private final int turnId = usrTurnId;
        private volatile boolean nativeFinished = false;

        boolean shouldForwardStop() {
            return isAlive() && !nativeFinished;
        }

        @Override
        public void run() {
            // ONE native call runs prefill + the whole decode loop, streaming via onTokenStream(). Returns
            // "Over_Inputs" or a "<prefill>|<decode>" tok/s payload.
            String runResult = "";
            try {
                runResult = Run_LLM(query, clear, think_mode, turnId);
            } finally {
                nativeFinished = true;
            }
            final String result = runResult;
            final LLMThread finishedThread = this;
            mainHandler.post(() -> {
                // A theme switch mid-generation recreates the Activity, so target the CURRENT active one
                // (weak ref), not the destroyed launcher.
                MainActivity activity = currentActivity();
                if (chatting && activity != null) {
                    activity.removeLoadingBubble();   // safety: beam may have streamed nothing (Over_Inputs/empty)
                    if ("Over_Inputs".equals(result)) {
                        activity.addHistory(ChatMessage.TYPE_SERVER, over_inputs);
                    } else if (result != null && result.indexOf('|') >= 0) {
                        activity.finalizeStreamingMarkdown();
                        activity.updatePerfStats(result);   // throughput readout -> its own window, off the bubble
                    }
                }
                chatting = false;
                if (activity != null) {
                    activity.setGenerating(false);   // back to Send; reply finished (or was manually stopped)
                    activity.refreshMemoryStatsAsync();
                }
                if (llmThread == finishedThread) {
                    llmThread = null;
                }
            });
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
        // UI first (instant feedback): gate stale posts, drop this turn + later, reseed, show edited bubble + placeholder.
        chatting = false;
        int removeCount = messages.size() - position;
        for (int i = messages.size() - 1; i >= position; i--) {
            messages.remove(i);
        }
        chatAdapter.notifyItemRangeRemoved(position, removeCount);
        chatAdapter.clearThinkExpandState();
        streamIndex = -1;   // rows shifted; force the streaming buffer to reseed on the next append
        autoScrollLocked = false;
        updateJumpToBottomVisibility();
        inputBox.setText("");
        final int newTurnId = nextTurnId++;
        addHistory(ChatMessage.TYPE_USER, editedText, newTurnId);
        showLoadingBubble();   // placeholder until the rollback completes and the reply starts streaming
        // Stop+join the worker and roll back the KV cache OFF the UI thread (both can block), then launch the
        // new generation once the context is idle + rewound.
        new Thread(() -> {
            cancelOngoingGeneration();   // Stop_LLM + join + drop stale posts: native context now free
            Rollback_LLM(turnId);        // KV rewind (fast KV_Slice, or re-prefill fallback) off the UI thread
            final int[] memoryStats = Get_Memory_Stats();
            mainHandler.post(() -> {
                applyMemoryStats(memoryStats);
                // User bubble + placeholder already shown; just launch the worker (fields read at construction).
                usrInputText = editedText;
                usrTurnId = newTurnId;
                startLLM();
            });
        }, "edit-rollback").start();
    }

    // ==================== System prompt editor ====================

    // Slot system-prompt editor (multiline AlertDialog). Save persists+applies (empty disables), Reset restores
    // the default. Takes effect NEXT turn (no retro-edit of cached history).
    private void onEditSystemPrompt() {
        if (!modelsReady) {
            return;   // tokenizer not ready yet (Pre_Process runs during the model load)
        }
        View content = getLayoutInflater().inflate(R.layout.dialog_system_prompt, null, false);
        TextInputEditText editText = content.findViewById(R.id.systemPromptInput);
        TextView counter = content.findViewById(R.id.systemPromptCounter);
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
        counter.setText(getString(R.string.system_prompt_counter, text == null ? 0 : text.length()));
    }

    // Persist the slot prompt + push to native (next turn). Empty disables the slot.
    private void saveSystemPrompt(String prompt) {
        getSharedPreferences(App.PREFS, MODE_PRIVATE).edit()
                .putString(App.KEY_SYSTEM_PROMPT, prompt).apply();
        Set_System_Prompt(prompt);
        showToast(MainActivity.this, getString(R.string.system_prompt_saved), false);
    }

    // Push a saved slot prompt to native after the tokenizer is ready (once after Pre_Process); else the
    // native default stands.
    private void applySavedSystemPrompt() {
        SharedPreferences prefs = getSharedPreferences(App.PREFS, MODE_PRIVATE);
        if (prefs.contains(App.KEY_SYSTEM_PROMPT)) {
            Set_System_Prompt(prefs.getString(App.KEY_SYSTEM_PROMPT, SYSTEM_PROMPT_DEFAULT));
        }
    }

    // ==================== Decode-strategy controls ====================

    // Wire the decode-strategy sliders to native Configure_LLM. Beam/Top-K span 1..10 (< 2 disables beam,
    // then penalty picks greedy or argmax); Top-K coupled >= Beam. Penalty 0.50..1.00 (1.00 = off).
    private void setupDecodeControls() {
        beamSlider = findViewById(R.id.beam_slider);
        topKSlider = findViewById(R.id.top_k_slider);
        penaltySlider = findViewById(R.id.penalty_slider);
        memorySlider = findViewById(R.id.memory_slider);
        prefillSlider = findViewById(R.id.prefill_slider);
        decodeLimitSlider = findViewById(R.id.decode_limit_slider);
        beamValue = findViewById(R.id.beam_value);
        topKValue = findViewById(R.id.top_k_value);
        penaltyValue = findViewById(R.id.penalty_value);
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
        memoryBandValue = findViewById(R.id.memory_band_value);
        prefillValue = findViewById(R.id.prefill_value);
        decodeLimitValue = findViewById(R.id.decode_limit_value);
        decodeContent = findViewById(R.id.decode_content);
        organizeMemorySwitch = findViewById(R.id.organize_memory_switch);
        findViewById(R.id.btn_decode_settings).setOnClickListener(v -> toggleDecodePanel());
        loadMemoryLimits();
        restoreMemoryControls();
        Slider.OnChangeListener listener = (slider, value, fromUser) -> applyDecodeConfig(slider);
        beamSlider.addOnChangeListener(listener);
        topKSlider.addOnChangeListener(listener);
        penaltySlider.addOnChangeListener(listener);
        Slider.OnChangeListener memoryListener = (slider, value, fromUser) -> applyMemoryConfig(slider);
        memorySlider.addOnChangeListener(memoryListener);
        prefillSlider.addOnChangeListener(memoryListener);
        decodeLimitSlider.addOnChangeListener(memoryListener);
        organizeMemoryEnabled = getSharedPreferences(App.PREFS, MODE_PRIVATE)
                .getBoolean(App.KEY_ORGANIZE_MEMORY, false);
        if (organizeMemorySwitch != null) {
            organizeMemorySwitch.setChecked(organizeMemoryEnabled);
            organizeMemorySwitch.setOnCheckedChangeListener(
                    (button, checked) -> applyOrganizeMemoryConfig(checked, true));
        }
        applyMemoryConfig(null);   // push saved/default memory profile to native before the first run
        applyDecodeConfig(null);   // push the initial (argmax) defaults to the native side
        applyOrganizeMemoryConfig(organizeMemoryEnabled, false);   // push saved/default organize-memory state
        setupMemoryThresholds();   // load + wire the two draggable hysteresis-band markers
    }

    // Expand/collapse the decode sliders with a height transition. Triggered by the top-bar gear.
    private void toggleDecodePanel() {
        boolean expand = decodeContent.getVisibility() != View.VISIBLE;
        View root = findViewById(R.id.use_think);
        if (root instanceof ViewGroup) {
            TransitionManager.beginDelayedTransition((ViewGroup) root, new AutoTransition().setDuration(220));
        }
        decodeContent.setVisibility(expand ? View.VISIBLE : View.GONE);
    }

    // Read the sliders, enforce Top-K >= Beam, refresh labels, forward to native. Beam engages only when
    // Beam AND Top-K >= 2; else greedy (penalty < 1.0) or argmax. Top-K != 1 also forces Beam >= 2.
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
        if (topk != 1 && beam < 2) {
            beamSlider.setValue(2);   // Top-K asked for beam search; a Beam of 1 can't provide it, so raise it
            return;
        }
        float penalty = penaltySlider.getValue();
        boolean useBeam = beam >= 2 && topk >= 2;
        useBeamSearch = useBeam;
        beamValue.setText(String.valueOf(beam));
        topKValue.setText(String.valueOf(topk));
        String penaltyText = penalty >= 1.0f
                ? getString(R.string.penalty_off)
                : String.format(Locale.US, "%.2f", penalty);
        penaltyValue.setText(penaltyText);
        Configure_LLM(useBeam, topk, beam, penalty);
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
        int memory = Math.round(memorySlider.getValue());
        int prefill = Math.round(prefillSlider.getValue());
        if (prefill > memory) {
            prefillSlider.setValue(memory);
            return;
        }
        int decode = Math.round(decodeLimitSlider.getValue());
        memoryValue.setText(String.valueOf(memory));
        prefillValue.setText(String.valueOf(prefill));
        decodeLimitValue.setText(String.valueOf(decode));
        getSharedPreferences(App.PREFS, MODE_PRIVATE).edit()
                .putInt(App.KEY_MEMORY_TOKENS, memory)
                .putInt(App.KEY_PREFILL_TOKENS, prefill)
                .putInt(App.KEY_DECODE_TOKENS, decode)
                .apply();
        Configure_Memory(memory, prefill, decode);
        updateMemoryRemainingFromKnownUsage();
    }

    // Push the "organize memory" (整理记忆) toggle to native (next turn), optionally persisting it.
    private void applyOrganizeMemoryConfig(boolean enabled, boolean persist) {
        organizeMemoryEnabled = enabled;
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
        new Thread(() -> {
            final int[] memoryStats = Get_Memory_Stats();
            mainHandler.post(() -> {
                if (requestId == memoryStatsRequestId && !chatting) {
                    applyMemoryStats(memoryStats);
                }
            });
        }, "memory-stats").start();
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
        int capacity = Math.max(1, capacityTokens);
        memoryUsedTokens = clamp(usedTokens, 0, capacity);
        memoryRemainingTokens = clamp(remainingTokens, 0, capacity);
        memoryCapacityTokens = capacity;
        memoryRemainingPercent = remainingPercent(memoryRemainingTokens, capacity);
        updateMemoryRemainingDisplay();
    }

    private void updateMemoryRemainingFromKnownUsage() {
        int capacity = memorySlider != null ? Math.round(memorySlider.getValue()) : defaultMemoryTokens;
        capacity = Math.max(1, capacity);
        int used = clamp(memoryUsedTokens, 0, capacity);
        memoryRemainingTokens = capacity - used;
        memoryCapacityTokens = capacity;
        memoryRemainingPercent = remainingPercent(memoryRemainingTokens, capacity);
        updateMemoryRemainingDisplay();
    }

    // ==================== Memory threshold band (draggable) ====================

    // ---- Draggable hysteresis-band markers (green = reset target, red = rebuild trigger) ----
    @SuppressLint("ClickableViewAccessibility")
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
        if (memoryRemainingTrack != null) {
            memoryRemainingTrack.setOnTouchListener(this::onMemoryTrackTouch);
            // Marker position depends on the track width, only known after layout (and after a rotation).
            memoryRemainingTrack.addOnLayoutChangeListener(
                    (v, l, t, r, b, ol, ot, or, ob) -> positionThresholdMarkers());
            positionThresholdMarkers();
        }
    }

    private boolean onMemoryTrackTouch(View v, MotionEvent event) {
        float width = v.getWidth();
        if (width <= 0f) {
            return false;
        }
        float x = Math.max(0f, Math.min(event.getX(), width));
        int percent = Math.round(x / width * 100f);
        switch (event.getActionMasked()) {
            case MotionEvent.ACTION_DOWN: {
                // Grab whichever marker is nearer the touch, within a finger-friendly tolerance.
                float greenX = memoryGreenPercent / 100f * width;
                float redX = memoryRedPercent / 100f * width;
                float toGreen = Math.abs(x - greenX);
                float toRed = Math.abs(x - redX);
                if (Math.min(toGreen, toRed) > dp(24)) {
                    draggingThreshold = 0;
                    return false;
                }
                draggingThreshold = (toGreen <= toRed) ? 1 : 2;
                if (v.getParent() != null) {
                    v.getParent().requestDisallowInterceptTouchEvent(true);
                }
                applyThresholdDrag(percent);
                return true;
            }
            case MotionEvent.ACTION_MOVE:
                if (draggingThreshold == 0) {
                    return false;
                }
                applyThresholdDrag(percent);
                return true;
            case MotionEvent.ACTION_UP:
            case MotionEvent.ACTION_CANCEL:
                if (draggingThreshold == 0) {
                    return false;
                }
                draggingThreshold = 0;
                v.performClick();   // accessibility: a touch sequence ended on this control
                commitThresholds();
                if (v.getParent() != null) {
                    v.getParent().requestDisallowInterceptTouchEvent(false);
                }
                return true;
            default:
                return false;
        }
    }

    // Move the marker under the finger, keeping green strictly below red. Visual + caption only; native push
    // + persist happen on release (commitThresholds).
    private void applyThresholdDrag(int percent) {
        if (draggingThreshold == 1) {
            memoryGreenPercent = clamp(percent, MIN_MEMORY_BAND_PERCENT, memoryRedPercent - MIN_MEMORY_BAND_GAP);
        } else if (draggingThreshold == 2) {
            memoryRedPercent = clamp(percent, memoryGreenPercent + MIN_MEMORY_BAND_GAP, MAX_MEMORY_BAND_PERCENT);
        }
        positionThresholdMarkers();
        updateMemoryBandCaption();
    }

    private void clampThresholds() {
        memoryRedPercent = clamp(memoryRedPercent,
                MIN_MEMORY_BAND_PERCENT + MIN_MEMORY_BAND_GAP, MAX_MEMORY_BAND_PERCENT);
        memoryGreenPercent = clamp(memoryGreenPercent,
                MIN_MEMORY_BAND_PERCENT, memoryRedPercent - MIN_MEMORY_BAND_GAP);
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
    }

    // ==================== Memory gauge rendering ====================

    // Refresh the memory panel: free-headroom headline (health colour) + usage bar. Used = tokens held in the
    // window; the track behind the bar is free headroom.
    private void updateMemoryRemainingDisplay() {
        int healthColor = memorySemanticColor(memoryRemainingPercent);
        if (memoryRemainingValue != null) {
            memoryRemainingValue.setText(getString(R.string.memory_remaining_value,
                    (int) Math.round(memoryRemainingPercent), memoryCapacityTokens));
            memoryRemainingValue.setTextColor(healthColor);
        }
        updateMemoryRemainingBar();
    }

    // Gauge fills from the left with the USED portion (shades toward red as it fills). Pulses only when usage
    // climbs, not when a reset frees it.
    private void updateMemoryRemainingBar() {
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
        animateMemoryBarTo(targetFraction, targetColor, pulse);
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

    private static void showToast(final Activity context, final String content, boolean display_long){
        if (display_long) {
            context.runOnUiThread(() -> Toast.makeText(context, content, Toast.LENGTH_LONG).show());
        } else {
            context.runOnUiThread(() -> Toast.makeText(context, content, Toast.LENGTH_SHORT).show());
        }
    }

    // ==================== Asset staging ====================

    private void Copy_from_Asset_to_Cache(String fileName, AssetManager mgr, byte[] buffer, boolean forceOverwrite){
        try {
            File cacheDir = getCacheDir();
            if (!cacheDir.exists()) {
                if (!cacheDir.mkdirs()) {
                    System.out.println("Directory creation failed.");
                }
            }
            File outFile = new File(cacheDir,fileName);
            if (!outFile.exists()){
                if (!outFile.createNewFile()) {
                    return;
                }
            } else if (!forceOverwrite && outFile.length() > 10) {
                return;   // already staged and not forcing a refresh
            }
            InputStream is = mgr.open(fileName);
            FileOutputStream os = new FileOutputStream(outFile);
            int len;
            while ((len = is.read(buffer)) != -1) {
                os.write(buffer,0, len);
            }
            is.close();
            os.flush();
            os.close();
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    // ==================== Native methods (JNI) ====================

    private native boolean Pre_Process();
    // Load ONNX models. assetManager null = low-memory mode (external data format). Returns true on success.
    private native boolean Load_Models_A(AssetManager assetManager, int EP_TYPE, boolean LOW_MEMORY_MODE);
    // Runs prefill + the full decode loop in C++, streaming via onTokenStream(); returns "<prefill>|<decode>"
    // tok/s (or "Over_Inputs"). CLEAR resets context; USE_THINK toggles the think template; TURN_ID checkpoints KV.
    private static native String Run_LLM(String QUERY, boolean CLEAR, boolean USE_THINK, int TURN_ID);
    // Rolls the KV cache back to before TURN_ID's prompt (dropping it + later turns) so an edited bubble can
    // regenerate. Returns false (and clears history) when no longer rollable.
    private static native boolean Rollback_LLM(int TURN_ID);
    // Runtime decode-strategy: beam when enabled with TOP_K/BEAM_SIZE >= 2, greedy for penalty, else argmax.
    // Values clamped natively to the exported graph limits.
    private static native boolean Configure_LLM(boolean USE_BEAM_SEARCH, int TOP_K, int BEAM_SIZE, float REPEAT_PENALTY);
    // Replaces the slot system prompt (next turn). Organize Memory ON strips its block during clean recompute.
    private static native boolean Set_System_Prompt(String SYSTEM_PROMPT);
    // Runtime memory profile (retained memory, prefill window, per-turn decode cap); snapshotted at turn start.
    private static native boolean Configure_Memory(int MEMORY_TOKENS, int PREFILL_TOKENS, int DECODE_TOKENS);
    // Toggles "organize memory" (整理记忆): true = native strips the system prompt from cross-turn memory. Next turn.
    private static native boolean Configure_Organize_Memory(boolean ENABLE);
    // Returns min/max/default triples for Memory, Prefill, Decode Limit (derived natively from max_seq_len).
    private static native int[] Get_Memory_Limits();
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
}
