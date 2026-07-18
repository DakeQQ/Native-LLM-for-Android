package com.example.myapplication;

import android.app.Application;
import android.content.SharedPreferences;

import androidx.appcompat.app.AppCompatDelegate;

/**
 * Applies the persisted Light/Dark choice before any Activity is created, so the first frame is themed
 * (no flash). Stored in {@link #PREFS}; default Light.
 */
public class App extends Application {
    static final String PREFS = "ui_prefs";
    static final String KEY_NIGHT_MODE = "night_mode";
    // Slot system-prompt text (empty = disabled).
    static final String KEY_SYSTEM_PROMPT = "system_prompt";
    static final String KEY_MEMORY_TOKENS = "memory_tokens";
    static final String KEY_PREFILL_TOKENS = "prefill_tokens";
    static final String KEY_DECODE_TOKENS = "decode_tokens";
    static final String KEY_MEMORY_RED_PERCENT = "memory_red_percent";
    static final String KEY_MEMORY_GREEN_PERCENT = "memory_green_percent";
    static final String KEY_DECODE_MODE = "decode_mode";
    static final String KEY_BEAM_SIZE = "beam_size";
    static final String KEY_BEAM_TOP_K = "beam_top_k";
    static final String KEY_DIRECT_REPEAT_PENALTY = "direct_repeat_penalty";
    static final String KEY_PENALTY_RANGE = "penalty_range";
    static final String KEY_SAMPLING_TEMPERATURE = "sampling_temperature";
    static final String KEY_SAMPLING_TOP_K = "sampling_top_k";
    static final String KEY_SAMPLING_TOP_P = "sampling_top_p";
    static final String KEY_SAMPLING_REPEAT_PENALTY = "sampling_repeat_penalty";
    // "Organize memory": retain clean cross-turn text and replace visual state with note-backed history.
    static final String KEY_ORGANIZE_MEMORY = "organize_memory";
    // Persisted throughput points; survive chat reset + restart, wiped only from the chart's data-clear.
    static final String KEY_THROUGHPUT_PREFILL = "throughput_prefill";
    static final String KEY_THROUGHPUT_DECODE = "throughput_decode";
    // Fingerprint of the generated SHA-256 bundle manifest used for the staged tokenizer vocab.
    static final String KEY_ASSET_BUNDLE_FINGERPRINT = "asset_bundle_fingerprint";

    @Override
    public void onCreate() {
        super.onCreate();
        SharedPreferences prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        int mode = prefs.getInt(KEY_NIGHT_MODE, AppCompatDelegate.MODE_NIGHT_NO);
        AppCompatDelegate.setDefaultNightMode(mode);
        // Reload persisted throughput off the main thread so cold-start's first frame is not delayed by
        // the SharedPreferences read + point parse; the load completes long before any generation writes.
        MetricsHistory.get().restoreAsync(this);
    }
}
