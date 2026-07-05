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
    // "Organize memory" (整理记忆): true = native drops the system prompt from cross-turn memory.
    static final String KEY_ORGANIZE_MEMORY = "organize_memory";
    // Persisted throughput points; survive chat reset + restart, wiped only from the chart's data-clear.
    static final String KEY_THROUGHPUT_PREFILL = "throughput_prefill";
    static final String KEY_THROUGHPUT_DECODE = "throughput_decode";
    // Bump when a bundled .onnx/vocab asset changes so the cache is force-re-staged (else a stale copy shadows it).
    static final String KEY_ASSET_STAGE_VERSION = "asset_stage_version";

    @Override
    public void onCreate() {
        super.onCreate();
        SharedPreferences prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        int mode = prefs.getInt(KEY_NIGHT_MODE, AppCompatDelegate.MODE_NIGHT_NO);
        AppCompatDelegate.setDefaultNightMode(mode);
        MetricsHistory.get().restore(this);   // reload persisted throughput before any Activity reads it
    }
}
