package com.example.myapplication;

import android.content.Intent;
import android.content.res.Configuration;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.system.Os;
import android.system.OsConstants;
import android.view.View;
import android.widget.ImageButton;
import android.widget.LinearLayout;
import android.widget.TextView;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsControllerCompat;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.regex.Pattern;

/**
 * Full-screen telemetry chart, reused for two datasets via {@link #EXTRA_MODE}: {@link #MODE_SYSTEM}
 * (CPU%/RAM over time) or {@link #MODE_THROUGHPUT} (token total vs tokens/s). {@link #EXTRA_FOCUS} picks the
 * series shown on entry. {@link MetricChartView} owns drawing/zoom; this Activity wires header/legend/zoom.
 */
public class MetricChartActivity extends AppCompatActivity {

    public static final String EXTRA_MODE = "metric_mode";
    public static final String EXTRA_FOCUS = "metric_focus";
    public static final String MODE_SYSTEM = "system";
    public static final String MODE_THROUGHPUT = "throughput";

    private boolean throughput;
    private MetricChartView chart;
    private LinearLayout legendA;
    private LinearLayout legendB;
    private TextView legendALabel;
    private TextView legendBLabel;
    private View legendADot;
    private View legendBDot;
    private HandlerThread chartMonitorThread;
    private Handler chartMonitorHandler;
    private volatile boolean chartMonitorActive = false;
    private long lastCpuJiffies = -1L;
    private long lastCpuWallMs = 0L;
    private final int cpuCount = Math.max(1, Runtime.getRuntime().availableProcessors());
    private long clockTicksPerSec = 100L;
    private long pageSizeBytes = 4096L;

    private static final long SYSTEM_SAMPLE_INTERVAL_MS = 1000L;
    private static final Pattern WHITESPACE = Pattern.compile("\\s+");

    private final Runnable systemSampleTask = new Runnable() {
        @Override
        public void run() {
            sampleSystemUsageForChart();
            if (chartMonitorActive && chartMonitorHandler != null) {
                chartMonitorHandler.postDelayed(this, SYSTEM_SAMPLE_INTERVAL_MS);
            }
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_metric_chart);
        applyThemeSystemBars();

        throughput = MODE_THROUGHPUT.equals(getIntent().getStringExtra(EXTRA_MODE));
        int focus = getIntent().getIntExtra(EXTRA_FOCUS, MetricChartView.SERIES_A);
        if (!throughput) {
            setupSystemSampler();
        }

        chart = findViewById(R.id.metric_chart);
        legendA = findViewById(R.id.legend_a);
        legendB = findViewById(R.id.legend_b);
        legendALabel = findViewById(R.id.legend_a_label);
        legendBLabel = findViewById(R.id.legend_b_label);
        legendADot = findViewById(R.id.legend_a_dot);
        legendBDot = findViewById(R.id.legend_b_dot);

        TextView title = findViewById(R.id.metric_title);
        TextView subtitle = findViewById(R.id.metric_subtitle);

        int colorA;
        int glowA;
        int colorB;
        int glowB;
        if (throughput) {
            // Prefill = blue, Decode = orange-yellow (distinct from CPU/RAM cyan/magenta); day+night variants.
            colorA = ContextCompat.getColor(this, R.color.chartPrefill);            glowA = ContextCompat.getColor(this, R.color.chartPrefillGlow);
            colorB = ContextCompat.getColor(this, R.color.chartDecode);
            glowB = ContextCompat.getColor(this, R.color.chartDecodeGlow);
        } else {
            colorA = ContextCompat.getColor(this, R.color.chartSeriesA);
            glowA = ContextCompat.getColor(this, R.color.chartSeriesAGlow);
            colorB = ContextCompat.getColor(this, R.color.chartSeriesB);
            glowB = ContextCompat.getColor(this, R.color.chartSeriesBGlow);
        }

        String labelA;
        String labelB;
        if (throughput) {
            title.setText(R.string.chart_title_throughput);
            subtitle.setText(R.string.chart_subtitle_throughput);
            labelA = getString(R.string.chart_series_prefill);
            labelB = getString(R.string.chart_series_decode);
            chart.setFormatting(false);
            chart.configureSeries(MetricChartView.SERIES_A, " tok/s", colorA, glowA, false);
            chart.configureSeries(MetricChartView.SERIES_B, " tok/s", colorB, glowB, true);
        } else {
            title.setText(R.string.chart_title_system);
            subtitle.setText(R.string.chart_subtitle_system);
            labelA = getString(R.string.chart_series_cpu);
            labelB = getString(R.string.chart_series_mem);
            chart.setFormatting(true);
            chart.configureSeries(MetricChartView.SERIES_A, "%", colorA, glowA, false);
            chart.configureSeries(MetricChartView.SERIES_B, " MB", colorB, glowB, true);
        }
        legendALabel.setText(labelA);
        legendBLabel.setText(labelB);
        // Throughput = scatter pins (raw samples); system time-series keep the line.
        chart.setScatterMode(throughput);
        tintDot(legendADot, colorA);
        tintDot(legendBDot, colorB);

        // Enter showing the tapped metric; the other can be overlaid from its chip.
        chart.setSeriesVisible(MetricChartView.SERIES_A, focus == MetricChartView.SERIES_A);
        chart.setSeriesVisible(MetricChartView.SERIES_B, focus == MetricChartView.SERIES_B);

        legendA.setOnClickListener(v -> toggleSeries(MetricChartView.SERIES_A));
        legendB.setOnClickListener(v -> toggleSeries(MetricChartView.SERIES_B));

        ImageButton back = findViewById(R.id.metric_back);
        back.setOnClickListener(v -> finish());
        findViewById(R.id.metric_zoom_in).setOnClickListener(v -> chart.zoomIn());
        findViewById(R.id.metric_zoom_out).setOnClickListener(v -> chart.zoomOut());
        findViewById(R.id.metric_fit).setOnClickListener(v -> chart.fitView());
        findViewById(R.id.metric_clear).setOnClickListener(v -> confirmClearData());

        loadData(true);
        refreshLegend();
    }

    @Override
    protected void onResume() {
        super.onResume();
        startSystemSampler();
    }

    @Override
    protected void onPause() {
        stopSystemSampler();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        stopSystemSampler();
        if (chartMonitorThread != null) {
            chartMonitorThread.quitSafely();
            chartMonitorThread = null;
            chartMonitorHandler = null;
        }
        super.onDestroy();
    }

    private void loadData(boolean animate) {
        MetricsHistory.Series[] s = throughput
                ? MetricsHistory.get().throughputSeries()
                : MetricsHistory.get().systemSeries();
        chart.setData(s[0], s[1], animate);
    }

    // Dedicated data-clear action. Throughput stats persist across sessions, so erasing them is irreversible
    // and needs an explicit second confirmation.
    private void confirmClearData() {
        new AlertDialog.Builder(this)
                .setTitle(R.string.chart_clear_confirm_title)
                .setMessage(R.string.chart_clear_confirm_message)
                .setPositiveButton(R.string.chart_clear_confirm_positive, (dialog, which) -> {
                    MetricsHistory.get().clear();
                    if (chartMonitorHandler != null) {
                        chartMonitorHandler.post(() -> lastCpuJiffies = -1L);
                    }
                    loadData(true);
                })
                .setNegativeButton(R.string.chart_clear_confirm_negative, null)
                .show();
    }

    private void setupSystemSampler() {
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
            // Safe defaults match Linux/Android common values if sysconf is unavailable.
        }
        chartMonitorThread = new HandlerThread("metric-chart-monitor");
        chartMonitorThread.start();
        chartMonitorHandler = new Handler(chartMonitorThread.getLooper());
    }

    private void startSystemSampler() {
        if (throughput || chartMonitorHandler == null) {
            return;
        }
        chartMonitorActive = true;
        chartMonitorHandler.removeCallbacksAndMessages(null);
        chartMonitorHandler.post(() -> {
            lastCpuJiffies = -1L;
            systemSampleTask.run();
        });
    }

    private void stopSystemSampler() {
        chartMonitorActive = false;
        if (chartMonitorHandler != null) {
            chartMonitorHandler.removeCallbacksAndMessages(null);
        }
    }

    private void sampleSystemUsageForChart() {
        long nowMs = SystemClock.elapsedRealtime();
        long jiffies = readProcessCpuJiffies();
        long rssBytes = readProcessRssBytes();
        boolean added = false;
        if (jiffies >= 0L && lastCpuJiffies >= 0L && nowMs > lastCpuWallMs) {
            double procSeconds = (jiffies - lastCpuJiffies) / (double) clockTicksPerSec;
            double wallSeconds = (nowMs - lastCpuWallMs) / 1000.0;
            double percent = procSeconds / wallSeconds * 100.0 / cpuCount;
            if (percent < 0.0) {
                percent = 0.0;
            } else if (percent > 100.0) {
                percent = 100.0;
            }
            MetricsHistory.get().addCpu(nowMs, (float) percent);
            added = true;
        }
        if (jiffies >= 0L) {
            lastCpuJiffies = jiffies;
            lastCpuWallMs = nowMs;
        }
        if (rssBytes >= 0L) {
            MetricsHistory.get().addMemory(nowMs, rssBytes / (1024.0f * 1024.0f));
            added = true;
        }
        if (added && chart != null) {
            chart.post(() -> {
                if (chartMonitorActive && !isFinishing()) {
                    loadData(false);
                }
            });
        }
    }

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

    private void toggleSeries(int index) {
        int other = index == MetricChartView.SERIES_A ? MetricChartView.SERIES_B : MetricChartView.SERIES_A;
        // Keep at least one series visible so the plot never goes fully blank.
        if (chart.isSeriesVisible(index) && !chart.isSeriesVisible(other)) {
            return;
        }
        chart.setSeriesVisible(index, !chart.isSeriesVisible(index));
        refreshLegend();
    }

    private void refreshLegend() {
        legendA.setAlpha(chart.isSeriesVisible(MetricChartView.SERIES_A) ? 1.0f : 0.4f);
        legendB.setAlpha(chart.isSeriesVisible(MetricChartView.SERIES_B) ? 1.0f : 0.4f);
    }

    private static void tintDot(View dot, int color) {
        if (dot.getBackground() != null) {
            dot.getBackground().mutate().setTint(color);
        }
    }

    // Match the system-bar icon polarity to the active light/dark mode (page paints edge-to-edge).
    private void applyThemeSystemBars() {
        boolean night = (getResources().getConfiguration().uiMode & Configuration.UI_MODE_NIGHT_MASK)
                == Configuration.UI_MODE_NIGHT_YES;
        WindowCompat.setDecorFitsSystemWindows(getWindow(), true);
        View decor = getWindow().getDecorView();
        WindowInsetsControllerCompat controller = WindowCompat.getInsetsController(getWindow(), decor);
        controller.setAppearanceLightStatusBars(!night);
        controller.setAppearanceLightNavigationBars(!night);
    }

    /** Convenience launcher used by the performance-panel cells. */
    static void launch(android.content.Context context, String mode, int focus) {
        Intent intent = new Intent(context, MetricChartActivity.class);
        intent.putExtra(EXTRA_MODE, mode);
        intent.putExtra(EXTRA_FOCUS, focus);
        context.startActivity(intent);
    }
}
