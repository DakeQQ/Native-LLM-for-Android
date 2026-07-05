package com.example.myapplication;

import android.content.res.Configuration;
import android.graphics.Typeface;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsControllerCompat;

import com.google.android.material.button.MaterialButton;
import com.google.android.material.slider.Slider;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Standalone Compute / Memory benchmark screen. Runs the native {@link MemBandwidth} benchmark on a worker
 * thread, renders the report as tables, and estimates an attainable decode token rate for several Qwen3 sizes.
 */
public class ComputeMemoryActivity extends AppCompatActivity {

    private static final Pattern NUMBER_PATTERN =
            Pattern.compile("[0-9]+(?:\\.[0-9]+)?(?:[eE][-+]?[0-9]+)?");
    private static final String[] CORE_GROUP_LABELS = {"Little", "Big", "Total"};
        private static final String[] BANDWIDTH_HEADERS = {"Kernel", "Med", "Peak", "Avg"};
        private static final String[] COMPUTE_HEADERS = {"Metric", "Med", "Peak", "Avg"};
        private static final float[] EMPTY_FLOATS = new float[0];

    // Approximate total parameter counts for the Qwen3 family used in the attainable-rate estimate.
    private static final String[] MODEL_LABELS = {"0.6B", "1.7B", "4B", "8B"};
    private static final float[] MODEL_PARAMS = {0.6e9f, 1.7e9f, 4.0e9f, 8.0e9f};

    // Dynamic quantized matvec: packed int weights dequantized to FP32, so bytes/weight includes per-group
    // F32 scale + zero-point metadata, not pure Q4/Q8.
    private static final float DYNAMIC_QUANT_GROUP_SIZE = 32.0f;    private static final float Q4F32_BYTES_PER_WEIGHT = 0.5f + (4.0f + 0.5f) / DYNAMIC_QUANT_GROUP_SIZE;
    private static final float Q8F32_BYTES_PER_WEIGHT = 1.0f + (4.0f + 1.0f) / DYNAMIC_QUANT_GROUP_SIZE;
    private static final String[] DECODE_FORMAT_LABELS = {"Q4F32", "Q8F32", "F16"};
    private static final float[] DECODE_BYTES_PER_WEIGHT = {Q4F32_BYTES_PER_WEIGHT, Q8F32_BYTES_PER_WEIGHT, 2.0f};
        private static final String[] ESTIMATE_HEADERS = {
            "Model", DECODE_FORMAT_LABELS[0], DECODE_FORMAT_LABELS[1], DECODE_FORMAT_LABELS[2]};

    // Fraction of read-streaming bandwidth a low-bit GEMV decode realizes; the rest goes to dequant, KV-cache
    // reads, activation traffic + threading. Scales the roofline into an attainable rate, not a ceiling.
    private static final int DEFAULT_DECODE_EFFICIENCY_PERCENT = 80;

    private int tableText;
    private int tableMuted;
    private int tableAccent;
    private int tableHeaderBg;
    private int tableEmphasisBg;
    private int tableStripeBg;
    private int tablePeakBg;
    private int tablePeakText;
    private int tableDivider;

    private final Handler mainHandler = new Handler(Looper.getMainLooper());

    private Slider arraySlider;
    private Slider iterSlider;
    private Slider littleThreadSlider;
    private Slider bigThreadSlider;
    private Slider efficiencySlider;
    private TextView arrayValue;
    private TextView iterValue;
    private TextView littleThreadValue;
    private TextView bigThreadValue;
    private TextView efficiencyValue;
    private TextView resultStatus;
    private TextView estimateStatus;
    private LinearLayout resultContainer;
    private LinearLayout estimateContainer;
    private MaterialButton runButton;
    private ProgressBar runProgress;

    private volatile boolean running = false;
    private volatile boolean destroyed = false;

    // Decode is weight-memory-bound, so the estimate keys off the SUSTAINED read-only median, scaled by an
    // attainable GEMV efficiency factor rather than the full roofline.
    private float decodeBwEfficiency = DEFAULT_DECODE_EFFICIENCY_PERCENT / 100.0f;
    private float realizedDecodeGBs = 0.0f;  // sustained read median × decodeBwEfficiency
    private float sustainedReadGBs = 0.0f;   // raw sustained read-only median (shown in the caption)

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_compute_memory);
        applyThemeSystemBars();
        initTablePalette();

        ImageButton back = findViewById(R.id.compute_back);
        arraySlider = findViewById(R.id.array_slider);
        iterSlider = findViewById(R.id.iter_slider);
        littleThreadSlider = findViewById(R.id.little_thread_slider);
        bigThreadSlider = findViewById(R.id.big_thread_slider);
        efficiencySlider = findViewById(R.id.efficiency_slider);
        arrayValue = findViewById(R.id.array_value);
        iterValue = findViewById(R.id.iter_value);
        littleThreadValue = findViewById(R.id.little_thread_value);
        bigThreadValue = findViewById(R.id.big_thread_value);
        efficiencyValue = findViewById(R.id.efficiency_value);
        resultStatus = findViewById(R.id.result_status);
        estimateStatus = findViewById(R.id.estimate_status);
        resultContainer = findViewById(R.id.result_container);
        estimateContainer = findViewById(R.id.estimate_container);
        runButton = findViewById(R.id.run_button);
        runProgress = findViewById(R.id.run_progress);

        back.setOnClickListener(v -> finish());
        runButton.setOnClickListener(v -> startBenchmark());

        arraySlider.addOnChangeListener((slider, value, fromUser) ->
                arrayValue.setText(getString(R.string.compute_mem_array_value, Math.round(value))));
        iterSlider.addOnChangeListener((slider, value, fromUser) ->
                iterValue.setText(String.valueOf(Math.round(value))));
        int maxThreadSetting = Math.max(1, Math.min(16, Runtime.getRuntime().availableProcessors()));
        setupThreadSlider(littleThreadSlider, littleThreadValue, maxThreadSetting);
        setupThreadSlider(bigThreadSlider, bigThreadValue, maxThreadSetting);
        efficiencySlider.addOnChangeListener((slider, value, fromUser) ->
            updateEfficiency(Math.round(value), sustainedReadGBs > 0.0f));

        // Seed the value labels from the layout's initial slider values.
        arrayValue.setText(getString(R.string.compute_mem_array_value, Math.round(arraySlider.getValue())));
        iterValue.setText(String.valueOf(Math.round(iterSlider.getValue())));
        updateEfficiency(Math.round(efficiencySlider.getValue()), false);
    }

    @Override
    protected void onDestroy() {
        destroyed = true;
        super.onDestroy();
    }

    private void applyThemeSystemBars() {
        WindowInsetsControllerCompat controller =
                WindowCompat.getInsetsController(getWindow(), getWindow().getDecorView());
        if (controller != null) {
            boolean lightMode = (getResources().getConfiguration().uiMode
                    & Configuration.UI_MODE_NIGHT_MASK) != Configuration.UI_MODE_NIGHT_YES;
            controller.setAppearanceLightStatusBars(lightMode);
            controller.setAppearanceLightNavigationBars(lightMode);
        }
    }

    private void initTablePalette() {
        tableText = ContextCompat.getColor(this, R.color.textPrimary);
        tableMuted = ContextCompat.getColor(this, R.color.textMuted);
        tableAccent = ContextCompat.getColor(this, R.color.energyYellow);
        tableHeaderBg = ContextCompat.getColor(this, R.color.computeTableHeaderBg);
        tableEmphasisBg = ContextCompat.getColor(this, R.color.computeTableEmphasisBg);
        tableStripeBg = ContextCompat.getColor(this, R.color.computeTableStripeBg);
        tablePeakBg = ContextCompat.getColor(this, R.color.energyYellow);
        tablePeakText = ContextCompat.getColor(this, R.color.textOnYellow);
        tableDivider = ContextCompat.getColor(this, R.color.computeTableDivider);
    }

    private void setupThreadSlider(Slider slider, TextView valueView, int maxThreadSetting) {
        slider.setValueTo(maxThreadSetting);
        slider.setValue(maxThreadSetting);
        valueView.setText(threadValueLabel(maxThreadSetting, maxThreadSetting));
        slider.addOnChangeListener((s, value, fromUser) ->
                valueView.setText(threadValueLabel(Math.round(value), maxThreadSetting)));
    }

    private String threadValueLabel(int threads, int maxThreadSetting) {
        if (threads <= 0) {
            return getString(R.string.compute_mem_skip);
        }
        if (threads >= maxThreadSetting) {
            return getString(R.string.compute_mem_all);
        }
        return String.valueOf(threads);
    }

    private void updateEfficiency(int percent, boolean refreshEstimate) {
        decodeBwEfficiency = Math.max(1, percent) / 100.0f;
        efficiencyValue.setText(getString(R.string.compute_mem_efficiency_value, percent));
        if (refreshEstimate) {
            renderEstimate();
        }
    }

    // Run the benchmark on a worker thread (blocks several seconds), post the report + estimate back.
    // Re-entrancy guarded so a second tap while running is ignored.
    private void startBenchmark() {
        if (running) {
            return;
        }
        running = true;
        final int arrayMillions = Math.round(arraySlider.getValue());
        final int iterations = Math.round(iterSlider.getValue());
        final int littleThreads = Math.round(littleThreadSlider.getValue());
        final int bigThreads = Math.round(bigThreadSlider.getValue());

        if (littleThreads <= 0 && bigThreads <= 0) {
            running = false;
            resultContainer.setVisibility(View.GONE);
            resultStatus.setVisibility(View.VISIBLE);
            resultStatus.setText(R.string.compute_mem_no_core_group);
            estimateContainer.setVisibility(View.GONE);
            estimateStatus.setVisibility(View.VISIBLE);
            estimateStatus.setText(R.string.compute_mem_estimate_idle);
            sustainedReadGBs = 0.0f;
            realizedDecodeGBs = 0.0f;
            return;
        }

        runButton.setEnabled(false);
        runProgress.setVisibility(View.VISIBLE);
        resultContainer.setVisibility(View.GONE);
        resultStatus.setVisibility(View.VISIBLE);
        resultStatus.setText(R.string.compute_mem_running);

        new Thread(() -> {
            String report;
            try {
                report = MemBandwidth.runMemoryBandwidthTest(
                        arrayMillions, iterations, littleThreads, bigThreads);
            } catch (Throwable t) {
                report = "ERROR: " + t.getMessage();
            }
            final String finalReport = report;
            mainHandler.post(() -> {
                if (destroyed) {
                    return;
                }
                running = false;
                runButton.setEnabled(true);
                runProgress.setVisibility(View.GONE);
                showResult(finalReport);
                updateEstimate(finalReport);
            });
        }, "membw-runner").start();
    }

    // Build the result tables (Bandwidth kernels + Compute rows) from the report. On an unparseable report
    // the raw text is shown in the status line instead.
    private void showResult(String report) {
        List<String> groups = presentCoreGroups(report);
        if (groups.isEmpty()) {
            resultContainer.setVisibility(View.GONE);
            resultStatus.setVisibility(View.VISIBLE);
            resultStatus.setText(report);
            return;
        }
        resultContainer.removeAllViews();
        List<String> arrays = linesStartingWith(report, "arrays");
        for (int i = 0; i < arrays.size(); i++) {
            resultContainer.addView(makeInfo(arrays.get(i), i == 0 ? 0 : dp(4)));
        }
        boolean spaced = !arrays.isEmpty();
        for (String group : groups) {
            resultContainer.addView(makeCaption(group + " · " + getString(R.string.compute_mem_bw_caption), spaced));
            List<String[]> rows = new ArrayList<>(6);
            rows.add(kernelRow(report, group + " Read", "Read"));
            rows.add(kernelRow(report, group + " Copy", "Copy"));
            rows.add(kernelRow(report, group + " Scale", "Scale"));
            rows.add(kernelRow(report, group + " Add", "Add"));
            rows.add(kernelRow(report, group + " Triad", "Triad"));
            float[] groupPeak = numbersOnLine(report, group + " peak");
            rows.add(new String[]{"Peak", cell(groupPeak, 0), cell(groupPeak, 1), "—"});
            resultContainer.addView(buildTable(BANDWIDTH_HEADERS, rows, -1, 2));
            spaced = true;
        }

        for (String group : groups) {
            resultContainer.addView(makeCaption(group + " · " + getString(R.string.compute_mem_compute_caption), true));
            List<String[]> computeRows = new ArrayList<>(3);
            computeRows.add(computeRow(report, group + " F32 compute", "F32"));
            computeRows.add(computeRow(report, group + " F16 compute", "F16"));
            computeRows.add(computeRow(report, group + " INT8 compute", "INT8"));
            resultContainer.addView(buildTable(COMPUTE_HEADERS, computeRows, -1, 2));
        }

        for (String group : groups) {
            String wall = lineStartingWith(report, group + " wall");
            if (wall != null) {
                resultContainer.addView(makeInfo(wall, dp(10)));
            }
        }
        resultStatus.setVisibility(View.GONE);
        resultContainer.setVisibility(View.VISIBLE);
    }

    // Decode estimate: rate ≈ (sustainedRead · efficiency) / (P · bytesPerWeight). Efficiency folds in
    // dequant/KV/activation/threading; Q4F32/Q8F32 bytes include group scale/zero metadata.
    private void updateEstimate(String report) {
        float[] read = firstNumbersOnLine(report,
                "Total Read", "Big Read", "Little Read", "Read");
        float[] peak = firstNumbersOnLine(report,
                "Total peak", "Big peak", "Little peak", "peak");
        float sustained = 0.0f;
        if (read.length >= 1 && read[0] > 0.0f) {
            sustained = read[0];                           // sustained read-only median
        } else if (peak.length >= 1 && peak[0] > 0.0f) {
            sustained = peak[0];                           // fallback: peak median when no Read row
        }
        if (sustained <= 0.0f) {
            estimateContainer.setVisibility(View.GONE);
            estimateStatus.setVisibility(View.VISIBLE);
            estimateStatus.setText(R.string.compute_mem_estimate_idle);
            sustainedReadGBs = 0.0f;
            realizedDecodeGBs = 0.0f;
            return;
        }
        sustainedReadGBs = sustained;
        renderEstimate();
    }

    private void renderEstimate() {
        if (sustainedReadGBs <= 0.0f) {
            return;
        }
        realizedDecodeGBs = sustainedReadGBs * decodeBwEfficiency;

        List<String[]> rows = new ArrayList<>(MODEL_LABELS.length);
        for (int i = 0; i < MODEL_LABELS.length; i++) {
            String[] row = new String[1 + DECODE_FORMAT_LABELS.length];
            row[0] = MODEL_LABELS[i];
            for (int f = 0; f < DECODE_FORMAT_LABELS.length; f++) {
                row[f + 1] = fmtRate(decodeRate(MODEL_PARAMS[i], DECODE_BYTES_PER_WEIGHT[f]));
            }
            rows.add(row);
        }
        estimateContainer.removeAllViews();
        estimateContainer.addView(makeCaption(
                getString(R.string.compute_mem_estimate_caption,
                        fmt1(realizedDecodeGBs), fmt1(sustainedReadGBs),
                Math.round(decodeBwEfficiency * 100.0f)), false));
        estimateContainer.addView(buildTable(
                ESTIMATE_HEADERS, rows, -1, -1));
        estimateContainer.addView(makeNote(getString(R.string.compute_mem_estimate_note)));
        estimateStatus.setVisibility(View.GONE);
        estimateContainer.setVisibility(View.VISIBLE);
    }

    // One row {label, med, peak, avg} from the numbers on the report line starting with linePrefix.
    private String[] kernelRow(String report, String linePrefix, String label) {        float[] n = numbersOnLine(report, linePrefix);
        return new String[]{label, cell(n, 0), cell(n, 1), cell(n, 2)};
    }

    private String[] computeRow(String report, String linePrefix, String label) {
        float[] n = numbersOnLine(report, linePrefix);
        return new String[]{label, cell(n, 0), cell(n, 1), cell(n, 2)};
    }

    // Build a table view: rounded panel, header row, dividers, zebra stripes, optional highlighted row/column.
    private View buildTable(String[] headers, List<String[]> rows, int emphasizedRow, int highlightedColumn) {
        LinearLayout table = new LinearLayout(this);
        table.setOrientation(LinearLayout.VERTICAL);
        table.setBackgroundResource(R.drawable.bg_compute_table);
        table.setClipToOutline(true);
        LinearLayout.LayoutParams tableParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        tableParams.topMargin = dp(8);
        table.setLayoutParams(tableParams);

        table.addView(buildRow(headers, true, false, false, highlightedColumn));
        for (int i = 0; i < rows.size(); i++) {
            View divider = new View(this);
            divider.setLayoutParams(new LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, dp(1)));
            divider.setBackgroundColor(tableDivider);
            table.addView(divider);
            table.addView(buildRow(rows.get(i), false, i == emphasizedRow, i % 2 == 0, highlightedColumn));
        }
        return table;
    }

    private LinearLayout buildRow(String[] cells, boolean header, boolean emphasized, boolean striped,
                                  int highlightedColumn) {
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        if (header) {
            row.setBackgroundColor(tableHeaderBg);
        } else if (emphasized) {
            row.setBackgroundColor(tableEmphasisBg);
        } else if (striped) {
            row.setBackgroundColor(tableStripeBg);
        }
        for (int c = 0; c < cells.length; c++) {
            int gravity = (c == 0 ? Gravity.START : Gravity.END) | Gravity.CENTER_VERTICAL;
            row.addView(makeCell(cells[c], header, emphasized, c == highlightedColumn, gravity));
        }
        return row;
    }

    private TextView makeCell(String text, boolean header, boolean emphasized, boolean highlighted, int gravity) {
        TextView cell = new TextView(this);
        cell.setLayoutParams(new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        cell.setPadding(dp(12), dp(9), dp(12), dp(9));
        cell.setGravity(gravity);
        cell.setTextSize(TypedValue.COMPLEX_UNIT_SP, 13);
        cell.setTypeface(Typeface.MONOSPACE, (header || emphasized || highlighted) ? Typeface.BOLD : Typeface.NORMAL);
        if (highlighted) {
            cell.setBackgroundColor(tablePeakBg);
            cell.setTextColor(tablePeakText);
        } else {
            cell.setTextColor((header || emphasized) ? tableAccent : tableText);
        }
        cell.setText(text);
        return cell;
    }

    // Bold section label above a table; spaced adds extra top margin when it follows other content.
    private TextView makeCaption(String text, boolean spaced) {
        TextView caption = new TextView(this);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        params.topMargin = spaced ? dp(14) : dp(2);
        caption.setLayoutParams(params);
        caption.setText(text);
        caption.setTextColor(tableText);
        caption.setTextSize(TypedValue.COMPLEX_UNIT_SP, 12);
        caption.setTypeface(null, Typeface.BOLD);
        return caption;
    }

    // Muted single-line context (the config / wall lines from the report).
    private TextView makeInfo(String text, int topMargin) {
        TextView info = new TextView(this);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        params.topMargin = topMargin;
        info.setLayoutParams(params);
        info.setText(text);
        info.setTextColor(tableMuted);
        info.setTextSize(TypedValue.COMPLEX_UNIT_SP, 11);
        return info;
    }

    // Wrapping explanatory note below the estimate table.
    private TextView makeNote(String text) {
        TextView note = new TextView(this);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        params.topMargin = dp(12);
        note.setLayoutParams(params);
        note.setText(text);
        note.setTextColor(tableMuted);
        note.setTextSize(TypedValue.COMPLEX_UNIT_SP, 11);
        note.setLineSpacing(dp(2), 1f);
        return note;
    }

    private int dp(float value) {
        return Math.round(getResources().getDisplayMetrics().density * value);
    }

    // The value at index i formatted to 2 decimals, or an em dash when missing.
    private static String cell(float[] values, int i) {
        return i < values.length ? String.format(Locale.US, "%.2f", values[i]) : "—";
    }

    private static String fmt1(float value) {
        return String.format(Locale.US, "%.1f", value);
    }

    private float decodeRate(float params, float bytesPerWeight) {
        return realizedDecodeGBs * 1.0e9f / (params * bytesPerWeight);
    }

    // Compact tokens/second formatting: "t/s" suffix, with resolution that shrinks as the rate grows.
    private static String fmtRate(float tokPerSec) {
        final String number;
        if (tokPerSec >= 100.0f) {
            number = String.format(Locale.US, "%.0f", tokPerSec);
        } else if (tokPerSec >= 10.0f) {
            number = String.format(Locale.US, "%.1f", tokPerSec);
        } else {
            number = String.format(Locale.US, "%.2f", tokPerSec);
        }
        return number + " t/s";
    }

    // All numbers on the first line whose trimmed text starts with linePrefix (empty array if none).
    private static float[] numbersOnLine(String report, String linePrefix) {
        if (report == null) {
            return EMPTY_FLOATS;
        }
        int len = report.length();
        int start = 0;
        while (start < len) {
            int end = nextLineEnd(report, start, len);
            int trimmedStart = trimStart(report, start, end);
            int trimmedEnd = trimEnd(report, trimmedStart, end);
            if (trimmedLineStartsWith(report, trimmedStart, trimmedEnd, linePrefix)) {
                int valueStart = indexOf(report, ':', trimmedStart, trimmedEnd);
                int numberStart = valueStart >= 0 ? valueStart + 1 : trimmedStart;
                return numbersInRegion(report, numberStart, trimmedEnd);
            }
            start = end + 1;
        }
        return EMPTY_FLOATS;
    }

    // The first line (trimmed) whose text starts with linePrefix, or null if none.
    private static String lineStartingWith(String report, String linePrefix) {
        if (report == null) {
            return null;
        }
        int len = report.length();
        int start = 0;
        while (start < len) {
            int end = nextLineEnd(report, start, len);
            int trimmedStart = trimStart(report, start, end);
            int trimmedEnd = trimEnd(report, trimmedStart, end);
            if (trimmedLineStartsWith(report, trimmedStart, trimmedEnd, linePrefix)) {
                return report.substring(trimmedStart, trimmedEnd);
            }
            start = end + 1;
        }
        return null;
    }

    private static List<String> linesStartingWith(String report, String linePrefix) {
        ArrayList<String> lines = new ArrayList<>(4);
        if (report == null) {
            return lines;
        }
        int len = report.length();
        int start = 0;
        while (start < len) {
            int end = nextLineEnd(report, start, len);
            int trimmedStart = trimStart(report, start, end);
            int trimmedEnd = trimEnd(report, trimmedStart, end);
            if (trimmedLineStartsWith(report, trimmedStart, trimmedEnd, linePrefix)) {
                lines.add(report.substring(trimmedStart, trimmedEnd));
            }
            start = end + 1;
        }
        return lines;
    }

    private static List<String> presentCoreGroups(String report) {
        ArrayList<String> groups = new ArrayList<>(CORE_GROUP_LABELS.length);
        for (String group : CORE_GROUP_LABELS) {
            if (lineStartingWith(report, group + " Read") != null
                    || lineStartingWith(report, group + " peak") != null) {
                groups.add(group);
            }
        }
        return groups;
    }

    private static float[] firstNumbersOnLine(String report, String... linePrefixes) {
        for (String linePrefix : linePrefixes) {
            float[] numbers = numbersOnLine(report, linePrefix);
            if (numbers.length > 0) {
                return numbers;
            }
        }
        return EMPTY_FLOATS;
    }

    private static float[] numbersInRegion(String text, int start, int end) {
        Matcher matcher = NUMBER_PATTERN.matcher(text);
        matcher.region(start, end);
        float[] values = new float[4];
        int count = 0;
        while (matcher.find()) {
            try {
                if (count == values.length) {
                    float[] grown = new float[values.length * 2];
                    System.arraycopy(values, 0, grown, 0, values.length);
                    values = grown;
                }
                values[count++] = Float.parseFloat(matcher.group());
            } catch (NumberFormatException ignored) {
                // Skip any token that is not a finite float.
            }
        }
        if (count == 0) {
            return EMPTY_FLOATS;
        }
        float[] out = new float[count];
        System.arraycopy(values, 0, out, 0, count);
        return out;
    }

    private static int nextLineEnd(String text, int start, int len) {
        int end = text.indexOf('\n', start);
        return end < 0 ? len : end;
    }

    private static int trimStart(String text, int start, int end) {
        while (start < end && text.charAt(start) <= ' ') {
            start++;
        }
        return start;
    }

    private static int trimEnd(String text, int start, int end) {
        while (end > start && text.charAt(end - 1) <= ' ') {
            end--;
        }
        return end;
    }

    private static boolean trimmedLineStartsWith(String text, int start, int end, String prefix) {
        int prefixLen = prefix.length();
        return end - start >= prefixLen && text.regionMatches(start, prefix, 0, prefixLen);
    }

    private static int indexOf(String text, char target, int start, int end) {
        for (int i = start; i < end; i++) {
            if (text.charAt(i) == target) {
                return i;
            }
        }
        return -1;
    }
}
