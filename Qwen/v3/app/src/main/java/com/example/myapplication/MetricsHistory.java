package com.example.myapplication;

import android.content.Context;
import android.content.SharedPreferences;

/**
 * Process-wide, thread-safe telemetry store backing {@link MetricChartActivity}: System time-series
 * (CPU% + RAM MB, ~1 Hz) and Throughput scatter (token-total vs tokens/s per finished prefill/decode).
 * Only the scatter is persisted to {@link App#PREFS} (survives chat reset + restart), wiped via {@link #clear()}.
 */
public final class MetricsHistory {

    /** ~30 minutes of 1 Hz system samples; older points are overwritten in-place. */
    private static final int MAX_SYSTEM_SAMPLES = 1800;
    private static final int MAX_THROUGHPUT_POINTS = 400;

    private static final MetricsHistory INSTANCE = new MetricsHistory();

    public static MetricsHistory get() {
        return INSTANCE;
    }

    /** An immutable snapshot of one plotted line: paired X/Y arrays plus their bounds. */
    public static final class Series {
        public final float[] x;
        public final float[] y;
        public final float minX;
        public final float maxX;
        public final float minY;
        public final float maxY;

        Series(float[] x, float[] y, float minX, float maxX, float minY, float maxY) {
            this.x = x;
            this.y = y;
            this.minX = minX;
            this.maxX = maxX;
            this.minY = minY;
            this.maxY = maxY;
        }

        public boolean hasData() {
            return x.length > 0;
        }
    }

    private final TimeRing cpu = new TimeRing(MAX_SYSTEM_SAMPLES);
    private final TimeRing mem = new TimeRing(MAX_SYSTEM_SAMPLES);
    private final PointRing prefill = new PointRing(MAX_THROUGHPUT_POINTS);
    private final PointRing decode = new PointRing(MAX_THROUGHPUT_POINTS);

    /** Context for persisting throughput points; null until {@link #restore(Context)}. */
    private Context appContext;

    private MetricsHistory() {
    }

    // ---- persistence ------------------------------------------------------

    /** Loads saved throughput points. Call once before any reader/writer (from {@link App#onCreate()}). */
    public synchronized void restore(Context context) {        appContext = context.getApplicationContext();
        SharedPreferences prefs = appContext.getSharedPreferences(App.PREFS, Context.MODE_PRIVATE);
        loadRing(prefill, prefs.getString(App.KEY_THROUGHPUT_PREFILL, null));
        loadRing(decode, prefs.getString(App.KEY_THROUGHPUT_DECODE, null));
    }

    /** Writes the changed throughput ring(s) to {@link App#PREFS}. No-op before {@link #restore(Context)}. */
    private void persistThroughput(boolean writePrefill, boolean writeDecode) {
        if (appContext == null) {
            return;
        }
        SharedPreferences prefs = appContext.getSharedPreferences(App.PREFS, Context.MODE_PRIVATE);
        SharedPreferences.Editor editor = prefs.edit();
        if (writePrefill) {
            editor.putString(App.KEY_THROUGHPUT_PREFILL, serializeRing(prefill));
        }
        if (writeDecode) {
            editor.putString(App.KEY_THROUGHPUT_DECODE, serializeRing(decode));
        }
        editor.apply();
    }

    // ---- writers ----------------------------------------------------------

    public synchronized void addCpu(long timestampMs, float percent) {
        cpu.add(timestampMs, percent);
    }

    public synchronized void addMemory(long timestampMs, float megabytes) {
        mem.add(timestampMs, megabytes);
    }

    public synchronized void addPrefillPoint(float tokenTotal, float tokensPerSecond) {
        if (tokenTotal > 0f && tokensPerSecond > 0f) {
            prefill.add(tokenTotal, tokensPerSecond);
            persistThroughput(true, false);
        }
    }

    public synchronized void addDecodePoint(float tokenTotal, float tokensPerSecond) {
        if (tokenTotal > 0f && tokensPerSecond > 0f) {
            decode.add(tokenTotal, tokensPerSecond);
            persistThroughput(false, true);
        }
    }

    public synchronized void clear() {
        cpu.clear();
        mem.clear();
        prefill.clear();
        decode.clear();
        persistThroughput(true, true);   // wipe the persisted copy so it does not reload on restart
    }

    // ---- readers ----------------------------------------------------------

    /** CPU% (0) and RAM MB (1) time-series sharing one time origin so both align on X (elapsed seconds). */
    public synchronized Series[] systemSeries() {
        long origin = Long.MAX_VALUE;
        origin = earliest(cpu, origin);
        origin = earliest(mem, origin);
        if (origin == Long.MAX_VALUE) {
            origin = 0L;
        }
        return new Series[]{timeSeries(cpu, origin), timeSeries(mem, origin)};
    }

    /** Prefill (index 0) and decode (index 1) as (token-total, tokens/s) scatter, sorted ascending by X. */
    public synchronized Series[] throughputSeries() {
        return new Series[]{scatterSeries(prefill), scatterSeries(decode)};
    }

    public synchronized boolean hasSystemData() {
        return !cpu.isEmpty() || !mem.isEmpty();
    }

    public synchronized boolean hasThroughputData() {
        return !prefill.isEmpty() || !decode.isEmpty();
    }

    // ---- helpers ----------------------------------------------------------

    private static long earliest(TimeRing ring, long current) {
        if (!ring.isEmpty()) {
            return Math.min(current, ring.oldestTime());
        }
        return current;
    }

    private static Series timeSeries(TimeRing ring, long origin) {
        int n = ring.size();
        float[] x = new float[n];
        float[] y = new float[n];
        float minY = Float.MAX_VALUE;
        float maxY = -Float.MAX_VALUE;
        float minX = Float.MAX_VALUE;
        float maxX = -Float.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            float sx = (ring.timeAt(i) - origin) / 1000.0f;
            float sy = ring.valueAt(i);
            x[i] = sx;
            y[i] = sy;
            if (sx < minX) minX = sx;
            if (sx > maxX) maxX = sx;
            if (sy < minY) minY = sy;
            if (sy > maxY) maxY = sy;
        }
        if (n == 0) {
            minX = maxX = minY = maxY = 0f;
        }
        return new Series(x, y, minX, maxX, minY, maxY);
    }

    private static Series scatterSeries(PointRing ring) {
        int n = ring.size();
        float[] x = new float[n];
        float[] y = new float[n];
        float minY = Float.MAX_VALUE;
        float maxY = -Float.MAX_VALUE;
        float minX = Float.MAX_VALUE;
        float maxX = -Float.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            float sx = ring.xAt(i);
            float sy = ring.yAt(i);
            x[i] = sx;
            y[i] = sy;
            if (sx < minX) minX = sx;
            if (sx > maxX) maxX = sx;
            if (sy < minY) minY = sy;
            if (sy > maxY) maxY = sy;
        }
        sortByX(x, y, ring.sortXs, ring.sortYs);
        if (n == 0) {
            minX = maxX = minY = maxY = 0f;
        }
        return new Series(x, y, minX, maxX, minY, maxY);
    }

    private static void sortByX(float[] x, float[] y, float[] tmpX, float[] tmpY) {
        int n = x.length;
        if (n < 2 || isSortedByX(x)) {
            return;
        }
        if (n < 32) {
            insertionSortByX(x, y, n);
            return;
        }
        stableMergeSortByX(x, y, tmpX, tmpY, n);
    }

    private static boolean isSortedByX(float[] x) {
        for (int i = 1; i < x.length; i++) {
            if (x[i - 1] > x[i]) {
                return false;
            }
        }
        return true;
    }

    private static void insertionSortByX(float[] x, float[] y, int n) {
        for (int i = 1; i < n; i++) {
            float keyX = x[i];
            float keyY = y[i];
            int j = i - 1;
            while (j >= 0 && x[j] > keyX) {
                x[j + 1] = x[j];
                y[j + 1] = y[j];
                j--;
            }
            x[j + 1] = keyX;
            y[j + 1] = keyY;
        }
    }

    private static void stableMergeSortByX(float[] x, float[] y, float[] tmpX, float[] tmpY, int n) {
        float[] srcX = x;
        float[] srcY = y;
        float[] dstX = tmpX;
        float[] dstY = tmpY;
        for (int width = 1; width < n; width <<= 1) {
            for (int left = 0; left < n; left += width << 1) {
                int mid = Math.min(left + width, n);
                int right = Math.min(left + (width << 1), n);
                mergeByX(srcX, srcY, dstX, dstY, left, mid, right);
            }
            float[] swapX = srcX;
            srcX = dstX;
            dstX = swapX;
            float[] swapY = srcY;
            srcY = dstY;
            dstY = swapY;
        }
        if (srcX != x) {
            System.arraycopy(srcX, 0, x, 0, n);
            System.arraycopy(srcY, 0, y, 0, n);
        }
    }

    private static void mergeByX(float[] srcX, float[] srcY, float[] dstX, float[] dstY,
                                 int left, int mid, int right) {
        int i = left;
        int j = mid;
        int out = left;
        while (i < mid && j < right) {
            if (srcX[i] <= srcX[j]) {
                dstX[out] = srcX[i];
                dstY[out] = srcY[i];
                i++;
            } else {
                dstX[out] = srcX[j];
                dstY[out] = srcY[j];
                j++;
            }
            out++;
        }
        while (i < mid) {
            dstX[out] = srcX[i];
            dstY[out] = srcY[i];
            i++;
            out++;
        }
        while (j < right) {
            dstX[out] = srcX[j];
            dstY[out] = srcY[j];
            j++;
            out++;
        }
    }

    /** Serializes a point ring, in chronological order, as {@code "x:y;x:y;..."} (empty when no points). */
    private static String serializeRing(PointRing ring) {
        int n = ring.size();
        if (n == 0) {
            return "";
        }
        StringBuilder sb = new StringBuilder(n * 12);
        for (int i = 0; i < n; i++) {
            if (i > 0) {
                sb.append(';');
            }
            sb.append(ring.xAt(i)).append(':').append(ring.yAt(i));
        }
        return sb.toString();
    }

    /** Parses a {@link #serializeRing} payload back into {@code ring}; malformed pairs are skipped. */
    private static void loadRing(PointRing ring, String data) {
        ring.clear();
        if (data == null || data.isEmpty()) {
            return;
        }
        int len = data.length();
        int start = 0;
        while (start < len) {
            int sep = data.indexOf(';', start);
            int end = sep < 0 ? len : sep;
            int colon = data.indexOf(':', start);
            if (colon > start && colon < end) {
                try {
                    float x = Float.parseFloat(data.substring(start, colon));
                    float y = Float.parseFloat(data.substring(colon + 1, end));
                    if (x > 0f && y > 0f) {
                        ring.add(x, y);
                    }
                } catch (NumberFormatException ignored) {
                    // Skip a corrupted pair, keep the rest of the saved history.
                }
            }
            if (sep < 0) {
                break;
            }
            start = sep + 1;
        }
    }

    private static final class TimeRing {
        private final long[] times;
        private final float[] values;
        private int head;
        private int count;

        TimeRing(int capacity) {
            times = new long[capacity];
            values = new float[capacity];
        }

        void add(long timestampMs, float value) {
            times[head] = timestampMs;
            values[head] = value;
            head = (head + 1) % times.length;
            if (count < times.length) {
                count++;
            }
        }

        void clear() {
            head = 0;
            count = 0;
        }

        boolean isEmpty() {
            return count == 0;
        }

        int size() {
            return count;
        }

        long oldestTime() {
            return times[indexAt(0)];
        }

        long timeAt(int chronologicalIndex) {
            return times[indexAt(chronologicalIndex)];
        }

        float valueAt(int chronologicalIndex) {
            return values[indexAt(chronologicalIndex)];
        }

        private int indexAt(int chronologicalIndex) {
            int start = count == times.length ? head : 0;
            return (start + chronologicalIndex) % times.length;
        }
    }

    private static final class PointRing {
        private final float[] xs;
        private final float[] ys;
        private final float[] sortXs;
        private final float[] sortYs;
        private int head;
        private int count;

        PointRing(int capacity) {
            xs = new float[capacity];
            ys = new float[capacity];
            sortXs = new float[capacity];
            sortYs = new float[capacity];
        }

        void add(float x, float y) {
            xs[head] = x;
            ys[head] = y;
            head = (head + 1) % xs.length;
            if (count < xs.length) {
                count++;
            }
        }

        void clear() {
            head = 0;
            count = 0;
        }

        boolean isEmpty() {
            return count == 0;
        }

        int size() {
            return count;
        }

        float xAt(int chronologicalIndex) {
            return xs[indexAt(chronologicalIndex)];
        }

        float yAt(int chronologicalIndex) {
            return ys[indexAt(chronologicalIndex)];
        }

        private int indexAt(int chronologicalIndex) {
            int start = count == xs.length ? head : 0;
            return (start + chronologicalIndex) % xs.length;
        }
    }
}
