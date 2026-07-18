package com.example.myapplication;

import android.util.Log;

/**
 * Bridge to the native memory-bandwidth + compute benchmark. Its own {@code libmembw.so} links nothing
 * from ONNX Runtime. Call {@link #runMemoryBandwidthTest} off the UI thread (blocks several seconds).
 */
public final class MemBandwidth {

    private static final String TAG = "MemBandwidth";
    private static final int CPU_GROUP_TOTAL = 0;
    private static final int CPU_GROUP_LITTLE = 1;
    private static final int CPU_GROUP_BIG = 2;

    static {
        try {
            System.loadLibrary("membw");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "Failed to load membw native library: " + e.getMessage());
        }
    }

    private MemBandwidth() {
    }

    /**
     * Runs read-only weight-streaming, STREAM DRAM bandwidth (Copy/Scale/Add/Triad), and an FMA compute
     * kernel, reporting GB/s and GFLOP/s. Synchronous — blocks several seconds; params clamped natively.
     *
     * @param arraySizeMillions per-array size in millions of floats (clamped 1..96; 3 arrays)
     * @param iterations        repetitions of every kernel (clamped 1..500)
     * @param littleThreadCount little-core worker threads; 0 = skip Little
     * @param bigThreadCount    big-core worker threads; 0 = skip Big
     * @return multi-line human-readable report of measured bandwidth and compute rates
     */
    public static synchronized String runMemoryBandwidthTest(int arraySizeMillions, int iterations,
                                                             int littleThreadCount, int bigThreadCount) {
        StringBuilder report = new StringBuilder(8192);
        if (littleThreadCount > 0) {
            appendGroupReport(report, CPU_GROUP_LITTLE, arraySizeMillions, iterations, littleThreadCount);
        }
        if (bigThreadCount > 0) {
            appendGroupReport(report, CPU_GROUP_BIG, arraySizeMillions, iterations, bigThreadCount);
        }
        if (littleThreadCount > 0 && bigThreadCount > 0) {
            appendGroupReport(report, CPU_GROUP_TOTAL, arraySizeMillions, iterations,
                    littleThreadCount + bigThreadCount);
        }
        return report.toString();
    }

    private static void appendGroupReport(StringBuilder report, int mode, int arraySizeMillions,
                                          int iterations, int threadCount) {
        if (report.length() > 0) {
            report.append('\n');
        }
        report.append(runMemoryBandwidthTestForCurrentGroup(
            arraySizeMillions, iterations, threadCount, mode));
    }

    private static native String runMemoryBandwidthTestForCurrentGroup(int arraySizeMillions,
                                                                       int iterations,
                                           int threadCount,
                                           int cpuGroupMode);
}
