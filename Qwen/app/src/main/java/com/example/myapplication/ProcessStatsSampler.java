package com.example.myapplication;

import android.os.SystemClock;
import android.system.Os;
import android.system.OsConstants;

import java.io.FileInputStream;
import java.io.IOException;

/** Process-local CPU and RSS sampler backed by /proc/self. Each consumer owns its baseline. */
final class ProcessStatsSampler implements AutoCloseable {
    static final class Sample {
        final long timestampMs;
        final double cpuPercent;
        final long rssBytes;

        Sample(long timestampMs, double cpuPercent, long rssBytes) {
            this.timestampMs = timestampMs;
            this.cpuPercent = cpuPercent;
            this.rssBytes = rssBytes;
        }

        boolean hasCpuPercent() {
            return !Double.isNaN(cpuPercent);
        }
    }

    private static final int PROC_BUFFER_BYTES = 1024;

    private final int cpuCount = Math.max(1, Runtime.getRuntime().availableProcessors());
    private final long clockTicksPerSecond;
    private final long pageSizeBytes;
    private final byte[] statBuffer = new byte[PROC_BUFFER_BYTES];
    private final byte[] statmBuffer = new byte[128];
    private final FileInputStream statInput;
    private final FileInputStream statmInput;
    private long lastCpuJiffies = -1L;
    private long lastCpuWallMs;

    ProcessStatsSampler() {
        clockTicksPerSecond = sysconfOr(OsConstants._SC_CLK_TCK, 100L);
        pageSizeBytes = sysconfOr(OsConstants._SC_PAGESIZE, 4096L);
        statInput = openProc("/proc/self/stat");
        statmInput = openProc("/proc/self/statm");
    }

    synchronized void resetCpuBaseline() {
        lastCpuJiffies = -1L;
        lastCpuWallMs = 0L;
    }

    synchronized Sample sample() {
        long nowMs = SystemClock.elapsedRealtime();
        long jiffies = readProcessCpuJiffies();
        double cpuPercent = Double.NaN;
        if (jiffies >= 0L && lastCpuJiffies >= 0L && nowMs > lastCpuWallMs) {
            double processSeconds = (jiffies - lastCpuJiffies) / (double) clockTicksPerSecond;
            double wallSeconds = (nowMs - lastCpuWallMs) / 1000.0;
            cpuPercent = Math.max(0.0, Math.min(100.0,
                    processSeconds / wallSeconds * 100.0 / cpuCount));
        }
        if (jiffies >= 0L) {
            lastCpuJiffies = jiffies;
            lastCpuWallMs = nowMs;
        }
        return new Sample(nowMs, cpuPercent, readProcessRssBytes());
    }

    private static long sysconfOr(int name, long fallback) {
        try {
            long value = Os.sysconf(name);
            return value > 0L ? value : fallback;
        } catch (Throwable ignored) {
            return fallback;
        }
    }

    private long readProcessCpuJiffies() {
        int length = readProc(statInput, statBuffer);
        if (length <= 0) {
            return -1L;
        }
        int commandEnd = length - 1;
        while (commandEnd >= 0 && statBuffer[commandEnd] != ')') {
            commandEnd--;
        }
        if (commandEnd < 0) {
            return -1L;
        }
        long user = parseLongField(statBuffer, commandEnd + 1, length, 11);
        long system = parseLongField(statBuffer, commandEnd + 1, length, 12);
        return user < 0L || system < 0L ? -1L : user + system;
    }

    private long readProcessRssBytes() {
        int length = readProc(statmInput, statmBuffer);
        long residentPages = parseLongField(statmBuffer, 0, length, 1);
        if (residentPages < 0L || residentPages > Long.MAX_VALUE / pageSizeBytes) {
            return -1L;
        }
        return residentPages * pageSizeBytes;
    }

    private static FileInputStream openProc(String path) {
        try {
            return new FileInputStream(path);
        } catch (IOException ignored) {
            return null;
        }
    }

    private static int readProc(FileInputStream input, byte[] buffer) {
        if (input == null) {
            return -1;
        }
        try {
            input.getChannel().position(0L);
            return input.read(buffer, 0, buffer.length);
        } catch (IOException ignored) {
            return -1;
        }
    }

    private static long parseLongField(byte[] data, int start, int length, int targetField) {
        if (length <= 0 || start < 0 || start >= length) {
            return -1L;
        }
        int cursor = start;
        int field = 0;
        while (cursor < length) {
            while (cursor < length && data[cursor] <= ' ') cursor++;
            if (cursor >= length) break;
            int tokenStart = cursor;
            while (cursor < length && data[cursor] > ' ') cursor++;
            if (field++ != targetField) continue;
            boolean negative = data[tokenStart] == '-';
            int digit = negative ? tokenStart + 1 : tokenStart;
            if (digit >= cursor) return -1L;
            long value = 0L;
            while (digit < cursor) {
                int next = data[digit++] - '0';
                if (next < 0 || next > 9 || value > (Long.MAX_VALUE - next) / 10L) {
                    return -1L;
                }
                value = value * 10L + next;
            }
            return negative ? -value : value;
        }
        return -1L;
    }

    @Override
    public synchronized void close() {
        closeQuietly(statInput);
        closeQuietly(statmInput);
    }

    private static void closeQuietly(FileInputStream input) {
        if (input != null) {
            try {
                input.close();
            } catch (IOException ignored) {
            }
        }
    }
}
