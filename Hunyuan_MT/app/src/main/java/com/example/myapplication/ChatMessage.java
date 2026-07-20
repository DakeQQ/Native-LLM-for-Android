package com.example.myapplication;

import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.ref.SoftReference;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

public final class ChatMessage {
    private static final String TAG = "ChatMessage";
    public static final int TYPE_USER = 0;
    public static final int TYPE_SERVER = 1;
    // Loading placeholder shown while deferred work waits for the first streamed token.
    public static final int TYPE_LOADING = 2;
    // turnId for non-editable rows; user rows carry a monotonic id for KV-cache rollback.
    public static final int NO_TURN = -1;

    private final int type;
    private String content;
    private String contentPagePath;
    // Softly-held cache of the last disk read-back of paged content, so repeated binds don't re-read;
    // reclaimable under memory pressure. Cleared when the message returns to streaming.
    private SoftReference<String> pagedContentCache;
    private StringBuilder streamingContent;
    private final int turnId;
    private boolean markdownReady;
    private final long firstTokenTimeMillis;
    private long lastTokenTimeMillis;

    public ChatMessage(int type, String content, int turnId, boolean markdownReady,
                       long firstTokenTimeMillis, long lastTokenTimeMillis) {
        this.type = type;
        this.content = content == null ? "" : content;
        this.turnId = turnId;
        this.markdownReady = markdownReady;
        long now = System.currentTimeMillis();
        if (firstTokenTimeMillis <= 0L) {
            firstTokenTimeMillis = now;
        }
        if (lastTokenTimeMillis <= 0L) {
            lastTokenTimeMillis = firstTokenTimeMillis;
        }
        if (lastTokenTimeMillis < firstTokenTimeMillis) {
            lastTokenTimeMillis = firstTokenTimeMillis;
        }
        this.firstTokenTimeMillis = firstTokenTimeMillis;
        this.lastTokenTimeMillis = lastTokenTimeMillis;
    }

    public ChatMessage(int type, String content) {
        this(type, content, NO_TURN, true, 0L, 0L);
    }

    private static File storageDirectory(File cacheDirectory) {
        return new File(cacheDirectory, "chat-transcript");
    }

    public static void clearStaleStorage(File cacheDirectory) {
        File[] files = storageDirectory(cacheDirectory).listFiles();
        if (files == null) {
            return;
        }
        for (File file : files) {
            if (file.isFile()) {
                file.delete();
            }
        }
    }

    public void deleteStorage() {
        deletePath(contentPagePath);
    }

    private static void deletePath(String path) {
        if (path != null && !path.isEmpty()) {
            new File(path).delete();
        }
    }

    public ChatMessage withContent(String content) {
        return new ChatMessage(type, content, turnId, markdownReady,
            firstTokenTimeMillis, lastTokenTimeMillis);
    }

    public ChatMessage withStreamingUpdate(String content, boolean markdownReady, long lastTokenTimeMillis) {
        return new ChatMessage(type, content, turnId, markdownReady,
                firstTokenTimeMillis, lastTokenTimeMillis);
    }

    public synchronized void appendStreaming(String chunk, long eventTimeMillis) {
        if (streamingContent == null) {
            String current = content();
            streamingContent = new StringBuilder(Math.max(512, current.length() + 256));
            streamingContent.append(current);
            content = "";
            contentPagePath = null;
            pagedContentCache = null;   // returning to streaming: any paged read-back is now stale
        }
        streamingContent.append(chunk);
        markdownReady = false;
        lastTokenTimeMillis = Math.max(lastTokenTimeMillis, eventTimeMillis);
    }

    public synchronized void finishStreaming() {
        if (streamingContent != null) {
            content = streamingContent.toString();
            streamingContent = null;
        }
        markdownReady = true;
    }

    public boolean pageContentToDisk(File cacheDirectory) {
        final String snapshot;
        synchronized (this) {
            if (streamingContent != null || contentPagePath != null || content == null ||
                    content.length() < 256) {
                return contentPagePath != null;
            }
            snapshot = content;
        }
        File directory = storageDirectory(cacheDirectory);
        File outputFile = null;
        try {
            if ((!directory.isDirectory() && !directory.mkdirs()) || !directory.isDirectory()) {
                return false;
            }
            outputFile = File.createTempFile("text-", ".txt", directory);
            try (FileOutputStream output = new FileOutputStream(outputFile)) {
                output.write(snapshot.getBytes(StandardCharsets.UTF_8));
                output.flush();
            }
            synchronized (this) {
                if (streamingContent == null && contentPagePath == null && content == snapshot) {
                    contentPagePath = outputFile.getAbsolutePath();
                    content = null;
                    return true;
                }
            }
        } catch (IOException error) {
            Log.e(TAG, "Could not page chat text", error);
        }
        if (outputFile != null && !outputFile.delete()) {
            Log.w(TAG, "Could not remove unused chat text page: " + outputFile);
        }
        return false;
    }

    public int type() { return type; }
    public String content() {
        final String pagePath;
        synchronized (this) {
            if (streamingContent != null) {
                return streamingContent.toString();
            }
            if (content != null) {
                return content;
            }
            if (contentPagePath == null) {
                return "";
            }
            SoftReference<String> cache = pagedContentCache;
            String cached = cache == null ? null : cache.get();
            if (cached != null) {
                return cached;
            }
            pagePath = contentPagePath;
        }
        // Read the paged text WITHOUT holding the monitor, so a slow disk read during a RecyclerView bind
        // (main thread) never blocks a concurrent appendStreaming/finishStreaming/pageContentToDisk.
        final String text;
        try {
            text = new String(Files.readAllBytes(new File(pagePath).toPath()),
                    StandardCharsets.UTF_8);
        } catch (IOException error) {
            Log.e(TAG, "Could not read paged chat text", error);
            return "";
        }
        synchronized (this) {
            // Cache only if still paged to the same file (not re-streamed/cleared while we read).
            if (streamingContent == null && content == null && pagePath.equals(contentPagePath)) {
                pagedContentCache = new SoftReference<>(text);
            }
        }
        return text;
    }
    public int turnId() { return turnId; }
    public boolean markdownReady() { return markdownReady; }
    public long firstTokenTimeMillis() { return firstTokenTimeMillis; }
    public long lastTokenTimeMillis() { return lastTokenTimeMillis; }
}
