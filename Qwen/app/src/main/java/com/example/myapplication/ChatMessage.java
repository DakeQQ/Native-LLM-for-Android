package com.example.myapplication;

import android.graphics.Bitmap;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.ref.SoftReference;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class ChatMessage {
    private static final String TAG = "ChatMessage";
    private static final int MAX_VIDEO_PREVIEW_FRAMES = 12;
    public static final int TYPE_USER = 0;
    public static final int TYPE_SERVER = 1;
    // "Typing" placeholder shown while beam search decodes silently.
    public static final int TYPE_LOADING = 2;
    // User-side image bubble carrying a vision snapshot thumbnail.
    public static final int TYPE_USER_IMAGE = 3;
    // User-side video bubble carrying sampled preview frames from a long-press recording.
    public static final int TYPE_USER_VIDEO = 4;
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
    private final String thumbnailPath;
    private final List<String> videoFramePaths;
    private final int mediaId;
    private final boolean mediaPreviewFailed;
    private final long firstTokenTimeMillis;
    private long lastTokenTimeMillis;

    public ChatMessage(int type, String content, int turnId, boolean markdownReady, String thumbnailPath,
                       List<String> videoFramePaths, long firstTokenTimeMillis, long lastTokenTimeMillis) {
        this(type, content, turnId, markdownReady, thumbnailPath, videoFramePaths, 0,
            false, firstTokenTimeMillis, lastTokenTimeMillis);
    }

    private ChatMessage(int type, String content, int turnId, boolean markdownReady,
                        String thumbnailPath, List<String> videoFramePaths, int mediaId,
                boolean mediaPreviewFailed, long firstTokenTimeMillis,
                long lastTokenTimeMillis) {
        this.type = type;
        this.content = content == null ? "" : content;
        this.turnId = turnId;
        this.markdownReady = markdownReady;
        this.thumbnailPath = thumbnailPath;
        this.videoFramePaths = videoFramePaths == null
                ? Collections.emptyList()
            : Collections.unmodifiableList(new ArrayList<>(videoFramePaths));
        this.mediaId = mediaId;
        this.mediaPreviewFailed = mediaPreviewFailed;
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
        this(type, content, NO_TURN, true, null, null, 0L, 0L);
    }

    public static ChatMessage visionPlaceholder(boolean video, int turnId, int mediaId) {
        return new ChatMessage(video ? TYPE_USER_VIDEO : TYPE_USER_IMAGE, "", turnId, true,
                null, null, mediaId, false, 0L, 0L);
    }

    // Called on the media worker. The model path already owns its separate CHW input.
    public static ChatMessage image(Bitmap thumbnail, String caption, int turnId, int mediaId,
                                    File cacheDirectory) {
        String compressed = compressAndRecycle(thumbnail, 84, mediaDirectory(cacheDirectory), "image-");
        return new ChatMessage(TYPE_USER_IMAGE, caption, turnId, true, compressed, null, mediaId,
            compressed == null, 0L, 0L);
    }

    // Called on the media worker. Only display previews are persisted; native owns the model input.
    public static ChatMessage video(List<Bitmap> frames, String caption, int turnId, int mediaId,
                                    File cacheDirectory) {
        List<String> safeFrames = new ArrayList<>();
        File directory = mediaDirectory(cacheDirectory);
        int expectedPreviewCount = frames == null
            ? 0 : Math.min(frames.size(), MAX_VIDEO_PREVIEW_FRAMES);
        if (frames != null) {
            int frameCount = frames.size();
            int previewCount = expectedPreviewCount;
            int previewIndex = 0;
            for (int frameIndex = 0; frameIndex < frameCount; ++frameIndex) {
                int selectedIndex = previewCount <= 1
                        ? 0
                        : (int) ((long) previewIndex * (frameCount - 1) / (previewCount - 1));
                Bitmap frame = frames.get(frameIndex);
                if (previewIndex < previewCount && frameIndex == selectedIndex) {
                    String compressed = compressAndRecycle(frame, 78, directory, "video-");
                    if (compressed != null) {
                        safeFrames.add(compressed);
                    }
                    previewIndex++;
                } else if (frame != null && !frame.isRecycled()) {
                    frame.recycle();
                }
            }
        }
        safeFrames = Collections.unmodifiableList(safeFrames);
        String poster = safeFrames.isEmpty() ? null : safeFrames.get(0);
        boolean previewFailed = expectedPreviewCount == 0 || safeFrames.size() < expectedPreviewCount;
        return new ChatMessage(TYPE_USER_VIDEO, caption == null ? "" : caption, turnId, true,
            poster, safeFrames, mediaId, previewFailed, 0L, 0L);
    }

    private static String compressAndRecycle(Bitmap bitmap, int quality, File directory,
                                             String prefix) {
        if (bitmap == null || bitmap.isRecycled()) {
            return null;
        }
        File outputFile = null;
        boolean complete = false;
        try {
            if ((!directory.isDirectory() && !directory.mkdirs()) || !directory.isDirectory()) {
                Log.e(TAG, "Could not create chat media directory: " + directory);
                return null;
            }
            outputFile = File.createTempFile(prefix, ".jpg", directory);
            try (FileOutputStream output = new FileOutputStream(outputFile)) {
                if (bitmap.compress(Bitmap.CompressFormat.JPEG, quality, output)) {
                    complete = true;
                    return outputFile.getAbsolutePath();
                }
                Log.e(TAG, "Bitmap JPEG compression returned false: " + outputFile);
            }
        } catch (IOException error) {
            Log.e(TAG, "Could not persist chat media preview", error);
            return null;
        } finally {
            if (!complete && outputFile != null) {
                outputFile.delete();
            }
            bitmap.recycle();
        }
        return null;
    }

    private static File mediaDirectory(File cacheDirectory) {
        return new File(cacheDirectory, "chat-media");
    }

    public static void clearStaleMedia(File cacheDirectory) {
        File[] files = mediaDirectory(cacheDirectory).listFiles();
        if (files == null) {
            return;
        }
        for (File file : files) {
            if (file.isFile()) {
                file.delete();
            }
        }
    }

    public void deleteMedia() {
        if (!videoFramePaths.isEmpty()) {
            for (String path : videoFramePaths) {
                deletePath(path);
            }
        } else {
            deletePath(thumbnailPath);
        }
        deletePath(contentPagePath);
    }

    private static void deletePath(String path) {
        if (path != null && !path.isEmpty()) {
            new File(path).delete();
        }
    }

    public ChatMessage withContent(String content) {
        return new ChatMessage(type, content, turnId, markdownReady, thumbnailPath, videoFramePaths,
                mediaId, mediaPreviewFailed, firstTokenTimeMillis, lastTokenTimeMillis);
    }

    public ChatMessage withStreamingUpdate(String content, boolean markdownReady, long lastTokenTimeMillis) {
        return new ChatMessage(type, content, turnId, markdownReady, thumbnailPath, videoFramePaths,
                mediaId, mediaPreviewFailed, firstTokenTimeMillis, lastTokenTimeMillis);
    }

    public ChatMessage withMediaFrom(ChatMessage encoded) {
        return new ChatMessage(type, content(), turnId, markdownReady, encoded.thumbnailPath,
                encoded.videoFramePaths, mediaId, encoded.mediaPreviewFailed,
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
        File directory = mediaDirectory(cacheDirectory);
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
    public String thumbnailPath() { return thumbnailPath; }
    public List<String> videoFramePaths() { return videoFramePaths; }
    public int mediaId() { return mediaId; }
    public boolean mediaPreviewFailed() { return mediaPreviewFailed; }
    public long firstTokenTimeMillis() { return firstTokenTimeMillis; }
    public long lastTokenTimeMillis() { return lastTokenTimeMillis; }
}
