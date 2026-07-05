package com.example.myapplication;

public record ChatMessage(int type, String content, int turnId, boolean markdownReady) {
    public static final int TYPE_USER = 0;
    public static final int TYPE_SERVER = 1;
    // "Typing" placeholder shown while beam search decodes silently.
    public static final int TYPE_LOADING = 2;
    // turnId for non-editable rows; user rows carry a monotonic id for KV-cache rollback.
    public static final int NO_TURN = -1;

    public ChatMessage(int type, String content, int turnId) {
        this(type, content, turnId, true);
    }

    public ChatMessage(int type, String content) {
        this(type, content, NO_TURN);
    }

    public ChatMessage withContent(String content) {
        return new ChatMessage(type, content, turnId, markdownReady);
    }

    public ChatMessage withMarkdownReady(boolean markdownReady) {
        return new ChatMessage(type, content, turnId, markdownReady);
    }
}
