package com.example.myapplication;

public record ChatMessage(int type, String content) {
    public static final int TYPE_USER = 0;
    public static final int TYPE_SERVER = 1;
}
