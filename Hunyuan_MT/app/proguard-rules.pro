# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# If your project uses WebView with JS, uncomment the following
# and specify the fully qualified class name to the JavaScript interface
# class:
#-keepclassmembers class fqcn.of.javascript.interface.for.webview {
#   public *;
#}

# Uncomment this to preserve the line number information for
# debugging stack traces.
#-keepattributes SourceFile,LineNumberTable

# If you keep the line number information, uncomment this to
# hide the original source file name.
#-renamesourcefileattribute SourceFile

# ── JNI bridge (Native-LLM) ──────────────────────────────────────────
# R8 full mode (android.enableR8.fullMode=true) aggressively removes or renames
# any member with no visible Java caller. The native libraries resolve these
# symbols by exact class + method name via JNI, so they MUST be kept or the app
# crashes at runtime (UnsatisfiedLinkError / GetStaticMethodID -> null).

# Native entry points (Java -> C++). Keeps the native method names and their
# declaring class names so the Java_<pkg>_<Class>_<method> symbols resolve.
-keepclasseswithmembernames,includedescriptorclasses class * {
    native <methods>;
}

# Static callbacks invoked FROM native (C++ -> Java) via GetStaticMethodID in
# processLLM.cpp. They have no Java caller, so full mode would strip them.
-keepclassmembers class com.example.myapplication.MainActivity {
    private static void onTokenStream(java.lang.String);
    private static void onPerfStats(float, float, int, int, int, int, boolean);
    private static void onPostProcessingState(boolean);
}
