package com.example.myapplication;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.os.IBinder;
import android.util.Log;

import androidx.annotation.Nullable;
import androidx.core.app.NotificationCompat;
import androidx.core.content.ContextCompat;

import java.util.concurrent.atomic.AtomicInteger;

public final class GenerationService extends Service {
    private static final String TAG = "GenerationService";
    private static final String CHANNEL_ID = "llm_generation";
    private static final int NOTIFICATION_ID = 1001;
    private static final String EXTRA_MODE = "mode";
    private static final int MODE_NONE = 0;
    private static final int MODE_GENERATING = 1;
    private static final int MODE_LOADING = 2;
    // Single source of truth for the desired foreground mode. Both generation and model-loading starts
    // publish into it; onStartCommand renders the notification idempotently from it. Replaces the former
    // `running` flag, which was set only in onCreate() and consulted only by generation starts.
    private static final AtomicInteger requestedMode = new AtomicInteger(MODE_NONE);

    public static void startGeneration(Context context) {
        start(context, MODE_GENERATING);
    }

    public static void startModelLoading(Context context) {
        start(context, MODE_LOADING);
    }

    private static void start(Context context, int mode) {
        requestedMode.set(mode);
        try {
            ContextCompat.startForegroundService(
                    context.getApplicationContext(),
                    new Intent(context, GenerationService.class).putExtra(EXTRA_MODE, mode));
        } catch (RuntimeException error) {
            Log.e(TAG, "Could not start generation foreground service", error);
        }
    }

    public static void stop(Context context) {
        requestedMode.set(MODE_NONE);
        context.getApplicationContext().stopService(
                new Intent(context, GenerationService.class));
    }

    @Override
    public void onCreate() {
        super.onCreate();
        createNotificationChannel();
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        int mode = requestedMode.get();
        if (mode == MODE_NONE) {
            mode = intent == null ? MODE_GENERATING
                    : intent.getIntExtra(EXTRA_MODE, MODE_GENERATING);
        }
        startForeground(NOTIFICATION_ID, buildNotification(mode));
        return START_NOT_STICKY;
    }

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    @Override
    public void onDestroy() {
        requestedMode.set(MODE_NONE);
        stopForeground(STOP_FOREGROUND_REMOVE);
        super.onDestroy();
    }

    private void createNotificationChannel() {
        NotificationManager manager = getSystemService(NotificationManager.class);
        if (manager == null || manager.getNotificationChannel(CHANNEL_ID) != null) {
            return;
        }
        NotificationChannel channel = new NotificationChannel(
                CHANNEL_ID,
                getString(R.string.generation_notification_channel),
                NotificationManager.IMPORTANCE_LOW);
        channel.setDescription(getString(R.string.generation_notification_channel_description));
        manager.createNotificationChannel(channel);
    }

    private Notification buildNotification(int mode) {
        Intent activityIntent = new Intent(this, MainActivity.class)
                .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP);
        PendingIntent contentIntent = PendingIntent.getActivity(
                this,
                0,
                activityIntent,
                PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);
        return new NotificationCompat.Builder(this, CHANNEL_ID)
                .setSmallIcon(R.drawable.ic_speed)
                .setContentTitle(getString(mode == MODE_LOADING
                        ? R.string.model_loading_notification_title
                        : R.string.generation_notification_title))
                .setContentText(getString(mode == MODE_LOADING
                        ? R.string.model_loading_notification_text
                        : R.string.generation_notification_text))
                .setContentIntent(contentIntent)
                .setOngoing(true)
                .setOnlyAlertOnce(true)
                .setCategory(NotificationCompat.CATEGORY_PROGRESS)
                .build();
    }
}
