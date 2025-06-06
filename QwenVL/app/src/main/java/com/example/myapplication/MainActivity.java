package com.example.myapplication;

import static com.example.myapplication.GLRender.camera_height;
import static com.example.myapplication.GLRender.camera_width;
import static com.example.myapplication.GLRender.executorService;
import static com.example.myapplication.GLRender.focus_area;
import static com.example.myapplication.GLRender.pixel_values;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.hardware.camera2.params.OutputConfiguration;
import android.hardware.camera2.params.SessionConfiguration;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.view.Surface;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Switch;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.google.android.material.textfield.TextInputEditText;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private static final boolean low_memory_mode = true;  // Enable it for reduce peak memory consume. But only works for (*.onnx + external_data), not for *.ort format.
    private static final int max_seq_len = 4096;         // Set the value with the exported model.
    public static final int font_size = 16;
    private Button sendButton;
    private TextInputEditText inputBox;
    @SuppressLint("UseSwitchCompatOrMaterialCode")
    private Switch switch_vision;
    private static RecyclerView answerView;
    private static ChatAdapter chatAdapter;
    private static List<ChatMessage> messages;
    private static String usrInputText = "";
    private static final String first_talk = "请输入问题 Enter Questions";
    private static final String load_failed = "模型加载失败。\nModel loading failed.";
    private static final String over_inputs = "一次输入太多单词 \nInput too many words at once.";
    private static final String file_name_vocab = "vocab_Qwen.txt";
    private static final String low_memory_mode_error = "低内存模式必须使用外部数据格式，例如 '*.onnx + ONNX文件数据'。\nThe low_memory_mode must use an external data format, such as '*.onnx + onnx_data_file.'";
    public static boolean chatting = false;
    private GLSurfaceView mGLSurfaceView;
    private Context mContext;
    private GLRender mGLRender;
    private static CameraManager mCameraManager;
    private static CameraDevice mCameraDevice;
    private static CameraCaptureSession mCaptureSession;
    private static CaptureRequest mPreviewRequest;
    private static CaptureRequest.Builder mPreviewRequestBuilder;
    private static final String mCameraId = "0";  // {BACK_MAIN_CAMERA=0, FRONT_CAMERA=1, BACK_WIDTH_CAMERA=4}, The ID Value may different on others device. Try yourself.
    private static Handler mBackgroundHandler;
    private static HandlerThread mBackgroundThread;
    public static final int REQUEST_CAMERA_PERMISSION = 1;
    static {
        System.loadLibrary("myapplication");
    }

    @SuppressLint("SetTextI18n")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        mContext = this;
        AssetManager mgr = getAssets();
        ImageView set_photo = findViewById(R.id.role_image);
        set_photo.setImageResource(R.drawable.psyduck);
        Button clearButton = findViewById(R.id.clear);
        sendButton = findViewById(R.id.send);
        inputBox = findViewById(R.id.input_text);
        messages = new ArrayList<>();
        chatAdapter = new ChatAdapter(messages);
        answerView = findViewById(R.id.result_text);
        answerView.setLayoutManager(new LinearLayoutManager(this));
        answerView.setAdapter(chatAdapter);
        switch_vision = findViewById(R.id.use_vision);
        clearButton.setOnClickListener(v -> clearHistory());
        boolean success;
        if (low_memory_mode) {
            try {
                String[] files = mgr.list("");
                if (files != null) {
                    for (String fileName : files) {
                        String[] subEntries = mgr.list(fileName);
                        if (subEntries == null || subEntries.length == 0) {
                            if (fileName.contains(".data") || fileName.contains(".onnx")) {
                                Copy_from_Asset_to_Cache(fileName, mgr);
                            }
                        }
                    }
                }
            } catch (IOException e) {
                addHistory(ChatMessage.TYPE_SERVER, low_memory_mode_error);
                throw new RuntimeException(e);
            }
            success = Load_Models_F(null, false, true);  // LLM part
            if (success) {
                success = Load_Models_B(null, true, false, false, false, false, true);  // Vision part
                if (success) { // Part C, D, E are default to using a single file for loading.
                    success = Load_Models_A(null, false, true) && Load_Models_C(mgr, false, false) && Load_Models_D(mgr, false, false) && Load_Models_E(mgr, false, false);
                }
            }
        } else {
            success = Load_Models_F(mgr, false, false);  // LLM part
            if (success) {
                success = Load_Models_B(mgr, true, false, false, false, false, false);  // Vision part
                if (success) { // Part C, D, E are default to using a single file for loading.
                    success = Load_Models_A(mgr, false, false) && Load_Models_C(mgr, false, false) && Load_Models_D(mgr, false, false) && Load_Models_E(mgr, false, false);
                }
            }
        }
        if (success) {
            Copy_from_Asset_to_Cache(file_name_vocab, mgr);
            setWindowFlag();
            initView();
            Start_Chat();
        } else {
            addHistory(ChatMessage.TYPE_SERVER, load_failed);
        }
    }

    private void setWindowFlag() {
        Window window = getWindow();
        window.addFlags(WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS);
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    private void initView() {
        mGLSurfaceView = findViewById(R.id.glSurfaceView);
        mGLSurfaceView.setEGLContextClientVersion(3);
        mGLRender = new GLRender(mContext);
        mGLSurfaceView.setRenderer(mGLRender);
    }

    @Override
    public void onResume() {
        super.onResume();
        startBackgroundThread();
        if (mGLSurfaceView != null) {
            mGLSurfaceView.onResume();
        }
        openCamera();
    }

    @Override
    public void onPause() {
        closeCamera();
        stopBackgroundThread();
        if (mGLSurfaceView != null) {
            mGLSurfaceView.onPause();
        }
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }

    private void startBackgroundThread() {
        mBackgroundThread = new HandlerThread("CameraBackground");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler(mBackgroundThread.getLooper());
    }

    private void stopBackgroundThread() {
        if (mBackgroundThread != null) {
            mBackgroundThread.quitSafely();
            try {
                mBackgroundThread.join();
                mBackgroundThread = null;
                mBackgroundHandler = null;
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    @SuppressLint("SetTextI18n")
    private void requestCameraPermission() {
        if (!ActivityCompat.shouldShowRequestPermissionRationale(this, Manifest.permission.CAMERA)) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA_PERMISSION);
        }
    }

    public void openCamera() {
        if (ContextCompat.checkSelfPermission(mContext, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            requestCameraPermission();
            return;
        }
        setUpCameraOutputs();
        try {
            mCameraManager.openCamera(mCameraId, mStateCallback, mBackgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    private void setUpCameraOutputs() {
        mCameraManager = (CameraManager) mContext.getSystemService(Context.CAMERA_SERVICE);
    }

    private final CameraDevice.StateCallback mStateCallback = new CameraDevice.StateCallback() {
        @SuppressLint({"ResourceType", "DefaultLocale", "SetTextI18n"})
        @Override
        public void onOpened(@NonNull CameraDevice cameraDevice) {
            try {
                SurfaceTexture surfaceTexture = mGLRender.getSurfaceTexture();
                if (surfaceTexture == null) {
                    return;
                }
                surfaceTexture.setDefaultBufferSize(camera_width, camera_height);
                surfaceTexture.setOnFrameAvailableListener(surfaceTexture1 -> {
                    mGLSurfaceView.requestRender();
                });
                Surface surface = new Surface(surfaceTexture);
                mCameraDevice = cameraDevice;
                mPreviewRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                mPreviewRequestBuilder.addTarget(surface);
                mPreviewRequest = mPreviewRequestBuilder.build();
                List<OutputConfiguration> outputConfigurations = new ArrayList<>();
                outputConfigurations.add(new OutputConfiguration(surface));
                SessionConfiguration sessionConfiguration = new SessionConfiguration(
                        SessionConfiguration.SESSION_REGULAR,
                        outputConfigurations,
                        executorService,
                        sessionsStateCallback
                );
                cameraDevice.createCaptureSession(sessionConfiguration);
            } catch (CameraAccessException e) {
                e.printStackTrace();
            }
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice cameraDevice) {
            cameraDevice.close();
            mCameraDevice = null;
        }

        @Override
        public void onError(@NonNull CameraDevice cameraDevice, int error) {
            cameraDevice.close();
            mCameraDevice = null;
            finish();
        }
    };

    CameraCaptureSession.StateCallback sessionsStateCallback = new CameraCaptureSession.StateCallback() {
        @Override
        public void onConfigured(@NonNull CameraCaptureSession session) {
            if (null == mCameraDevice) return;
            mCaptureSession = session;
            try {
                // Turn off for processing speed (lower power consumption). / Turn on for image quality.
                mPreviewRequestBuilder.set(CaptureRequest.CONTROL_CAPTURE_INTENT, CaptureRequest.CONTROL_CAPTURE_INTENT_PREVIEW);
                mPreviewRequestBuilder.set(CaptureRequest.CONTROL_MODE, CaptureRequest.CONTROL_MODE_AUTO);
                mPreviewRequestBuilder.set(CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE, CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_ON);
                mPreviewRequestBuilder.set(CaptureRequest.NOISE_REDUCTION_MODE, CaptureRequest.NOISE_REDUCTION_MODE_HIGH_QUALITY);
//                mPreviewRequestBuilder.set(CaptureRequest.CONTROL_AF_REGIONS, focus_area);
                mPreviewRequest = mPreviewRequestBuilder.build();
                mCaptureSession.setRepeatingRequest(mPreviewRequest, mCaptureCallback, mBackgroundHandler);
            } catch (CameraAccessException e) {
                e.printStackTrace();
            }
        }

        @Override
        public void onConfigureFailed(@NonNull CameraCaptureSession session) {
        }
    };

    private final CameraCaptureSession.CaptureCallback mCaptureCallback
            = new CameraCaptureSession.CaptureCallback() {
        @Override
        public void onCaptureProgressed(@NonNull CameraCaptureSession session,
                                        @NonNull CaptureRequest request,
                                        @NonNull CaptureResult partialResult) {
        }

        @Override
        public void onCaptureCompleted(@NonNull CameraCaptureSession session,
                                       @NonNull CaptureRequest request,
                                       @NonNull TotalCaptureResult result) {
        }
    };

    private void closeCamera() {
        if (null != mCaptureSession) {
            try {
                mCaptureSession.stopRepeating();
                mCaptureSession.close();
            } catch (CameraAccessException e) {
                e.printStackTrace();
            }
            mCaptureSession = null;
        }
        if (null != mCameraDevice) {
            mCameraDevice.close();
            mCameraDevice = null;
        }
    }

    private void Copy_from_Asset_to_Cache(String fileName, AssetManager mgr){
        try {
            File cacheDir = getCacheDir();
            if (!cacheDir.exists()) {
                if (!cacheDir.mkdirs()) {
                    System.out.println("Directory creation failed.");
                }
            }
            File outFile = new File(cacheDir,fileName);
            if (!outFile.exists()){
                if (!outFile.createNewFile()) {
                    return;
                }
            } else {
                if (outFile.length() > 10) {
                    return;
                }
            }
            InputStream is = mgr.open(fileName);
            FileOutputStream os = new FileOutputStream(outFile);
            byte[] buffer = new byte[102400];
            int len;
            while ((len = is.read(buffer)) != -1) {
                os.write(buffer,0, len);
            }
            is.close();
            os.flush();
            os.close();
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    @SuppressLint("NotifyDataSetChanged")
    private static void addHistory(int messageType, String result) {
        int lastMessageIndex = messages.size() - 1;
        if (lastMessageIndex >= 0 && messages.get(lastMessageIndex).type() == messageType) {
            messages.set(lastMessageIndex, new ChatMessage(messageType, messages.get(lastMessageIndex).content() + result));
        } else {
            messages.add(new ChatMessage(messageType, result));
        }
        chatAdapter.notifyDataSetChanged();
        answerView.smoothScrollToPosition(lastMessageIndex + 1);
    }

    @SuppressLint("NotifyDataSetChanged")
    private void clearHistory(){
        inputBox.setText("");
        usrInputText = "";
        messages.clear();
        chatAdapter.notifyDataSetChanged();
        answerView.smoothScrollToPosition(0);
        chatting = false;
        showToast(MainActivity.this, "已清除 Cleared",false);
    }

    private void Start_Chat() {
        sendButton.setOnClickListener(view -> {
            usrInputText = String.valueOf(inputBox.getText());
            if (usrInputText.isEmpty()){
                showToast(MainActivity.this, first_talk,false);
            } else {
                addHistory(ChatMessage.TYPE_USER, usrInputText);
                inputBox.setText("");
                startLLM();
            }
        });
    }

    private class LLMThread extends Thread {
        private String LLM_Talk = "";
        private int response_count = 0;
        boolean use_vision = false;
        @SuppressLint("DefaultLocale")
        @Override
        public void run() {
            use_vision = switch_vision.isChecked();
            long image_embed_count = System.currentTimeMillis();
            Run_LLM_Part_0(pixel_values, usrInputText, use_vision);
            if (use_vision) {
                runOnUiThread(() -> {
                    @SuppressLint("DefaultLocale")
                    String formattedTimeCost = String.format("%.4f", (System.currentTimeMillis() - image_embed_count) * 0.001f);
                    addHistory(ChatMessage.TYPE_SERVER, "Image Process Complete. \n\nTime Cost: " + formattedTimeCost + " seconds\n\n");
                });
            }
            LLM_Talk = Run_LLM_Part_1(true, use_vision);
            chatting = true;
            usrInputText = "";
            long start_time = System.currentTimeMillis();
            while (chatting) {
                runOnUiThread(() -> {
                    switch (LLM_Talk) {
                        case "END" -> {
                            if (chatting) {  // Java multithreading may not stop immediately. Therefore, use a switch to prevent repeat print.
                                chatting = false;
                                addHistory(ChatMessage.TYPE_SERVER, "\n\nDecode: " + String.format("%.4f", ((float) 1000 * response_count / (System.currentTimeMillis() - start_time))) + " token/s");
                            }
                        }
                        case "Over_Inputs" -> {
                            if (chatting) {
                                chatting = false;
                                addHistory(ChatMessage.TYPE_SERVER, over_inputs);
                            }
                        }
                        default -> {
                            addHistory(ChatMessage.TYPE_SERVER, LLM_Talk);
                            response_count += 1;
                        }
                    }
                });
                if (response_count > max_seq_len) {
                    break;
                }
                LLM_Talk = Run_LLM_Part_1(false, use_vision);
            }
        }
    }

    @SuppressLint("SetTextI18n")
    private void startLLM() {
        LLMThread llmThread = new LLMThread();
        llmThread.start();
    }

    private static void showToast(final Activity context, final String content, boolean display_long){
        if (display_long) {
            context.runOnUiThread(() -> Toast.makeText(context, content, Toast.LENGTH_LONG).show());
        } else {
            context.runOnUiThread(() -> Toast.makeText(context, content, Toast.LENGTH_SHORT).show());
        }
    }
    public static native boolean Process_Init(int textureId);
    public static native int[] Process_Texture();
    private native boolean Load_Models_A(AssetManager assetManager, boolean USE_XNNPACK, boolean LOW_MEMORY_MODE);
    private native boolean Load_Models_B(AssetManager assetManager, boolean USE_FLOAT_MODEL, boolean USE_QNN_CPU, boolean USE_QNN_GPU, boolean USE_QNN_NPU, boolean USE_XNNPACK, boolean LOW_MEMORY_MODE);
    private native boolean Load_Models_C(AssetManager assetManager, boolean USE_XNNPACK, boolean LOW_MEMORY_MODE);
    private native boolean Load_Models_D(AssetManager assetManager, boolean USE_XNNPACK, boolean LOW_MEMORY_MODE);
    private native boolean Load_Models_E(AssetManager assetManager, boolean USE_XNNPACK, boolean LOW_MEMORY_MODE);
    private native boolean Load_Models_F(AssetManager assetManager, boolean USE_XNNPACK, boolean LOW_MEMORY_MODE);
    private static native void Run_LLM_Part_0(byte[] pixel_values, String query, boolean use_vision);
    private static native String Run_LLM_Part_1(boolean add_prompt, boolean use_vision);
}
