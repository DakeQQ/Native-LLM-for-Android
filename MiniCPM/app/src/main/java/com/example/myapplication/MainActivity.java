package com.example.myapplication;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Switch;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.example.myapplication.databinding.ActivityMainBinding;
import com.google.android.material.textfield.TextInputEditText;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private static final boolean low_memory_mode = true;  // Enable it for reduce peak memory consume. But only works for (*.onnx + external_data), not for *.ort format.
    public static final int font_size = 18;
    @SuppressLint("UseSwitchCompatOrMaterialCode")
    private Button sendButton;
    private TextInputEditText inputBox;
    private static RecyclerView answerView;
    private static ChatAdapter chatAdapter;
    private static List<ChatMessage> messages;
    private static String usrInputText = "";
    private static final String file_name_vocab_A = "vocab_MiniCPM.txt";
    private static final String first_talk = "请输入问题 Enter Questions";
    private static final String load_failed = "模型加载失败。\nModel loading failed.";
    private static final String over_inputs = "一次输入太多单词 \nInput too many words at once.";
    private static final String low_memory_mode_error = "低内存模式必须使用外部数据格式，例如 '*.onnx + ONNX文件数据'。\nThe low_memory_mode must use an external data format, such as '*.onnx + onnx_data_file.'";
    private static boolean clear_flag = false;
    private static boolean chatting = false;

    static {
        System.loadLibrary("myapplication");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(ActivityMainBinding.inflate(getLayoutInflater()).getRoot());
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
        clearButton.setOnClickListener(v -> clearHistory());
        boolean success;
        if (low_memory_mode) {
            try {
                String[] files = mgr.list("");
                if (files != null) {
                    int count_file = 0;
                    for (String fileName : files) {
                        String[] subEntries = mgr.list(fileName);
                        if (subEntries == null || subEntries.length == 0) {
                            if (fileName.endsWith(".onnx") || fileName.contains(".data")) {
                                Copy_from_Asset_to_Cache(fileName, mgr);
                                count_file += 1;
                                if (count_file > 1) {
                                    break;
                                }
                            }
                        }
                    }
                    if (count_file != 2) {
                        addHistory(ChatMessage.TYPE_SERVER, low_memory_mode_error);
                        throw new RuntimeException(low_memory_mode_error);
                    }
                }
            } catch (IOException e) {
                addHistory(ChatMessage.TYPE_SERVER, low_memory_mode_error);
                throw new RuntimeException(e);
            }
            success = Load_Models_A(null, false, true);
        } else {
            success = Load_Models_A(mgr, false, false);
        }
        if (success) {
            Copy_from_Asset_to_Cache(file_name_vocab_A, mgr);
            Pre_Process();
            Start_Chat();
        } else {
            addHistory(ChatMessage.TYPE_SERVER, load_failed);
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
        clear_flag = true;
        chatting = false;
        showToast(MainActivity.this, "已清除 Cleared",false);
    }

    private void Start_Chat() {
        sendButton.setOnClickListener(view -> {
            usrInputText = String.valueOf(inputBox.getText());
            if (usrInputText.isEmpty()){
                showToast(MainActivity.this, first_talk,false);
            } else {
                startLLM();
                inputBox.setText("");
                addHistory(ChatMessage.TYPE_USER, usrInputText);
            }
        });
    }

    private class LLMThread extends Thread {
        private String LLM_Talk = "";
        private int response_count = 0;
        @SuppressLint("DefaultLocale")
        @Override
        public void run() {
            LLM_Talk = Run_LLM(usrInputText,true, clear_flag);
            usrInputText = "";
            if (clear_flag) {
                clear_flag = false;
            }
            chatting = true;
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
                LLM_Talk = Run_LLM(usrInputText, false, clear_flag);
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

    private native boolean Pre_Process();
    private native boolean Load_Models_A(AssetManager assetManager, boolean USE_XNNPACK, boolean LOW_MEMORY_MODE);
    private static native String Run_LLM(String QUERY, boolean ADD_PROMPT, boolean CLEAR);
}
