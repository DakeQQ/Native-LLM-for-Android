package com.example.myapplication;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Spinner;
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
    // Dedicated main-thread handler used ONLY for streaming token batches from the native decode loop,
    // so removeCallbacksAndMessages(null) can drop stale tokens on cancel without touching other work.
    private static final Handler mainHandler = new Handler(Looper.getMainLooper());
    // Current generation worker; joined before a new one starts so the native inference state is never
    // shared between two generations (the native side is a single global, non-reentrant context).
    private LLMThread llmThread;
    private static final String file_name_vocab_A = "vocab_Hunyuan_MT.txt";
    private static final String first_talk = "请输入问题 Enter Questions";
    private static final String load_failed = "模型加载失败。\nModel loading failed.";
    private static final String over_inputs = "一次输入太多单词 \nInput too many words at once.";
    private static final String low_memory_mode_error = "低内存模式必须使用外部数据格式，例如 '*.onnx + ONNX文件数据'。\nThe low_memory_mode must use an external data format, such as '*.onnx + onnx_data_file.'";
    private static boolean chatting = false;
    private static String target_language = "英语-English";

    // Execution Provider Types
    private static final int EP_CPU = 0;      // Default CPU execution provider
    private static final int EP_XNNPACK = 1;  // XNNPACK execution provider (optimized for mobile CPU)
    private static final int EP_QNN = 2;      // QNN execution provider (Qualcomm NPU/HTP)

    private static final String[] languageList = {
            "阿拉伯语-Arabic", "孟加拉语-Bengali", "缅甸语-Burmese", "粤语-Cantonese",
            "中文-Chinese", "繁体中文-Chinese (Traditional)", "捷克语-Czech", "荷兰语-Dutch",
            "英语-English", "菲律宾语-Filipino", "法语-French", "德语-German",
            "古吉拉特语-Gujarati", "希伯来语-Hebrew", "印地语-Hindi", "印尼语-Indonesian",
            "意大利语-Italian", "日语-Japanese", "哈萨克语-Kazakh", "高棉语-Khmer",
            "韩语-Korean", "马来语-Malay", "马拉地语-Marathi", "蒙古语-Mongolian",
            "波斯语-Persian", "波兰语-Polish", "葡萄牙语-Portuguese", "俄语-Russian",
            "西班牙语-Spanish", "泰米尔语-Tamil", "泰卢固语-Telugu", "泰语-Thai",
            "藏语-Tibetan", "土耳其语-Turkish", "乌克兰语-Ukrainian", "乌尔都语-Urdu",
            "维吾尔语-Uyghur", "越南语-Vietnamese"
    };

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

        // Initialize Language Spinner
        Spinner languageSpinner = findViewById(R.id.language_spinner);
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_item, languageList);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        languageSpinner.setAdapter(adapter);
        languageSpinner.setSelection(8);
        languageSpinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                target_language = languageList[position];
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {
                target_language = "英语-English";
            }
        });


        boolean success;
        if (low_memory_mode) {
            try {
                String[] files = mgr.list("");
                if (files != null) {
                    int count_file = 0;
                    for (String fileName : files) {
                        String[] subEntries = mgr.list(fileName);
                        if (subEntries == null || subEntries.length == 0) {
                            if (fileName.endsWith(".onnx") || fileName.contains("-")) {
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
            success = Load_Models_A(null, EP_CPU, true);
        } else {
            success = Load_Models_A(mgr, EP_CPU, false);
        }
        if (success) {
            Copy_from_Asset_to_Cache(file_name_vocab_A, mgr);
            Pre_Process();
            Start_Chat();
        } else {
            addHistory(ChatMessage.TYPE_SERVER, load_failed);
        }
    }



    // Cached C++ -> Java streaming entry point. The native decode loop invokes this (batched) from the
    // LLMThread; we marshal to the main Looper because addHistory mutates the RecyclerView. The chatting
    // gate drops any token batches that arrive after a Clear/cancel so they cannot leak into the UI.
    private static void onTokenStream(String text) {
        mainHandler.post(() -> {
            if (chatting) {
                addHistory(ChatMessage.TYPE_SERVER, text);
            }
        });
    }

    private static void addHistory(int messageType, String result) {
        int lastMessageIndex = messages.size() - 1;
        if (lastMessageIndex >= 0 && messages.get(lastMessageIndex).type() == messageType) {
            // Append to the current bubble and rebind ONLY that row (O(changed row)) instead of the
            // whole list, so streaming a long reply does not re-bind every RecyclerView item per batch.
            messages.set(lastMessageIndex, new ChatMessage(messageType, messages.get(lastMessageIndex).content() + result));
            chatAdapter.notifyItemChanged(lastMessageIndex);
            answerView.smoothScrollToPosition(lastMessageIndex);
        } else {
            messages.add(new ChatMessage(messageType, result));
            int newIndex = messages.size() - 1;
            chatAdapter.notifyItemInserted(newIndex);
            answerView.smoothScrollToPosition(newIndex);
        }
    }

    @SuppressLint("NotifyDataSetChanged")
    private void clearHistory(){
        // Cancel any in-flight generation (g_cancel) and gate out its remaining streamed token posts
        // (chatting=false), then drop those stale posts from the dedicated handler before clearing.
        Stop_LLM();
        chatting = false;
        mainHandler.removeCallbacksAndMessages(null);
        inputBox.setText("");
        usrInputText = "";
        messages.clear();
        chatAdapter.notifyDataSetChanged();
        answerView.smoothScrollToPosition(0);
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

    private static class LLMThread extends Thread {
        private final String language = target_language.split("-")[1];
        private final String zh_language = target_language.split("-")[0];
        private final String input_str;

        {
            // HunYuan-MT2: Use Chinese prompt when Chinese is involved, English prompt otherwise
            if (language.equals("Chinese") || language.equals("Chinese (Traditional)") || zh_language.contains("中文")) {
                input_str = "<｜hy_begin▁of▁sentence｜><｜hy_User｜>将以下文本翻译为" + zh_language + "，注意只需要输出翻译后的结果，不要额外解释：\n\n" + usrInputText + "<｜hy_Assistant｜>";
            } else {
                input_str = "<｜hy_begin▁of▁sentence｜><｜hy_User｜>Translate the following text into " + language + ". Note that you should only output the translated result without any additional explanation:\n\n" + usrInputText + "<｜hy_Assistant｜>";
            }
        }
        @Override
        public void run() {
            // ONE native call now performs prefill + the ENTIRE decode loop, streaming tokens to the UI
            // through onTokenStream() (batched in C++). It returns either "Over_Inputs" or the final
            // "\n\nDecode: X token/s" line, which we append once. No per-token JNI round-trip remains.
            final String result = Run_LLM(input_str);
            usrInputText = "";
            mainHandler.post(() -> {
                if (chatting) {
                    if ("Over_Inputs".equals(result)) {
                        addHistory(ChatMessage.TYPE_SERVER, over_inputs);
                    } else if (result != null && !result.isEmpty()) {
                        addHistory(ChatMessage.TYPE_SERVER, result);
                    }
                }
                chatting = false;
            });
        }
    }

    @SuppressLint("SetTextI18n")
    private void startLLM() {
        // Cancel + reap any in-flight generation so the native inference state is free before we launch a
        // new one (otherwise the native re-entrancy guard would drop this query), then drop the previous
        // generation's stale streamed token posts so they cannot leak into the new conversation.
        Stop_LLM();
        if (llmThread != null) {
            try {
                llmThread.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        mainHandler.removeCallbacksAndMessages(null);
        chatting = true;
        llmThread = new LLMThread();
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
    /**
     * Load ONNX models with specified execution provider.
     * @param assetManager Asset manager for loading model from assets (null for low memory mode)
     * @param EP_TYPE Execution provider type: EP_CPU(0), EP_XNNPACK(1), EP_QNN(2)
     * @param LOW_MEMORY_MODE Enable low memory mode (requires external data format)
     * @return true if model loaded successfully
     */
    private native boolean Load_Models_A(AssetManager assetManager, int EP_TYPE, boolean LOW_MEMORY_MODE);
    // Runs prefill + the full decode loop in C++, streaming tokens via onTokenStream(); returns the
    // final "Decode: X token/s" line (or "Over_Inputs"). One call per reply.
    private static native String Run_LLM(String QUERY);
    // Requests cooperative cancellation of an in-flight Run_LLM (sets the native g_cancel flag).
    private static native void Stop_LLM();
}
