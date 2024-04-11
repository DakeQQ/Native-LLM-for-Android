package com.example.myapplication;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.example.myapplication.databinding.ActivityMainBinding;
import com.google.android.material.textfield.TextInputEditText;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class MainActivity extends AppCompatActivity {
    public static final int font_size = 18;
    private static final int single_chat_limit = 341; // The same variable which declared in the project.h. Please keep the same.
    private static final int token_unknown = -1;
    private Button sendButton;
    private TextInputEditText inputBox;
    private static RecyclerView answerView;
    private static ChatAdapter chatAdapter;
    private static List<ChatMessage> messages;
    private static String usrInputText = "";
    private static final String first_talk = "请输入问题 Enter Questions";
    private static final String load_failed = "模型加载失败。\nModel loading failed.";
    private static final String over_inputs = "一次输入太多单词 \nInput too many words at once.";
    private static final String file_name_vocab = "vocab_StableLM2.txt";
    private boolean clear_flag = false;
    private static final List<String> vocab_StableLM2 = new ArrayList<>();
    private static final int[] input_token = new int[single_chat_limit];
    private static final int[] max_logit_id = new int[1];

    static {
        System.loadLibrary("myapplication");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        com.example.myapplication.databinding.ActivityMainBinding binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());
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
        if (!Load_Models_A(mgr,false,false,false,false,false,false)) {
            addHistory(ChatMessage.TYPE_SERVER, load_failed);
        } else {
            Read_Assets(mgr);
            Pre_Process();
            Start_Chat();
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
        private int response_count = 0;
        private boolean chatting = true;
        @Override
        public void run() {
            max_logit_id[0] = Run_LLM(Tokenizer(usrInputText), input_token,true, clear_flag);
            if (clear_flag) {
                clear_flag = false;
            }
            long start_time = System.currentTimeMillis();
            while (chatting) {
                runOnUiThread(() -> {
                    switch (max_logit_id[0]) {
                        case -1 -> {
                            addHistory(ChatMessage.TYPE_SERVER, over_inputs);
                            chatting = false;
                        }
                        case -2 -> {
                            addHistory(ChatMessage.TYPE_SERVER,"\n\nDecode: " + ((float) 1000 * response_count / (System.currentTimeMillis() - start_time)) + " token/s");
                            chatting = false;
                        }
                        default -> {
                            addHistory(ChatMessage.TYPE_SERVER, vocab_StableLM2.get(max_logit_id[0]) + ' ');
                            response_count += 1;
                        }
                    }
                });
                max_logit_id[0] = Run_LLM(1, max_logit_id,false, clear_flag);
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
    private static int Search_Token_Index(@NonNull String word) {
        int index = vocab_StableLM2.indexOf(word);
        if (index != -1) {
            return index;
        }
        return token_unknown;
    }
    private static int Tokenizer(String question) {
        Matcher matcher = Pattern.compile("\\p{InCJK_UNIFIED_IDEOGRAPHS}|([a-zA-Z]+)|\\d|\\p{Punct}").matcher(question);
        int count = 0;
        while (matcher.find()) {
            String match = matcher.group();
            if (!match.trim().isEmpty()) {
                int search_result = Search_Token_Index(match);
                if (search_result != token_unknown) {
                    input_token[count] = search_result;
                    count++;
                    if (count == single_chat_limit - 8) {  // '8' is come from the chat prompt template.
                        break;  // If the input query exceeds (single_chat_limit - 8) words, it will be truncated directly.
                    }
                } else {
                    String[] match_split = match.split("");
                    if (match_split.length > 1) {
                        for (String words : match_split) {
                            input_token[count] = Search_Token_Index(words);
                            count++;
                            if (count == single_chat_limit - 8) {
                                break;  // If the input query exceeds (single_chat_limit - 8) words, it will be truncated directly.
                            }
                        }
                    } else {
                        input_token[count] = token_unknown;
                        count++;
                        if (count == single_chat_limit - 8) {
                            break;  // If the input query exceeds (single_chat_limit - 8) words, it will be truncated directly.
                        }
                    }
                }
            }
        }
        return count;
    }
    private void Read_Assets(AssetManager mgr) {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(mgr.open(file_name_vocab)));
            String line;
            while ((line = reader.readLine()) != null) {
                vocab_StableLM2.add(line);
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    private native boolean Pre_Process();
    private native boolean Load_Models_A(AssetManager assetManager, boolean USE_GPU, boolean FP16, boolean USE_NNAPI, boolean USE_XNNPACK, boolean USE_QNN, boolean USE_DSP_NPU);
    private static native int Run_LLM(int count_words, int[] query_ids, boolean add_prompt, boolean clear);
}