package com.example.myapplication;

import android.annotation.SuppressLint;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import java.util.List;
public class ChatAdapter extends RecyclerView.Adapter<ChatAdapter.ViewHolder> {
    private final List<ChatMessage> messages;
    public ChatAdapter(List<ChatMessage> messages) {
        this.messages = messages;
    }
    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        int layoutRes = viewType == ChatMessage.TYPE_USER ?
                R.layout.item_user_message : R.layout.item_server_message;
        View view = LayoutInflater.from(parent.getContext()).inflate(layoutRes, parent, false);
        return new ViewHolder(view);
    }
    @SuppressLint("NotifyDataSetChanged")
    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        ChatMessage message = messages.get(position);
        holder.messageText.setTextSize(MainActivity.font_size);
        holder.messageText.setText(message.content());
        int backgroundColorRes = 0;
        switch (message.type()) {
            case ChatMessage.TYPE_USER -> backgroundColorRes = R.color.userMessageBackground;
            case ChatMessage.TYPE_SERVER -> backgroundColorRes = R.color.serverMessageBackground;
        }
        holder.messageText.setBackgroundResource(backgroundColorRes);
    }
    @Override
    public int getItemCount() {
        return messages.size();
    }
    @Override
    public int getItemViewType(int position) {
        return messages.get(position).type();
    }
    static class ViewHolder extends RecyclerView.ViewHolder {
        TextView messageText;
        ViewHolder(View itemView) {
            super(itemView);
            messageText = itemView.findViewById(R.id.messageText);
        }
    }
}
