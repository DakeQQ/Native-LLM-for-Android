package com.example.myapplication;

import android.annotation.SuppressLint;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;
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
        // Apply the rounded bubble drawable + matching high-contrast text colour per role.
        // (Setting a flat colour resource here would override the rounded drawable.)
        switch (message.type()) {
            case ChatMessage.TYPE_USER -> {
                holder.messageText.setBackgroundResource(R.drawable.user_message_background);
                holder.messageText.setTextColor(
                        ContextCompat.getColor(holder.itemView.getContext(), R.color.textPrimary));
            }
            case ChatMessage.TYPE_SERVER -> {
                holder.messageText.setBackgroundResource(R.drawable.server_message_background);
                holder.messageText.setTextColor(
                        ContextCompat.getColor(holder.itemView.getContext(), R.color.textOnYellow));
            }
        }
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
