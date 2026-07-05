package com.example.myapplication;

import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.animation.AccelerateDecelerateInterpolator;
import android.view.animation.Animation;
import android.view.animation.DecelerateInterpolator;
import android.view.animation.TranslateAnimation;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.core.content.ContextCompat;
import androidx.recyclerview.widget.RecyclerView;

import java.util.HashMap;
import java.util.List;
public class ChatAdapter extends RecyclerView.Adapter<ChatAdapter.ViewHolder> {
    // Tapping a user bubble enters edit mode; the activity handles the edit + resend.
    public interface OnUserMessageClickListener {
        void onEditUserMessage(int position);
    }
    private final List<ChatMessage> messages;
    private MarkdownTextRenderer markdownRenderer;
    private OnUserMessageClickListener userMessageClickListener;
    // Highest row index that already played its one-shot entrance animation; guards streaming rebinds.
    private int lastAnimatedPosition = -1;
    // Per-row thinking-panel expand override by position (absent => expanded while thinking, else collapsed).
    // Cleared on reset/resend since those renumber rows.
    private final HashMap<Integer, Boolean> thinkExpanded = new HashMap<>();
    public ChatAdapter(List<ChatMessage> messages) {
        this.messages = messages;
    }
    void setOnUserMessageClickListener(OnUserMessageClickListener listener) {
        this.userMessageClickListener = listener;
    }
    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        int layoutRes = switch (viewType) {
            case ChatMessage.TYPE_USER -> R.layout.item_user_message;
            case ChatMessage.TYPE_LOADING -> R.layout.item_loading_message;
            default -> R.layout.item_server_message;
        };
        View view = LayoutInflater.from(parent.getContext()).inflate(layoutRes, parent, false);
        ViewHolder holder = new ViewHolder(view);
        // Apply role-constant styling + click wiring ONCE (holders are pooled per view-type). Click handlers
        // read the LIVE adapter position at click time, so they stay correct as rows stream/shift.
        if (holder.messageText != null) {
            holder.messageText.setTextSize(MainActivity.font_size);
            Context context = parent.getContext();
            if (viewType == ChatMessage.TYPE_USER) {
                holder.messageText.setBackgroundResource(R.drawable.user_message_background);
                // userText stays light in BOTH themes (bubble is vivid blue); textPrimary would flip in light mode.
                holder.messageText.setTextColor(ContextCompat.getColor(context, R.color.userText));
                // Disable selection so a single tap enters edit mode instead of placing a cursor.
                holder.messageText.setTextIsSelectable(false);
                holder.messageText.setOnClickListener(v -> {
                    if (userMessageClickListener == null) {
                        return;
                    }
                    int pos = holder.getBindingAdapterPosition();
                    if (pos != RecyclerView.NO_POSITION) {
                        userMessageClickListener.onEditUserMessage(pos);
                    }
                });
            } else if (viewType == ChatMessage.TYPE_SERVER) {
                holder.messageText.setBackgroundResource(R.drawable.server_message_background);
                holder.messageText.setTextColor(ContextCompat.getColor(context, R.color.assistantText));
            }
        }
        if (holder.copyButton != null) {
            // One-tap copy: live position + CURRENT answer (think/sentinels stripped), not a bind-time snapshot.
            holder.copyButton.setOnClickListener(v -> {
                int pos = holder.getBindingAdapterPosition();
                if (pos != RecyclerView.NO_POSITION) {
                    copyToClipboard(v.getContext(), parseReply(messages.get(pos).content()).answer);
                }
            });
        }
        return holder;
    }
    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        animateEntrance(holder.itemView, position);
        ChatMessage message = messages.get(position);
        if (message.type() == ChatMessage.TYPE_LOADING) {
            // Three staggered bouncing "typing" dots; the row is removed when the beam reply lands.
            startDotBounce(holder.dot1, 0L);
            startDotBounce(holder.dot2, 140L);
            startDotBounce(holder.dot3, 280L);
            return;
        }
        // Styling/click wiring done in onCreateViewHolder; bind only sets per-message text/visibility.
        switch (message.type()) {
            case ChatMessage.TYPE_USER -> holder.messageText.setText(message.content());
            case ChatMessage.TYPE_SERVER -> {
                ParsedReply parsed = parseReplyCached(holder, message.content());
                bindThinkSection(holder, position, parsed);
                boolean hasAnswer = !isBlank(parsed.answer);
                if (hasAnswer) {
                    if (message.markdownReady()) {
                        // Render Markdown only after finalize (streaming binds plain text); reuse the cache otherwise.
                        if (!parsed.answer.equals(holder.renderedAnswerKey)) {
                            holder.renderedAnswerValue = getMarkdownRenderer().render(parsed.answer);
                            holder.renderedAnswerKey = parsed.answer;
                        }
                        holder.messageText.setText(holder.renderedAnswerValue);
                    } else {
                        holder.renderedAnswerKey = null;
                        holder.renderedAnswerValue = null;
                        holder.messageText.setText(parsed.answer);
                    }
                    holder.messageText.setVisibility(View.VISIBLE);
                } else {
                    // Mid-thought, no answer yet: hide the empty bubble so only the thinking panel shows.
                    holder.messageText.setText("");
                    holder.messageText.setVisibility(parsed.hasThink ? View.GONE : View.VISIBLE);
                }
                if (holder.copyButton != null) {
                    holder.copyButton.setVisibility(hasAnswer ? View.VISIBLE : View.GONE);
                }
            }
        }
    }

    // Copy reply text to the clipboard + confirm with a toast. No-op if empty or clipboard unavailable.
    private static void copyToClipboard(Context context, String text) {
        if (text == null || text.isEmpty()) {
            return;
        }
        ClipboardManager clipboard =
                (ClipboardManager) context.getSystemService(Context.CLIPBOARD_SERVICE);
        if (clipboard == null) {
            return;
        }
        clipboard.setPrimaryClip(ClipData.newPlainText("assistant_reply", text));
        Toast.makeText(context, R.string.copied_to_clipboard, Toast.LENGTH_SHORT).show();
    }

    // ===================== Chain-of-thought ("thinking") rendering =====================
    // CoT sentinels emitted by stream_token() (processLLM.cpp), STX/ETX-delimited. MUST stay byte-identical
    // to the C++ kThink*Sentinel.
    private static final String THINK_OPEN = "\u0002\u0002think\u0003\u0003";
    private static final String THINK_CLOSE = "\u0002\u0002/think\u0003\u0003";

    // Split a streamed reply into (optional) chain-of-thought and visible answer. Only THINK_OPEN present
    // => still reasoning (thinking == true).
    private static ParsedReply parseReply(String content) {
        if (content == null || content.isEmpty()) {
            return new ParsedReply("", "", false, false);
        }
        int open = content.indexOf(THINK_OPEN);
        if (open < 0) {
            return new ParsedReply("", content, false, false);
        }
        String before = content.substring(0, open);
        int bodyStart = open + THINK_OPEN.length();
        int close = content.indexOf(THINK_CLOSE, bodyStart);
        if (close < 0) {
            // Still thinking: no closing sentinel yet, so everything after the open marker is reasoning.
            return new ParsedReply(content.substring(bodyStart), before, true, true);
        }
        String think = content.substring(bodyStart, close);
        String answer = before + content.substring(close + THINK_CLOSE.length());
        return new ParsedReply(think, answer, true, false);
    }

    // Per-holder cache of the last parseReply result, keyed on content String identity. Streaming swaps in
    // a fresh content instance each batch (miss); no-content-change rebinds (finalize flip, think toggle)
    // reuse the parse. Cleared on recycle.
    private static ParsedReply parseReplyCached(ViewHolder holder, String content) {
        // Identity (==), not equals(): a grown reply misses in O(1); an equals() check would re-scan the
        // whole reply just to detect a hit, defeating the cache.
        if (content != null && content == holder.parsedContentKey && holder.parsedReplyValue != null) {
            return holder.parsedReplyValue;
        }
        ParsedReply parsed = parseReply(content);
        holder.parsedContentKey = content;
        holder.parsedReplyValue = parsed;
        return parsed;
    }

    // Empty-or-whitespace test without allocating a trimmed copy; short-circuits at the first visible char.
    private static boolean isBlank(String s) {
        for (int i = 0, n = s.length(); i < n; i++) {
            if (s.charAt(i) > ' ') {
                return false;
            }
        }
        return true;
    }

    // Bind the collapsible thinking panel. Hidden with no chain-of-thought; else expanded while thinking,
    // collapsed once answered, unless the user tapped to override this row.
    private void bindThinkSection(ViewHolder holder, int position, ParsedReply parsed) {
        if (holder.thinkContainer == null) {
            return;
        }
        // Non-allocating blank check; body is trimmed lazily (only when expanded) in applyThinkExpansion.
        boolean hasThinkBody = parsed.think != null && !isBlank(parsed.think);
        if (!parsed.hasThink || !hasThinkBody) {
            if (parsed.hasThink && !parsed.thinking) {
                bindSkippedThinkSection(holder);
            } else {
                hideThinkSection(holder);
            }
            return;
        }
        holder.thinkContainer.setVisibility(View.VISIBLE);
        holder.thinkChevron.setVisibility(View.VISIBLE);
        holder.thinkTitle.setText(parsed.thinking
                ? R.string.think_section_active : R.string.think_section_done);
        boolean defaultExpanded = parsed.thinking;   // watch it reason live, then tuck it away
        Boolean override = thinkExpanded.get(position);
        boolean expanded = override != null ? override : defaultExpanded;
        applyThinkExpansion(holder, expanded, parsed.think);
        View.OnClickListener toggle = v -> toggleThinkSection(holder);
        holder.thinkContainer.setOnClickListener(toggle);
        holder.thinkHeader.setOnClickListener(toggle);
        holder.thinkText.setOnClickListener(toggle);
    }

    private static void hideThinkSection(ViewHolder holder) {
        holder.thinkContainer.setVisibility(View.GONE);
        holder.thinkContainer.setOnClickListener(null);
        holder.thinkHeader.setOnClickListener(null);
        holder.thinkText.setOnClickListener(null);
        holder.thinkText.setText("");
    }

    private static void bindSkippedThinkSection(ViewHolder holder) {
        holder.thinkContainer.setVisibility(View.VISIBLE);
        holder.thinkChevron.setVisibility(View.GONE);
        holder.thinkTitle.setText(R.string.think_section_skipped);
        holder.thinkText.setText("");
        holder.thinkText.setVisibility(View.GONE);
        holder.thinkContainer.setOnClickListener(null);
        holder.thinkHeader.setOnClickListener(null);
        holder.thinkText.setOnClickListener(null);
    }

    private void toggleThinkSection(ViewHolder holder) {
        int pos = holder.getBindingAdapterPosition();
        if (pos == RecyclerView.NO_POSITION || messages.get(pos).type() != ChatMessage.TYPE_SERVER) {
            return;
        }
        ParsedReply parsed = parseReply(messages.get(pos).content());
        Boolean cur = thinkExpanded.get(pos);
        boolean curExpanded = cur != null ? cur : parsed.thinking;
        thinkExpanded.put(pos, !curExpanded);
        notifyItemChanged(pos);
    }

    // Flip the disclosure chevron + show/drop the reasoning body (cleared while collapsed).
    private static void applyThinkExpansion(ViewHolder holder, boolean expanded, String think) {
        holder.thinkChevron.setVisibility(View.VISIBLE);
        holder.thinkChevron.setText(expanded ? "\u25BE" : "\u25B8");   // ▾ / ▸
        if (expanded) {
            holder.thinkText.setText(think.trim());
            holder.thinkText.setVisibility(View.VISIBLE);
        } else {
            holder.thinkText.setText("");
            holder.thinkText.setVisibility(View.GONE);
        }
    }

    // Forget all per-row expand overrides. Called on clear/edit-resend since those shift row positions.
    void clearThinkExpandState() {
        thinkExpanded.clear();
    }
    @Override
    public int getItemCount() {
        return messages.size();
    }
    @Override
    public int getItemViewType(int position) {
        return messages.get(position).type();
    }
    // One-shot entrance (fade + upward glide) per newly shown row. Guarded by lastAnimatedPosition so
    // streaming rebinds never re-trigger it; onViewRecycled resets recycled views.
    private void animateEntrance(View view, int position) {
        if (position <= lastAnimatedPosition) {
            return;
        }
        lastAnimatedPosition = position;
        float rise = view.getResources().getDisplayMetrics().density * 14f;
        view.setAlpha(0f);
        view.setTranslationY(rise);
        view.animate()
                .alpha(1f)
                .translationY(0f)
                .setDuration(260L)
                .setInterpolator(new DecelerateInterpolator())
                .start();
    }
    @Override
    public void onViewRecycled(@NonNull ViewHolder holder) {
        super.onViewRecycled(holder);
        holder.itemView.animate().cancel();
        holder.itemView.setAlpha(1f);
        holder.itemView.setTranslationY(0f);
        clearAnimation(holder.messageText);
        clearAnimation(holder.dot1);
        clearAnimation(holder.dot2);
        clearAnimation(holder.dot3);
        // Release the cached render so the recycled holder cannot pin a stale Spannable in memory.
        holder.renderedAnswerKey = null;
        holder.renderedAnswerValue = null;
        holder.parsedContentKey = null;
        holder.parsedReplyValue = null;
    }
    // Called when the conversation is cleared so the first rows of the next chat animate in fresh.
    void resetEntranceAnimation() {
        lastAnimatedPosition = -1;
    }
    // Bounce one "typing" dot up and back down forever, offset so the three dots ripple in sequence.
    private static void startDotBounce(View dot, long startOffset) {
        if (dot == null) {
            return;
        }
        float hop = dot.getResources().getDisplayMetrics().density * 6f;   // ~6dp travel
        TranslateAnimation bounce = new TranslateAnimation(0f, 0f, 0f, -hop);
        bounce.setDuration(360);
        bounce.setStartOffset(startOffset);
        bounce.setRepeatCount(Animation.INFINITE);
        bounce.setRepeatMode(Animation.REVERSE);
        bounce.setInterpolator(new AccelerateDecelerateInterpolator());
        dot.startAnimation(bounce);
    }
    private static void clearAnimation(View view) {
        if (view != null) {
            view.clearAnimation();
        }
    }
    private MarkdownTextRenderer getMarkdownRenderer() {
        if (markdownRenderer == null) {
            markdownRenderer = new MarkdownTextRenderer();
        }
        return markdownRenderer;
    }

    // Parsed server reply: chain-of-thought, visible answer, whether a think span exists, and whether
    // still mid-thought (open seen, close not yet).
    private static final class ParsedReply {
        final String think;
        final String answer;
        final boolean hasThink;
        final boolean thinking;
        ParsedReply(String think, String answer, boolean hasThink, boolean thinking) {
            this.think = think;
            this.answer = answer;
            this.hasThink = hasThink;
            this.thinking = thinking;
        }
    }
    static class ViewHolder extends RecyclerView.ViewHolder {
        // findViewById returns null for views absent from a given row layout (dots = loading, copy/think = server).
        TextView messageText;
        ImageButton copyButton;
        // Cached markdown render for THIS row (server only); unchanged rebind reuses it. Cleared on recycle.
        String renderedAnswerKey;
        CharSequence renderedAnswerValue;
        // Last parseReply result for THIS row, keyed on content identity (see parseReplyCached).
        String parsedContentKey;
        ParsedReply parsedReplyValue;
        View thinkContainer;
        View thinkHeader;
        TextView thinkTitle;
        TextView thinkChevron;
        TextView thinkText;
        View dot1;
        View dot2;
        View dot3;
        ViewHolder(View itemView) {
            super(itemView);
            messageText = itemView.findViewById(R.id.messageText);
            copyButton = itemView.findViewById(R.id.copyButton);
            thinkContainer = itemView.findViewById(R.id.thinkContainer);
            thinkHeader = itemView.findViewById(R.id.thinkHeader);
            thinkTitle = itemView.findViewById(R.id.thinkTitle);
            thinkChevron = itemView.findViewById(R.id.thinkChevron);
            thinkText = itemView.findViewById(R.id.thinkText);
            dot1 = itemView.findViewById(R.id.dot1);
            dot2 = itemView.findViewById(R.id.dot2);
            dot3 = itemView.findViewById(R.id.dot3);
        }
    }
}
