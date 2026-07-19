package com.example.myapplication;

import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.util.LruCache;
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

import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class ChatAdapter extends RecyclerView.Adapter<ChatAdapter.ViewHolder> {
    private static final Handler MAIN_HANDLER = new Handler(Looper.getMainLooper());
    private static final ExecutorService MARKDOWN_EXECUTOR = newSingleDaemonExecutor("chat-markdown");
    private static final DateTimeFormatter FIRST_TOKEN_TIME_FORMAT =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss z");
    private static final DateTimeFormatter LAST_TOKEN_TIME_FORMAT =
            DateTimeFormatter.ofPattern("HH:mm:ss");
    // Tapping a user bubble enters edit mode; the activity handles the edit + resend.
    public interface OnUserMessageClickListener {
        void onEditUserMessage(int position);
    }
    private final List<ChatMessage> messages;
    private MarkdownTextRenderer markdownRenderer;
    private final LruCache<String, CharSequence> markdownCache =
            new LruCache<>(512 * 1024) {
                @Override
                protected int sizeOf(@NonNull String key, @NonNull CharSequence value) {
                    return (key.length() + value.length()) * Character.BYTES;
                }
            };
    private OnUserMessageClickListener userMessageClickListener;
    // Highest row index that already played its one-shot entrance animation; guards streaming rebinds.
    private int lastAnimatedPosition = -1;
    // Per-row thinking-panel expand override by position (absent => expanded while thinking, else collapsed).
    // Cleared on reset/resend since those renumber rows.
    private final HashMap<Integer, Boolean> thinkExpanded = new HashMap<>();

    private static ExecutorService newSingleDaemonExecutor(String name) {
        return Executors.newSingleThreadExecutor(runnable -> {
            Thread thread = new Thread(runnable, name);
            thread.setDaemon(true);
            return thread;
        });
    }

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
        if (holder.boundMessage != message) {
            holder.pendingMarkdownKey = null;
        }
        holder.boundMessage = message;
        holder.answerBuffer = null;
        holder.thinkBuffer = null;
        holder.streamingThinking = false;
        holder.lastRenderedBottomSecond = Long.MIN_VALUE;
        if (message.type() == ChatMessage.TYPE_LOADING) {
            // Three staggered bouncing "typing" dots; the row is removed when the beam reply lands.
            startDotBounce(holder.dot1, 0L);
            startDotBounce(holder.dot2, 140L);
            startDotBounce(holder.dot3, 280L);
            return;
        }
        // Styling/click wiring done in onCreateViewHolder; bind only sets per-message text/visibility.
        bindTimestampLabels(holder, message);
        switch (message.type()) {
            case ChatMessage.TYPE_USER -> holder.messageText.setText(message.content());
            case ChatMessage.TYPE_SERVER -> {
                ParsedReply parsed = parseReplyCached(holder, message.content());
                holder.answerBuffer = new StringBuilder(parsed.answer);
                holder.thinkBuffer = new StringBuilder(parsed.think);
                holder.streamingThinking = parsed.thinking;
                bindThinkSection(holder, position, parsed);
                boolean hasAnswer = !isBlank(parsed.answer);
                if (hasAnswer) {
                    if (message.markdownReady()) {
                        bindMarkdownAnswer(holder, message, parsed.answer);
                    } else {
                        holder.pendingMarkdownKey = null;
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

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position,
                                 @NonNull List<Object> payloads) {
        if (payloads.isEmpty() || holder.boundMessage != messages.get(position)) {
            onBindViewHolder(holder, position);
            return;
        }
        boolean handled = false;
        for (Object payload : payloads) {
            if (payload instanceof StreamingPayload stream &&
                    messages.get(position).type() == ChatMessage.TYPE_SERVER) {
                appendStreamingPayload(holder, position, stream);
                handled = true;
            }
        }
        if (!handled) {
            onBindViewHolder(holder, position);
        }
    }

    void notifyStreamingAppend(int position, String chunk, long eventTimeMillis) {
        notifyItemChanged(position, new StreamingPayload(chunk, eventTimeMillis));
    }

    private void appendStreamingPayload(ViewHolder holder, int position, StreamingPayload payload) {
        if (holder.answerBuffer == null) {
            holder.answerBuffer = new StringBuilder();
        }
        if (holder.thinkBuffer == null) {
            holder.thinkBuffer = new StringBuilder();
        }
        String chunk = payload.chunk;
        int cursor = 0;
        while (cursor < chunk.length()) {
            String marker = holder.streamingThinking ? THINK_CLOSE : THINK_OPEN;
            int markerAt = chunk.indexOf(marker, cursor);
            int end = markerAt >= 0 ? markerAt : chunk.length();
            if (end > cursor) {
                appendStreamingSegment(holder, chunk.substring(cursor, end));
            }
            if (markerAt < 0) {
                break;
            }
            holder.streamingThinking = !holder.streamingThinking;
            if (holder.streamingThinking) {
                beginStreamingThink(holder, position);
            } else {
                finishStreamingThink(holder, position);
            }
            cursor = markerAt + marker.length();
        }
        if (holder.messageTimeBottom != null) {
            // The label shows whole seconds (HH:mm:ss); only re-format + setText when the displayed second
            // actually changes, so most streamed frames skip the formatter and TextView invalidation.
            long epochSecond = Math.floorDiv(payload.eventTimeMillis, 1000L);
            if (epochSecond != holder.lastRenderedBottomSecond) {
                holder.lastRenderedBottomSecond = epochSecond;
                holder.messageTimeBottom.setText(formatLastTokenTime(payload.eventTimeMillis));
            }
        }
    }

    private static void appendStreamingSegment(ViewHolder holder, String segment) {
        if (segment.isEmpty()) {
            return;
        }
        if (holder.streamingThinking) {
            holder.thinkBuffer.append(segment);
            if (holder.thinkText != null && holder.thinkText.getVisibility() == View.VISIBLE) {
                holder.thinkText.append(segment);
            }
        } else {
            holder.answerBuffer.append(segment);
            holder.messageText.append(segment);
            holder.messageText.setVisibility(View.VISIBLE);
            if (holder.copyButton != null) {
                holder.copyButton.setVisibility(View.VISIBLE);
            }
        }
    }

    private void beginStreamingThink(ViewHolder holder, int position) {
        if (holder.thinkContainer == null) {
            return;
        }
        holder.thinkContainer.setVisibility(View.VISIBLE);
        holder.thinkTitle.setText(R.string.think_section_active);
        Boolean override = thinkExpanded.get(position);
        applyThinkExpansion(holder, override == null || override, holder.thinkBuffer.toString());
    }

    private void finishStreamingThink(ViewHolder holder, int position) {
        if (holder.thinkContainer == null) {
            return;
        }
        holder.thinkTitle.setText(R.string.think_section_done);
        Boolean override = thinkExpanded.get(position);
        applyThinkExpansion(holder, override != null && override, holder.thinkBuffer.toString());
    }

    private static final class StreamingPayload {
        final String chunk;
        final long eventTimeMillis;

        StreamingPayload(String chunk, long eventTimeMillis) {
            this.chunk = chunk;
            this.eventTimeMillis = eventTimeMillis;
        }
    }

    private void bindMarkdownAnswer(ViewHolder holder, ChatMessage message, String answer) {
        CharSequence cached = markdownCache.get(answer);
        if (cached != null) {
            holder.pendingMarkdownKey = null;
            holder.messageText.setText(cached);
            return;
        }
        holder.messageText.setText(answer);
        if (answer.equals(holder.pendingMarkdownKey)) {
            return;
        }
        holder.pendingMarkdownKey = answer;
        MARKDOWN_EXECUTOR.execute(() -> {
            CharSequence rendered = getMarkdownRenderer().render(answer);
            MAIN_HANDLER.post(() -> {
                markdownCache.put(answer, rendered);
                if (holder.boundMessage != message || !answer.equals(holder.pendingMarkdownKey)) {
                    return;
                }
                holder.pendingMarkdownKey = null;
                holder.messageText.setText(rendered);
            });
        });
    }

    private static void bindTimestampLabels(ViewHolder holder, ChatMessage message) {
        if (holder.messageTimeTop != null) {
            holder.messageTimeTop.setText(formatFirstTokenTime(message.firstTokenTimeMillis()));
        }
        if (holder.messageTimeBottom != null) {
            holder.messageTimeBottom.setText(formatLastTokenTime(message.lastTokenTimeMillis()));
        }
    }

    private static String formatFirstTokenTime(long timeMillis) {
        return FIRST_TOKEN_TIME_FORMAT.format(
                Instant.ofEpochMilli(timeMillis).atZone(ZoneId.systemDefault()));
    }

    private static String formatLastTokenTime(long timeMillis) {
        return LAST_TOKEN_TIME_FORMAT.format(
                Instant.ofEpochMilli(timeMillis).atZone(ZoneId.systemDefault()));
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
        holder.pendingMarkdownKey = null;
        holder.parsedContentKey = null;
        holder.parsedReplyValue = null;
        holder.boundMessage = null;
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
        TextView messageTimeTop;
        TextView messageTimeBottom;
        String pendingMarkdownKey;
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
        ChatMessage boundMessage;
        StringBuilder answerBuffer;
        StringBuilder thinkBuffer;
        boolean streamingThinking;
        // Last whole epoch second rendered into messageTimeBottom during streaming; guards redundant
        // per-frame timestamp formatting when the displayed second has not advanced.
        long lastRenderedBottomSecond = Long.MIN_VALUE;
        ViewHolder(View itemView) {
            super(itemView);
            messageText = itemView.findViewById(R.id.messageText);
            copyButton = itemView.findViewById(R.id.copyButton);
            messageTimeTop = itemView.findViewById(R.id.messageTimeTop);
            messageTimeBottom = itemView.findViewById(R.id.messageTimeBottom);
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
