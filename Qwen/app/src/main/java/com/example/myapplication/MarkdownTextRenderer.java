package com.example.myapplication;

import android.graphics.Typeface;
import android.text.Spannable;
import android.text.SpannableStringBuilder;
import android.text.Spanned;
import android.text.style.QuoteSpan;
import android.text.style.RelativeSizeSpan;
import android.text.style.StrikethroughSpan;
import android.text.style.StyleSpan;
import android.text.style.TypefaceSpan;
import android.text.style.URLSpan;

import org.commonmark.Extension;
import org.commonmark.ext.gfm.strikethrough.Strikethrough;
import org.commonmark.ext.gfm.strikethrough.StrikethroughExtension;
import org.commonmark.ext.gfm.tables.TableBlock;
import org.commonmark.ext.gfm.tables.TableCell;
import org.commonmark.ext.gfm.tables.TableRow;
import org.commonmark.ext.gfm.tables.TablesExtension;
import org.commonmark.ext.task.list.items.TaskListItemMarker;
import org.commonmark.ext.task.list.items.TaskListItemsExtension;
import org.commonmark.node.BlockQuote;
import org.commonmark.node.BulletList;
import org.commonmark.node.Code;
import org.commonmark.node.Document;
import org.commonmark.node.Emphasis;
import org.commonmark.node.FencedCodeBlock;
import org.commonmark.node.HardLineBreak;
import org.commonmark.node.Heading;
import org.commonmark.node.HtmlBlock;
import org.commonmark.node.HtmlInline;
import org.commonmark.node.Image;
import org.commonmark.node.IndentedCodeBlock;
import org.commonmark.node.Link;
import org.commonmark.node.ListItem;
import org.commonmark.node.Node;
import org.commonmark.node.OrderedList;
import org.commonmark.node.Paragraph;
import org.commonmark.node.SoftLineBreak;
import org.commonmark.node.StrongEmphasis;
import org.commonmark.node.Text;
import org.commonmark.node.ThematicBreak;
import org.commonmark.parser.Parser;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

final class MarkdownTextRenderer {
    private final Parser parser;

    MarkdownTextRenderer() {
        List<Extension> extensions = Arrays.asList(
                StrikethroughExtension.builder().requireTwoTildes(true).build(),
                TablesExtension.create(),
                TaskListItemsExtension.create());
        parser = Parser.builder().extensions(extensions).build();
    }

    CharSequence render(String markdown) {
        if (markdown == null || markdown.isEmpty()) {
            return "";
        }
        SpannableStringBuilder out = new SpannableStringBuilder();
        renderNode(parser.parse(normalizeCjkTables(markdown)), out, 0);
        trimTrailingNewlines(out);
        return out;
    }

    // Qwen-style models often emit Markdown tables with full-width punctuation (U+FF5C "｜" for the ASCII
    // pipe, sometimes full-width colons/dashes in the delimiter row). GFM's parser only recognises ASCII
    // '|', '-', ':', so those tables render as raw pipe text. Rewrite the full-width chars to ASCII, but ONLY
    // within lines that actually form a table (header + delimiter row + contiguous body); skip fenced code.
    private static String normalizeCjkTables(String markdown) {
        if (!containsFullWidthTablePunctuation(markdown)) {
            return markdown;   // fast path: no full-width table punctuation anywhere
        }
        String[] lines = markdown.split("\n", -1);
        boolean inFence = false;
        boolean changed = false;
        for (int i = 0; i < lines.length; i++) {
            String trimmed = lines[i].trim();
            if (trimmed.startsWith("```") || trimmed.startsWith("~~~")) {
                inFence = !inFence;
                continue;
            }
            if (inFence) {
                continue;
            }
            // A table starts where a pipe-bearing header line is immediately followed by a delimiter row.
            if (i + 1 < lines.length && lineContainsPipe(lines[i]) && isSeparatorCandidate(lines[i + 1])) {
                String header = normalizePipeVariants(lines[i]);
                if (!header.equals(lines[i])) {
                    lines[i] = header;
                    changed = true;
                }
                String separator = normalizeSeparatorLine(lines[i + 1]);
                if (!separator.equals(lines[i + 1])) {
                    lines[i + 1] = separator;
                    changed = true;
                }
                int j = i + 2;
                while (j < lines.length) {
                    String rowTrimmed = lines[j].trim();
                    if (rowTrimmed.startsWith("```") || rowTrimmed.startsWith("~~~")
                            || !lineContainsPipe(lines[j])) {
                        break;   // a fence or a non-row line ends the table body
                    }
                    String row = normalizePipeVariants(lines[j]);
                    if (!row.equals(lines[j])) {
                        lines[j] = row;
                        changed = true;
                    }
                    j++;
                }
                i = j - 1;   // resume just after the consumed table (the for-loop re-increments)
            }
        }
        return changed ? String.join("\n", lines) : markdown;
    }

    // True once the text holds any non-ASCII pipe/dash/colon/space variant that could hide a table from the
    // ASCII-only GFM parser; lets the all-ASCII common case skip the per-line scan.
    private static boolean containsFullWidthTablePunctuation(CharSequence text) {
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c < 0x80) {
                continue;   // ASCII never needs normalization
            }
            if (isPipeVariant(c) || isDashVariant(c) || isColonVariant(c) || c == '\u3000') {
                return true;
            }
        }
        return false;
    }

    // A delimiter-row candidate: only pipe/dash/colon variants (and spaces), with at least one pipe and one
    // dash -- the shape GFM needs to turn the line above it into a table header.
    private static boolean isSeparatorCandidate(String line) {
        boolean pipe = false;
        boolean dash = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == ' ' || c == '\t' || c == '\r' || c == '\u3000') {
                continue;
            }
            if (isPipeVariant(c)) {
                pipe = true;
            } else if (isDashVariant(c)) {
                dash = true;
            } else if (!isColonVariant(c)) {
                return false;   // any other visible character rules out a delimiter row
            }
        }
        return pipe && dash;
    }

    private static boolean lineContainsPipe(String line) {
        for (int i = 0; i < line.length(); i++) {
            if (isPipeVariant(line.charAt(i))) {
                return true;
            }
        }
        return false;
    }

    // Rewrite only the vertical-bar variants to ASCII '|'; other cell text is left intact so genuine
    // full-width colons/dashes survive.
    private static String normalizePipeVariants(String line) {
        StringBuilder sb = null;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (isPipeVariant(c) && c != '|') {
                if (sb == null) {
                    sb = new StringBuilder(line);
                }
                sb.setCharAt(i, '|');
            }
        }
        return sb == null ? line : sb.toString();
    }

    // Rewrite the delimiter row wholesale: pipes, dashes, colons and the ideographic space each collapse to
    // ASCII so GFM can read the alignment markers and column count.
    private static String normalizeSeparatorLine(String line) {
        StringBuilder sb = new StringBuilder(line.length());
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (isPipeVariant(c)) {
                sb.append('|');
            } else if (isDashVariant(c)) {
                sb.append('-');
            } else if (isColonVariant(c)) {
                sb.append(':');
            } else if (c == '\u3000') {
                sb.append(' ');
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    private static boolean isPipeVariant(char c) {
        return c == '|' || c == '\uFF5C' || c == '\u2502' || c == '\u2503' || c == '\u2223';
    }

    private static boolean isDashVariant(char c) {
        return c == '-' || c == '\u2010' || c == '\u2011' || c == '\u2012' || c == '\u2013'
                || c == '\u2014' || c == '\u2015' || c == '\u2212' || c == '\u2500' || c == '\uFF0D';
    }

    private static boolean isColonVariant(char c) {
        return c == ':' || c == '\uFF1A';
    }

    private void renderNode(Node node, SpannableStringBuilder out, int listDepth) {
        if (node instanceof Document) {
            renderChildren(node, out, listDepth);
        } else if (node instanceof Paragraph) {
            renderParagraph(node, out, listDepth);
        } else if (node instanceof Heading) {
            renderHeading((Heading) node, out, listDepth);
        } else if (node instanceof Text) {
            out.append(((Text) node).getLiteral());
        } else if (node instanceof SoftLineBreak || node instanceof HardLineBreak) {
            out.append('\n');
        } else if (node instanceof Emphasis) {
            renderStyled(node, out, new StyleSpan(Typeface.ITALIC), listDepth);
        } else if (node instanceof StrongEmphasis) {
            renderStyled(node, out, new StyleSpan(Typeface.BOLD), listDepth);
        } else if (node instanceof Strikethrough) {
            renderStyled(node, out, new StrikethroughSpan(), listDepth);
        } else if (node instanceof Image) {
            renderImage((Image) node, out, listDepth);
        } else if (node instanceof Link) {
            renderLink((Link) node, out, listDepth);
        } else if (node instanceof Code) {
            renderInlineCode((Code) node, out);
        } else if (node instanceof FencedCodeBlock) {
            renderCodeBlock(((FencedCodeBlock) node).getLiteral(), out);
        } else if (node instanceof IndentedCodeBlock) {
            renderCodeBlock(((IndentedCodeBlock) node).getLiteral(), out);
        } else if (node instanceof BulletList) {
            renderList(node, out, false, 0, listDepth);
        } else if (node instanceof OrderedList) {
            OrderedList list = (OrderedList) node;
            Integer startNumber = list.getMarkerStartNumber();
            renderList(node, out, true, startNumber != null ? startNumber : 1, listDepth);
        } else if (node instanceof BlockQuote) {
            renderBlockQuote(node, out, listDepth);
        } else if (node instanceof ThematicBreak) {
            ensureBlockStart(out);
            out.append("-----\n\n");
        } else if (node instanceof TableBlock) {
            renderTable(node, out, listDepth);
        } else if (node instanceof TaskListItemMarker) {
            out.append(((TaskListItemMarker) node).isChecked() ? "[x] " : "[ ] ");
        } else if (node instanceof HtmlInline) {
            out.append(((HtmlInline) node).getLiteral());
        } else if (node instanceof HtmlBlock) {
            ensureBlockStart(out);
            out.append(((HtmlBlock) node).getLiteral()).append("\n\n");
        } else {
            renderChildren(node, out, listDepth);
        }
    }

    private void renderParagraph(Node paragraph, SpannableStringBuilder out, int listDepth) {
        ensureBlockStart(out);
        renderChildren(paragraph, out, listDepth);
        out.append("\n\n");
    }

    private void renderHeading(Heading heading, SpannableStringBuilder out, int listDepth) {
        ensureBlockStart(out);
        int start = out.length();
        renderChildren(heading, out, listDepth);
        int end = out.length();
        if (end > start) {
            out.setSpan(new StyleSpan(Typeface.BOLD), start, end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
            out.setSpan(new RelativeSizeSpan(headingSize(heading.getLevel())), start, end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        }
        out.append("\n\n");
    }

    private void renderStyled(Node node, SpannableStringBuilder out, Object span, int listDepth) {
        int start = out.length();
        renderChildren(node, out, listDepth);
        if (out.length() > start) {
            out.setSpan(span, start, out.length(), Spannable.SPAN_EXCLUSIVE_EXCLUSIVE);
        }
    }

    private void renderLink(Link link, SpannableStringBuilder out, int listDepth) {
        int start = out.length();
        renderChildren(link, out, listDepth);
        if (out.length() > start && link.getDestination() != null && !link.getDestination().isEmpty()) {
            out.setSpan(new URLSpan(link.getDestination()), start, out.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        }
    }

    private void renderImage(Image image, SpannableStringBuilder out, int listDepth) {
        int start = out.length();
        out.append("[image: ");
        renderChildren(image, out, listDepth);
        if (out.length() == start + 8) {
            out.append(image.getDestination() != null ? image.getDestination() : "");
        }
        out.append(']');
        out.setSpan(new StyleSpan(Typeface.ITALIC), start, out.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
    }

    private void renderInlineCode(Code code, SpannableStringBuilder out) {
        int start = out.length();
        out.append(code.getLiteral());
        if (out.length() > start) {
            out.setSpan(new TypefaceSpan("monospace"), start, out.length(), Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        }
    }

    private void renderCodeBlock(String literal, SpannableStringBuilder out) {
        ensureBlockStart(out);
        int start = out.length();
        out.append(literal);
        if (out.length() == 0 || out.charAt(out.length() - 1) != '\n') {
            out.append('\n');
        }
        int end = out.length();
        out.setSpan(new TypefaceSpan("monospace"), start, end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        out.setSpan(new RelativeSizeSpan(0.92f), start, end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        out.append('\n');
    }

    private void renderList(Node list, SpannableStringBuilder out, boolean ordered, int startNumber, int listDepth) {
        ensureBlockStart(out);
        int number = startNumber;
        for (Node item = list.getFirstChild(); item != null; item = item.getNext()) {
            if (!(item instanceof ListItem)) {
                renderNode(item, out, listDepth + 1);
                continue;
            }
            renderListItem((ListItem) item, out, ordered, number, listDepth);
            if (ordered) {
                number += 1;
            }
        }
        out.append('\n');
    }

    private void renderListItem(ListItem item, SpannableStringBuilder out, boolean ordered, int number, int listDepth) {
        int lineStart = out.length();
        appendIndent(out, listDepth);
        out.append(ordered ? number + ". " : "- ");
        for (Node child = item.getFirstChild(); child != null; child = child.getNext()) {
            if (child instanceof TaskListItemMarker) {
                renderNode(child, out, listDepth);
            } else if (child instanceof Paragraph) {
                renderChildren(child, out, listDepth);
            } else {
                if (out.length() > 0 && out.charAt(out.length() - 1) != '\n') {
                    out.append('\n');
                }
                renderNode(child, out, listDepth + 1);
            }
        }
        if (out.length() == lineStart) {
            out.append('\n');
        } else if (out.charAt(out.length() - 1) != '\n') {
            out.append('\n');
        }
    }

    private void renderBlockQuote(Node quote, SpannableStringBuilder out, int listDepth) {
        ensureBlockStart(out);
        int start = out.length();
        renderChildren(quote, out, listDepth);
        int end = out.length();
        if (end > start) {
            out.setSpan(new QuoteSpan(), start, end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        }
        out.append('\n');
    }

    private void renderTable(Node table, SpannableStringBuilder out, int listDepth) {
        ensureBlockStart(out);
        List<List<RenderedTableCell>> rows = new ArrayList<>();
        collectTableRows(table, rows, listDepth);
        if (rows.isEmpty()) {
            return;
        }
        int columns = 0;
        for (List<RenderedTableCell> row : rows) {
            columns = Math.max(columns, row.size());
        }
        int[] widths = new int[columns];
        for (List<RenderedTableCell> row : rows) {
            for (int i = 0; i < row.size(); i++) {
                RenderedTableCell cell = row.get(i);
                widths[i] = Math.max(widths[i], Math.max(cell.displayWidth, cell.minWidth));
            }
        }
        int start = out.length();
        appendTableDivider(out, widths, listDepth);
        boolean dividerAlreadyAppended = true;
        for (List<RenderedTableCell> row : rows) {
            int rowStart = out.length();
            appendTableRow(out, row, widths, listDepth);
            int rowEnd = out.length();
            dividerAlreadyAppended = false;
            if (rowHasHeader(row)) {
                out.setSpan(new StyleSpan(Typeface.BOLD), rowStart, rowEnd, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
                appendTableDivider(out, widths, listDepth);
                dividerAlreadyAppended = true;
            }
        }
        if (!dividerAlreadyAppended) {
            appendTableDivider(out, widths, listDepth);
        }
        int end = out.length();
        if (end > start) {
            out.setSpan(new TypefaceSpan("monospace"), start, end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
            out.setSpan(new RelativeSizeSpan(0.9f), start, end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE);
        }
        out.append('\n');
    }

    private void collectTableRows(Node node, List<List<RenderedTableCell>> rows, int listDepth) {
        if (node instanceof TableRow) {
            List<RenderedTableCell> row = new ArrayList<>();
            for (Node cell = node.getFirstChild(); cell != null; cell = cell.getNext()) {
                if (cell instanceof TableCell) {
                    row.add(renderTableCell((TableCell) cell, listDepth));
                }
            }
            if (!row.isEmpty()) {
                rows.add(row);
            }
            return;
        }
        for (Node child = node.getFirstChild(); child != null; child = child.getNext()) {
            collectTableRows(child, rows, listDepth);
        }
    }

    private RenderedTableCell renderTableCell(TableCell cell, int listDepth) {
        SpannableStringBuilder content = new SpannableStringBuilder();
        renderChildren(cell, content, listDepth);
        normalizeTableCellContent(content);
        return new RenderedTableCell(content, displayWidth(content), Math.max(3, cell.getWidth()),
                cell.getAlignment(), cell.isHeader());
    }

    private void appendTableRow(SpannableStringBuilder out, List<RenderedTableCell> row,
                                int[] widths, int listDepth) {
        appendIndent(out, listDepth);
        out.append('|');
        for (int i = 0; i < widths.length; i++) {
            RenderedTableCell cell = i < row.size() ? row.get(i) : EMPTY_TABLE_CELL;
            out.append(' ');
            appendAlignedCell(out, cell, widths[i]);
            out.append(' ').append('|');
        }
        out.append('\n');
    }

    private static void appendAlignedCell(SpannableStringBuilder out, RenderedTableCell cell, int width) {
        int pad = Math.max(0, width - cell.displayWidth);
        int left = 0;
        int right = pad;
        if (cell.alignment == TableCell.Alignment.RIGHT) {
            left = pad;
            right = 0;
        } else if (cell.alignment == TableCell.Alignment.CENTER) {
            left = pad / 2;
            right = pad - left;
        }
        appendRepeated(out, ' ', left);
        out.append(cell.content);
        appendRepeated(out, ' ', right);
    }

    private static void appendTableDivider(SpannableStringBuilder out, int[] widths, int listDepth) {
        appendIndent(out, listDepth);
        out.append('+');
        for (int width : widths) {
            appendRepeated(out, '-', width + 2);
            out.append('+');
        }
        out.append('\n');
    }

    private static boolean rowHasHeader(List<RenderedTableCell> row) {
        for (RenderedTableCell cell : row) {
            if (cell.header) {
                return true;
            }
        }
        return false;
    }

    private static void normalizeTableCellContent(SpannableStringBuilder content) {
        for (int i = 0; i < content.length(); i++) {
            char ch = content.charAt(i);
            if (ch == '\n' || ch == '\r' || ch == '\t') {
                content.replace(i, i + 1, " ");
            }
        }
        int start = 0;
        while (start < content.length() && content.charAt(start) <= ' ') {
            start += 1;
        }
        int end = content.length();
        while (end > start && content.charAt(end - 1) <= ' ') {
            end -= 1;
        }
        if (end < content.length()) {
            content.delete(end, content.length());
        }
        if (start > 0) {
            content.delete(0, start);
        }
    }

    private static int displayWidth(CharSequence text) {
        int width = 0;
        for (int offset = 0; offset < text.length(); ) {
            int codePoint = Character.codePointAt(text, offset);
            offset += Character.charCount(codePoint);
            int type = Character.getType(codePoint);
            if (type == Character.NON_SPACING_MARK
                    || type == Character.ENCLOSING_MARK
                    || type == Character.FORMAT) {
                continue;
            }
            width += isWideCodePoint(codePoint) ? 2 : 1;
        }
        return width;
    }

    private static boolean isWideCodePoint(int codePoint) {
        return codePoint >= 0x1100 && (codePoint <= 0x115F
                || codePoint == 0x2329
                || codePoint == 0x232A
                || codePoint >= 0x2E80 && codePoint <= 0xA4CF && codePoint != 0x303F
                || codePoint >= 0xAC00 && codePoint <= 0xD7A3
                || codePoint >= 0xF900 && codePoint <= 0xFAFF
                || codePoint >= 0xFE10 && codePoint <= 0xFE19
                || codePoint >= 0xFE30 && codePoint <= 0xFE6F
                || codePoint >= 0xFF00 && codePoint <= 0xFF60
                || codePoint >= 0xFFE0 && codePoint <= 0xFFE6
                || codePoint >= 0x1F300 && codePoint <= 0x1FAFF);
    }

    private static void appendRepeated(SpannableStringBuilder out, char ch, int count) {
        for (int i = 0; i < count; i++) {
            out.append(ch);
        }
    }

    private void renderChildren(Node node, SpannableStringBuilder out, int listDepth) {
        for (Node child = node.getFirstChild(); child != null; child = child.getNext()) {
            renderNode(child, out, listDepth);
        }
    }

    private static float headingSize(int level) {
        return switch (level) {
            case 1 -> 1.35f;
            case 2 -> 1.25f;
            case 3 -> 1.16f;
            case 4 -> 1.08f;
            default -> 1.0f;
        };
    }

    private static void appendIndent(SpannableStringBuilder out, int listDepth) {
        for (int i = 0; i < listDepth; i++) {
            out.append("  ");
        }
    }

    private static void ensureBlockStart(SpannableStringBuilder out) {
        int length = out.length();
        if (length > 0 && out.charAt(length - 1) != '\n') {
            out.append('\n');
        }
    }

    private static void trimTrailingNewlines(SpannableStringBuilder out) {
        while (out.length() > 0 && out.charAt(out.length() - 1) == '\n') {
            out.delete(out.length() - 1, out.length());
        }
    }

    private static final RenderedTableCell EMPTY_TABLE_CELL =
            new RenderedTableCell("", 0, 3, null, false);

    private static final class RenderedTableCell {
        final CharSequence content;
        final int displayWidth;
        final int minWidth;
        final TableCell.Alignment alignment;
        final boolean header;

        RenderedTableCell(CharSequence content, int displayWidth, int minWidth,
                          TableCell.Alignment alignment, boolean header) {
            this.content = content;
            this.displayWidth = displayWidth;
            this.minWidth = minWidth;
            this.alignment = alignment;
            this.header = header;
        }
    }
}