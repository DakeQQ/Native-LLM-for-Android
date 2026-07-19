package com.example.myapplication;

import android.animation.ValueAnimator;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.RectF;
import android.graphics.Shader;
import android.graphics.Typeface;
import android.util.AttributeSet;
import android.util.TypedValue;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.View;
import android.view.animation.DecelerateInterpolator;

import androidx.core.content.ContextCompat;

import java.util.Locale;

/**
 * Lightweight line/area + scatter chart drawn entirely with {@link Canvas} (no charting library). Up to two
 * series with independent left/right value axes; pinch-zoom X + drag-pan, Y auto-fits the visible X window.
 * Glow = layered translucent strokes (BlurMaskFilter is unreliable under HW accel). On (re)load the line
 * draws itself left→right with a glowing head dot.
 */
public class MetricChartView extends View {

    public static final int SERIES_A = 0;
    public static final int SERIES_B = 1;

    private static final int GRID_LINES = 4;          // horizontal grid / axis tick count
    private static final float MIN_VIEW_SPAN = 0.04f; // max zoom-in = 25× of the full X range
    private static final long ENTRANCE_MS = 1100L;

    private static final class SeriesState {
        MetricsHistory.Series data;
        String unitLabel = "";
        int color = Color.CYAN;
        int glowColor = Color.CYAN;
        boolean axisRight = false;
        boolean visible = true;
        LinearGradient fillShader;
        float fillShaderTop = Float.NaN;
        float fillShaderBottom = Float.NaN;
        int fillShaderColor = Color.TRANSPARENT;
        MetricsHistory.Series rangeData;
        float rangeMinX = Float.NaN;
        float rangeMaxX = Float.NaN;
        float rangeMinY;
        float rangeMaxY;
        final String[] axisLabels = new String[GRID_LINES + 1];
    }

    private final SeriesState[] series = {new SeriesState(), new SeriesState()};
    private boolean xIsTime = true;
    // Scatter/pin mode: glowing dots at each raw sample only, no line/fill/curve (used by throughput charts).
    private boolean scatterMode = false;

    private final Paint linePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint glowPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint fillPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint gridPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint axisTextPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint unitTextPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint panelPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint panelStrokePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint pointPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint headGlowPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint emptyPaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    private final Path linePath = new Path();
    private final Path fillPath = new Path();
    private final RectF plot = new RectF();
    private final RectF panelRect = new RectF();

    private int axisTextColor;

    // Visible X window as a fraction [0,1] of the full data X domain.
    private float viewLo = 0f;
    private float viewHi = 1f;

    private float drawProgress = 1f;
    private ValueAnimator entranceAnimator;

    private final ScaleGestureDetector scaleDetector;
    private float lastTouchX;
    private boolean dragging;
    private int activePointerId = -1;

    // Reusable scratch for the pixel polyline (avoids per-frame allocation).
    private float[] pxBuf = new float[0];
    private float[] pyBuf = new float[0];
    private final float[] leftAxisRange = new float[2];
    private final float[] rightAxisRange = new float[2];
    private final float[] drawYRange = new float[2];
    private final String[] xAxisLabels = new String[5];
    private float xAxisLabelMin = Float.NaN;
    private float xAxisLabelMax = Float.NaN;
    private boolean xAxisLabelTime;

    public MetricChartView(Context context, AttributeSet attrs) {
        super(context, attrs);

        axisTextColor = ContextCompat.getColor(context, R.color.chartAxisText);

        gridPaint.setStyle(Paint.Style.STROKE);
        gridPaint.setStrokeWidth(dp(0.8f));
        gridPaint.setColor(ContextCompat.getColor(context, R.color.chartGrid));

        axisTextPaint.setColor(axisTextColor);
        axisTextPaint.setTextSize(sp(12.5f));
        axisTextPaint.setTypeface(Typeface.DEFAULT_BOLD);

        // Axis unit captions (tok/s, %, MB): bold, larger than tick numbers. Colour set per series at draw time.
        unitTextPaint.setTextSize(sp(13.5f));
        unitTextPaint.setTypeface(Typeface.DEFAULT_BOLD);

        panelPaint.setStyle(Paint.Style.FILL);
        panelPaint.setColor(ContextCompat.getColor(context, R.color.chartPanelFill));
        panelStrokePaint.setStyle(Paint.Style.STROKE);
        panelStrokePaint.setStrokeWidth(dp(1f));
        panelStrokePaint.setColor(ContextCompat.getColor(context, R.color.chartPanelStroke));

        linePaint.setStyle(Paint.Style.STROKE);
        linePaint.setStrokeCap(Paint.Cap.ROUND);
        linePaint.setStrokeJoin(Paint.Join.ROUND);
        linePaint.setStrokeWidth(dp(2.2f));

        glowPaint.setStyle(Paint.Style.STROKE);
        glowPaint.setStrokeCap(Paint.Cap.ROUND);
        glowPaint.setStrokeJoin(Paint.Join.ROUND);

        fillPaint.setStyle(Paint.Style.FILL);
        pointPaint.setStyle(Paint.Style.FILL);
        headGlowPaint.setStyle(Paint.Style.FILL);

        emptyPaint.setColor(axisTextColor);
        emptyPaint.setTextSize(sp(13));
        emptyPaint.setTextAlign(Paint.Align.CENTER);

        scaleDetector = new ScaleGestureDetector(context, new ScaleListener());
        setLayerType(LAYER_TYPE_HARDWARE, null);
    }

    // ---- public API -------------------------------------------------------

    public void configureSeries(int index, String unit,
                                int color, int glowColor, boolean axisRight) {
        SeriesState s = series[index];
        s.unitLabel = unit == null ? "" : unit.trim();
        s.color = color;
        s.glowColor = glowColor;
        s.axisRight = axisRight;
        s.fillShader = null;
    }

    public void setFormatting(boolean xIsTime) {
        this.xIsTime = xIsTime;
        xAxisLabelMin = Float.NaN;
    }

    public void setScatterMode(boolean scatter) {
        this.scatterMode = scatter;
    }

    public void setData(MetricsHistory.Series a, MetricsHistory.Series b, boolean animate) {
        series[SERIES_A].data = a;
        series[SERIES_B].data = b;
        series[SERIES_A].rangeData = null;
        series[SERIES_B].rangeData = null;
        xAxisLabelMin = Float.NaN;
        if (animate) {
            fitView();
            replayAnimation();
        } else {
            invalidate();
        }
    }

    public boolean isSeriesVisible(int index) {
        return series[index].visible;
    }

    public void setSeriesVisible(int index, boolean visible) {
        if (series[index].visible != visible) {
            series[index].visible = visible;
            replayAnimation();
        }
    }

    public void zoomIn() {
        zoomBy(0.65f);
    }

    public void zoomOut() {
        zoomBy(1.0f / 0.65f);
    }

    public void fitView() {
        viewLo = 0f;
        viewHi = 1f;
        invalidate();
    }

    public void replayAnimation() {
        if (entranceAnimator != null) {
            entranceAnimator.cancel();
        }
        entranceAnimator = ValueAnimator.ofFloat(0f, 1f);
        entranceAnimator.setDuration(ENTRANCE_MS);
        entranceAnimator.setInterpolator(new DecelerateInterpolator(1.4f));
        entranceAnimator.addUpdateListener(anim -> {
            drawProgress = (float) anim.getAnimatedValue();
            invalidate();
        });
        entranceAnimator.start();
    }

    // ---- geometry ---------------------------------------------------------

    private void computePlot() {
        float padLeft = dp(46);
        float padRight = hasRightAxis() ? dp(46) : dp(16);
        float padTop = dp(22);
        float padBottom = dp(30);
        panelRect.set(dp(2), dp(2), getWidth() - dp(2), getHeight() - dp(2));
        plot.set(padLeft, padTop, getWidth() - padRight, getHeight() - padBottom);
    }

    private boolean hasRightAxis() {
        for (SeriesState s : series) {
            if (s.visible && s.axisRight && s.data != null && s.data.hasData()) {
                return true;
            }
        }
        return false;
    }

    private boolean anyVisibleData() {
        for (SeriesState s : series) {
            if (s.visible && s.data != null && s.data.hasData()) {
                return true;
            }
        }
        return false;
    }

    private float domainMinX() {
        float min = Float.MAX_VALUE;
        for (SeriesState s : series) {
            if (s.visible && s.data != null && s.data.hasData()) {
                min = Math.min(min, s.data.minX);
            }
        }
        return min == Float.MAX_VALUE ? 0f : min;
    }

    private float domainMaxX() {
        float max = -Float.MAX_VALUE;
        for (SeriesState s : series) {
            if (s.visible && s.data != null && s.data.hasData()) {
                max = Math.max(max, s.data.maxX);
            }
        }
        return max == -Float.MAX_VALUE ? 1f : max;
    }

    // ---- drawing ----------------------------------------------------------

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        computePlot();

        float r = dp(14);
        canvas.drawRoundRect(panelRect, r, r, panelPaint);
        canvas.drawRoundRect(panelRect, r, r, panelStrokePaint);

        if (!anyVisibleData()) {
            canvas.drawText(getResources().getString(R.string.chart_no_data),
                    getWidth() / 2f, getHeight() / 2f, emptyPaint);
            return;
        }

        float dMinX = domainMinX();
        float dMaxX = domainMaxX();
        float spanX = dMaxX - dMinX;
        if (spanX <= 0f) {
            spanX = 1f;
        }
        float visMinX = dMinX + viewLo * spanX;
        float visMaxX = dMinX + viewHi * spanX;

        drawGridAndAxes(canvas, visMinX, visMaxX);

        for (int i = 0; i < series.length; i++) {
            drawSeries(canvas, series[i], dMinX, spanX, visMinX, visMaxX);
        }
    }

    private void drawGridAndAxes(Canvas canvas, float visMinX, float visMaxX) {
        SeriesState left = axisSeries(false);
        SeriesState right = axisSeries(true);
        boolean hasLeftRange = left != null;
        boolean hasRightRange = right != null;
        if (hasLeftRange) {
            cachedVisibleYRange(left, visMinX, visMaxX, leftAxisRange);
        }
        if (hasRightRange) {
            cachedVisibleYRange(right, visMinX, visMaxX, rightAxisRange);
        }

        axisTextPaint.setColor(axisTextColor);
        // Per-axis label colour/padding/baseline are frame-constant; resolve once, not per grid line.
        final int leftLabelColor = hasLeftRange ? mutedColor(left.color) : 0;
        final int rightLabelColor = hasRightRange ? mutedColor(right.color) : 0;
        final float axisLabelPadX = dp(6);
        final float axisLabelBaseline = sp(4.4f);
        for (int i = 0; i <= GRID_LINES; i++) {
            float fb = i / (float) GRID_LINES;             // 0 = bottom, 1 = top
            float y = plot.bottom - fb * plot.height();
            canvas.drawLine(plot.left, y, plot.right, y, gridPaint);

            if (hasLeftRange) {
                axisTextPaint.setColor(leftLabelColor);
                axisTextPaint.setTextAlign(Paint.Align.RIGHT);
                canvas.drawText(left.axisLabels[i], plot.left - axisLabelPadX,
                    y + axisLabelBaseline, axisTextPaint);
            }
            if (hasRightRange) {
                axisTextPaint.setColor(rightLabelColor);
                axisTextPaint.setTextAlign(Paint.Align.LEFT);
                canvas.drawText(right.axisLabels[i], plot.right + axisLabelPadX,
                    y + axisLabelBaseline, axisTextPaint);
            }
        }

        // Axis unit captions (bold, tinted per series), theme-adaptive.
        if (left != null && !left.unitLabel.isEmpty()) {            unitTextPaint.setColor(mutedColor(left.color));
            unitTextPaint.setTextAlign(Paint.Align.LEFT);
            canvas.drawText(left.unitLabel, plot.left - dp(2), plot.top - dp(6), unitTextPaint);
        }
        if (right != null && !right.unitLabel.isEmpty()) {
            unitTextPaint.setColor(mutedColor(right.color));
            unitTextPaint.setTextAlign(Paint.Align.RIGHT);
            canvas.drawText(right.unitLabel, plot.right + dp(2), plot.top - dp(6), unitTextPaint);
        }

        // X axis labels.
        axisTextPaint.setColor(axisTextColor);
        axisTextPaint.setTextAlign(Paint.Align.CENTER);
        final float xLabelBaseline = plot.bottom + dp(16);
        updateXAxisLabels(visMinX, visMaxX);
        for (int i = 0; i <= 4; i++) {
            float sf = i / 4f;
            float x = plot.left + sf * plot.width();
            canvas.drawText(xAxisLabels[i], x, xLabelBaseline, axisTextPaint);
        }
    }

    private SeriesState axisSeries(boolean right) {
        for (SeriesState s : series) {
            if (s.visible && s.axisRight == right && s.data != null && s.data.hasData()) {
                return s;
            }
        }
        return null;
    }

    private void drawSeries(Canvas canvas, SeriesState s, float dMinX, float spanX,
                            float visMinX, float visMaxX) {
        if (!s.visible || s.data == null || !s.data.hasData()) {
            return;
        }
        float[] xs = s.data.x;
        float[] ys = s.data.y;
        int sourceCount = xs.length;
        if (pxBuf.length < sourceCount) {
            pxBuf = new float[sourceCount];
            pyBuf = new float[sourceCount];
        }
        cachedVisibleYRange(s, visMinX, visMaxX, drawYRange);
        float yLo = drawYRange[0];
        float yHi = drawYRange[1];
        float ySpan = yHi - yLo;
        if (ySpan <= 0f) {
            ySpan = 1f;
        }
        float viewSpan = (viewHi - viewLo);
        if (viewSpan <= 0f) {
            viewSpan = 1f;
        }
        int n = 0;
        if (scatterMode) {
            for (int i = 0; i < sourceCount; i++) {
                n = projectPoint(xs[i], ys[i], dMinX, spanX, viewSpan, yLo, ySpan, n);
            }
        } else {
            int first = Math.max(0, lowerBound(xs, visMinX) - 1);
            int end = Math.min(sourceCount, upperBound(xs, visMaxX) + 1);
            int visibleCount = Math.max(0, end - first);
            int bucketCount = Math.max(1, Math.round(plot.width() / 2f));
            if (visibleCount <= bucketCount * 2) {
                for (int i = first; i < end; i++) {
                    n = projectPoint(xs[i], ys[i], dMinX, spanX, viewSpan, yLo, ySpan, n);
                }
            } else {
                for (int bucket = 0; bucket < bucketCount; bucket++) {
                    int bucketStart = first + (int) ((long) bucket * visibleCount / bucketCount);
                    int bucketEnd = first + (int) ((long) (bucket + 1) * visibleCount / bucketCount);
                    bucketEnd = Math.max(bucketStart + 1, Math.min(end, bucketEnd));
                    int minIndex = bucketStart;
                    int maxIndex = bucketStart;
                    for (int i = bucketStart + 1; i < bucketEnd; i++) {
                        if (ys[i] < ys[minIndex]) minIndex = i;
                        if (ys[i] > ys[maxIndex]) maxIndex = i;
                    }
                    int earlier = Math.min(minIndex, maxIndex);
                    int later = Math.max(minIndex, maxIndex);
                    n = projectPoint(xs[earlier], ys[earlier], dMinX, spanX,
                            viewSpan, yLo, ySpan, n);
                    if (later != earlier) {
                        n = projectPoint(xs[later], ys[later], dMinX, spanX,
                                viewSpan, yLo, ySpan, n);
                    }
                }
            }
        }
        if (n == 0) {
            return;
        }

        float revealX = plot.left + clamp01(drawProgress) * plot.width();

        if (scatterMode) {
            // Pin mode: glowing dots at each raw sample (no line/fill), revealed left->right by the entrance
            // animation. Radii/colours are frame-constant (hundreds of pins), so resolve them ONCE.
            canvas.save();
            float pinClip = dp(12);
            canvas.clipRect(plot.left - pinClip, plot.top - pinClip, plot.right + pinClip, plot.bottom + pinClip);
            float pinGlowRadius = dp(10f);
            float pinRadius = dp(5.5f);
            headGlowPaint.setColor(withAlpha(s.glowColor, 0x55));
            pointPaint.setColor(s.color);
            for (int i = 0; i < n; i++) {
                if (pxBuf[i] > revealX) {
                    continue;
                }
                canvas.drawCircle(pxBuf[i], pyBuf[i], pinGlowRadius, headGlowPaint);
                canvas.drawCircle(pxBuf[i], pyBuf[i], pinRadius, pointPaint);
            }
            canvas.restore();
            return;
        }

        buildSmoothPath(pxBuf, pyBuf, n);
        // Area fill below the line, clipped to the plot via the closed path + canvas clip.
        fillPath.set(linePath);
        fillPath.lineTo(pxBuf[n - 1], plot.bottom);
        fillPath.lineTo(pxBuf[0], plot.bottom);
        fillPath.close();

        canvas.save();
        canvas.clipRect(plot.left, plot.top, revealX, plot.bottom);

        fillPaint.setShader(fillShaderFor(s));
        canvas.drawPath(fillPath, fillPaint);

        // Manual glow: two translucent fat strokes under the crisp line.
        glowPaint.setColor(withAlpha(s.glowColor, 0x30));
        glowPaint.setStrokeWidth(dp(9f));
        canvas.drawPath(linePath, glowPaint);
        glowPaint.setColor(withAlpha(s.glowColor, 0x50));
        glowPaint.setStrokeWidth(dp(5f));
        canvas.drawPath(linePath, glowPaint);

        linePaint.setColor(s.color);
        canvas.drawPath(linePath, linePaint);
        canvas.restore();

        // Data dots (only the revealed ones). Radius + cull bounds are frame-constant -> resolve once.
        pointPaint.setColor(s.color);
        float dotRadius = dp(2.2f);
        float dotLeftBound = plot.left - dp(2);
        float dotRightBound = plot.right + dp(2);
        for (int i = 0; i < n; i++) {
            if (pxBuf[i] <= revealX && pxBuf[i] >= dotLeftBound && pxBuf[i] <= dotRightBound) {
                canvas.drawCircle(pxBuf[i], pyBuf[i], dotRadius, pointPaint);
            }
        }

        // Glowing head dot riding the reveal front while the entrance animation plays.
        if (drawProgress < 1f) {
            float headY = interpY(pxBuf, pyBuf, n, revealX);
            if (!Float.isNaN(headY)) {
                headGlowPaint.setColor(withAlpha(s.glowColor, 0x66));
                canvas.drawCircle(revealX, headY, dp(7f), headGlowPaint);
                pointPaint.setColor(s.color);
                canvas.drawCircle(revealX, headY, dp(3.2f), pointPaint);
            }
        }
    }

    private int projectPoint(float x, float y, float domainMinX, float domainSpanX,
                             float viewSpan, float yMin, float ySpan, int outputIndex) {
        float normalizedX = (x - domainMinX) / domainSpanX;
        float visibleFraction = (normalizedX - viewLo) / viewSpan;
        pxBuf[outputIndex] = plot.left + visibleFraction * plot.width();
        float normalizedY = (y - yMin) / ySpan;
        pyBuf[outputIndex] = plot.bottom - normalizedY * plot.height();
        return outputIndex + 1;
    }

    private static int lowerBound(float[] values, float target) {
        int low = 0;
        int high = values.length;
        while (low < high) {
            int middle = (low + high) >>> 1;
            if (values[middle] < target) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        return low;
    }

    private static int upperBound(float[] values, float target) {
        int low = 0;
        int high = values.length;
        while (low < high) {
            int middle = (low + high) >>> 1;
            if (values[middle] <= target) {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        return low;
    }

    private void visibleYRange(SeriesState s, float visMinX, float visMaxX, float[] out) {
        float[] xs = s.data.x;
        float[] ys = s.data.y;
        float min = Float.MAX_VALUE;
        float max = -Float.MAX_VALUE;
        for (int i = 0; i < xs.length; i++) {
            if (xs[i] >= visMinX && xs[i] <= visMaxX) {
                min = Math.min(min, ys[i]);
                max = Math.max(max, ys[i]);
            }
        }
        if (min == Float.MAX_VALUE) {            // nothing in window: fall back to full series
            min = s.data.minY;
            max = s.data.maxY;
        }
        if (min == max) {                        // flat line: pad so it sits mid-plot
            float pad = Math.max(1f, Math.abs(min) * 0.1f);
            min -= pad;
            max += pad;
        } else {
            float pad = (max - min) * 0.12f;     // headroom above + below
            min -= pad;
            max += pad;
        }
        out[0] = min;
        out[1] = max;
    }

    private void cachedVisibleYRange(SeriesState s, float visMinX, float visMaxX, float[] out) {
        if (s.rangeData != s.data || s.rangeMinX != visMinX || s.rangeMaxX != visMaxX) {
            visibleYRange(s, visMinX, visMaxX, out);
            s.rangeData = s.data;
            s.rangeMinX = visMinX;
            s.rangeMaxX = visMaxX;
            s.rangeMinY = out[0];
            s.rangeMaxY = out[1];
            for (int i = 0; i <= GRID_LINES; i++) {
                float fraction = i / (float) GRID_LINES;
                s.axisLabels[i] = compact(s.rangeMinY + fraction * (s.rangeMaxY - s.rangeMinY));
            }
        }
        out[0] = s.rangeMinY;
        out[1] = s.rangeMaxY;
    }

    private void updateXAxisLabels(float visMinX, float visMaxX) {
        if (xAxisLabelMin == visMinX && xAxisLabelMax == visMaxX && xAxisLabelTime == xIsTime) {
            return;
        }
        xAxisLabelMin = visMinX;
        xAxisLabelMax = visMaxX;
        xAxisLabelTime = xIsTime;
        for (int i = 0; i < xAxisLabels.length; i++) {
            float fraction = i / (float) (xAxisLabels.length - 1);
            float value = visMinX + fraction * (visMaxX - visMinX);
            xAxisLabels[i] = xIsTime ? formatTime(value) : compact(value);
        }
    }

    private LinearGradient fillShaderFor(SeriesState s) {
        if (s.fillShader == null
                || s.fillShaderTop != plot.top
                || s.fillShaderBottom != plot.bottom
                || s.fillShaderColor != s.color) {
            s.fillShader = new LinearGradient(0, plot.top, 0, plot.bottom,
                    withAlpha(s.color, 0x66), withAlpha(s.color, 0x05), Shader.TileMode.CLAMP);
            s.fillShaderTop = plot.top;
            s.fillShaderBottom = plot.bottom;
            s.fillShaderColor = s.color;
        }
        return s.fillShader;
    }

    private void buildSmoothPath(float[] px, float[] py, int n) {
        linePath.rewind();
        if (n == 0) {
            return;
        }
        linePath.moveTo(px[0], py[0]);
        if (n == 1) {
            return;
        }
        for (int i = 1; i < n; i++) {
            float midX = (px[i - 1] + px[i]) / 2f;
            float midY = (py[i - 1] + py[i]) / 2f;
            linePath.quadTo(px[i - 1], py[i - 1], midX, midY);
        }
        linePath.lineTo(px[n - 1], py[n - 1]);
    }

    private static float interpY(float[] px, float[] py, int n, float targetX) {
        if (n == 0) {
            return Float.NaN;
        }
        if (targetX <= px[0]) {
            return py[0];
        }
        for (int i = 1; i < n; i++) {
            if (targetX <= px[i]) {
                float dx = px[i] - px[i - 1];
                float t = dx <= 0f ? 0f : (targetX - px[i - 1]) / dx;
                return py[i - 1] + t * (py[i] - py[i - 1]);
            }
        }
        return py[n - 1];
    }

    // ---- touch: pinch zoom + drag pan ------------------------------------

    @Override
    public boolean onTouchEvent(MotionEvent event) {
        scaleDetector.onTouchEvent(event);
        switch (event.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
                lastTouchX = event.getX();
                activePointerId = event.getPointerId(0);
                dragging = true;
                drawProgress = 1f;   // stop the entrance reveal once the user interacts
                getParent().requestDisallowInterceptTouchEvent(true);
                return true;
            case MotionEvent.ACTION_MOVE:
                if (dragging && !scaleDetector.isInProgress()) {
                    int idx = event.findPointerIndex(activePointerId);
                    if (idx >= 0) {
                        float x = event.getX(idx);
                        panByPixels(lastTouchX - x);
                        lastTouchX = x;
                    }
                }
                return true;
            case MotionEvent.ACTION_POINTER_UP:
                // Keep panning smoothly with a remaining finger.
                if (event.getPointerCount() <= 2) {
                    int liftedIdx = event.getActionIndex();
                    int liftedId = event.getPointerId(liftedIdx);
                    if (liftedId == activePointerId) {
                        int newIdx = liftedIdx == 0 ? 1 : 0;
                        activePointerId = event.getPointerId(newIdx);
                        lastTouchX = event.getX(newIdx);
                    }
                }
                return true;
            case MotionEvent.ACTION_UP:
                performClick();
                dragging = false;
                activePointerId = -1;
                getParent().requestDisallowInterceptTouchEvent(false);
                return true;
            case MotionEvent.ACTION_CANCEL:
                dragging = false;
                activePointerId = -1;
                getParent().requestDisallowInterceptTouchEvent(false);
                return true;
            default:
                return super.onTouchEvent(event);
        }
    }

    @Override
    public boolean performClick() {
        return super.performClick();
    }

    private void panByPixels(float dxPixels) {
        if (plot.width() <= 0f) {
            return;
        }
        float span = viewHi - viewLo;
        float deltaFrac = dxPixels / plot.width() * span;
        float lo = viewLo + deltaFrac;
        float hi = viewHi + deltaFrac;
        if (lo < 0f) {
            hi -= lo;
            lo = 0f;
        }
        if (hi > 1f) {
            lo -= (hi - 1f);
            hi = 1f;
        }
        viewLo = Math.max(0f, lo);
        viewHi = Math.min(1f, hi);
        invalidate();
    }

    private void zoomBy(float factor) {
        zoomAround(factor, 0.5f);
    }

    private void zoomAround(float factor, float focusFrac) {
        float span = viewHi - viewLo;
        float focusValue = viewLo + focusFrac * span;
        float newSpan = clamp(span * factor, MIN_VIEW_SPAN, 1f);
        float lo = focusValue - focusFrac * newSpan;
        float hi = lo + newSpan;
        if (lo < 0f) {
            lo = 0f;
            hi = newSpan;
        }
        if (hi > 1f) {
            hi = 1f;
            lo = 1f - newSpan;
        }
        viewLo = lo;
        viewHi = hi;
        drawProgress = 1f;
        invalidate();
    }

    private final class ScaleListener extends ScaleGestureDetector.SimpleOnScaleGestureListener {
        @Override
        public boolean onScale(ScaleGestureDetector detector) {
            float focusFrac = plot.width() <= 0f ? 0.5f
                    : clamp01((detector.getFocusX() - plot.left) / plot.width());
            zoomAround(1.0f / detector.getScaleFactor(), focusFrac);
            return true;
        }
    }

    // ---- formatting / colour helpers -------------------------------------

    private static String formatTime(float seconds) {
        if (seconds < 60f) {
            return Math.round(seconds) + "s";
        }
        int total = Math.round(seconds);
        return String.format(Locale.US, "%d:%02d", total / 60, total % 60);
    }

    private static String compact(float v) {
        float abs = Math.abs(v);
        if (abs >= 1000f) {
            return trimZero(String.format(Locale.US, "%.1f", v / 1000f)) + "k";
        }
        if (v == Math.floor(v)) {
            return String.valueOf((long) v);
        }
        return trimZero(String.format(Locale.US, "%.1f", v));
    }

    private static String trimZero(String s) {
        if (s.endsWith(".0")) {
            return s.substring(0, s.length() - 2);
        }
        return s;
    }

    private static int withAlpha(int color, int alpha) {
        return Color.argb(alpha, Color.red(color), Color.green(color), Color.blue(color));
    }

    // Blend a series colour toward the neutral axis tone so its axis labels stay legible but tinted.
    private int mutedColor(int color) {
        return Color.argb(255,
                (Color.red(color) + Color.red(axisTextColor)) / 2,
                (Color.green(color) + Color.green(axisTextColor)) / 2,
                (Color.blue(color) + Color.blue(axisTextColor)) / 2);
    }

    private static float clamp(float v, float lo, float hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private static float clamp01(float v) {
        return clamp(v, 0f, 1f);
    }

    private float dp(float value) {
        return TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, value,
                getResources().getDisplayMetrics());
    }

    private float sp(float value) {
        return TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_SP, value,
                getResources().getDisplayMetrics());
    }

    @Override
    protected void onDetachedFromWindow() {
        if (entranceAnimator != null) {
            entranceAnimator.cancel();
            entranceAnimator = null;
        }
        super.onDetachedFromWindow();
    }
}
