package com.example.myapplication;

import android.content.Context;
import android.util.AttributeSet;

import androidx.core.widget.NestedScrollView;

public final class MaxHeightNestedScrollView extends NestedScrollView {
    private int maxHeight = Integer.MAX_VALUE;

    public MaxHeightNestedScrollView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public void setMaxHeight(int maxHeight) {
        this.maxHeight = Math.max(1, maxHeight);
        requestLayout();
    }

    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        int mode = MeasureSpec.getMode(heightMeasureSpec);
        int size = MeasureSpec.getSize(heightMeasureSpec);
        int cappedSize = Math.min(size, maxHeight);
        int cappedSpec = MeasureSpec.makeMeasureSpec(cappedSize,
                mode == MeasureSpec.EXACTLY ? MeasureSpec.EXACTLY : MeasureSpec.AT_MOST);
        super.onMeasure(widthMeasureSpec, cappedSpec);
    }
}