package com.example.myapplication;
import android.annotation.SuppressLint;
import android.graphics.RectF;
import androidx.annotation.NonNull;


/**
 * Generic interface for interacting with different recognition engines.
 */
public interface Classifier {
    /**
     * An immutable result returned by a Classifier describing what was recognized.
     */
    class Recognition {
        /**
         * A unique identifier for what has been recognized. Specific to the class, not the instance of
         * the object.
         */
        private final String id;
        /**
         * Display name for the recognition.
         */
        private final String title;
        /**
         * A sortable score for how good the recognition is relative to others. Higher should be better.
         */
        private final Float confidence;
        /**
         * Optional location within the source image for the location of the recognized object.
         */
        private final RectF location;
        public Recognition(final String id, final String title, final Float confidence, final RectF location) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
            this.location = location;
        }

        public Float getConfidence() {
            return confidence;
        }

        public RectF getLocation() {
            return new RectF(location);
        }

        public String getTitle() {
            return title;
        }

        @NonNull
        @SuppressLint("DefaultLocale")
        @Override
        public String toString() {
            String resultString = "";
            if (id != null) {
                resultString += "[" + id + "] ";
            }

            if (title != null) {
                resultString += title + " ";
            }

            if (confidence != null) {
                resultString += String.format("(%.2f%%) ", confidence * 100.f);
            }

            if (location != null) {
                resultString += location + " ";
            }

            return resultString.trim();
        }
    }
}
