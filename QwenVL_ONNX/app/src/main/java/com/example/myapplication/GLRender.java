package com.example.myapplication;

import static com.example.myapplication.MainActivity.Process_Init;
import static com.example.myapplication.MainActivity.Process_Texture;
import static com.example.myapplication.MainActivity.chatting;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.params.MeteringRectangle;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public class GLRender implements GLSurfaceView.Renderer {
    public static final ExecutorService executorService = Executors.newFixedThreadPool(4);
    @SuppressLint("StaticFieldLeak")
    private static Context mContext;
    private static int mVertexLocation;
    private static int mTextureLocation;
    private static int mUTextureLocation;
    private static int mVMatrixLocation;
    private static int ShaderProgram_Camera;
    private static final int screen_resolution_short_side = 1080;
    public static final int camera_width = 960;  // Please modify the project.h file simultaneously when editing these values.
    public static final int camera_height = 960;
    private static final int camera_pixels = camera_height * camera_width;
    private static final int camera_pixels_2 = camera_pixels * 2;
    private static final int camera_pixels_half = camera_pixels / 2;
    private static final int place_view_center = (screen_resolution_short_side - camera_width) / 2;
    private static final int[] mTextureId = new int[1];
    public static int[] imageRGBA = new int[camera_pixels];
    public static final MeteringRectangle[] focus_area = new MeteringRectangle[]{new MeteringRectangle(camera_width >> 1, camera_height >> 1, 100, 100, MeteringRectangle.METERING_WEIGHT_MAX)};
    private static final float inv_255 = 1.f / 255.f;
    public static final float[] pixel_values = new float[camera_pixels * 3];
    private static final float[] vMatrix = new float[16];
    private static final String VERTEX_ATTRIB_POSITION = "aPosVertex";
    private static final String VERTEX_ATTRIB_TEXTURE_POSITION = "aTexVertex";
    private static final String UNIFORM_TEXTURE = "camera_texture";
    private static final String UNIFORM_VMATRIX = "vMatrix";
    private static final String camera_vertex_shader_name = "camera_vertex_shader.glsl";
    private static final String camera_fragment_shader_name = "camera_fragment_shader.glsl";
    public static SurfaceTexture mSurfaceTexture;

    public GLRender(Context context) {
        mContext = context;
    }

    public SurfaceTexture getSurfaceTexture() {
        return mSurfaceTexture;
    }

    private static final FloatBuffer mVertexCoord_buffer = getFloatBuffer(new float[]{
            -1f, -1f,
            1f, -1f,
            -1f, 1f,
            1f, 1f
    });

    private static final FloatBuffer mTextureCoord_buffer = getFloatBuffer(new float[]{
            0.0f, 0.0f,
            1.0f, 0.0f,
            0.0f, 1.0f,
            1.0f, 1.0f
    });

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        GLES20.glEnable(GLES20.GL_BLEND);
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA);
        GLES20.glClearColor(0.f, 0.0f, 0.0f, 1.0f);
        ShaderProgram_Camera = createAndLinkProgram(camera_vertex_shader_name, camera_fragment_shader_name);
        initTexture();
        initAttribLocation();
        Process_Init(mTextureId[0]);
        ((MainActivity) mContext).openCamera();
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        GLES20.glViewport(place_view_center, 0, camera_height, camera_width);
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        mSurfaceTexture.updateTexImage();
        mSurfaceTexture.getTransformMatrix(vMatrix);
        Draw_Camera_Preview();
        if (!chatting) {
            imageRGBA = Process_Texture();
            // Choose CPU normalization over GPU, as GPU float32 buffer access is much slower than int8 buffer access.
            // Therefore, use a new thread to parallelize the normalization process.
            executorService.execute(() -> {
                for (int i = 0; i < camera_pixels_half; i++) {
                    int rgba = imageRGBA[i];
                    pixel_values[i] = (float) ((rgba >> 16) & 0xFF) * inv_255;
                    pixel_values[i + camera_pixels] = (float) ((rgba >> 8) & 0xFF) * inv_255;
                    pixel_values[i + camera_pixels_2] = (float) (rgba & 0xFF) * inv_255;
                }
            });
            executorService.execute(() -> {
                for (int i = camera_pixels_half; i < camera_pixels; i++) {
                    int rgba = imageRGBA[i];
                    pixel_values[i] = (float) ((rgba >> 16) & 0xFF) * inv_255;
                    pixel_values[i + camera_pixels] = (float) ((rgba >> 8) & 0xFF) * inv_255;
                    pixel_values[i + camera_pixels_2] = (float) (rgba & 0xFF) * inv_255;
                }
            });
        }
    }

    private static void Draw_Camera_Preview() {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
        GLES20.glUseProgram(ShaderProgram_Camera);
        GLES20.glVertexAttribPointer(mVertexLocation, 2, GLES20.GL_FLOAT, false, 0, mVertexCoord_buffer);
        GLES20.glVertexAttribPointer(mTextureLocation, 2, GLES20.GL_FLOAT, false, 0, mTextureCoord_buffer);
        GLES20.glEnableVertexAttribArray(mVertexLocation);
        GLES20.glEnableVertexAttribArray(mTextureLocation);
        GLES20.glUniformMatrix4fv(mVMatrixLocation, 1, false, vMatrix, 0);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);  // mVertexCoord.length / 2
    }

    private static void initAttribLocation() {
        mVertexLocation = GLES20.glGetAttribLocation(ShaderProgram_Camera, VERTEX_ATTRIB_POSITION);
        mTextureLocation = GLES20.glGetAttribLocation(ShaderProgram_Camera, VERTEX_ATTRIB_TEXTURE_POSITION);
        mUTextureLocation = GLES20.glGetUniformLocation(ShaderProgram_Camera, UNIFORM_TEXTURE);
        mVMatrixLocation = GLES20.glGetUniformLocation(ShaderProgram_Camera, UNIFORM_VMATRIX);
    }

    private static void initTexture() {
        GLES20.glGenTextures(mTextureId.length, mTextureId, 0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, mTextureId[0]);
        GLES20.glTexParameterf(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameterf(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);
        GLES20.glTexParameterf(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameterf(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        mSurfaceTexture = new SurfaceTexture(mTextureId[0]);
        mSurfaceTexture.setDefaultBufferSize(camera_width, camera_height);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glUniform1i(mUTextureLocation, 0);
    }

    private static int createAndLinkProgram(String vertexShaderFN, String fragShaderFN) {
        int shaderProgram = GLES20.glCreateProgram();
        if (shaderProgram == 0) {
            return 0;
        }
        AssetManager mgr = mContext.getResources().getAssets();
        int vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, loadShaderSource(mgr, vertexShaderFN));
        if (0 == vertexShader) {
            return 0;
        }
        int fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, loadShaderSource(mgr, fragShaderFN));
        if (0 == fragmentShader) {
            return 0;
        }
        GLES20.glAttachShader(shaderProgram, vertexShader);
        GLES20.glAttachShader(shaderProgram, fragmentShader);
        GLES20.glLinkProgram(shaderProgram);
        int[] linked = new int[1];
        GLES20.glGetProgramiv(shaderProgram, GLES20.GL_LINK_STATUS, linked, 0);
        if (linked[0] == 0) {
            GLES20.glDeleteProgram(shaderProgram);
            return 0;
        }
        return shaderProgram;
    }

    private static int loadShader(int type, String shaderSource) {
        int shader = GLES20.glCreateShader(type);
        if (shader == 0) {
            return 0;
        }
        GLES20.glShaderSource(shader, shaderSource);
        GLES20.glCompileShader(shader);
        int[] compiled = new int[1];
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compiled, 0);
        if (compiled[0] == 0) {
            GLES20.glDeleteShader(shader);
            return 0;
        }
        return shader;
    }

    private static String loadShaderSource(AssetManager mgr, String file_name) {
        StringBuilder strBld = new StringBuilder();
        String nextLine;
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(mgr.open(file_name)));
            while ((nextLine = br.readLine()) != null) {
                strBld.append(nextLine).append('\n');
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return strBld.toString();
    }

    private static FloatBuffer getFloatBuffer(float[] array) {
        FloatBuffer buffer = ByteBuffer.allocateDirect((array.length << 2)).order(ByteOrder.nativeOrder()).asFloatBuffer();
        buffer.put(array).position(0);
        return buffer;
    }
}
