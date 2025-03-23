#version 320 es
#extension GL_OES_EGL_image_external_essl3 : require
precision mediump float;

in vec2 texCoord;//纹理坐标，图片当中的坐标点
out vec4 outColor;

uniform samplerExternalOES camera_texture;//图片，采样器

void main(){
    outColor = texture(camera_texture, texCoord);
}
