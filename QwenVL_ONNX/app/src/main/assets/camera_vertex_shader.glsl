#version 320 es
precision mediump float;
//把顶点坐标给这个变量， 确定要画画的形状
layout (location = 0) in vec4 aPosVertex;

//接收纹理坐标，接收采样器采样图片的坐标
layout (location = 1) in vec4 aTexVertex;

//传给片元着色器 像素点
out vec2 texCoord;

uniform mat4 vMatrix;

void main()
{
    //内置变量 gl_Position ,我们把顶点数据赋值给这个变量 opengl就知道它要画什么形状了
    gl_Position = aPosVertex;
    texCoord = (vMatrix * aTexVertex).xy;
    //texCoord = aTexVertex.xy;
}
