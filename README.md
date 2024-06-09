# Native-LLM-for-Android
1. Demonstration of running a native LLM on Android device. Now support:
    - Qwen2-Instruct: 0.5B, 1.5B ...
    - MiniCPM-DPO/SFT: 1B, 2.7B
    - Yuan2.0-Mars-hf: 2B+
    - Octopus-V2: 2.5B
    - Gemma1.1-it: 2.5B
    - StableLM2-Chat/Zephyr: 1.6B, 3B
    - Phi2: 2.7B
2. The demo models were uploaded to the drive: https://drive.google.com/drive/folders/1E43ApPcOq3I2xvb9b7aOxazTcR3hn5zK?usp=drive_link
3. After downloading, place the model into the assets folder.
4. Remember to decompress the *.so zip file stored in the libs/arm64-v8a folder.
5. The demo models were converted from HuggingFace or ModelScope and underwent code optimizations to achieve extreme execution speed.
6. Therefore, the inputs & outputs of the demo models are slightly different from the original one.
7. To better adapt to ONNX Runtime on Android, the export did not use dynamic axes. Therefore, the exported ONNX model may not be optimal for x86_64 systems.
8. The tokenizer.cpp and tokenizer.hpp files originated from the mnn-llm repository.
9. To export the model on your own, please go to the 'Export_ONNX' folder, follow the comments to set the folder path, and then execute the ***_Export.py Python script. Next, quantize / optimize the onnx model by yourself.
10. During the export process of MiniCPM-V, the Resampler always reports an error 'aten::_upsample_bilinear2d_aa' operator not supported, therefore, it is temporarily infeasible to use vision interaction.
11. Updated on June 3, 2024, to use fp16 for caching past_key_values, resulting in approximately 20% faster response time compared to the previous version.
12. See more projects: https://dakeqq.github.io/overview/
# 安卓本地运行LLM
1. 在Android设备上运行本地LLM的演示。目前支持:
   - 通义千问2-Instruct: 0.5B, 1.5B ...
   - MiniCPM-DPO/SFT: 1B, 2.7B
   - 源2.0-Mars-hf: 2B+
   - Octopus-V2: 2.5B
   - Gemma1.1-it: 2.5B
   - StableLM2-Chat/Zephyr: 1.6B, 3B
   - Phi2: 2.7B
2. 演示模型已上传至云端硬盘：https://drive.google.com/drive/folders/1E43ApPcOq3I2xvb9b7aOxazTcR3hn5zK?usp=drive_link
3. 百度: https://pan.baidu.com/s/1NHbUyjZ_VC-o62G13KCrSA?pwd=dake 提取码: dake
4. 下载后，请将模型文件放入assets文件夹。
5. 记得解压存放在libs/arm64-v8a文件夹中的*.so压缩文件。
6. 演示模型是从HuggingFace或ModelScope转换来的，并经过代码优化，以实现极致执行速度。
7. 因此，演示模型的输入输出与原始模型略有不同。
8. 为了更好的适配ONNXRuntime-Android，导出时未使用dynamic-axes. 因此导出的ONNX模型对x86_64而言不一定是最优解.
9. tokenizer.cpp和tokenizer.hpp文件源自mnn-llm仓库。
10. 想自行导出模型请前往“Export_ONNX”文件夹，按照注释操作设定文件夹路径，然后执行 ***_Export.py的python脚本。下一步，自己动手量化或优化导出的ONNX模型。
11. 在导出MiniCPM-V的过程中, Resampler总报错“aten::_upsample_bilinear2d_aa”算子不支持，因此暂时无法使用多模态交互。
12. 2024.6月3日更新，改使用fp16缓存past_key_values，回复速度比旧版快约20%
13. 看更多項目: https://dakeqq.github.io/overview/
# 通义千问 Qwen - 性能 Performance
| OS | Device | Backend | Model | Inference<br>( 1024 Context ) |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | Qwen2-1.5B-Instruct<br>q8f32 | 20.5 token/s |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | Qwen2-1.5B-Instruct<br>q8f32 | 12.5 token/s|
| Harmony 3 | 荣耀20S | Kirin_810-CPU<br>(2*A76) | Qwen2-1.5B-Instruct<br>q8f32 | 6.5 token/s |
# MiniCPM - 性能 Performance
| OS | Device | Backend | Model | Inference<br>( 1024 Context ) |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | MiniCPM-2.7B<br>q8f32 | 9.2 token/s |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | MiniCPM-2.7B<br>q8f32 | 5.5 token/s |
# 源 Yuan - 性能 Performance
| OS | Device | Backend | Model | Inference<br>( 1024 Context ) |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | Yuan2.0-2B-Mars-hf<br>q8f32 | 12 token/s |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | Yuan2.0-2B-Mars-hf<br>q8f32 | 6.5 token/s |
# Octopus - 性能 Performance
| OS | Device | Backend | Model | Inference<br>( 1024 Context ) |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | OctopusV2-2B<br>q8f32 | 15.5 token/s |
# Gemma - 性能 Performance
| OS | Device | Backend | Model | Inference<br>( 1024 Context ) |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | Gemma1.1-it-2B<br>q8f32 | 15.5 token/s |
# StableLM - 性能 Performance
| OS | Device | Backend | Model | Inference<br>( 1024 Context ) |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | StableLM2-1.6B-Chat<br>q8f32 | 17.8 token/s |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | StableLM2-1.6B-Chat<br>q8f32 | 11 token/s |
| Harmony 3 | 荣耀20S | Kirin_810-CPU<br>(2*A76) | StableLM2-1.6B-Chat<br>q8f32 | 5.5 token/s |
# Phi - 性能 Performance
| OS | Device | Backend | Model | Inference<br>( 1024 Context ) |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X3+A715) | Phi2-2B-Orange-V2<br>q8f32 | 9.5 token/s |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | Phi2-2B-Orange-V2<br>q8f32 | 5.8 token/s |
# 演示结果 Demo Results
(Qwen1.5-1.8B / 1024 Context)<br>
![Demo Animation](https://github.com/DakeQQ/Native-LLM-for-Android/blob/main/LLM_Qwen.gif?raw=true)
