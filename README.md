# Native-LLM-for-Android
1. Demonstration of running a native LLM on Android device.
2. The demo models were uploaded to the drive: https://drive.google.com/drive/folders/1E43ApPcOq3I2xvb9b7aOxazTcR3hn5zK?usp=drive_link
3. After downloading, place the model into the assets folder.
4. Remember to decompress the *.so zip file stored in the libs/arm64-v8a folder.
5. The demo models were converted from ModelScope and underwent code optimizations to achieve extreme execution speed.
6. Therefore, the inputs & outputs of the demo models are slightly different from the original one.
7. The tokenizer.cpp and tokenizer.hpp files originated from the mnn-llm repository.
8. To export the model on your own, please go to the 'Export' folder, follow the comments to replace the original 'modeling_***.py', and then execute the ***_Export.py Python script. Next, quantize / optimize the onnx model by yourself.
9. See more projects: https://dakeqq.github.io/overview/
# 安卓本地运行LLM
1. 在Android设备上运行本地LLM的演示。
2. 演示模型已上传至云端硬盘：https://drive.google.com/drive/folders/1E43ApPcOq3I2xvb9b7aOxazTcR3hn5zK?usp=drive_link
3. 百度: https://pan.baidu.com/s/1NHbUyjZ_VC-o62G13KCrSA?pwd=dake 提取码: dake
4. 下载后，请将模型文件放入assets文件夹。
5. 记得解压存放在libs/arm64-v8a文件夹中的*.so压缩文件。
6. 演示模型是从ModelScope转换来的，并经过代码优化，以实现极致执行速度。
7. 因此，演示模型的输入输出与原始模型略有不同。
8. tokenizer.cpp和tokenizer.hpp文件源自mnn-llm仓库。
9. 想自行导出模型请前往“Export”文件夹，按照注释操作取代原模型的“modeling_***.py”，然后执行 ***_Export.py的python脚本。下一步，自己动手量化或优化导出的ONNX模型。
10. 看更多項目: https://dakeqq.github.io/overview/
# 性能 Performance
| OS | Device | Backend | Model | Inference<br>( 1024 Context ) |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X2+A715) | Qwen1.5-1.8B<br>q8f32 | 14 token/s |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | Qwen1.5-1.8B<br>q8f32 | 9 token/s |
| Android 13 | Nubia Z50 | 8_Gen2-CPU<br>(X2+A715) | MiniCPM-2.7B<br>q8f32 | 7.7 token/s |
| Harmony 4 | P40 | Kirin_990_5G-CPU<br>(2*A76) | MiniCPM-2.7B<br>q8f32 | 4.5 token/s |
# 演示结果 Demo Results
(Qwen1.5-1.8B / 1024 Context)<br>
![Demo Animation](https://github.com/DakeQQ/Native-LLM-for-Android/blob/main/LLM_Qwen.gif?raw=true)
