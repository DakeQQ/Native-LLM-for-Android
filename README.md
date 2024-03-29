# Native-LLM-for-Android
1. Demonstration of running a native LLM on Android device.
2. The demo models were uploaded to the drive: https://drive.google.com/drive/folders/1E43ApPcOq3I2xvb9b7aOxazTcR3hn5zK?usp=drive_link
3. After downloading, place the model into the assets folder.
4. Remember to decompress the *.so zip file stored in the libs/arm64-v8a folder.
5. The demo models, named 'Qwen, version:1.5, params:1.8B', were converted from ModelScope and underwent code optimizations to achieve extreme execution speed.
6. Therefore, the inputs & outputs of the demo models are slightly different from the original one.
7. The tokenizer.cpp and tokenizer.hpp files originated from the mnn-llm repository.
8. We will make the exported method public later, and it does not support old versions of LLM.
9. See more projects: https://dakeqq.github.io/overview/
# 安卓本地运行LLM
1. 在Android设备上运行本地LLM的演示。
2. 演示模型已上传至云端硬盘：https://drive.google.com/drive/folders/1E43ApPcOq3I2xvb9b7aOxazTcR3hn5zK?usp=drive_link
3. 百度: 链接: 链接: 链接: https://pan.baidu.com/s/1ao_w12zesFS6ZfJnsw0lWA?pwd=dake 提取码: dake
4. 下载后，请将模型文件放入assets文件夹。
5. 记得解压存放在libs/arm64-v8a文件夹中的*.so压缩文件。
6. 演示模型名为'Qwen, 版本:1.5, 参数量:1.8B'，它们是从ModelScope转换来的，并经过代码优化，以实现极致执行速度。
7. 因此，演示模型的输入输出与原始模型略有不同。
8. tokenizer.cpp和tokenizer.hpp文件源自mnn-llm仓库。
9. 我们未来会提供转换导出的方法, 并且不再支持旧版的LLM。
10. 看更多項目: https://dakeqq.github.io/overview/
# 性能 Benchmark
| OS | Device | Backend | Format | Inference (256 Context) | Inference (1024 Context) |
|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Android 13 | Nubia Z50 | 8_Gen2 - cpu (1*X2+1*A715) | q8f32 | 19 token/s | 13 token/s |
| Harmony 4 | P40 | Kirin_990_5G - cpu (2*A76) | q8f32 | 11 token/s | 8 token/s |
# 演示结果 Demo Results
![Demo Animation](https://github.com/DakeQQ/Native-LLM-for-Android/blob/main/LLM.gif?raw=true)
