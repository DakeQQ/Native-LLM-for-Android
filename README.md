# Native-LLM-for-Android
1. Demonstration of running a native LLM on Android device.
2. The demo models were uploaded to the drive: https://drive.google.com/drive/folders/1ig73aGeXtd7NGjfU6qvZcMLXj947JCHb?usp=drive_link
3. After downloading, place the model into the assets folder.
4. Remember to decompress the *.so zip file stored in the libs/arm64-v8a folder.
5. The demo models, named 'Qwen', were converted from ModelScope and underwent code optimizations to achieve extreme execution speed.
6. Therefore, the inputs & outputs of the demo models are slightly different from the original one.
7. The tokenizer.cpp and tokenizer.hpp files originated from the mnn-llm repository.
8. We will make the exported method public later.
# 安卓本地运行LLM
1. 在Android设备上运行本地LLM的演示。
2. 演示模型已上传至云端硬盘：https://drive.google.com/drive/folders/1ig73aGeXtd7NGjfU6qvZcMLXj947JCHb?usp=drive_link
3. 下载后，请将模型文件放入assets文件夹。
4. 记得解压存放在libs/arm64-v8a文件夹中的*.so压缩文件。
5. 演示模型名为'Qwen'，它们是从ModelScope转换来的，并经过代码优化，以实现极致执行速度。
6. 因此，演示模型的输入输出与原始模型略有不同。
7. tokenizer.cpp和tokenizer.hpp文件源自mnn-llm仓库。
8. 我们未来会提供转换导出的方法。
# 演示结果 Demo Results
![Demo Animation](https://github.com/DakeQQ/Native-LLM-for-Android/blob/main/LLM.gif?raw=true)
