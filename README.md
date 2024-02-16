# Native-LLM-for-Android
1. Demonstration of running a native LLM on Android device.
2. The demo models were uploaded to the drive: https://drive.google.com/drive/folders/1ig73aGeXtd7NGjfU6qvZcMLXj947JCHb?usp=drive_link
3. After downloading, place the model into the assets folder.
4. Remember to decompress the *.so zip file stored in the libs/arm64-v8a folder.
5. The demo models, named 'Qwen', were converted from ModelScope and underwent code optimizations to achieve extreme execution speed.
6. Therefore, the inputs & outputs of the demo models are slightly different from the original one.
7. The tokenizer.cpp and tokenizer.hpp files originated from the mnn-llm repository.
8. We will make the exported method public later.

