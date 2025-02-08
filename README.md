# Native-LLM-for-Android

## Overview
Demonstration of running a native Large Language Model (LLM) on Android devices. Currently supported models include:

- **DeepSeek-R1-Distill-Qwen**: 1.5B
- **Qwen2.5-Instruct**: 0.5B, 1.5B
- **Qwen2/2.5VL**: 2B, 3B
- **MiniCPM-DPO/SFT**: 1B, 2.7B
- **Gemma2-it**: 2B
- **Phi3.5-mini-instruct**: 3.8B
- **Llama-3.2-Instruct**: 1B

## Getting Started
1. **Download Models:**
   - Demo models are available on [Google Drive](https://drive.google.com/drive/folders/1E43ApPcOq3I2xvb9b7aOxazTcR3hn5zK?usp=drive_link).
   - Alternatively, use [Baidu Cloud](https://pan.baidu.com/s/1NHbUyjZ_VC-o62G13KCrSA?pwd=dake) with the extraction code: `dake`.

2. **Setup Instructions:**
   - Place the downloaded model files into the `assets` folder.
   - Decompress the `*.so` files stored in the `libs/arm64-v8a` folder.

3. **Model Notes:**
   - Demo models are converted from HuggingFace or ModelScope and optimized for extreme execution speed.
   - Inputs and outputs may differ slightly from the original models.

4. **ONNX Export Considerations:**
   - Dynamic axes were not used during export to better adapt to ONNX Runtime on Android. Exported ONNX models may not be optimal for x86_64 systems.

## Tokenizer Files
- The `tokenizer.cpp` and `tokenizer.hpp` files are sourced from the [mnn-llm repository](https://github.com/mnn-llm).

## Exporting Models
1. Navigate to the `Export_ONNX` folder.
2. Follow the comments in the Python scripts to set the folder paths.
3. Execute the `***_Export.py` script to export the model.
4. Quantize or optimize the ONNX model manually.

## Quantization Notes
- Use `onnxruntime.tools.convert_onnx_models_to_ort` to convert models to `*.ort` format. Note that this process automatically adds `Cast` operators that change FP16 multiplication to FP32.
- The quantization methods are detailed in the `Do_Quantize` folder.
- The `q4` (uint4) quantization method is not recommended due to poor performance of the `MatMulNBits` operator in ONNX Runtime.

## Recent Updates
- 2025/02/07：**DeepSeek-R1-Distill-Qwen**: 1.5B (Please using Qwen_Export.py)
- Fix the continuous chat bugs.

## Additional Resources
- Explore more projects: [DakeQQ Projects](https://github.com/DakeQQ?tab=repositories)

## Performance Metrics
### DeepSeek-R1
| OS         | Device       | Backend                 | Model                  | Inference (1024 Context) |
|:----------:|:------------:|:-----------------------:|:----------------------:|:------------------------:|
| Harmony 4  | P40          | Kirin_990_5G-CPU (2*A76) | Distill-Qwen-1.5B<br>q8f32 | 13 token/s         |

### Qwen2VL
| OS         | Device       | Backend                 | Model             | Inference (1024 Context) |
|:----------:|:------------:|:-----------------------:|:-----------------:|:------------------------:|
| Android 13 | Nubia Z50    | 8_Gen2-CPU (X3+A715)   | Qwen2VL-2B<br>q8f32 | 15 token/s              |
| Harmony 4  | P40          | Kirin_990_5G-CPU (2*A76) | Qwen2VL-2B<br>q8f32 | 9 token/s               |

### Qwen
| OS         | Device       | Backend                 | Model                  | Inference (1024 Context) |
|:----------:|:------------:|:-----------------------:|:----------------------:|:------------------------:|
| Android 13 | Nubia Z50    | 8_Gen2-CPU (X3+A715)   | Qwen2-1.5B-Instruct<br>q8f32 | 20 token/s         |
| Harmony 4  | P40          | Kirin_990_5G-CPU (2*A76) | Qwen2-1.5B-Instruct<br>q8f32 | 13 token/s         |
| Harmony 3  | 荣耀\u20 (20S)  | Kirin_810-CPU (2*A76)     | Qwen2-1.5B-Instruct<br>q8f32 | 7 token/s          |

### MiniCPM
| OS         | Device       | Backend                 | Model                  | Inference (1024 Context) |
|:----------:|:------------:|:-----------------------:|:----------------------:|:------------------------:|
| Android 13 | Nubia Z50    | 8_Gen2-CPU (X3+A715)   | MiniCPM-2.7B<br>q8f32   | 9.5 token/s              |
| Harmony 4  | P40          | Kirin_990_5G-CPU (2*A76) | MiniCPM-2.7B<br>q8f32   | 6 token/s               |
| Android 13 | Nubia Z50    | 8_Gen2-CPU (X3+A715)   | MiniCPM-1.3B<br>q8f32   | 16.5 token/s             |
| Harmony 4  | P40          | Kirin_990_5G-CPU (2*A76) | MiniCPM-1.3B<br>q8f32   | 11 token/s              |

### Yuan
| OS         | Device       | Backend                 | Model                  | Inference (1024 Context) |
|:----------:|:------------:|:-----------------------:|:----------------------:|:------------------------:|
| Android 13 | Nubia Z50    | 8_Gen2-CPU (X3+A715)   | Yuan2.0-2B-Mars-hf<br>q8f32 | 12 token/s         |
| Harmony 4  | P40          | Kirin_990_5G-CPU (2*A76) | Yuan2.0-2B-Mars-hf<br>q8f32 | 6.5 token/s       |

### Gemma
| OS         | Device       | Backend                 | Model                  | Inference (1024 Context) |
|:----------:|:------------:|:-----------------------:|:----------------------:|:------------------------:|
| Android 13 | Nubia Z50    | 8_Gen2-CPU (X3+A715)   | Gemma1.1-it-2B<br>q8f32 | 16 token/s               |

### StableLM
| OS         | Device       | Backend                 | Model                  | Inference (1024 Context) |
|:----------:|:------------:|:-----------------------:|:----------------------:|:------------------------:|
| Android 13 | Nubia Z50    | 8_Gen2-CPU (X3+A715)   | StableLM2-1.6B-Chat<br>q8f32 | 17.8 token/s      |
| Harmony 4  | P40          | Kirin_990_5G-CPU (2*A76) | StableLM2-1.6B-Chat<br>q8f32 | 11 token/s        |
| Harmony 3  | 荣耀\u20 (20S)  | Kirin_810-CPU (2*A76)     | StableLM2-1.6B-Chat<br>q8f32 | 5.5 token/s       |

### Phi
| OS         | Device       | Backend                 | Model                  | Inference (1024 Context) |
|:----------:|:------------:|:-----------------------:|:----------------------:|:------------------------:|
| Android 13 | Nubia Z50    | 8_Gen2-CPU (X3+A715)   | Phi2-2B-Orange-V2<br>q8f32 | 9.5 token/s       |
| Harmony 4  | P40          | Kirin_990_5G-CPU (2*A76) | Phi2-2B-Orange-V2<br>q8f32 | 5.8 token/s       |

### Llama
| OS         | Device       | Backend                 | Model                  | Inference (1024 Context) |
|:----------:|:------------:|:-----------------------:|:----------------------:|:------------------------:|
| Android 13 | Nubia Z50    | 8_Gen2-CPU (X3+A715)   | Llama3.2-1B-Instruct<br>q8f32 | 25 token/s     |
| Harmony 4  | P40          | Kirin_990_5G-CPU (2*A76) | Llama3.2-1B-Instruct<br>q8f32 | 16 token/s     |

## Demo Results
### Qwen2VL-2B / 1024 Context
![Demo Animation](https://github.com/DakeQQ/Native-LLM-for-Android/blob/main/LLM_QwenVL.gif?raw=true)

### Qwen2-1.5B / 1024 Context
![Demo Animation](https://github.com/DakeQQ/Native-LLM-for-Android/blob/main/LLM_Qwen.gif?raw=true)

## 概述

展示在 Android 设备上运行原生大型语言模型 (LLM) 的示范。目前支持的模型包括：

- **DeepSeek-R1-Distill-Qwen**: 1.5B
- **Qwen2.5-Instruct**: 0.5B, 1.5B
- **Qwen2/2.5VL**: 2B, 3B
- **MiniCPM-DPO/SFT**: 1B, 2.7B
- **Gemma2-it**: 2B
- **Phi3.5-mini-instruct**: 3.8B
- **Llama-3.2-Instruct**: 1B

## 入门指南

1. **下载模型：**
   - Demo模型可以在 [Google Drive](https://drive.google.com/drive/folders/1E43ApPcOq3I2xvb9b7aOxazTcR3hn5zK?usp=drive_link) 上获取。
   - 或者使用 [百度网盘](https://pan.baidu.com/s/1NHbUyjZ_VC-o62G13KCrSA?pwd=dake) 提取码：`dake`。

2. **设置说明：**
   - 将下载的模型文件放入 `assets` 文件夹。
   - 解压存储在 `libs/arm64-v8a` 文件夹中的 `*.so` 文件。

3. **模型说明：**
   - 演示模型是从 HuggingFace 或 ModelScope 转换而来，并针对极限执行速度进行了优化。
   - 输入和输出可能与原始模型略有不同。

4. **ONNX 导出注意事项：**
   - 导出时未使用动态轴，以更好地适应 Android 上的 ONNX Runtime。导出的 ONNX 模型可能不适合 x86_64 系统。

## 分词器文件

- `tokenizer.cpp` 和 `tokenizer.hpp` 文件来源于 [mnn-llm 仓库](https://github.com/mnn-llm)。

## 导出模型

1. 进入 `Export_ONNX` 文件夹。
2. 按照 Python 脚本中的注释设置文件夹路径。
3. 执行 `***_Export.py` 脚本以导出模型。
4. 手动量化或优化 ONNX 模型。

## 量化说明

- 使用 `onnxruntime.tools.convert_onnx_models_to_ort` 将模型转换为 `*.ort` 格式。注意该过程会自动添加 `Cast` 操作符，将 FP16 乘法改为 FP32。
- 量化方法详见 `Do_Quantize` 文件夹。
- 不推荐使用 `q4` (uint4) 量化方法，因为 ONNX Runtime 中 `MatMulNBits` 操作符性能较差。

## 最近更新

- 2025/02/07：**DeepSeek-R1-Distill-Qwen**: 1.5B （请使用Qwen_Export.py）
- 修复连续对话的错误。

## 额外资源

- 探索更多项目：[DakeQQ Projects](https://github.com/DakeQQ?tab=repositories)
