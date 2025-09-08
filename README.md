# Native-LLM-for-Android

## Overview
Demonstration of running a native Large Language Model (LLM) on Android devices. Currently supported models include:

- **Qwen3**: 0.6B, 1.7B, 4B...
- **Qwen2.5-Instruct**: 0.5B, 1.5B, 3B...
- **Qwen2.5VL**: 3B
- **DeepSeek-R1-Distill-Qwen**: 1.5B
- **MiniCPM-DPO/SFT**: 1B, 2.7B
- **Gemma-3-it**: 1B, 4B...
- **Phi-4-mini-Instruct**: 3.8B
- **Llama-3.2-Instruct**: 1B
- **InternVL-Mono**: 2B
- **InternLM-3**: 8B
- **Seed-X**: [PRO-7B](https://modelscope.cn/models/ByteDance-Seed/Seed-X-PPO-7B), [Instruct-7B](https://modelscope.cn/models/ByteDance-Seed/Seed-X-Instruct-7B)
- **HunYuan**: [MT-7B](https://github.com/Tencent-Hunyuan/Hunyuan-MT)

## Update
- 2025/09/07：Update HunYuan-MT.
- 2025/08/02：Update Seed-X.
- 2025/04/29：Update Qwen3.
- 2025/04/05：Update Qwen2.5, InternVL-Mono `q4f32` + `dynamic_axes`.
- 2025/02/22：Support loading with low memory mode: `Qwen`, `QwenVL`, `MiniCPM_2B_single`; Set `low_memory_mode = true` in `MainActivity.java`.
- 2025/02/07：**DeepSeek-R1-Distill-Qwen**: 1.5B (Please using `Qwen v2.5 Qwen_Export.py`)

## Getting Started
1. **Download Models:**
   - Quick Try: [Qwen3-1.7B-Android](https://huggingface.co/H5N1AIDS/Qwen_Android_ONNX_Runtime/tree/main)

2. **Setup Instructions:**
   - Place the downloaded model files into the `assets` folder.
   - Decompress the `*.so` files stored in the `libs/arm64-v8a` folder.

3. **Model Notes:**
   - Demo models are converted from HuggingFace or ModelScope and optimized for extreme execution speed.
   - Inputs and outputs may differ slightly from the original models.
   - For Qwen2VL / Qwen2.5VL, adjust the key variables to match the model parameters.
      - `GLRender.java: Line 37, 38, 39`
      - `project.h: Line 14, 15, 16, 35, 36, 41, 59, 60`

4. **ONNX Export Considerations:**
   - It is recommended to use dynamic axes and q4f32 quantization.
   
## Tokenizer Files
- The `tokenizer.cpp` and `tokenizer.hpp` files are sourced from the [mnn-llm repository](https://github.com/alibaba/MNN/tree/master/transformers/llm/engine/src).

## Exporting Models
1. Navigate to the `Export_ONNX` folder.
2. Follow the comments in the Python scripts to set the folder paths.
3. Execute the `***_Export.py` script to export the model.
4. Quantize or optimize the ONNX model manually.

## Quantization Notes
- Use `onnxruntime.tools.convert_onnx_models_to_ort` to convert models to `*.ort` format. Note that this process automatically adds `Cast` operators that change FP16 multiplication to FP32.
- The quantization methods are detailed in the `Do_Quantize` folder.

## Additional Resources
- Explore more projects: [DakeQQ Projects](https://github.com/DakeQQ?tab=repositories)

## Performance Metrics
### Qwen
| OS         | Device       | Backend                 | Model                  | Inference (1024 Context) |
|:----------:|:------------:|:-----------------------:|:----------------------:|:------------------------:|
| Android 13 | Nubia Z50    | 8_Gen2-CPU              | Qwen-2-1.5B-Instruct<br>q8f32 | 20 token/s         |
| Android 15  | Vivo x200 Pro | MediaTek_9400-CPU     | Qwen-3-1.7B-Instruct<br>q4f32<br>dynamic | 37 token/s  |
| Harmony 4  | P40          | Kirin_990_5G-CPU        | Qwen-3-1.7B-Instruct<br>q4f32<br>dynamic | 18.5 token/s  |
| Harmony 4  | P40          | Kirin_990_5G-CPU        | Qwen-2.5-1.5B-Instruct<br>q4f32<br>dynamic | 20.5 token/s  |
| Harmony 4  | P40          | Kirin_990_5G-CPU        | Qwen-2-1.5B-Instruct<br>q8f32 | 13 token/s         |
| Harmony 3  | 荣耀 20S      | Kirin_810-CPU           | Qwen-2-1.5B-Instruct<br>q8f32 | 7 token/s          |

### QwenVL
| OS         | Device       | Backend                 | Model             | Inference (1024 Context) |
|:----------:|:------------:|:-----------------------:|:-----------------:|:------------------------:|
| Android 13 | Nubia Z50    | 8_Gen2-CPU              | QwenVL-2-2B<br>q8f32 | 15 token/s              |
| Harmony 4  | P40          | Kirin_990_5G-CPU        | QwenVL-2-2B<br>q8f32 | 9 token/s               |
| Harmony 4  | P40          | Kirin_990_5G-CPU        | QwenVL-2.5-3B<br>q4f32<br>dynamic | 9 token/s  |


### DeepSeek-R1
| OS         | Device       | Backend                 | Model                  | Inference (1024 Context) |
|:----------:|:------------:|:-----------------------:|:----------------------:|:------------------------:|
| Android 13 | Nubia Z50    | 8_Gen2-CPU              | Distill-Qwen-1.5B<br>q4f32<br>dynamic | 34.5 token/s |
| Harmony 4  | P40          | Kirin_990_5G-CPU        | Distill-Qwen-1.5B<br>q4f32<br>dynamic | 20.5 token/s |
| Harmony 4  | P40          | Kirin_990_5G-CPU        | Distill-Qwen-1.5B<br>q8f32 | 13 token/s         |
| HyperOS 2  | Xiaomi-14T-Pro | MediaTek_9300+-CPU    | Distill-Qwen-1.5B<br>q8f32 | 22 token/s         |

### MiniCPM
| OS         | Device       | Backend                 | Model                  | Inference (1024 Context) |
|:----------:|:------------:|:-----------------------:|:----------------------:|:------------------------:|
| Android 15 | Nubia Z50    | 8_Gen2-CPU              | MiniCPM4-0.5B<br>q4f32 | 78 token/s               |
| Android 13 | Nubia Z50    | 8_Gen2-CPU              | MiniCPM-2.7B<br>q8f32   | 9.5 token/s             |
| Android 13 | Nubia Z50    | 8_Gen2-CPU              | MiniCPM-1.3B<br>q8f32   | 16.5 token/s            |
| Harmony 4  | P40          | Kirin_990_5G-CPU        | MiniCPM-2.7B<br>q8f32   | 6 token/s               |
| Harmony 4  | P40          | Kirin_990_5G-CPU        | MiniCPM-1.3B<br>q8f32   | 11 token/s              |

### Gemma
| OS         | Device       | Backend                 | Model                  | Inference (1024 Context) |
|:----------:|:------------:|:-----------------------:|:----------------------:|:------------------------:|
| Android 13 | Nubia Z50    | 8_Gen2-CPU              | Gemma-1.1-it-2B<br>q8f32 | 16 token/s               |

### Phi
| OS         | Device       | Backend                 | Model                  | Inference (1024 Context) |
|:----------:|:------------:|:-----------------------:|:----------------------:|:------------------------:|
| Android 13 | Nubia Z50    | 8_Gen2-CPU              | Phi-2-2B-Orange-V2<br>q8f32 | 9.5 token/s       |
| Harmony 4  | P40          | Kirin_990_5G-CPU        | Phi-2-2B-Orange-V2<br>q8f32 | 5.8 token/s       |

### Llama
| OS         | Device       | Backend                 | Model                  | Inference (1024 Context) |
|:----------:|:------------:|:-----------------------:|:----------------------:|:------------------------:|
| Android 13 | Nubia Z50    | 8_Gen2-CPU              | Llama-3.2-1B-Instruct<br>q8f32 | 25 token/s     |
| Harmony 4  | P40          | Kirin_990_5G-CPU        | Llama-3.2-1B-Instruct<br>q8f32 | 16 token/s     |

### InternVL
| OS         | Device       | Backend                 | Model                  | Inference (1024 Context) |
|:----------:|:------------:|:-----------------------:|:----------------------:|:------------------------:|
| Harmony 4  | P40          | Kirin_990_5G-CPU        | Mono-2B-S1-3<br>q4f32<br>dynamic | 10.5 token/s     |

### MiniCPM
| OS         | Device       | Backend                 | Model                  | Inference (1024 Context) |
|:----------:|:------------:|:-----------------------:|:----------------------:|:------------------------:|
| Android 15 | Nubia Z50    | 8_Gen2-CPU              | MiniCPM4-0.5B<br>q4f32 | 78 token/s               |


## Demo Results
### Qwen2VL-2B / 1024 Context
![Demo Animation](https://github.com/DakeQQ/Native-LLM-for-Android/blob/main/LLM_QwenVL.gif?raw=true)

### Qwen2-1.5B / 1024 Context
![Demo Animation](https://github.com/DakeQQ/Native-LLM-for-Android/blob/main/LLM_Qwen.gif?raw=true)

## 概述

展示在 Android 设备上运行原生大型语言模型 (LLM) 的示范。目前支持的模型包括：

- **Qwen3**: 0.6B, 1.7B, 4B...
- **Qwen2.5-Instruct**: 0.5B, 1.5B, 3B...
- **Qwen2.5VL**: 3B
- **DeepSeek-R1-Distill-Qwen**: 1.5B
- **MiniCPM-DPO/SFT**: 1B, 2.7B
- **Gemma-3-it**: 1B, 4B...
- **Phi-4-mini-Instruct**: 3.8B
- **Llama-3.2-Instruct**: 1B
- **InternVL-Mono**: 2B
- **InternLM-3**: 8B
- **Seed-X**: [PRO-7B](https://modelscope.cn/models/ByteDance-Seed/Seed-X-PPO-7B), [Instruct-7B](https://modelscope.cn/models/ByteDance-Seed/Seed-X-Instruct-7B)
- **HunYuan**: [MT-7B](https://github.com/Tencent-Hunyuan/Hunyuan-MT)

## 最近更新
- 2025/09/07：更新 HunYuan-MT。
- 2025/08/02：更新 Seed-X。
- 2025/04/29：更新 Qwen3。
- 2025/04/05: 更新 Qwen2.5, InternVL-Mono `q4f32` + `dynamic_axes`。
- 2025/02/22：支持低内存模式加载: `Qwen`, `QwenVL`, `MiniCPM_2B_single`; Set `low_memory_mode = true` in `MainActivity.java`.
- 2025/02/07：**DeepSeek-R1-Distill-Qwen**: 1.5B （请使用 `Qwen v2.5 Qwen_Export.py`）。

## 入门指南

1. **下载模型：**
   - Quick Try: [Qwen3-1.7B-Android](https://huggingface.co/H5N1AIDS/Qwen_Android_ONNX_Runtime/tree/main)

2. **设置说明：**
   - 将下载的模型文件放入 `assets` 文件夹。
   - 解压存储在 `libs/arm64-v8a` 文件夹中的 `*.so` 文件。

3. **模型说明：**
   - 演示模型是从 HuggingFace 或 ModelScope 转换而来，并针对极限执行速度进行了优化。
   - 输入和输出可能与原始模型略有不同。
   - 对于Qwen2VL / Qwen2.5VL，请调整关键变量以匹配模型参数。
      - `GLRender.java: Line 37, 38, 39`
      - `project.h: Line 14, 15, 16, 35, 36, 41, 59, 60`

4. **ONNX 导出注意事项：**
   - 推荐使用动态轴以及`q4f32`量化。

## 分词器文件

- `tokenizer.cpp` 和 `tokenizer.hpp` 文件来源于 [mnn-llm 仓库](https://github.com/alibaba/MNN/tree/master/transformers/llm/engine/src)。

## 导出模型

1. 进入 `Export_ONNX` 文件夹。
2. 按照 Python 脚本中的注释设置文件夹路径。
3. 执行 `***_Export.py` 脚本以导出模型。
4. 手动量化或优化 ONNX 模型。

## 量化说明

- 使用 `onnxruntime.tools.convert_onnx_models_to_ort` 将模型转换为 `*.ort` 格式。注意该过程会自动添加 `Cast` 操作符，将 FP16 乘法改为 FP32。
- 量化方法详见 `Do_Quantize` 文件夹。

## 额外资源

- 探索更多项目：[DakeQQ Projects](https://github.com/DakeQQ?tab=repositories)
