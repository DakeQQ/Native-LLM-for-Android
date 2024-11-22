# Three Steps to Launch the Qualcomm NPU-HTP

## Overview
This guide will walk you through the steps to set up the Qualcomm NPU-HTP on your device. The process involves obtaining necessary libraries from your device and the official QNN SDK, and placing them in the appropriate directories.

---

## Step 1: Get Required Libraries from Your Device
To run the Qualcomm NPU-HTP, connect your mobile device and execute the following commands in the Terminal to retrieve the required libraries.

**Note:** These libraries are verified for Snapdragon 8Gen1, 8Gen2, 8Gen3; other devices may need different libraries.

### Execute the Commands in Terminal:
```bash
cd ~/Downloads/YOLO_Depth_Drivable_Qualcomm_NPU

adb pull /vendor/lib64/libcdsprpc.so ./app/libs/arm64-v8a
adb pull /vendor/lib64/vendor.qti.hardware.dsp@1.0.so ./app/libs/arm64-v8a
adb pull /vendor/lib64/libvmmem.so ./app/libs/arm64-v8a
adb pull /system/lib64/libhidlbase.so ./app/libs/arm64-v8a
adb pull /system/lib64/libhardware.so ./app/libs/arm64-v8a
adb pull /system/lib64/libutils.so ./app/libs/arm64-v8a
adb pull /system/lib64/libcutils.so ./app/libs/arm64-v8a
adb pull /system/lib64/libdmabufheap.so ./app/libs/arm64-v8a
adb pull /system/lib64/libc++.so ./app/libs/arm64-v8a
adb pull /system/lib64/libbase.so ./app/libs/arm64-v8a
adb pull /system/lib64/libvndksupport.so ./app/libs/arm64-v8a
adb pull /system/lib64/libdl_android.so ./app/libs/arm64-v8a
adb pull /system/lib64/ld-android.so ./app/libs/arm64-v8a

adb pull /vendor/lib64/libcdsprpc.so ./app/src/main/assets
adb pull /vendor/lib64/vendor.qti.hardware.dsp@1.0.so ./app/src/main/assets
adb pull /vendor/lib64/libvmmem.so ./app/src/main/assets
adb pull /system/lib64/libhidlbase.so ./app/src/main/assets
adb pull /system/lib64/libhardware.so ./app/src/main/assets
adb pull /system/lib64/libutils.so ./app/src/main/assets
adb pull /system/lib64/libcutils.so ./app/src/main/assets
adb pull /system/lib64/libdmabufheap.so ./app/src/main/assets
adb pull /system/lib64/libc++.so ./app/src/main/assets
adb pull /system/lib64/libbase.so ./app/src/main/assets
adb pull /system/lib64/libvndksupport.so ./app/src/main/assets
adb pull /system/lib64/libdl_android.so ./app/src/main/assets
adb pull /system/lib64/ld-android.so ./app/src/main/assets
```

---

## Step 2: Get Required Libraries from the Official QNN SDK

You need additional libraries from the official Qualcomm AI Engine Direct SDK (QNN SDK, demo version: 2.28.*). Download the SDK from the official website, and locate the libraries in the `2.28.*/lib/aarch64-android` directory. Ensure the SDK version matches the one used for compiling the `libonnxruntime.so` library.

**Note:** To obtain the latest SDK version, you must use the 'Qualcomm® Package Manager' (QPM3); otherwise, you will receive another version via the direct download link.<br> 
**Note:** Find your device from here: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/overview.html#supported-snapdragon-devices

### 8Gen1 Required Libraries:
1. `libQnnCpu.so`
2. `libQnnHtp.so`
3. `libQnnHtpPrepare.so`
4. `libQnnHtpV69Skel.so`
5. `libQnnHtpV69Stub.so`
6. `libQnnSystem.so`

### 8Gen2 Required Libraries:
1. `libQnnCpu.so`
2. `libQnnHtp.so`
3. `libQnnHtpPrepare.so`
4. `libQnnHtpV73Skel.so`
5. `libQnnHtpV73Stub.so`
6. `libQnnSystem.so`

### 8Gen3 Required Libraries:
1. `libQnnCpu.so`
2. `libQnnHtp.so`
3. `libQnnHtpPrepare.so`
4. `libQnnHtpV75Skel.so`
5. `libQnnHtpV75Stub.so`
6. `libQnnSystem.so`


---

## Step 3: Organize Libraries

Use 8Gen2 as example. Place all the required libraries in both the `assets` and `libs/arm64-v8a` folders.

### List of Libraries:
1. `libQnnCpu.so`
2. `libQnnHtp.so`
3. `libQnnHtpPrepare.so`
4. `libQnnHtpV73Skel.so`
5. `libQnnHtpV73Stub.so`
6. `libQnnSystem.so`
7. `libcdsprpc.so`
8. `vendor.qti.hardware.dsp@1.0.so`
9. `libvmmem.so`
10. `libhidlbase.so`
11. `libhardware.so`
12. `libutils.so`
13. `libcutils.so`
14. `libdmabufheap.so`
15. `libc++.so`
16. `libbase.so`
17. `libdl_android.so`
18. `ld-android.so`
19. `libonnxruntime.so` (libs/arm64-v8a folder only)
20. `libomp.so` (libs/arm64-v8a folder only)

---

Following these steps will ensure the necessary setup for running the Qualcomm NPU-HTP on a Snapdragon 8Gen2 device.

# 启动 Qualcomm NPU-HTP 的三个步骤

## 概述
本指南将引导您完成在设备上设置 Qualcomm NPU-HTP 的步骤。该过程涉及从您的设备和官方 QNN SDK 获取必要的库，并将它们放置在适当的目录中。

---

## 步骤 1：从您的设备获取所需库
要运行 Qualcomm NPU-HTP，请连接您的移动设备并在终端中执行以下命令以获取所需的库。

**注意：** 这些库仅针对 Snapdragon 8Gen1, 8Gen2, 8Gen3 确认；其他设备可能需要不同的库。

### 在终端中执行以下命令：
```bash
cd ~/Downloads/YOLO_Depth_Drivable_Qualcomm_NPU

adb pull /vendor/lib64/libcdsprpc.so ./app/libs/arm64-v8a
adb pull /vendor/lib64/vendor.qti.hardware.dsp@1.0.so ./app/libs/arm64-v8a
adb pull /vendor/lib64/libvmmem.so ./app/libs/arm64-v8a
adb pull /system/lib64/libhidlbase.so ./app/libs/arm64-v8a
adb pull /system/lib64/libhardware.so ./app/libs/arm64-v8a
adb pull /system/lib64/libutils.so ./app/libs/arm64-v8a
adb pull /system/lib64/libcutils.so ./app/libs/arm64-v8a
adb pull /system/lib64/libdmabufheap.so ./app/libs/arm64-v8a
adb pull /system/lib64/libc++.so ./app/libs/arm64-v8a
adb pull /system/lib64/libbase.so ./app/libs/arm64-v8a
adb pull /system/lib64/libvndksupport.so ./app/libs/arm64-v8a
adb pull /system/lib64/libdl_android.so ./app/libs/arm64-v8a
adb pull /system/lib64/ld-android.so ./app/libs/arm64-v8a

adb pull /vendor/lib64/libcdsprpc.so ./app/src/main/assets
adb pull /vendor/lib64/vendor.qti.hardware.dsp@1.0.so ./app/src/main/assets
adb pull /vendor/lib64/libvmmem.so ./app/src/main/assets
adb pull /system/lib64/libhidlbase.so ./app/src/main/assets
adb pull /system/lib64/libhardware.so ./app/src/main/assets
adb pull /system/lib64/libutils.so ./app/src/main/assets
adb pull /system/lib64/libcutils.so ./app/src/main/assets
adb pull /system/lib64/libdmabufheap.so ./app/src/main/assets
adb pull /system/lib64/libc++.so ./app/src/main/assets
adb pull /system/lib64/libbase.so ./app/src/main/assets
adb pull /system/lib64/libvndksupport.so ./app/src/main/assets
adb pull /system/lib64/libdl_android.so ./app/src/main/assets
adb pull /system/lib64/ld-android.so ./app/src/main/assets
```

---

## 步骤 2：从官方 QNN SDK 获取所需库

您需要从官方 Qualcomm AI Engine Direct SDK（QNN SDK，此演示的版本：2.24.*）获取额外的库。从官网下载 SDK，并在 `2.24.*/lib/aarch64-android` 目录中找到这些库。确保SDK版本与编译 `libonnxruntime.so` 库时使用的版本一致。

**注意：** 要获取最新的 SDK 版本，您必须使用“Qualcomm® Package Manager”（QPM3）；否则，您将通过直接下载链接收到其他版本。<br> 
**注意：** 可以在这查到型号: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-50/overview.html#supported-snapdragon-devices

### 8Gen1 Required Libraries:
1. `libQnnCpu.so`
2. `libQnnHtp.so`
3. `libQnnHtpPrepare.so`
4. `libQnnHtpV69Skel.so`
5. `libQnnHtpV69Stub.so`
6. `libQnnSystem.so`

### 8Gen2 Required Libraries:
1. `libQnnCpu.so`
2. `libQnnHtp.so`
3. `libQnnHtpPrepare.so`
4. `libQnnHtpV73Skel.so`
5. `libQnnHtpV73Stub.so`
6. `libQnnSystem.so`

### 8Gen3 Required Libraries:
1. `libQnnCpu.so`
2. `libQnnHtp.so`
3. `libQnnHtpPrepare.so`
4. `libQnnHtpV75Skel.so`
5. `libQnnHtpV75Stub.so`
6. `libQnnSystem.so`
---

## 步骤 3：组织库文件

使用8Gen2為例子，将所有需要的库文件放置在 `assets` 和 `libs/arm64-v8a` 文件夹中。

### 库文件列表：
1. `libQnnCpu.so`
2. `libQnnHtp.so`
3. `libQnnHtpPrepare.so`
4. `libQnnHtpV73Skel.so`
5. `libQnnHtpV73Stub.so`
6. `libQnnSystem.so`
7. `libcdsprpc.so`
8. `vendor.qti.hardware.dsp@1.0.so`
9. `libvmmem.so`
10. `libhidlbase.so`
11. `libhardware.so`
12. `libutils.so`
13. `libcutils.so`
14. `libdmabufheap.so`
15. `libc++.so`
16. `libbase.so`
17. `libdl_android.so`
18. `ld-android.so`
19. `libonnxruntime.so` （只需放在libs/arm64-v8a文件夹裡）
20. `libomp.so` （只需放在libs/arm64-v8a文件夹裡）

---

按照这些步骤操作，将确保在 Snapdragon 8Gen2 设备上运行 Qualcomm NPU-HTP 的必要设置。

