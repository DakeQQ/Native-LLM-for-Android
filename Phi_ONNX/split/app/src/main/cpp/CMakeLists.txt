project("myapplication")
set(ANDROID_ABI "arm64-v8a")
cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_BUILD_TYPE Relese)

include_directories(${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI})
add_compile_options(
#                -fexceptions # Debug mode
        -fno-exceptions # Release mode
        -flto  # Release mode
        -fno-rtti  # Release mode
        -Ofast
        -ffast-math
        -ffp-model=fast
        -ffp-contract=fast
        -march=armv8.7a+fp16fml+i8mm+dotprod+crypto+simd+tme
#        -mtune=cortex-x3
        -mtune=cortex-a76  # Replace with your chip
        -mcpu=cortex-a76+cortex-a55 # Replace with your chip
        -fPIC
        -fopenmp
        -fomit-frame-pointer
        -fvectorize
        -fno-stack-protector
        -finline-functions
        -ffunction-sections
        -fdata-sections
        -fstrict-aliasing
        -fvisibility=hidden
        -fvisibility-inlines-hidden
        -funroll-loops
        -malign-double
)

add_library(${CMAKE_PROJECT_NAME} SHARED project.cpp tokenizer.cpp)

add_library(
        onnxruntime
        SHARED
        IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libonnxruntime.so)

add_library(
        omp
        SHARED
        IMPORTED)
set_target_properties(omp PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libomp.so)

find_library(
        log
        android
)

target_link_libraries(${CMAKE_PROJECT_NAME}
        android
        omp
        onnxruntime
        log)
