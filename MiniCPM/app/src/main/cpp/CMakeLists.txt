project("myapplication")
set(ANDROID_ABI "arm64-v8a")
cmake_minimum_required(VERSION 3.22.1)
set(CMAKE_BUILD_TYPE Release)

include_directories(${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI})

add_compile_options(
        # Optimization Flags: Enable aggressive optimizations for performance
        -O3
        -ffast-math
        -ffp-model=fast
        -ffp-contract=fast
        -funsafe-math-optimizations
        -fno-math-errno
        -ffinite-math-only
        -fvectorize
        -funroll-loops
        -finline-functions
        -fomit-frame-pointer
        -fopenmp-simd
        -minline-all-stringops

        # CPU-Specific Optimization: Replace with your chip
        -march=armv8.7a+fp16fml+i8mm+dotprod+crypto+simd+tme
        -mtune=cortex-x3
#        -mtune=cortex-a76
#        -mcpu=cortex-a76+cortex-a55

        # Link-Time Optimization (LTO) & Code Size Reduction: Reduce binary size and improve performance
        -flto
        -ffunction-sections
        -fdata-sections
        -falign-functions

        # Exception Handling & Runtime Behavior: Disable RTTI, exceptions, and stack protection for faster execution
        -fno-sanitize=address
        -fno-sanitize=alignment
        -fno-exceptions
        -fno-rtti
        -fno-stack-protector
        -fPIC

        # Visibility and Symbol Management: Hide unnecessary symbols to reduce binary size
        -fvisibility=hidden
        -fvisibility-inlines-hidden

        # Miscellaneous / Compiler-Specific Tweaks
        -pipe

)

add_library(${CMAKE_PROJECT_NAME} SHARED project.cpp tokenizer.cpp)

add_library(
        omp
        SHARED
        IMPORTED)
set_target_properties(omp PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libomp.so)

add_library(
        onnxruntime
        SHARED
        IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libonnxruntime.so)

find_library(
        log
        android
)

target_link_libraries(${CMAKE_PROJECT_NAME}
        android
        omp
        onnxruntime
        log)
