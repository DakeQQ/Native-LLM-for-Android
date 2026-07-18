cmake_minimum_required(VERSION 3.22.1)
project("myapplication")

include_directories(${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI})
add_compile_definitions(ORT_NO_EXCEPTIONS)
set(NATIVE_OPTIMIZED_CONFIG "$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>")

add_compile_options(
        # The runtime is exception-free in every configuration; Gradle/NDK owns debug vs release.
        -fno-exceptions
        -fno-rtti
        -pipe

        # Production-only tuning. Debug retains symbols, stack protection and sanitizer compatibility.
        $<${NATIVE_OPTIMIZED_CONFIG}:-O3>
        $<${NATIVE_OPTIMIZED_CONFIG}:-ffast-math>
        $<${NATIVE_OPTIMIZED_CONFIG}:-ffp-model=fast>
        $<${NATIVE_OPTIMIZED_CONFIG}:-ffp-contract=fast>
        $<${NATIVE_OPTIMIZED_CONFIG}:-funsafe-math-optimizations>
        $<${NATIVE_OPTIMIZED_CONFIG}:-fno-math-errno>
        $<${NATIVE_OPTIMIZED_CONFIG}:-ffinite-math-only>
        $<${NATIVE_OPTIMIZED_CONFIG}:-fvectorize>
        $<${NATIVE_OPTIMIZED_CONFIG}:-funroll-loops>
        $<${NATIVE_OPTIMIZED_CONFIG}:-finline-functions>
        $<${NATIVE_OPTIMIZED_CONFIG}:-fomit-frame-pointer>
        $<${NATIVE_OPTIMIZED_CONFIG}:-fopenmp-simd>
        $<${NATIVE_OPTIMIZED_CONFIG}:-minline-all-stringops>
        $<${NATIVE_OPTIMIZED_CONFIG}:-march=armv8.7a+fp16fml+i8mm+dotprod+crypto+simd+tme>
        $<${NATIVE_OPTIMIZED_CONFIG}:-mtune=cortex-x925>
        $<${NATIVE_OPTIMIZED_CONFIG}:-flto=full>
        $<${NATIVE_OPTIMIZED_CONFIG}:-ffunction-sections>
        $<${NATIVE_OPTIMIZED_CONFIG}:-fdata-sections>
        $<${NATIVE_OPTIMIZED_CONFIG}:-falign-functions>
        $<${NATIVE_OPTIMIZED_CONFIG}:-fvisibility=hidden>
        $<${NATIVE_OPTIMIZED_CONFIG}:-fvisibility-inlines-hidden>
        $<${NATIVE_OPTIMIZED_CONFIG}:-fwhole-program-vtables>
        $<${NATIVE_OPTIMIZED_CONFIG}:-fforce-emit-vtables>
        $<${NATIVE_OPTIMIZED_CONFIG}:-fno-semantic-interposition>
        $<${NATIVE_OPTIMIZED_CONFIG}:-fmerge-all-constants>
        $<${NATIVE_OPTIMIZED_CONFIG}:-ffinite-loops>
)

# Link-time optimization and dead-code / section stripping.
add_link_options(
        $<${NATIVE_OPTIMIZED_CONFIG}:-flto=full>
        $<${NATIVE_OPTIMIZED_CONFIG}:-O3>
        $<${NATIVE_OPTIMIZED_CONFIG}:-Wl,--gc-sections>
        $<${NATIVE_OPTIMIZED_CONFIG}:-Wl,--icf=all>
        $<${NATIVE_OPTIMIZED_CONFIG}:-Wl,-O3>
)

add_library(${CMAKE_PROJECT_NAME} SHARED processLLM.cpp tokenizer.cpp unicode.cpp unicode_data.cpp glRender.cpp)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE QWEN_USE_SHARED_PREPACKED_WEIGHTS=1)

add_library(
        onnxruntime
        SHARED
        IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI}/libonnxruntime.so)

find_library(
        log
        android
)

# OpenGL ES 3.1 (compute shader + fenced SSBO readback) for the zero-copy camera path (glRender.cpp).
find_library(GLESv3-lib GLESv3)

target_link_libraries(${CMAKE_PROJECT_NAME}
        android
        onnxruntime
        ${GLESv3-lib}
        log)

# ── Standalone memory-bandwidth + heavy-compute benchmark ──
# Built as its OWN shared library, intentionally independent of the LLM pipeline and
# ONNX Runtime: it links only liblog (the c++_shared STL / pthreads are added by the
# toolchain automatically) and inherits the directory-wide optimization flags set via
# add_compile_options above. Loaded from Java via System.loadLibrary("membw").
add_library(membw SHARED memBandwidth.cpp)
target_compile_options(membw PRIVATE
        $<$<CONFIG:Debug>:-march=armv8.7a+fp16fml+i8mm+dotprod+crypto+simd+tme>
        $<$<CONFIG:Debug>:-mtune=cortex-x3>
)
target_link_libraries(membw log)
