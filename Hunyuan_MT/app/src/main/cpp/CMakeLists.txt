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

add_library(${CMAKE_PROJECT_NAME} SHARED processLLM.cpp tokenizer.cpp unicode.cpp unicode_data.cpp)

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
        onnxruntime
        log)
