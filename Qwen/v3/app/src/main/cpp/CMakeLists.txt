cmake_minimum_required(VERSION 3.22.1)
project("myapplication")
set(ANDROID_ABI "arm64-v8a")
set(CMAKE_BUILD_TYPE Release)

include_directories(${CMAKE_SOURCE_DIR}/../../../libs/${ANDROID_ABI})
add_compile_definitions(ORT_NO_EXCEPTIONS)

add_compile_options(
        # Optimization Flags
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

        # CPU-Specific Optimization for high-end ARM64 cores
        -march=armv8.7a+fp16fml+i8mm+dotprod+crypto+simd+tme
        -mtune=cortex-x925

        # LTO & Code Size
        -flto=full                    # monolithic whole-program LTO (most aggressive form)
        -ffunction-sections           # 1 section/function -> linker --gc-sections drops unused
        -fdata-sections               # 1 section/datum    -> ditto
        -falign-functions

        # Disable RTTI/exceptions for performance
        -fno-sanitize=address
        -fno-sanitize=alignment
        -fno-exceptions
        -fno-rtti
        -fno-stack-protector
        -fPIC

        # Visibility and Symbol Management: Hide unnecessary symbols to reduce binary size
        -fvisibility=hidden
        -fvisibility-inlines-hidden

        # Aggressive interprocedural optimization (whole-library; needs the LTO below)
        -fwhole-program-vtables       # speculative devirtualization across all TUs
        -fforce-emit-vtables          # emit vtables so the devirtualizer can actually fire
        -fno-semantic-interposition   # symbols are not interposed -> free to inline/clone
        -fmerge-all-constants         # fold identical constants / string literals
        -ffinite-loops                # assume loops terminate -> stronger loop transforms

        -pipe
)

# Link-time optimization and dead-code / section stripping.
add_link_options(
        -flto=full          # run the whole-program LTO backend at link time
        -O3                 # LTO codegen optimization level (otherwise defaults to -O2)
        -Wl,--gc-sections   # garbage-collect unreferenced functions/data
        -Wl,--icf=all       # identical-code folding: merge duplicate function bodies (lld)
        -Wl,-O3             # lld aggressive string/section merging
)

add_library(${CMAKE_PROJECT_NAME} SHARED processLLM.cpp tokenizer.cpp unicode.cpp unicode_data.cpp)

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

# ── Standalone memory-bandwidth + heavy-compute benchmark ──
# Built as its OWN shared library, intentionally independent of the LLM pipeline and
# ONNX Runtime: it links only liblog (the c++_shared STL / pthreads are added by the
# toolchain automatically) and inherits the directory-wide optimization flags set via
# add_compile_options above. Loaded from Java via System.loadLibrary("membw").
add_library(membw SHARED memBandwidth.cpp)
target_link_libraries(membw log)
