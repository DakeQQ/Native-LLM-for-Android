// ============================================================================
// Standalone memory-bandwidth + compute benchmark. INDEPENDENT of the LLM pipeline: its own libmembw.so
// (CMakeLists.txt) links nothing from ONNX Runtime/tokenizer/llm_runtime_state.h — only the C++ runtime, std::thread,
// liblog. Runs a read-only weight-streaming pass, STREAM DRAM tests (Copy/Scale/Add/Triad), and an FMA
// compute kernel. Arrays are far larger than cache so figures reflect main memory; the read-only pass is the
// proxy for LLM decode weight streaming. A persistent worker team + warmup + multi-pass samples (batch loop
// INSIDE the parallel region so team broadcast/join is amortized) + median/best/avg expose scheduler jitter.
// The median read row drives the decode-speed estimate. A printed checksum stops -O3/-ffast-math/-flto eliding.
// ============================================================================

#include <jni.h>
#include <android/log.h>
#include <algorithm>
#include <cerrno>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sched.h>
#include <string>
#include <unistd.h>
#include <arm_neon.h>

#define MEMBW_LOG_TAG "MemBandwidth"
#define MEMBW_LOGI(...) __android_log_print(ANDROID_LOG_INFO, MEMBW_LOG_TAG, __VA_ARGS__)
#define MEMBW_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, MEMBW_LOG_TAG, __VA_ARGS__)

// Multiply-adds per element in the compute kernel (2 FLOPs each). The per-element chain is independent
// across elements so the loop vectorizes across i while staying heavy.
static constexpr int MEMBW_COMPUTE_FMA = 256;
static constexpr int MEMBW_COMPUTE_INT8_DOT = 512;
static constexpr size_t MEMBW_ALIGNMENT = 64;
static constexpr int MEMBW_WARMUP_PASSES = 2;
static constexpr size_t MEMBW_BANDWIDTH_BATCH_TARGET_ELEMENTS = 16000000;

static int gMembwCpuGroupMode = 0;  // 0 = Total, 1 = Little, 2 = Big

struct MembwSamples {
    float totalSeconds = 0.0f;
    float bestSeconds = 0.0f;
    float medianSeconds = 0.0f;
};

struct MembwCpuGroup {
    const char* label = "";
    std::vector<int> cpus;
};

static int membwOnlineCpuCount() {
    const long online = sysconf(_SC_NPROCESSORS_ONLN);
    if (online > 0 && online < 1024) return static_cast<int>(online);
    int hw = static_cast<int>(std::thread::hardware_concurrency());
    return hw > 0 ? hw : 4;
}

static long membwCpuMaxFreqKHz(int cpu) {
    char path[128];
    std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", cpu);
    FILE* f = std::fopen(path, "r");
    if (!f) return 0;
    long freq = 0;
    if (std::fscanf(f, "%ld", &freq) != 1) freq = 0;
    std::fclose(f);
    return freq > 0 ? freq : 0;
}

static std::vector<int> membwAllCpuIds(int hw) {
    std::vector<int> cpus;
    cpus.reserve(static_cast<size_t>(std::max(1, hw)));
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
        for (int cpu = 0; cpu < hw; ++cpu) {
            if (CPU_ISSET(cpu, &allowed)) cpus.push_back(cpu);
        }
        if (!cpus.empty()) return cpus;
    }
    for (int cpu = 0; cpu < hw; ++cpu) cpus.push_back(cpu);
    if (cpus.empty()) cpus.push_back(0);
    return cpus;
}

static std::string membwCpuListString(const std::vector<int>& cpus) {
    std::string out;
    for (size_t i = 0; i < cpus.size(); ++i) {
        if (i > 0) out += ",";
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%d", cpus[i]);
        out += buf;
    }
    return out.empty() ? "-" : out;
}

static void membwBuildCpuGroups(int hw, MembwCpuGroup* little, MembwCpuGroup* big, MembwCpuGroup* total) {
    total->label = "Total";
    total->cpus = membwAllCpuIds(hw);
    little->label = "Little";
    big->label = "Big";

    std::vector<long> freqs(static_cast<size_t>(hw), 0);
    int validFreqs = 0;
    long minFreq = 0;
    long maxFreq = 0;
    for (int cpu : total->cpus) {
        const long freq = membwCpuMaxFreqKHz(cpu);
        freqs[static_cast<size_t>(cpu)] = freq;
        if (freq <= 0) continue;
        if (validFreqs == 0 || freq < minFreq) minFreq = freq;
        if (validFreqs == 0 || freq > maxFreq) maxFreq = freq;
        ++validFreqs;
    }

    if (validFreqs == static_cast<int>(total->cpus.size()) && maxFreq > minFreq) {
        for (int cpu : total->cpus) {
            if (freqs[static_cast<size_t>(cpu)] == minFreq) {
                little->cpus.push_back(cpu);
            } else {
                big->cpus.push_back(cpu);
            }
        }
    } else {
        const size_t split = std::max<size_t>(1, total->cpus.size() / 2);
        for (size_t i = 0; i < total->cpus.size(); ++i) {
            (i < split ? little->cpus : big->cpus).push_back(total->cpus[i]);
        }
    }

    if (little->cpus.empty()) little->cpus = total->cpus;
    if (big->cpus.empty()) big->cpus = total->cpus;
}

static bool membwBindCurrentThreadToCpus(const std::vector<int>& cpus) {
    if (cpus.empty()) return true;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int cpu : cpus) {
        if (cpu >= 0 && cpu < CPU_SETSIZE) CPU_SET(cpu, &set);
    }
    if (sched_setaffinity(0, sizeof(set), &set) == 0) return true;
    MEMBW_LOGE("MemBw: sched_setaffinity failed for cpu[%s]: %s",
               membwCpuListString(cpus).c_str(), std::strerror(errno));
    return false;
}

static float* membwAllocFloats(size_t count) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, MEMBW_ALIGNMENT, count * sizeof(float)) != 0) return nullptr;
    return static_cast<float*>(ptr);
}

static float16_t* membwAllocHalfs(size_t count) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, MEMBW_ALIGNMENT, count * sizeof(float16_t)) != 0) return nullptr;
    return static_cast<float16_t*>(ptr);
}

static int8_t* membwAllocInt8(size_t count) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, MEMBW_ALIGNMENT, count * sizeof(int8_t)) != 0) return nullptr;
    return static_cast<int8_t*>(ptr);
}

static MembwSamples membwSummarize(std::vector<float> samples) {
    MembwSamples out;
    if (samples.empty()) return out;
    for (float sample : samples) out.totalSeconds += sample;
    std::sort(samples.begin(), samples.end());
    out.bestSeconds = samples.front();
    out.medianSeconds = samples[samples.size() / 2];
    return out;
}

static float membwGBs(float bytes, float seconds) {
    return seconds > 0.0f ? bytes / seconds / 1.0e9f : 0.0f;
}

static float membwGFLOPs(float flops, float seconds) {
    return seconds > 0.0f ? flops / seconds / 1.0e9f : 0.0f;
}

class MembwThreadTeam {
public:
    explicit MembwThreadTeam(int threads, const std::vector<int>& cpus)
            : threads_(std::max(1, threads)), cpus_(cpus) {
        previousAffinityValid_ = sched_getaffinity(0, sizeof(previousAffinity_), &previousAffinity_) == 0;
        membwBindCurrentThreadToCpus(cpus_);
        workers_.reserve(static_cast<size_t>(threads_ - 1));
        for (int worker = 1; worker < threads_; ++worker) {
            workers_.emplace_back(&MembwThreadTeam::workerLoop, this, worker);
        }
    }

    ~MembwThreadTeam() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
            ++generation_;
        }
        workReady_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) worker.join();
        }
        if (previousAffinityValid_) {
            sched_setaffinity(0, sizeof(previousAffinity_), &previousAffinity_);
        }
    }

    template <typename Fn>
    void parallelFor(size_t n, Fn fn) {
        if (threads_ <= 1 || n == 0) {
            fn(static_cast<size_t>(0), n);
            return;
        }

        auto runner = [](void* context, size_t start, size_t end) {
            (*static_cast<Fn*>(context))(start, end);
        };

        {
            std::lock_guard<std::mutex> lock(mutex_);
            n_ = n;
            taskContext_ = &fn;
            taskRunner_ = runner;
            completedWorkers_ = 0;
            ++generation_;
        }
        workReady_.notify_all();

        runRange(0, n, runner, &fn);

        std::unique_lock<std::mutex> lock(mutex_);
        done_.wait(lock, [this] { return completedWorkers_ >= workers_.size(); });
        taskContext_ = nullptr;
        taskRunner_ = nullptr;
    }

private:
    using TaskRunner = void (*)(void*, size_t, size_t);

    void workerLoop(int workerIndex) {
        membwBindCurrentThreadToCpus(cpus_);
        size_t seenGeneration = 0;
        for (;;) {
            TaskRunner runner = nullptr;
            void* context = nullptr;
            size_t n = 0;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                workReady_.wait(lock, [this, seenGeneration] {
                    return stop_ || generation_ != seenGeneration;
                });
                if (stop_) return;
                seenGeneration = generation_;
                runner = taskRunner_;
                context = taskContext_;
                n = n_;
            }

            runRange(workerIndex, n, runner, context);

            {
                std::lock_guard<std::mutex> lock(mutex_);
                ++completedWorkers_;
                if (completedWorkers_ >= workers_.size()) done_.notify_one();
            }
        }
    }

    void runRange(int workerIndex, size_t n, TaskRunner runner, void* context) const {
        const size_t chunk = (n + static_cast<size_t>(threads_) - 1) / static_cast<size_t>(threads_);
        const size_t start = static_cast<size_t>(workerIndex) * chunk;
        if (start >= n || runner == nullptr) return;
        const size_t end = std::min(n, start + chunk);
        runner(context, start, end);
    }

    const int threads_;
    const std::vector<int> cpus_;
    cpu_set_t previousAffinity_;
    bool previousAffinityValid_ = false;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable workReady_;
    std::condition_variable done_;
    bool stop_ = false;
    size_t generation_ = 0;
    size_t n_ = 0;
    size_t completedWorkers_ = 0;
    TaskRunner taskRunner_ = nullptr;
    void* taskContext_ = nullptr;
};

extern "C"
JNIEXPORT void JNICALL
Java_com_example_myapplication_MemBandwidth_setCpuGroupMode(JNIEnv* env, jclass clazz, jint mode) {
    (void)env;
    (void)clazz;
    gMembwCpuGroupMode = (mode >= 0 && mode <= 2) ? static_cast<int>(mode) : 0;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_myapplication_MemBandwidth_runMemoryBandwidthTestForCurrentGroup(JNIEnv* env, jclass clazz,
                                                                                  jint jArraySizeMillions,
                                                                                  jint jIterations,
                                                                                  jint jThreadCount) {
    (void)clazz;
    // Clamp every input to a sane range so a bad UI value cannot OOM the device or
    // spin forever. Arrays are sized in millions of float elements (3 arrays total).
    long elems = static_cast<long>(jArraySizeMillions) * 1000000L;
    if (elems < 1000000L)       elems = 1000000L;          // >= 1M elements
    if (elems > 96000000L)      elems = 96000000L;         // <= 96M elements (~1.1 GB)
    int iterations = jIterations;
    if (iterations < 1)   iterations = 1;
    if (iterations > 500) iterations = 500;

    MembwCpuGroup littleGroup;
    MembwCpuGroup bigGroup;
    MembwCpuGroup totalGroup;
    const int hw = membwOnlineCpuCount();
    membwBuildCpuGroups(hw, &littleGroup, &bigGroup, &totalGroup);
    const MembwCpuGroup* activeGroup = &totalGroup;
    if (gMembwCpuGroupMode == 1) {
        activeGroup = &littleGroup;
    } else if (gMembwCpuGroupMode == 2) {
        activeGroup = &bigGroup;
    }
    const int availableCpus = std::max(1, static_cast<int>(activeGroup->cpus.size()));
    int threads = jThreadCount <= 0 ? availableCpus : jThreadCount;
    if (threads < 1)  threads = 1;
    if (threads > availableCpus) threads = availableCpus;
    if (threads > 64) threads = 64;

    const size_t N = static_cast<size_t>(elems);
    const size_t bytes = N * sizeof(float);

    // Exceptions are disabled in this build, so allocate explicitly and null-check.
    float* __restrict a = membwAllocFloats(N);
    float* __restrict b = membwAllocFloats(N);
    float* __restrict c = membwAllocFloats(N);
    if (!a || !b || !c) {
        free(a); free(b); free(c);
        MEMBW_LOGE("MemBw: failed to allocate %zu bytes x3 (%.0f MB)",
                   bytes, 3.0 * bytes / 1048576.0);
        return env->NewStringUTF("Memory bandwidth test\nERROR: out of memory");
    }

    // STREAM multiplier. The kernels chain destructively (Triad feeds `a` back), so each pass scales arrays
    // by g = scalar*(2+scalar) and the compute recurrence grows like scalar^MEMBW_COMPUTE_FMA. The classic
    // 3.0 overflows f32 (3^256 ≈ 1e122) -> +inf checksum; scalar < sqrt(2)-1 (~0.414) keeps g <= 1 and the
    // recurrence finite. Numerical hygiene only (byte/FLOP counts + GB/s unchanged); also needed since
    // -ffast-math implies -ffinite-math-only.
    const float scalar = 0.4f;

    const auto callStart = std::chrono::steady_clock::now();
    MembwThreadTeam team(threads, activeGroup->cpus);

    // First-touch initialization on the same threads that will read the data.
    team.parallelFor(N, [=](size_t s, size_t e) {
        for (size_t i = s; i < e; ++i) { a[i] = 1.0f; b[i] = 2.0f; c[i] = 0.0f; }
    });

    using clock = std::chrono::steady_clock;
    auto secs = [](clock::duration d) {
        return std::chrono::duration<float>(d).count();
    };

    auto timeKernel = [&](auto&& kernel) {
        const auto t0 = clock::now();
        kernel();
        return secs(clock::now() - t0);
    };

    // Each sample replays the kernel bandwidthPassesPerSample times, with the replay loop INSIDE the parallel
    // region so the team broadcast/join is paid once per sample, not per pass — keeps small-array timings
    // dominated by memory streaming, not thread-wakeup latency.
    const int bandwidthPassesPerSample = std::max(
        1,
        static_cast<int>((MEMBW_BANDWIDTH_BATCH_TARGET_ELEMENTS + N - 1) / N));

    std::vector<float> readPartials(static_cast<size_t>(threads), 0.0f);
    auto runRead = [&] {
        const size_t chunk = (N + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads);
        team.parallelFor(N, [=, &readPartials](size_t s, size_t e) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
            float acc4 = 0.0f;
            float acc5 = 0.0f;
            float acc6 = 0.0f;
            float acc7 = 0.0f;
            for (int pass = 0; pass < bandwidthPassesPerSample; ++pass) {
                size_t i = s;
                for (; i + 7 < e; i += 8) {
                    acc0 += a[i];
                    acc1 += a[i + 1];
                    acc2 += a[i + 2];
                    acc3 += a[i + 3];
                    acc4 += a[i + 4];
                    acc5 += a[i + 5];
                    acc6 += a[i + 6];
                    acc7 += a[i + 7];
                }
                for (; i < e; ++i) acc0 += a[i];
            }
            const float acc = acc0 + acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
            const size_t slot = chunk > 0 ? std::min(s / chunk, readPartials.size() - 1) : 0;
            readPartials[slot] = acc;
        });
    };

    auto runCopy = [&] {
        team.parallelFor(N, [=](size_t s, size_t e) {
            for (int pass = 0; pass < bandwidthPassesPerSample; ++pass)
                for (size_t i = s; i < e; ++i) c[i] = a[i];
        });
    };
    auto runScale = [&] {
        team.parallelFor(N, [=](size_t s, size_t e) {
            for (int pass = 0; pass < bandwidthPassesPerSample; ++pass)
                for (size_t i = s; i < e; ++i) b[i] = scalar * c[i];
        });
    };
    auto runAdd = [&] {
        team.parallelFor(N, [=](size_t s, size_t e) {
            for (int pass = 0; pass < bandwidthPassesPerSample; ++pass)
                for (size_t i = s; i < e; ++i) c[i] = a[i] + b[i];
        });
    };
    auto runTriad = [&] {
        team.parallelFor(N, [=](size_t s, size_t e) {
            for (int pass = 0; pass < bandwidthPassesPerSample; ++pass)
                for (size_t i = s; i < e; ++i) a[i] = b[i] + scalar * c[i];
        });
    };
    auto runCompute = [&] {
        const float32x4_t scalar32 = vdupq_n_f32(scalar);
        team.parallelFor(N, [=](size_t s, size_t e) {
            size_t i = s;
            for (; i + 3 < e; i += 4) {
                float32x4_t x = vld1q_f32(c + i);
                const float32x4_t y = vld1q_f32(a + i);
                for (int k = 0; k < MEMBW_COMPUTE_FMA; ++k) x = vfmaq_f32(y, x, scalar32);
                vst1q_f32(b + i, x);
            }
            for (; i < e; ++i) {
                float x = c[i];
                const float y = a[i];
                for (int k = 0; k < MEMBW_COMPUTE_FMA; ++k) x = x * scalar + y;
                b[i] = x;
            }
        });
    };

    for (int warm = 0; warm < MEMBW_WARMUP_PASSES; ++warm) {
        runRead();
        runCopy();
        runScale();
        runAdd();
        runTriad();
    }
    for (int warm = 0; warm < MEMBW_WARMUP_PASSES; ++warm) runCompute();

    std::vector<float> readSamples;
    std::vector<float> copySamples;
    std::vector<float> scaleSamples;
    std::vector<float> addSamples;
    std::vector<float> triadSamples;
    std::vector<float> computeSamples;
    readSamples.reserve(static_cast<size_t>(iterations));
    copySamples.reserve(static_cast<size_t>(iterations));
    scaleSamples.reserve(static_cast<size_t>(iterations));
    addSamples.reserve(static_cast<size_t>(iterations));
    triadSamples.reserve(static_cast<size_t>(iterations));
    computeSamples.reserve(static_cast<size_t>(iterations));

    const auto measureStart = clock::now();

    for (int it = 0; it < iterations; ++it) {
        readSamples.push_back(timeKernel(runRead));
        copySamples.push_back(timeKernel(runCopy));
        scaleSamples.push_back(timeKernel(runScale));
        addSamples.push_back(timeKernel(runAdd));
        triadSamples.push_back(timeKernel(runTriad));
    }

    // F32 compute-bound kernel: explicit NEON FMA over 4 lanes. Memory traffic is one
    // read + one write per element, so throughput is dominated by arithmetic.
    for (int it = 0; it < iterations; ++it) {
        computeSamples.push_back(timeKernel(runCompute));
    }

    // Full reduction keeps every store live (defeats dead-code elimination) and gives
    // a stable checksum to print.
    float checksum = 0.0f;
    for (size_t i = 0; i < N; ++i) checksum += a[i] + b[i] + c[i];
    for (float partial : readPartials) checksum += partial;

    free(a); free(b); free(c);

    float16_t* __restrict hA = membwAllocHalfs(N);
    float16_t* __restrict hB = membwAllocHalfs(N);
    float16_t* __restrict hC = membwAllocHalfs(N);
    std::vector<float> computeF16Samples;
    computeF16Samples.reserve(static_cast<size_t>(iterations));
    if (hA && hB && hC) {
        team.parallelFor(N, [=](size_t s, size_t e) {
            for (size_t i = s; i < e; ++i) {
                hA[i] = static_cast<float16_t>(0.25f);
                hB[i] = static_cast<float16_t>(0.5f);
                hC[i] = static_cast<float16_t>(0.75f);
            }
        });

        auto runComputeF16 = [&] {
            const float16x8_t scalar16 = vdupq_n_f16(static_cast<float16_t>(0.4f));
            team.parallelFor(N, [=](size_t s, size_t e) {
                size_t i = s;
                for (; i + 7 < e; i += 8) {
                    float16x8_t x = vld1q_f16(hC + i);
                    const float16x8_t y = vld1q_f16(hA + i);
                    for (int k = 0; k < MEMBW_COMPUTE_FMA; ++k) {
                        x = vfmaq_f16(y, x, scalar16);
                    }
                    vst1q_f16(hB + i, x);
                }
                for (; i < e; ++i) {
                    float16_t x = hC[i];
                    const float16_t y = hA[i];
                    for (int k = 0; k < MEMBW_COMPUTE_FMA; ++k) x = static_cast<float16_t>(x * static_cast<float16_t>(0.4f) + y);
                    hB[i] = x;
                }
            });
        };

        for (int warm = 0; warm < MEMBW_WARMUP_PASSES; ++warm) runComputeF16();
        for (int it = 0; it < iterations; ++it) computeF16Samples.push_back(timeKernel(runComputeF16));
        for (size_t i = 0; i < N; ++i) checksum += static_cast<float>(hA[i]) + static_cast<float>(hB[i]) + static_cast<float>(hC[i]);
    } else {
        MEMBW_LOGE("MemBw: failed to allocate F16 compute buffers (%zu elements x3)", N);
    }
    free(hA); free(hB); free(hC);

    int8_t* __restrict iA = membwAllocInt8(N);
    int8_t* __restrict iB = membwAllocInt8(N);
    std::vector<float> computeInt8Samples;
    std::vector<int32_t> int8Partials(static_cast<size_t>(threads), 0);
    computeInt8Samples.reserve(static_cast<size_t>(iterations));
    if (iA && iB) {
        team.parallelFor(N, [=](size_t s, size_t e) {
            for (size_t i = s; i < e; ++i) {
                iA[i] = static_cast<int8_t>((static_cast<int>(i) & 31) - 16);
                iB[i] = static_cast<int8_t>(16 - (static_cast<int>(i) & 31));
            }
        });

        auto runComputeInt8 = [&] {
            const size_t chunk = (N + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads);
            team.parallelFor(N, [=, &int8Partials](size_t s, size_t e) {
                int32x4_t acc = vdupq_n_s32(0);
                size_t i = s;
                for (; i + 15 < e; i += 16) {
                    const int8x16_t x = vld1q_s8(iA + i);
                    const int8x16_t y = vld1q_s8(iB + i);
                    for (int k = 0; k < MEMBW_COMPUTE_INT8_DOT; ++k) {
#if defined(__ARM_FEATURE_MATMUL_INT8)
                        acc = vmmlaq_s32(acc, x, y);
#else
                        acc = vdotq_s32(acc, x, y);
#endif
                    }
                }
                int32_t tail = 0;
                for (; i < e; ++i) tail += static_cast<int32_t>(iA[i]) * static_cast<int32_t>(iB[i]);
                const int32_t lanes = vaddvq_s32(acc);
                const size_t slot = chunk > 0 ? std::min(s / chunk, int8Partials.size() - 1) : 0;
                int8Partials[slot] = lanes + tail;
            });
        };

        for (int warm = 0; warm < MEMBW_WARMUP_PASSES; ++warm) runComputeInt8();
        for (int it = 0; it < iterations; ++it) computeInt8Samples.push_back(timeKernel(runComputeInt8));
        for (int32_t partial : int8Partials) checksum += static_cast<float>(partial) * 1.0e-9f;
    } else {
        MEMBW_LOGE("MemBw: failed to allocate INT8 compute buffers (%zu elements x2)", N);
    }
    free(iA); free(iB);

    const float tMeasureWall = secs(clock::now() - measureStart);

    // Per-kernel bandwidth reported as avg / median / best. Read moves 1 array; Copy/Scale 2 (1R+1W);
    // Add/Triad 3 (2R+1W).
    const float bwPasses = static_cast<float>(bandwidthPassesPerSample);
    const float bytesF = static_cast<float>(bytes);
    const float oneArr   = static_cast<float>(iterations) * bwPasses * bytesF;
    const float twoArr   = static_cast<float>(iterations) * bwPasses * 2.0f * bytesF;
    const float threeArr = static_cast<float>(iterations) * bwPasses * 3.0f * bytesF;
    const MembwSamples read = membwSummarize(readSamples);
    const MembwSamples copy = membwSummarize(copySamples);
    const MembwSamples scale = membwSummarize(scaleSamples);
    const MembwSamples add = membwSummarize(addSamples);
    const MembwSamples triad = membwSummarize(triadSamples);
    const MembwSamples compute = membwSummarize(computeSamples);
    const MembwSamples computeF16 = membwSummarize(computeF16Samples);
    const MembwSamples computeInt8 = membwSummarize(computeInt8Samples);

    const float readAvgGBs = membwGBs(oneArr, read.totalSeconds);
    const float copyAvgGBs = membwGBs(twoArr, copy.totalSeconds);
    const float scaleAvgGBs = membwGBs(twoArr, scale.totalSeconds);
    const float addAvgGBs = membwGBs(threeArr, add.totalSeconds);
    const float triadAvgGBs = membwGBs(threeArr, triad.totalSeconds);
    const float readMedGBs = membwGBs(bwPasses * bytesF, read.medianSeconds);
    const float copyMedGBs = membwGBs(bwPasses * 2.0f * bytesF, copy.medianSeconds);
    const float scaleMedGBs = membwGBs(bwPasses * 2.0f * bytesF, scale.medianSeconds);
    const float addMedGBs = membwGBs(bwPasses * 3.0f * bytesF, add.medianSeconds);
    const float triadMedGBs = membwGBs(bwPasses * 3.0f * bytesF, triad.medianSeconds);
    const float readBestGBs = membwGBs(bwPasses * bytesF, read.bestSeconds);
    const float copyBestGBs = membwGBs(bwPasses * 2.0f * bytesF, copy.bestSeconds);
    const float scaleBestGBs = membwGBs(bwPasses * 2.0f * bytesF, scale.bestSeconds);
    const float addBestGBs = membwGBs(bwPasses * 3.0f * bytesF, add.bestSeconds);
    const float triadBestGBs = membwGBs(bwPasses * 3.0f * bytesF, triad.bestSeconds);
    float peakBestGBs = readBestGBs;
    if (scaleBestGBs > peakBestGBs) peakBestGBs = scaleBestGBs;
    if (copyBestGBs > peakBestGBs) peakBestGBs = copyBestGBs;
    if (addBestGBs > peakBestGBs) peakBestGBs = addBestGBs;
    if (triadBestGBs > peakBestGBs) peakBestGBs = triadBestGBs;
    float peakMedianGBs = readMedGBs;
    if (scaleMedGBs > peakMedianGBs) peakMedianGBs = scaleMedGBs;
    if (copyMedGBs > peakMedianGBs) peakMedianGBs = copyMedGBs;
    if (addMedGBs > peakMedianGBs) peakMedianGBs = addMedGBs;
    if (triadMedGBs > peakMedianGBs) peakMedianGBs = triadMedGBs;

    const float totalFlops =
        static_cast<float>(iterations) * static_cast<float>(N) *
        static_cast<float>(MEMBW_COMPUTE_FMA) * 2.0f;
    const float flopsPerPass = static_cast<float>(N) * static_cast<float>(MEMBW_COMPUTE_FMA) * 2.0f;
    const float gflopsAvg = membwGFLOPs(totalFlops, compute.totalSeconds);
    const float gflopsMedian = membwGFLOPs(flopsPerPass, compute.medianSeconds);
    const float gflopsBest = membwGFLOPs(flopsPerPass, compute.bestSeconds);
    const float gflopsF16Avg = membwGFLOPs(totalFlops, computeF16.totalSeconds);
    const float gflopsF16Median = membwGFLOPs(flopsPerPass, computeF16.medianSeconds);
    const float gflopsF16Best = membwGFLOPs(flopsPerPass, computeF16.bestSeconds);
    const float totalInt8Ops =
        static_cast<float>(iterations) * static_cast<float>(N / 16 * 16) *
        static_cast<float>(MEMBW_COMPUTE_INT8_DOT) * 2.0f;
    const float int8OpsPerPass = static_cast<float>(N / 16 * 16) *
        static_cast<float>(MEMBW_COMPUTE_INT8_DOT) * 2.0f;
    const float gopsInt8Avg = membwGFLOPs(totalInt8Ops, computeInt8.totalSeconds);
    const float gopsInt8Median = membwGFLOPs(int8OpsPerPass, computeInt8.medianSeconds);
    const float gopsInt8Best = membwGFLOPs(int8OpsPerPass, computeInt8.bestSeconds);
    const float tWall = secs(clock::now() - callStart);

    char out[2048];
    const std::string cpuList = membwCpuListString(activeGroup->cpus);
    std::snprintf(out, sizeof(out),
        "Memory bandwidth test\n"
        "arrays: %.0fM x 3 = %.0f MB float · group: %s cpu[%s] · threads: %d/%d · iters: %d · warmup: %d\n"
        "reported as median / best sample; avg shown in [] · bw batch: %d pass%s/sample\n"
        "%s Read : %7.2f / %7.2f GB/s [%7.2f]\n"
        "%s Copy : %7.2f / %7.2f GB/s [%7.2f]\n"
        "%s Scale: %7.2f / %7.2f GB/s [%7.2f]\n"
        "%s Add  : %7.2f / %7.2f GB/s [%7.2f]\n"
        "%s Triad: %7.2f / %7.2f GB/s [%7.2f]\n"
        "%s peak : %7.2f / %7.2f GB/s\n"
        "%s F32 compute : %7.2f / %7.2f GFLOP/s [%7.2f] (%.1f Gflop)\n"
        "%s F16 compute : %7.2f / %7.2f GFLOP/s [%7.2f] (%.1f Gflop)\n"
        "%s INT8 compute: %7.2f / %7.2f GOP/s   [%7.2f] (%.1f Gop)\n"
        "%s wall: %.2fs measured · %.2fs total · checksum: %.4e",
        static_cast<float>(N) / 1.0e6f, 3.0f * bytesF / 1048576.0f,
        activeGroup->label, cpuList.c_str(), threads, availableCpus, iterations, MEMBW_WARMUP_PASSES,
        bandwidthPassesPerSample, bandwidthPassesPerSample == 1 ? "" : "es",
        activeGroup->label,
        readMedGBs, readBestGBs, readAvgGBs,
        activeGroup->label,
        copyMedGBs, copyBestGBs, copyAvgGBs,
        activeGroup->label,
        scaleMedGBs, scaleBestGBs, scaleAvgGBs,
        activeGroup->label,
        addMedGBs, addBestGBs, addAvgGBs,
        activeGroup->label,
        triadMedGBs, triadBestGBs, triadAvgGBs,
        activeGroup->label,
        peakMedianGBs, peakBestGBs,
        activeGroup->label,
        gflopsMedian, gflopsBest, gflopsAvg, totalFlops / 1.0e9f,
        activeGroup->label,
        gflopsF16Median, gflopsF16Best, gflopsF16Avg, totalFlops / 1.0e9f,
        activeGroup->label,
        gopsInt8Median, gopsInt8Best, gopsInt8Avg, totalInt8Ops / 1.0e9f,
        activeGroup->label,
        tMeasureWall, tWall, checksum);

    MEMBW_LOGI("MemBw[%s cpu[%s]]: readBest=%.2f peakMedian=%.2f peakBest=%.2f GB/s F32=%.2f F16=%.2f GFLOP/s INT8=%.2f GOP/s wall=%.2fs threads=%d/%d N=%zu iters=%d bwBatch=%d",
               activeGroup->label, cpuList.c_str(),
               readBestGBs, peakMedianGBs, peakBestGBs, gflopsMedian, gflopsF16Median, gopsInt8Median,
               tWall, threads, availableCpus, N, iterations, bandwidthPassesPerSample);

    return env->NewStringUTF(out);
}
