#pragma once

#include <array>
#include <string>
#include <cstdint>
#include <vector>

// Native-LLM user settings. Mutable runtime state lives in llm_runtime_state.h.

// Row order defines ModelId and pipeline order. Each strategy graph fuses Rotary + Main (embed folded in)
// + the decode head into one session. All merged graphs share one external weight blob attached in memory
// through OrtApi::AddInitializer; only its descriptor carries an external-data sidecar.
#define LLM_MODEL_TABLE(X) \
    X(LLM_TextPrefillGreedy,         "LLM_TextPrefillGreedy.onnx",         "") \
    X(LLM_TextPrefillPenaltyGreedy,  "LLM_TextPrefillPenaltyGreedy.onnx",  "") \
    X(LLM_TextPrefillSampling,       "LLM_TextPrefillSampling.onnx",       "") \
    X(LLM_TextDecodeGreedy,          "LLM_DecodeGreedy.onnx",              "") \
    X(LLM_TextDecodePenaltyGreedy,   "LLM_DecodePenaltyGreedy.onnx",       "") \
    X(LLM_TextDecodeSampling,        "LLM_DecodeSampling.onnx",            "") \
    X(LLM_KV_Slice,                  "LLM_KV_Slice.onnx",                  "") \
    X(LLM_KV_Split2,                 "LLM_KV_Split2.onnx",                 "") \
    X(LLM_KV_Concat,                 "LLM_KV_Concat.onnx",                 "") \
    X(LLM_RopeShift,                 "LLM_RopeShift.onnx",                 "") \
    X(LLM_SharedInitializers,        "LLM_SharedInitializers.onnx",        "LLM_SharedInitializers.onnx.data") \
    X(LLM_Metadata,                  "LLM_Metadata.onnx",                  "")

// Merged-graph geometry: three prefill slots followed by three decode slots.
#define LLM_MERGED_MODEL_COUNT 6
#define LLM_FIRST_MERGED_MODEL LLM_TextPrefillGreedy

// ORT threading.
constexpr bool kUseGlobalThreadPool  = true;
// Spin control is per pool, NOT global. The prefill/shared pool owns kPrefillCpuCoreCount worker threads that
// sit idle for the entire (long) decode phase. If that pool spins it busy-waits on every big core while only
// the 2-thread decode pool is doing real work, so the decode stage reads as full-CPU usage. Keep the prefill
// pool BLOCKING when idle (0) so it sleeps during decode; keep the decode pool SPINNING (1) for low per-token
// wake latency on its Top-N big cores. During prefill the shared pool is busy, so 0 only costs a negligible
// per-parallel-section wakeup while eliminating the decode-phase idle burn (ORT default is 0 on-device).
constexpr int  kPrefillAllowSpinning = 0;
constexpr int  kDecodeAllowSpinning  = 1;
constexpr int  kPrefillCpuCoreCount  = 8;
constexpr int  kDecodeCpuCoreCount   = 2;
static_assert(kPrefillCpuCoreCount > 0 && kDecodeCpuCoreCount > 0, "CPU core counts must be positive");

// Register one CPU arena on the shared OrtEnv so session.use_env_allocators is effective.
// -1/0 keep ORT defaults (kNextPowerOfTwo growth, unlimited max memory); false restores per-session arenas.
constexpr bool        kRegisterSharedEnvAllocator   = true;
constexpr int         kEnvArenaExtendStrategy       = -1;   // -1 ORT default (=kNextPowerOfTwo); 1 = kSameAsRequested
constexpr int         kEnvArenaInitialChunkBytes    = -1;   // -1 = ORT default
constexpr int         kEnvArenaMaxDeadBytesPerChunk = -1;   // -1 = ORT default
constexpr std::size_t kEnvArenaMaxMemBytes          = 0;    // 0 = unlimited

// Pre-grow the shared arena for cache-heavy graphs; requires kRegisterSharedEnvAllocator.
// Target size is kPrewarmArenaPercent% of the largest single-graph cache working set.
constexpr bool kPrewarmSharedArena  = true;
constexpr int  kPrewarmArenaPercent = 100;   // 0 disables; % of the all-graph peak cache working set to reserve

// Share ORT pre-packed initializer buffers across sessions when graph initializers are shared/injected.
// This does not merge graphs by itself; it is the runtime half of ORT's AddInitializer +
// CreateSession*WithPrepackedWeightsContainer sharing mechanism.
constexpr bool kUseSharedPrepackedWeightsContainer = true;

// Model geometry. Metadata-filled values are set during Load_Models_A.
constexpr int  kDefaultNumLayers = 0;
constexpr int  max_num_layers    = 64;

inline int  max_seq_len = 0;       // metadata "max_seq_len": baked rotary-table size (export MAX_SEQ_LEN)

constexpr int kv_blocks_max   = 6;
constexpr int max_keys_values = max_num_layers * kv_blocks_max;

// Conversation limits, in RoPE/token positions. Runtime max/default values derive from max_seq_len.
inline int DEFAULT_MEMORY_TOKENS     = 0;   // max_seq_len * 3 / 8
inline int DEFAULT_PREFILL_TOKENS    = 0;   // DEFAULT_MEMORY_TOKENS / 8
inline int DEFAULT_MAX_DECODE_TOKENS = 0;   // max_seq_len / 2 - 1

constexpr int kDecodeRecycleHeadroom = 32;

constexpr int MIN_MEMORY_TOKENS          = 256;
inline    int MAX_MEMORY_TOKENS_RUNTIME  = 0;   // set to max_seq_len - kDecodeRecycleHeadroom at load
constexpr int MIN_PREFILL_TOKENS         = 0;
inline    int MAX_PREFILL_TOKENS_RUNTIME = 0;   // MAX_MEMORY_TOKENS_RUNTIME
constexpr int MIN_DECODE_TOKENS          = 2;
inline    int MAX_DECODE_TOKENS_RUNTIME  = 0;   // max_seq_len

// Memory hysteresis band (% of memory cap); green must stay below red.
constexpr int DEFAULT_MEMORY_GREEN_TARGET_PERCENT = 60;
constexpr int DEFAULT_MEMORY_RED_ZONE_PERCENT     = 95;
constexpr int MIN_MEMORY_BAND_PERCENT = 15;
constexpr int MAX_MEMORY_BAND_PERCENT = 95;
constexpr int MIN_MEMORY_BAND_GAP     = 1;

// Decode defaults and UI-safe runtime bounds. Greedy uses the exporter's direct 0..1 logit
// multiplier; sampling uses the standard >=1 repetition penalty from the Python inference path.
constexpr int   DECODE_MODE_GREEDY       = 0;
constexpr int   DECODE_MODE_SAMPLING     = 1;
static_assert(DECODE_MODE_GREEDY == 0 && DECODE_MODE_SAMPLING == 1,
              "Decode mode protocol must contain only Greedy and Sampling");
constexpr int   DEFAULT_TOP_K           = 3;
constexpr float DEFAULT_REPEAT_PENALTY  = 1.0f;
constexpr int   DEFAULT_PENALTY_RANGE   = 20;
constexpr float DEFAULT_TEMPERATURE     = 0.8f;
constexpr float DEFAULT_TOP_P           = 0.95f;
constexpr int   MAX_SAMPLING_TOP_K_RUNTIME = 50;
constexpr int   MAX_PENALTY_RANGE_RUNTIME  = 256;
constexpr float MIN_TEMPERATURE_RUNTIME    = 0.1f;
constexpr float MAX_TEMPERATURE_RUNTIME    = 2.0f;
constexpr float MIN_TOP_P_RUNTIME          = 0.05f;
constexpr float MAX_TOP_P_RUNTIME          = 1.0f;
constexpr float MAX_SAMPLING_PENALTY       = 2.0f;

// Chat IDs are auto-filled from ONNX metadata; buildChatTemplates() fills the arrays.
inline int chat_endoftext_id      = 0;  // metadata "chat_endoftext_id"      (<|endoftext|>)
inline int chat_im_start_id       = 0;  // metadata "chat_im_start_id"       (<|im_start|>)
inline int chat_im_end_id         = 0;  // metadata "chat_im_end_id"         (<|im_end|>)
inline int chat_user_id           = 0;
inline int chat_assistant_id      = 0;
inline int chat_newline_id        = 0;  // metadata "chat_newline_id"        (\n)
inline int chat_double_newline_id = 0;  // metadata "chat_double_newline_id" (\n\n)
inline int chat_think_start_id    = 0;  // metadata "chat_think_start_id"    (<think>)
inline int chat_think_end_id      = 0;  // metadata "chat_think_end_id"      (</think>)

inline std::vector<int> chat_conversation_prefix_ids;
inline std::vector<int> chat_user_prefix_ids;
inline std::vector<int> chat_assistant_prefix_ids;
inline std::vector<int> chat_empty_think_block_ids;
inline std::vector<int> chat_previous_assistant_close_ids;
// Streaming flush policy.
constexpr int     STREAM_BATCH    = 10;
constexpr int32_t STREAM_FLUSH_MS = 50;
constexpr int32_t PERF_STATS_INTERVAL_MS = 250;

// App-specific paths are supplied by Java from Context directories before any model is loaded. This keeps
// work profiles, secondary Android users, and applicationId changes on the same lookup path as getCacheDir().
inline std::string storage_path;
inline std::string cache_path;
// Preferred tokenizer path. Pre_Process auto-discovers staged vocab_*.txt assets (case-insensitive,
// see resolveVocabPath) when this exact file is absent. The file header selects the tokenizer format.
inline std::string vocab_path;
inline const std::string MANUAL_STOP_NOTICE = "\n\n\u23f9 \u7528\u6237\u624b\u52a8\u505c\u6b62 User manually stopped";

// ORT run/session/provider configs. Runtime-specific overrides stay in loadModels.cpp/ort_helpers.h.
struct OrtConfigEntry { const char* key; const char* value; };

// CPU EP policy for the six merged LLM strategy graphs. These exceptions were isolated under ORT 1.27;
// keep the switch explicit so each Android CPU/model combination can be A/B benchmarked on-device.
// XNNPACK, QNN, and KV/RoPE utility graphs retain their provider defaults.
constexpr bool kUseCpuMergedOptimizerExceptions = true;
constexpr const char* kCpuMergedDisabledOptimizers =
    "MatMulAddFusion;NchwcTransformer;ConvAddActivationFusion;MatmulTransposeFusion";

inline constexpr OrtConfigEntry kOrtRunConfigs[] = {
    {"disable_synchronize_execution_providers", "0"},
};

inline constexpr OrtConfigEntry kQnnRunConfigs[] = {
    {"qnn.htp_perf_mode",          "burst"},
    {"qnn.htp_perf_mode_post_run", "burst"},
    {"qnn.rpc_control_latency",    "100"},
};

inline constexpr OrtConfigEntry kOrtSessionConfigs[] = {
    {"session.inter_op.allow_spinning",                        "1"},
    {"session.intra_op.allow_spinning",                        "1"},
    {"session.intra_op.spin_backoff_max",                      "1"},
    {"session.inter_op.spin_backoff_max",                      "1"},
    {"session.force_spinning_stop",                            "0"},
    {"session.graph_optimizations_loop_level",                 "2"},
    {"optimization.minimal_build_optimizations",               ""},
    {"optimization.constant_folding_max_output_size_in_bytes", ""},
    {"optimization.enable_gelu_approximation",                 "1"},
    {"optimization.enable_cast_chain_elimination",             "1"},
    {"session.disable_prepacking",                             "0"},
    {"session.use_ort_model_bytes_directly",                   "0"},
    {"session.use_ort_model_bytes_for_initializers",           "0"},
    {"session.use_memory_mapped_ort_model",                    "1"},
    {"session.use_env_allocators",                             "1"},
    {"session.use_device_allocator_for_initializers",          "1"},
    {"session.set_denormal_as_zero",                           "1"},
    {"session.disable_quant_qdq",                              "0"},
    {"session.disable_qdq_constant_folding",                   "0"},
    {"session.disable_double_qdq_remover",                     "0"},
    {"session.enable_quant_qdq_cleanup",                       "1"},
    {"session.qdqisint8allowed",                               "1"},
    {"session.qdq_matmulnbits_accuracy_level",                 "4"},
    {"session.qdq_matmulnbits_block_size",                     "32"},
    {"session.enable_dq_matmulnbits_fusion",                   "1"},
    {"session.disable_aot_function_inlining",                  "0"},
    {"mlas.enable_gemm_fastmath_arm64_bfloat16",               "0"},
    {"mlas.use_lut_gemm",                                      "1"},
    {"mlas.disable_kleidiai",                                  "0"},
    {"session.strict_shape_type_inference",                    "0"},
    {"session.allow_released_opsets_only",                     "0"},
    {"ep.context_enable",                                      "0"},
    {"ep.context_embed_mode",                                  "0"},
    {"session.disable_model_compile",                          "0"},
    {"session.fail_on_suboptimal_compiled_model",              "0"},
    {"ep.dynamic.workload_type",                               "Default"},
    {"session.debug_layout_transformation",                    "0"},
    {"session.record_ep_graph_assignment_info",                "0"},
};

inline constexpr OrtConfigEntry kXnnpackProviderOptions[] = {
    {"intra_op_num_threads", "4"},
};

inline constexpr OrtConfigEntry kQnnProviderOptions[] = {
    {"backend_type",                             "htp"},
    {"profiling_level",                          "off"},
    {"htp_performance_mode",                     "burst"},
    {"rpc_control_latency",                      "100"},
    {"vtcm_mb",                                  "0"},
    {"qnn_context_priority",                     "high"},
    {"htp_graph_finalization_optimization_mode", "3"},
    {"soc_model",                                "0"},
    {"htp_arch",                                 "0"},
    {"device_id",                                "0"},
    {"enable_htp_fp16_precision",                "1"},
    {"offload_graph_io_quantization",            "1"},
    {"enable_htp_spill_fill_buffer",             "0"},
    {"enable_htp_shared_memory_allocator",       "1"},
    {"dump_json_qnn_graph",                      "0"},
};
