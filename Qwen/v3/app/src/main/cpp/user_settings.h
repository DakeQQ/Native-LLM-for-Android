#pragma once

#include <array>
#include <string>
#include <cstdint>

// Native-LLM user settings. Mutable runtime state lives in llm_runtime_state.h.

// Row order defines ModelId and pipeline order. Embed/Main use external data files.
// External names MUST match the `location` string baked into each .onnx (see export) so the
// runtime can locate/mmap the sidecar; the exporter writes "<model>.onnx.data".
#define LLM_MODEL_TABLE(X) \
    X(LLM_Embed,         "LLM_Embed.onnx",          "LLM_Embed.onnx.data") \
    X(LLM_Main,          "LLM_Main.onnx",           "LLM_Main.onnx.data") \
    X(LLM_RotaryPrefill, "LLM_RotaryPrefill.onnx",  "") \
    X(LLM_RotaryDecode,  "LLM_RotaryDecode.onnx",   "") \
    X(LLM_Greedy,        "LLM_Greedy.onnx",         "") \
    X(LLM_FirstBeam,     "LLM_FirstBeam.onnx",      "") \
    X(LLM_SecondBeam,    "LLM_SecondBeam.onnx",     "") \
    X(LLM_Penalty,       "LLM_Penalty.onnx",        "") \
    X(LLM_Argmax,        "LLM_Argmax.onnx",         "") \
    X(LLM_KV_Slice,      "LLM_KV_Slice.onnx",       "") \
    X(LLM_KV_Split2,     "LLM_KV_Split2.onnx",      "") \
    X(LLM_KV_Concat,     "LLM_KV_Concat.onnx",      "") \
    X(LLM_RopeShift,     "LLM_RopeShift.onnx",      "")

// Dedicated tiny graph used only to preload exporter-stamped metadata before session flags are known.
constexpr const char* kMetadataModelFileName = "LLM_Metadata.onnx";

// ORT threading.
constexpr bool kUseGlobalThreadPool  = true;
constexpr int  kGlobalIntraOpThreads = 7;
constexpr int  kGlobalInterOpThreads = 7;
constexpr int  kGlobalAllowSpinning  = 1;
constexpr const char* kGlobalIntraAffinities = "1;2;3;4;5;6,7";

constexpr int         kSessionIntraOpThreads   = kGlobalIntraOpThreads;
constexpr int         kSessionInterOpThreads   = kGlobalInterOpThreads;
constexpr int         kSessionDynamicBlockBase = kGlobalIntraOpThreads - 1;
constexpr const char* kSessionIntraAffinities  = kGlobalIntraAffinities;

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

// Model geometry. Metadata-filled values are set during Load_Models_A.
constexpr int  kDefaultNumLayers = 0;
constexpr int  max_num_layers    = 64;

inline int  max_seq_len = 0;       // metadata "max_seq_len": baked rotary-table size (export MAX_SEQ_LEN)
inline bool ORT_FP16    = false;   // metadata "activations_fp16": 1 => fully-FP16 attention; 0 => FP32 acts + FP16 KV

constexpr int kv_blocks_max   = 6;
constexpr int max_keys_values = max_num_layers * kv_blocks_max;

// Conversation limits, in RoPE/token positions. Runtime max/default values derive from max_seq_len.
inline int DEFAULT_MEMORY_TOKENS     = 0;   // max_seq_len / 4
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

// Decode defaults.
constexpr bool  DEFAULT_USE_BEAM_SEARCH = false;
constexpr int   DEFAULT_TOP_K           = 3;
constexpr int   DEFAULT_BEAM_SIZE       = 3;
constexpr float DEFAULT_REPEAT_PENALTY  = 1.0f;
constexpr float ESCAPE_REPEAT_PENALTY   = 0.6f;
constexpr int   PENALTY_RANGE           = 20;
constexpr int   MAX_TOP_K_RUNTIME       = 10;
constexpr int   MAX_BEAM_SIZE_RUNTIME   = 10;

// Organize-memory default.
constexpr bool DEFAULT_ORGANIZE_MEMORY = false;

// Chat IDs are auto-filled from ONNX metadata; buildChatTemplates() fills the arrays.
inline int chat_endoftext_id      = 0;  // metadata "chat_endoftext_id"      (<|endoftext|>)
inline int chat_im_start_id       = 0;  // metadata "chat_im_start_id"       (<|im_start|>)
inline int chat_im_end_id         = 0;  // metadata "chat_im_end_id"         (<|im_end|>)
inline int chat_system_id         = 0;
inline int chat_user_id           = 0;
inline int chat_assistant_id      = 0;
inline int chat_newline_id        = 0;  // metadata "chat_newline_id"        (\n)
inline int chat_double_newline_id = 0;  // metadata "chat_double_newline_id" (\n\n)
inline int chat_think_start_id    = 0;  // metadata "chat_think_start_id"    (<think>)
inline int chat_think_end_id      = 0;  // metadata "chat_think_end_id"      (</think>)

inline int end_id_0 = 0;
inline int end_id_1 = 0;

inline std::array<int, 3> chat_user_prefix_ids{};
inline std::array<int, 5> chat_assistant_prefix_ids{};
inline std::array<int, 4> chat_empty_think_block_ids{};
inline std::array<int, 2> chat_previous_assistant_close_ids{};
inline std::array<int, 3> chat_system_prefix_ids{};
inline std::array<int, 2> chat_system_suffix_ids{};

// Must match SYSTEM_PROMPT_DEFAULT in MainActivity.java.
inline const std::string DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant.";

// Streaming flush policy.
constexpr int     STREAM_BATCH    = 10;
constexpr int32_t STREAM_FLUSH_MS = 50;

// Storage paths and stop notice.
inline const std::string storage_path = "/storage/emulated/0/Android/data/com.example.myapplication/";
inline const std::string cache_path   = "/data/user/0/com.example.myapplication/cache/";
inline const std::string vocab_path   = "/data/user/0/com.example.myapplication/cache/vocab_Qwen.txt";  // staged from assets
inline const std::string MANUAL_STOP_NOTICE = "\n\n\u23f9 \u7528\u6237\u624b\u52a8\u505c\u6b62 User manually stopped";

// ORT run/session/provider configs. Runtime-specific overrides stay in loadModels.cpp/ort_helpers.h.
struct OrtConfigEntry { const char* key; const char* value; };

inline constexpr OrtConfigEntry kOrtRunConfigs[] = {
    {"disable_synchronize_execution_providers", "1"},
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
    {"optimization.constant_folding_max_output_size_in_bytes", "0"},
    {"optimization.enable_gelu_approximation",                 "1"},
    {"optimization.enable_cast_chain_elimination",             "1"},
    {"session.disable_prepacking",                             "0"},
    {"session.use_ort_model_bytes_directly",                   "1"},
    {"session.use_ort_model_bytes_for_initializers",           "0"},
    {"session.use_memory_mapped_ort_model",                    "1"},
    {"session.use_env_allocators",                             "1"},
    {"session.use_device_allocator_for_initializers",          "0"},
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
    {"session.disable_cpu_ep_fallback",                        "0"},
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
