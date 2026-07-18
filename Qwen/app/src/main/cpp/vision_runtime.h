#pragma once

// Vision runtime state. Text-only exports leave all readiness flags false.

#include "llm_runtime_state.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

// Modality of a pending / active vision input.
enum VisionModality : uint8_t { VISION_NONE = 0, VISION_IMAGE = 1, VISION_VIDEO = 2 };

// A pending vision request handed from the camera/UI thread to the LLM worker (Run_LLM). The pixel
// data lives in the pre-bound static pixel_values buffers below; this atomic only signals which
// modality (if any) the next Run_LLM should consume. Cleared by Run_LLM once the turn starts.
inline std::atomic<uint8_t> g_pending_vision{VISION_NONE};

// Guards the static pixel_values buffers between the camera memcpy (producer) and Run_LLM (consumer).
inline std::mutex g_vision_pixels_mutex;

// Prewarm and Run_LLM share the vision sessions and cached hidden under one mutex. Capture sequence IDs
// reject stale results; cache misses encode inline.
inline std::mutex            g_vision_encode_mutex;
inline OrtValue*             g_pending_vision_hidden  = nullptr;   // cached ViT output (owned; device tensor)
inline uint8_t               g_vision_hidden_modality = VISION_NONE;
inline uint32_t              g_vision_hidden_seq      = 0;         // capture seq the cached hidden was built for
inline bool                  g_vision_hidden_ready    = false;
inline std::atomic<uint32_t> g_vision_capture_seq{0};             // bumped on each image push / video submit
inline std::atomic<bool>     g_vision_cancel_requested{false};    // aborts speculative / inline vision encode

// Preallocated pixel tensors wrap storage owned by the backing vectors. Input types come from each
// preprocess graph; fixed-shape buffers are never reallocated per frame.
inline OrtValue* g_image_pixel_values = nullptr;   // [N,1,3,H,W]
inline OrtValue* g_video_pixel_values = nullptr;   // [F,3,H,W] (ring-filled)
inline std::vector<uint8_t> g_image_pixel_backing; // owns g_image_pixel_values storage
inline std::vector<uint8_t> g_video_pixel_backing; // owns g_video_pixel_values storage
inline size_t g_image_slot_bytes          = 0;     // raw camera bytes per image (uint8 CHW)
inline size_t g_image_slot_storage_bytes  = 0;     // ONNX backing bytes per image
inline size_t g_video_frame_bytes         = 0;     // raw camera bytes per frame (uint8 CHW)
inline size_t g_video_frame_storage_bytes = 0;     // ONNX backing bytes per frame

// The GL callback writes ring slots directly; submit reorders a wrapped ring oldest-to-newest.
inline std::atomic<int>  g_video_ring_count{0};       // frames currently buffered (<= g_video_num_frames)
inline std::atomic<int>  g_video_ring_write{0};       // next slot to write (mod g_video_num_frames)
inline std::atomic<bool> g_video_stream_active{false};// sampling on/off (paused during processing)
inline std::atomic<bool> g_video_stream_desired{false};// UI wants continuous streaming (survives a pause)
inline bool              g_video_fixed_length_input = true; // fixed F axis -> ring + black padding
inline int               g_video_recorded_frames = 0;       // actual sampled frames in the current clip
inline int               g_video_dynamic_capacity_frames = 0; // allocated append capacity for dynamic-F input

// mRoPE position accounting. For a text turn this equals saved_kv_base+saved_kv_len; for a vision turn
// the mRoPE position advances by the vision segment SPAN (max(grid_h,grid_w)) while the physical KV
// length advances by the vision token count, so the two diverge and are tracked separately. This is
// the value fed as history_len to the next prefill's rotary graph.
inline int64_t saved_mrope_pos = 0;

// Each active-window segment records physical length, mRoPE span and modality. Vision segments cannot
// be reconstructed from token IDs alone.
struct KVSegment {
    int64_t token_len;   // physical KV tokens this segment occupies
    int64_t mrope_span;  // mRoPE base-position advance for this segment
    uint8_t modality;    // VisionModality
};
inline std::vector<KVSegment> g_kv_segments;

// True while saved KV contains a vision segment; ID-based rebuilds replace it with the memory note.
inline bool saved_kv_has_vision = false;
