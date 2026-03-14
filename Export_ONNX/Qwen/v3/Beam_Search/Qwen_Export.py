import gc
import time
import torch
import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from transformers import AutoModelForCausalLM, AutoTokenizer


download_path                   = r'/home/DakeQQ/Downloads/Qwen3-1.7B'                             # Set the folder path where the Qwen whole project downloaded.
onnx_model_Embed                = r'/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Embed.onnx'
onnx_model_Main                 = r'/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Main.onnx'
onnx_model_Rotary_Mask_Prefill  = r'/home/DakeQQ/Downloads/Qwen_ONNX/Rotary_Mask_Text_Prefill.onnx'
onnx_model_Rotary_Mask_Decode   = r'/home/DakeQQ/Downloads/Qwen_ONNX/Rotary_Mask_Text_Decode.onnx'
onnx_model_Greedy               = r'/home/DakeQQ/Downloads/Qwen_ONNX/Greedy_Search.onnx'
onnx_model_First_Beam           = r'/home/DakeQQ/Downloads/Qwen_ONNX/First_Beam_Search.onnx'
onnx_model_Second_Beam          = r'/home/DakeQQ/Downloads/Qwen_ONNX/Second_Beam_Search.onnx'
onnx_model_Penalty              = r'/home/DakeQQ/Downloads/Qwen_ONNX/Apply_Penalty.onnx'
onnx_model_Argmax               = r'/home/DakeQQ/Downloads/Qwen_ONNX/Argmax.onnx'
onnx_model_KV_Slice             = r'/home/DakeQQ/Downloads/Qwen_ONNX/KV_Slice.onnx'

# Test input
TEST_THINK_MODE = False
TEST_QUERY = "地球最高的山峰是什么？"

# Model Config
DO_EXPORT = True                    # Whether to export the ONNX models
PREVENT_F16_OVERFLOW = False        # Prevent float16 overflow. Set True for Q4F16 or Q8F16 or F16 quantization.
STOP_TOKEN = [151643, 151645]       # Qwen stop token ids
MAX_SEQ_LEN = 4096                  # Max context length. Can not edit after export.

# KV cache quantization
KV_QUANT_DTYPE = "F16"              # "Q8" | "Q8_CUDA" | "F16" | "F32"
USE_FLOAT16_SCALE_BIAS = True       # If choose Q8, whether to use float16 for scale and bias.

# Decoding strategy
USE_BEAM_SEARCH = False             # Use beam search or greedy search
REPEAT_PENALTY = 1.0                # 0.0 ~ 1.0; No penalty = 1.0
PENALTY_RANGE = 20                  # Recent-token window to apply penalty
MAX_BEAM_SIZE = 10                  # Max beam size for beam search. Can not edit after export.
TOP_K = 3                           # Top-K for beam search
BEAM_SIZE = 3                       # Beam size for beam search. Must be <= MAX_BEAM_SIZE

# Runtime config
ORT_LOG = False                     # Enable ONNX Runtime logging for debugging. Set to False for best performance.
ORT_FP16 = False                    # Set to True for FP16 ONNX Runtime settings. For CPUs, this requires ARM64-v8.2a or newer.
ORT_Accelerate_Providers = []       # ORT execution providers; ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
MAX_THREADS = 0                     # 0 = auto
DEVICE_ID = 0                       # Device ID for GPU
OPSET = 17                          # ONNX opset version


# ══════════════════════════════════════════════════════════════════════════════
# Decoding Strategy Modules
# ══════════════════════════════════════════════════════════════════════════════
class GREEDY_SEARCH(torch.nn.Module):
    """Greedy decoding: select the token with the highest logit."""

    def __init__(self):
        super().__init__()

    def forward(self, logits, save_id):
        max_logits_idx = torch.argmax(logits, dim=-1, keepdim=True).int()
        save_id = torch.cat([save_id, max_logits_idx], dim=-1)
        return max_logits_idx, save_id


class FIRST_BEAM_SEARCH(torch.nn.Module):
    """First beam-search step: expand a single hypothesis into `beam_size` beams."""

    def __init__(self, total_layers):
        super().__init__()
        self.total_layers = total_layers
        self.save_keys_values = [None] * self.total_layers
        self.kv_q8_cuda = (KV_QUANT_DTYPE == "Q8_CUDA")

    def forward(self, *all_inputs):
        logits = all_inputs[-3]
        save_id = all_inputs[-2]
        beam_size = all_inputs[-1]

        # Compute log-probabilities for the top-k beams
        row_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_beam_logits, top_beam_indices = torch.topk(logits, dim=-1, k=beam_size, sorted=False, largest=True)
        top_beam_prob = top_beam_logits - row_logsumexp

        # Replicate KV caches across all beams
        for i in range(self.total_layers):
            kv = all_inputs[i]
            self.save_keys_values[i] = kv.repeat(beam_size, *([1] * (kv.dim() - 1)))

        top_beam_indices = top_beam_indices.transpose(0, 1).int()
        save_id = torch.cat([save_id, top_beam_indices], dim=-1)
        max_logits_idx = top_beam_indices[[0]]

        return (
            *self.save_keys_values,
            save_id,
            top_beam_prob.transpose(0, 1),
            top_beam_indices,
            max_logits_idx
        )


class SECOND_BEAM_SEARCH(torch.nn.Module):
    """Subsequent beam-search steps: prune and re-expand beams."""

    def __init__(self, total_layers):
        super().__init__()
        self.total_layers = total_layers
        self.save_keys_values = [None] * self.total_layers

    def forward(self, *all_inputs):
        logits = all_inputs[-5]
        save_id = all_inputs[-4]
        previous_prob = all_inputs[-3]
        beam_size = all_inputs[-2]
        top_k = all_inputs[-1]

        # Compute log-probabilities and accumulate with previous scores
        row_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=-1, largest=True, sorted=False)
        top_k_prob = top_k_logits - row_logsumexp
        current_prob = (top_k_prob + previous_prob).view(-1)

        # Select the top beams from all candidates
        top_beam_prob, flat_beam_indices = torch.topk(current_prob, k=beam_size, dim=-1, largest=True, sorted=False)
        beam_index = flat_beam_indices // top_k
        top_beam_indices = top_k_indices.view(-1)[flat_beam_indices]

        # Gather KV caches for surviving beams
        for i in range(self.total_layers):
            self.save_keys_values[i] = torch.index_select(all_inputs[i], dim=0, index=beam_index)

        gathered_save_id = torch.index_select(save_id, dim=0, index=beam_index)
        top_beam_indices = top_beam_indices.unsqueeze(-1).int()
        max_logits_idx = top_beam_indices[[0]]
        save_id = torch.cat([gathered_save_id, top_beam_indices], dim=-1)

        return (
            *self.save_keys_values,
            save_id,
            top_beam_prob.unsqueeze(-1),
            top_beam_indices,
            max_logits_idx
        )


# ══════════════════════════════════════════════════════════════════════════════
# Penalty & Utility Modules
# ══════════════════════════════════════════════════════════════════════════════
class APPLY_PENALTY(torch.nn.Module):
    """Apply repetition penalty to recently generated token logits."""

    def __init__(self):
        super().__init__()

    def forward(self, logits, save_id, penalty_value, penalty_range):
        target_indices = save_id[:, -penalty_range:].long()
        penalized = logits.gather(1, target_indices) * penalty_value
        logits = logits.scatter(1, target_indices, penalized)
        return logits


class ARGMAX(torch.nn.Module):
    """Simple argmax over the vocabulary dimension."""

    def __init__(self):
        super().__init__()

    def forward(self, logits):
        return torch.argmax(logits, dim=-1, keepdim=True).int()


# ══════════════════════════════════════════════════════════════════════════════
# KV Cache Slice
# ══════════════════════════════════════════════════════════════════════════════
class KV_SLICE(torch.nn.Module):
    """Apply slice to KV cache tensors."""

    def __init__(self, num_layers):
        super().__init__()
        self.kv_quantized = (KV_QUANT_DTYPE == "Q8") or (KV_QUANT_DTYPE == "Q8_CUDA")
        self.num_layers = num_layers
        self.num_layers_2 = num_layers * 2
        self.num_layers_3 = num_layers * 3
        self.num_layers_4 = num_layers * 4
        self.num_layers_5 = num_layers * 5
        self.save_key   = [None] * num_layers
        self.save_value = [None] * num_layers
        if self.kv_quantized:
            self.save_k_scale = [None] * num_layers
            self.save_k_bias  = [None] * num_layers
            self.save_v_scale = [None] * num_layers
            self.save_v_bias  = [None] * num_layers

    def forward(self, *all_inputs):
        slice_start = all_inputs[-2]
        slice_end = all_inputs[-1]
        for i in range(self.num_layers):
            self.save_key[i]   = all_inputs[i][..., slice_start: slice_end]
            self.save_value[i] = all_inputs[i + self.num_layers][..., slice_start: slice_end, :]
            if self.kv_quantized:
                self.save_k_scale[i] = all_inputs[i + self.num_layers_2][..., slice_start: slice_end]
                self.save_k_bias[i]  = all_inputs[i + self.num_layers_3][..., slice_start: slice_end]
                self.save_v_scale[i] = all_inputs[i + self.num_layers_4][..., slice_start: slice_end, :]
                self.save_v_bias[i]  = all_inputs[i + self.num_layers_5][..., slice_start: slice_end, :]
        if self.kv_quantized:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_k_bias, *self.save_v_scale, *self.save_v_bias
        return *self.save_key, *self.save_value


# ══════════════════════════════════════════════════════════════════════════════
# KV Cache Quantization
# ══════════════════════════════════════════════════════════════════════════════
class KVQuantizer(torch.nn.Module):
    """Quantize key/value tensors to Q8 or Q8_CUDA packed formats."""
    
    def __init__(self):
        super().__init__()
        self.QMAX = 255.0
        self.register_buffer("inv_qmax", torch.tensor([1.0 / self.QMAX]).view(1, 1, 1, 1, -1))
        # Constants for Q8_CUDA int32 packing/unpacking
        for name, val in [("_256", 256), ("_128", 128), ("_65536", 65536), ("_16777216", 16777216)]:
            self.register_buffer(name, torch.tensor([val], dtype=torch.int32).view(1, 1, 1, 1, -1))

    def _quantize_block(self, x, dim):
        """Per-block min-max quantization to [0, 255]."""
        block_min, block_max = torch.aminmax(x, dim=dim, keepdim=True)
        scale = (block_max - block_min) * self.inv_qmax
        x_normalized = (x - block_min) / scale
        x_packed = torch.round(x_normalized)
        if KV_QUANT_DTYPE == "Q8":
            x_packed = x_packed.to(torch.uint8)
        if USE_FLOAT16_SCALE_BIAS:
            scale = scale.half()
            block_min = block_min.half()
        return x_packed, scale, block_min

    def pack_q8_cuda(self, x, dim, batch_size, num_kv_heads, head_dim_quarter):
        """Pack 4 uint8 values into a single int32 for CUDA-friendly storage."""
        x_i32 = x.to(torch.int32)
        if dim != -1:
            x_i32 = x_i32.reshape(batch_size, num_kv_heads, 1, head_dim_quarter, 4, -1)
        else:
            x_i32 = x_i32.reshape(batch_size, num_kv_heads, 1, -1, head_dim_quarter, 4)
        x0, x1, x2, x3 = torch.unbind(x_i32, dim=dim)
        return x0 + x1 * self._256 + x2 * self._65536 + (x3 - self._128) * self._16777216

    def unpack_q8_cuda(self, x_i32, dim, batch_size, num_kv_heads, head_dim):
        """Unpack int32 back into 4 uint8 channels."""
        r3 = x_i32 % self._16777216
        x3 = ((x_i32 - r3) // self._16777216) + self._128
        x2 = r3 // self._65536
        r2 = r3 % self._65536
        x1 = r2 // self._256
        x0 = r2 % self._256
        unpacked = torch.stack([x0, x1, x2, x3], dim=dim)
        if dim != -1:
            return unpacked.reshape(batch_size, num_kv_heads, 1, head_dim, -1)
        return unpacked.reshape(batch_size, num_kv_heads, 1, -1, head_dim)

    def forward(self, keys, values, batch_size, num_kv_heads, head_dim_quarter):
        k_packed, k_scale, k_bias = self._quantize_block(keys, dim=-2)
        v_packed, v_scale, v_bias = self._quantize_block(values, dim=-1)
        if KV_QUANT_DTYPE == "Q8_CUDA":
            k_packed = self.pack_q8_cuda(k_packed, -2, batch_size, num_kv_heads, head_dim_quarter)
            v_packed = self.pack_q8_cuda(v_packed, -1, batch_size, num_kv_heads, head_dim_quarter)
        return k_packed, k_scale, k_bias, v_packed, v_scale, v_bias


# ══════════════════════════════════════════════════════════════════════════════
# Embedding Module
# ══════════════════════════════════════════════════════════════════════════════
class LLM_EMBED(torch.nn.Module):
    """Extract and apply the token embedding layer in float32."""

    def __init__(self, llm):
        super().__init__()
        self.embed_tokens = llm.model.embed_tokens.float()

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


# ══════════════════════════════════════════════════════════════════════════════
# Rotary Positional Embedding & Attention Mask
# ══════════════════════════════════════════════════════════════════════════════
class ROTARY_MASK_PREFILL(torch.nn.Module):
    """Precompute rotary embeddings and causal mask for the prefill phase."""

    def __init__(self, llm, max_seq_len):
        super().__init__()

        # Causal attention mask: upper triangle → -128
        self.attention_mask = (1 - torch.tril(torch.ones(1, 1, 1, max_seq_len, max_seq_len, dtype=torch.int8))) * -128

        # Precompute rotary embeddings
        cos, sin = self._build_rotary_table(llm, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half(), persistent=False)

    @staticmethod
    def _build_rotary_table(llm, max_seq_len):
        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq = llm.model.rotary_emb.inv_freq
        idx_theta = (position_ids * inv_freq).unsqueeze(1).unsqueeze(1).unsqueeze(0)
        return torch.cos(idx_theta), torch.sin(idx_theta)

    def forward(self, ids_len, history_len):
        kv_seq_len = ids_len + history_len
        rotary_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        attention_mask = self.attention_mask[..., :ids_len, :kv_seq_len].float()
        return rotary_cos, rotary_sin, attention_mask, kv_seq_len


class ROTARY_MASK_DECODE(torch.nn.Module):
    """Provide rotary embeddings for a single decode step."""

    def __init__(self, llm, max_seq_len):
        super().__init__()
        cos, sin = ROTARY_MASK_PREFILL._build_rotary_table(llm, max_seq_len)
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half(), persistent=False)

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        rotary_cos = self.cos_rotary_pos_emb[:, kv_seq_len_next].float()
        rotary_sin = self.sin_rotary_pos_emb[:, kv_seq_len_next].float()
        return rotary_cos, rotary_sin, kv_seq_len_next


# ══════════════════════════════════════════════════════════════════════════════
# Main Transformer Module
# ══════════════════════════════════════════════════════════════════════════════
class LLM_MAIN(torch.nn.Module):
    """
    Main transformer module that processes hidden states through all decoder layers.

    Handles:
      - Fused QKV projection with pre-merged layer norms
      - Rotary positional embeddings (RoPE)
      - KV cache management with optional Q8/Q8_CUDA quantization
      - Grouped-query attention (GQA)
      - Fused gate-up MLP projection
    """

    def __init__(self, llm, num_heads, num_key_value_heads, head_dim, num_layers, hidden_size):
        super().__init__()
        self.llm = llm

        # ── Attention geometry ───────────────────────────────────────────
        self.head_dim = head_dim
        self.head_dim_half = head_dim // 2
        self.head_dim_quarter = head_dim // 4
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.qk_heads = num_heads + num_key_value_heads

        # ── Layer count multipliers (for indexing into flat KV input list) ──
        self.num_layers = num_layers
        self.num_layers_2 = num_layers * 2
        self.num_layers_3 = num_layers * 3
        self.num_layers_4 = num_layers * 4
        self.num_layers_5 = num_layers * 5

        # ── KV cache dtype flags ─────────────────────────────────────────
        self.kv_f16 = (KV_QUANT_DTYPE == "F16")
        self.kv_q8 = (KV_QUANT_DTYPE == "Q8")
        self.kv_q8_cuda = (KV_QUANT_DTYPE == "Q8_CUDA")
        self.kv_quantized = self.kv_q8 or self.kv_q8_cuda

        # ── Quantizer & overflow guard ───────────────────────────────────
        self.quantizer = KVQuantizer().eval()
        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)

        # ── Per-layer output buffers ─────────────────────────────────────
        self.save_key   = [None] * num_layers
        self.save_value = [None] * num_layers
        if self.kv_quantized:
            self.save_k_scale = [None] * num_layers
            self.save_k_bias  = [None] * num_layers
            self.save_v_scale = [None] * num_layers
            self.save_v_bias  = [None] * num_layers

        # ── Fuse & reshape weights for efficient inference ───────────────
        self._replace_gelu_with_tanh_approximation(self.llm)
        self._fuse_weights(hidden_size)

    # ══════════════════════════════════════════════════════════════════════
    # Weight Fusion (runs once at init)
    # ══════════════════════════════════════════════════════════════════════
    def _fuse_weights(self, hidden_size):
        """
        Merge separate Q/K/V projections into a single QKV linear,
        absorb RMSNorm weights into projection matrices, and fuse
        gate/up projections for the MLP.
        """
        scale_factor = self.head_dim ** -0.25
        norm_factor = hidden_size ** 0.5
        norm_factor_qk = self.head_dim ** 0.5

        with torch.no_grad():
            for layer in self.llm.model.layers:
                self._fuse_qkv_projection(layer, scale_factor, norm_factor, norm_factor_qk)
                self._fuse_gate_up_projection(layer, norm_factor)

            # Absorb final RMSNorm into lm_head
            final_norm_weight = self.llm.model.norm.weight.unsqueeze(0) * norm_factor
            self.llm.lm_head.weight.mul_(final_norm_weight)
            del self.llm.model.norm

    def _fuse_qkv_projection(self, layer, scale_factor, norm_factor, norm_factor_qk):
        """Fuse Q, K, V projections and absorb input LayerNorm + QK norms."""
        attn = layer.self_attn
        q_proj, k_proj, v_proj = attn.q_proj, attn.k_proj, attn.v_proj

        # ── Create merged QKV linear ─────────────────────────────────
        in_features = int(q_proj.in_features)
        out_features = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
        has_bias = any(p.bias is not None for p in (q_proj, k_proj, v_proj))

        qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
        qkv.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))

        if has_bias:

            def _get_bias(proj):
                return proj.bias if proj.bias is not None else torch.zeros(proj.out_features, dtype=qkv.weight.dtype)

            qkv.bias.copy_(torch.cat([_get_bias(q_proj), _get_bias(k_proj), _get_bias(v_proj)], dim=0))

        # Store split dimensions for later use
        attn.q_out_features = int(q_proj.out_features)
        attn.k_out_features = int(k_proj.out_features)
        attn.v_out_features = int(v_proj.out_features)
        attn.qkv_in_features = in_features

        del attn.q_proj, attn.k_proj, attn.v_proj

        # ── Fuse QK norms (absorb scale factors) ────────────────────
        combined_scale = scale_factor * norm_factor_qk
        attn.q_norm.weight.mul_(combined_scale)
        attn.k_norm.weight.mul_(combined_scale)

        q_norm_repeated = attn.q_norm.weight.repeat(self.num_heads)
        k_norm_repeated = attn.k_norm.weight.repeat(self.num_key_value_heads)
        attn.qk_norm_weight = torch.nn.Parameter(torch.cat([q_norm_repeated, k_norm_repeated], dim=0).view(1, 1, 1, -1, self.head_dim))
        del attn.q_norm, attn.k_norm

        # ── Absorb input LayerNorm into QKV weights ─────────────────
        input_norm_weight = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
        qkv.weight.mul_(input_norm_weight)
        attn.qkv = qkv
        del layer.input_layernorm

    def _fuse_gate_up_projection(self, layer, norm_factor):
        """Fuse gate and up projections, absorbing post-attention LayerNorm."""
        post_norm_weight = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
        gate, up = layer.mlp.gate_proj, layer.mlp.up_proj

        gate_up = torch.nn.Linear(gate.in_features, gate.out_features + up.out_features, bias=False)
        gate_up.weight.copy_(torch.cat([
            gate.weight * post_norm_weight,
            up.weight * post_norm_weight
        ], dim=0))

        layer.mlp.gate_up_proj = gate_up
        del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

    # ══════════════════════════════════════════════════════════════════════
    # Utility Methods
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _replace_gelu_with_tanh_approximation(module):
        """Recursively replace exact GELU with tanh-approximated GELU for ONNX compatibility."""
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                LLM_MAIN._replace_gelu_with_tanh_approximation(child)

    def _rms_norm(self, x):
        """Apply modified RMS normalization (with optional overflow scaling)."""
        if PREVENT_F16_OVERFLOW:
            x = x * self.overflow_scale
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True)) # Note, not the .mean()

    def _rotate_half(self, x, batch_size):
        """Rotate the last dimension by swapping and negating halves (for RoPE).
           Using flip() is more efficient than split() + concat() in ONNX Runtime.
        """
        x = x.view(batch_size, -1, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(batch_size, -1, 1, self.qk_heads, self.head_dim)

    def forward(self, *all_inputs):
        hidden_states      = all_inputs[-4]
        rotary_pos_emb_cos = all_inputs[-3]
        rotary_pos_emb_sin = all_inputs[-2]
        attention_mask     = all_inputs[-1]
        batch_size         = hidden_states.shape[0]

        for i, layer in enumerate(self.llm.model.layers):

            # ── Self-Attention ───────────────────────────────────────
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)

            # Fused QKV projection & reshape
            qkv = layer.self_attn.qkv(hidden_states)
            qkv = qkv.reshape(batch_size, -1, 1, self.qk_heads + self.num_key_value_heads, self.head_dim)
            qk, v = torch.split(qkv, [self.qk_heads, self.num_key_value_heads], dim=-2)

            # QK normalization & rotary embedding
            qk = self._rms_norm(qk) * layer.self_attn.qk_norm_weight
            qk_rot = qk * rotary_pos_emb_cos + self._rotate_half(qk, batch_size) * rotary_pos_emb_sin

            # Split into query and key, reshape query for GQA
            q, k = torch.split(qk_rot, [self.num_heads, self.num_key_value_heads], dim=-2)
            q = q.reshape(batch_size, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q = q.permute(0, 2, 3, 1, 4)

            # Optional FP16 cast for KV
            if self.kv_f16:
                k = k.half()
                v = v.half()

            # Transpose K and V into cache layout
            k = k.permute(0, 3, 2, 4, 1)
            v = v.transpose(1, 3)

            # ── KV Cache Update & Attention Compute ──────────────────
            if self.kv_quantized:
                # Quantize current K/V and concatenate with cached values
                packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.head_dim_quarter)
                k   = torch.cat([all_inputs[i],                        packed_k], dim=-1)
                v   = torch.cat([all_inputs[i + self.num_layers],      packed_v], dim=-2)
                k_s = torch.cat([all_inputs[i + self.num_layers_2],    scale_k],  dim=-1)
                k_b = torch.cat([all_inputs[i + self.num_layers_3],    bias_k],   dim=-1)
                v_s = torch.cat([all_inputs[i + self.num_layers_4],    scale_v],  dim=-2)
                v_b = torch.cat([all_inputs[i + self.num_layers_5],    bias_v],   dim=-2)

                # Save updated caches
                self.save_key[i]     = k
                self.save_value[i]   = v
                self.save_k_scale[i] = k_s
                self.save_k_bias[i]  = k_b
                self.save_v_scale[i] = v_s
                self.save_v_bias[i]  = v_b

                # Upcast scale/bias if stored as FP16
                if USE_FLOAT16_SCALE_BIAS:
                    k_s = k_s.float()
                    k_b = k_b.float()
                    v_s = v_s.float()
                    v_b = v_b.float()

                # Unpack int32-packed Q8 for CUDA path
                if self.kv_q8_cuda:
                    k = self.quantizer.unpack_q8_cuda(k, -2, batch_size, self.num_key_value_heads, self.head_dim)
                    v = self.quantizer.unpack_q8_cuda(v, -1, batch_size, self.num_key_value_heads, self.head_dim)

                # Dequantized attention: attn = softmax((Q @ K) * k_scale + Q_sum * k_bias + mask) @ (V * v_scale + v_bias)
                attn_raw        = torch.matmul(q, k.float())
                attn_bias       = q.sum(dim=-1, keepdim=True) * k_b + attention_mask
                attn            = torch.addcmul(attn_bias, attn_raw, k_s)
                attn            = torch.softmax(attn, dim=-1)
                v_dequant       = torch.addcmul(v_b, v.float(), v_s)
                attn            = torch.matmul(attn, v_dequant)

            else:
                # Concatenate with cached K/V (F16 or F32)
                k = torch.cat((all_inputs[i], k), dim=-1)
                v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)
                self.save_key[i]   = k
                self.save_value[i] = v

                if self.kv_f16:
                    k = k.float()
                    v = v.float()

                attn = torch.matmul(q, k) + attention_mask
                attn = torch.softmax(attn, dim=-1)
                attn = torch.matmul(attn, v)

            # Output projection & residual
            attn = attn.permute(0, 3, 1, 2, 4).reshape(batch_size, -1, layer.self_attn.o_proj.in_features)
            hidden_states = residual + layer.self_attn.o_proj(attn)

            # ── Feed-Forward Network ─────────────────────────────────
            residual = hidden_states
            hidden_states = self._rms_norm(hidden_states)

            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = residual + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

        # ── Final Projection ─────────────────────────────────────────
        hidden_states = self._rms_norm(hidden_states[:, -1])
        logits = self.llm.lm_head(hidden_states)

        if self.kv_quantized:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_k_bias, *self.save_v_scale, *self.save_v_bias, logits
        return *self.save_key, *self.save_value, logits


if DO_EXPORT:
    print('Export start ...')
    with torch.inference_mode():

        # ══════════════════════════════════════════════════════════════════
        # Load Model & Extract Config
        # ══════════════════════════════════════════════════════════════════
        model = AutoModelForCausalLM.from_pretrained(download_path, dtype=torch.float32, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()

        num_layers   = model.config.num_hidden_layers
        num_heads    = model.config.num_attention_heads
        num_kv_heads = model.config.num_key_value_heads
        head_dim     = model.config.head_dim
        vocab_size   = model.model.vocab_size
        hidden_size  = model.model.embed_tokens.embedding_dim
        scale_dtype  = torch.float16 if USE_FLOAT16_SCALE_BIAS else torch.float32

        # ══════════════════════════════════════════════════════════════════
        # Build Dummy Tensors for Tracing
        # ══════════════════════════════════════════════════════════════════
        batch_size  = BEAM_SIZE
        ids_len     = torch.tensor([10], dtype=torch.int64)
        history_len = torch.tensor([0], dtype=torch.int64)
        kv_seq_len  = ids_len + history_len
        beam_size   = torch.tensor([BEAM_SIZE], dtype=torch.int64)
        logits      = torch.ones((BEAM_SIZE, vocab_size), dtype=torch.float32)

        # KV cache spec: list of (name, concat_dim)
        kv_specs = [('key', 4), ('value', 3)]

        if KV_QUANT_DTYPE == "F16":
            kv_dtype = torch.float16
        elif KV_QUANT_DTYPE in ("Q8", "Q8_CUDA"):
            kv_specs.extend([
                ('key_scale', 4), ('key_bias', 4),
                ('value_scale', 3), ('value_bias', 3)
            ])
            kv_dtype = torch.int32 if KV_QUANT_DTYPE == "Q8_CUDA" else torch.uint8
        else:
            kv_dtype = torch.float32

        # Determine KV tensor shapes based on quantization mode
        if KV_QUANT_DTYPE == "Q8_CUDA":
            k_head = head_dim // 4
            v_head = head_dim // 4
        else:
            k_head = head_dim
            v_head = head_dim

        kv_tensors = {
            'key':   torch.zeros((batch_size, num_kv_heads, 1, k_head, history_len), dtype=kv_dtype),
            'value': torch.zeros((batch_size, num_kv_heads, 1, history_len, v_head), dtype=kv_dtype)
        }
        if KV_QUANT_DTYPE in ("Q8", "Q8_CUDA"):
            kv_tensors.update({
                'key_scale':   torch.ones((batch_size, num_kv_heads, 1, 1, history_len), dtype=scale_dtype),
                'key_bias':    torch.ones((batch_size, num_kv_heads, 1, 1, history_len), dtype=scale_dtype),
                'value_scale': torch.ones((batch_size, num_kv_heads, 1, history_len, 1), dtype=scale_dtype),
                'value_bias':  torch.ones((batch_size, num_kv_heads, 1, history_len, 1), dtype=scale_dtype)
            })

        # ══════════════════════════════════════════════════════════════════
        # Helper: Build KV I/O names, tensors, and dynamic axes
        # ══════════════════════════════════════════════════════════════════
        def get_kv_io(tensors_dict, batch_axis='batch_size', seq_axis='history_len', out_seq_axis='kv_seq_len'):
            inputs, in_names, out_names, axes = [], [], [], {}
            for name, dim in kv_specs:
                tensor = tensors_dict[name]
                for i in range(num_layers):
                    in_n  = f'in_{name}_{i}'
                    out_n = f'out_{name}_{i}'
                    inputs.append(tensor)
                    in_names.append(in_n)
                    out_names.append(out_n)
                    axes[in_n]  = {0: batch_axis, dim: seq_axis}
                    axes[out_n] = {0: batch_axis, dim: out_seq_axis}
            return inputs, in_names, out_names, axes

        # ══════════════════════════════════════════════════════════════════
        # Export: LLM_Embed
        # ══════════════════════════════════════════════════════════════════
        input_ids = torch.ones((1, ids_len), dtype=torch.int32)
        torch.onnx.export(
            LLM_EMBED(model),
            (input_ids,),
            onnx_model_Embed,
            input_names=['input_ids'],
            output_names=['hidden_states'],
            dynamic_axes={
                'input_ids':     {0: 'batch', 1: 'ids_len'},
                'hidden_states': {0: 'batch', 1: 'ids_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del input_ids

        # ══════════════════════════════════════════════════════════════════
        # Export: Rotary + Mask (Prefill)
        # ══════════════════════════════════════════════════════════════════
        torch.onnx.export(
            ROTARY_MASK_PREFILL(model, MAX_SEQ_LEN),
            (ids_len, history_len),
            onnx_model_Rotary_Mask_Prefill,
            input_names=['ids_len', 'history_len'],
            output_names=['rotary_cos', 'rotary_sin', 'attention_mask', 'kv_seq_len'],
            dynamic_axes={
                'rotary_cos':     {1: 'ids_len'},
                'rotary_sin':     {1: 'ids_len'},
                'attention_mask': {3: 'ids_len', 4: 'kv_seq_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )

        # ══════════════════════════════════════════════════════════════════
        # Export: Rotary + Mask (Decode)
        # ══════════════════════════════════════════════════════════════════
        torch.onnx.export(
            ROTARY_MASK_DECODE(model, MAX_SEQ_LEN),
            (kv_seq_len,),
            onnx_model_Rotary_Mask_Decode,
            input_names=['kv_seq_len'],
            output_names=['rotary_cos', 'rotary_sin', 'kv_seq_len'],
            dynamic_axes=None,
            opset_version=OPSET,
            dynamo=False
        )

        # ══════════════════════════════════════════════════════════════════
        # Export: LLM_Main (Transformer Layers)
        # ══════════════════════════════════════════════════════════════════
        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors)

        hidden_states  = torch.ones((batch_size, ids_len, hidden_size), dtype=torch.float32)
        rotary_cos     = torch.zeros((1, ids_len, 1, 1, head_dim), dtype=torch.float32)
        rotary_sin     = rotary_cos
        attention_mask = torch.zeros((1, 1, 1, ids_len, kv_seq_len), dtype=torch.float32)

        all_inputs   = kv_ins + [hidden_states, rotary_cos, rotary_sin, attention_mask]
        input_names  = kv_in_names + ['hidden_states', 'rotary_cos', 'rotary_sin', 'attention_mask']
        output_names = kv_out_names + ['logits']
        dynamic_axes = {
            **kv_axes,
            'hidden_states':  {0: 'batch', 1: 'ids_len'},
            'logits':         {0: 'batch'},
            'rotary_cos':     {1: 'ids_len'},
            'rotary_sin':     {1: 'ids_len'},
            'attention_mask': {3: 'ids_len', 4: 'kv_seq_len'}
        }

        model_Main = LLM_MAIN(model, num_heads, num_kv_heads, head_dim, num_layers, hidden_size)
        del model

        torch.onnx.export(
            model_Main,
            tuple(all_inputs),
            onnx_model_Main,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            dynamo=False
        )
        del model_Main, hidden_states, attention_mask, all_inputs
        gc.collect()

        # ══════════════════════════════════════════════════════════════════
        # Export: Greedy Search
        # ══════════════════════════════════════════════════════════════════
        save_id_in = torch.zeros((BEAM_SIZE, 10), dtype=torch.int32)  # 10 is a dummy value.
        
        torch.onnx.export(
            GREEDY_SEARCH(),
            (logits, save_id_in),
            onnx_model_Greedy,
            input_names=['logits', 'save_id_in'],
            output_names=['max_logits_idx', 'save_id_out'],
            dynamic_axes={
                'logits':         {0: 'batch'},
                'save_id_in':     {0: 'batch', 1: 'history_len'},
                'save_id_out':    {0: 'batch', 1: 'history_len'},
                'max_logits_idx': {0: 'batch'}
            },
            opset_version=OPSET,
            dynamo=False
        )

        # ══════════════════════════════════════════════════════════════════
        # Export: First Beam Search
        # ══════════════════════════════════════════════════════════════════
        num_layers_beam = num_layers * len(kv_specs)
        # First beam uses single-batch KV (batch dim = 1)
        kv_tensors_Greedy = {k: v[[0]] for k, v in kv_tensors.items()}
        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors_Greedy)
        # Remove output axes — first beam outputs have variable batch, not tracked here
        kv_input_only_axes = {k: v for k, v in kv_axes.items() if k not in kv_out_names}
        
        torch.onnx.export(
            FIRST_BEAM_SEARCH(num_layers_beam),
            tuple(kv_ins + [logits[[0]], save_id_in, beam_size]),
            onnx_model_First_Beam,
            input_names=kv_in_names + ['logits', 'save_id_in', 'beam_size'],
            output_names=(
                ['out_' + n[3:] for n in kv_in_names] + ['save_id_out', 'top_beam_prob', 'top_beam_indices', 'max_logits_idx']
            ),
            dynamic_axes={
                **kv_input_only_axes,
                'logits':           {0: 'batch'},
                'save_id_in':       {0: 'batch', 1: 'history_len'},
                'top_beam_prob':    {0: 'batch'},
                'top_beam_indices': {0: 'batch'},
                'max_logits_idx':   {0: 'batch'},
                'batch_indices':    {0: 'batch'},
                'save_id_out':      {0: 'batch', 1: 'history_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )

        # ══════════════════════════════════════════════════════════════════
        # Export: Second Beam Search
        # ══════════════════════════════════════════════════════════════════
        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors)
        previous_prob = torch.zeros((BEAM_SIZE, 1), dtype=torch.float32)
        topK = torch.tensor([TOP_K], dtype=torch.int64)
        
        torch.onnx.export(
            SECOND_BEAM_SEARCH(num_layers_beam),
            tuple(kv_ins + [logits, save_id_in, previous_prob, beam_size, topK]),
            onnx_model_Second_Beam,
            input_names=kv_in_names + ['logits', 'save_id_in', 'previous_prob', 'beam_size', 'topK'],
            output_names=kv_out_names + ['save_id_out', 'top_beam_prob', 'top_beam_indices', 'max_logits_idx'],
            dynamic_axes={
                **kv_axes,
                'logits':           {0: 'batch'},
                'save_id_in':       {0: 'batch', 1: 'history_len'},
                'previous_prob':    {0: 'batch'},
                'save_id_out':      {0: 'batch', 1: 'history_len'},
                'top_beam_prob':    {0: 'batch'},
                'top_beam_indices': {0: 'batch'},
                'max_logits_idx':   {0: 'batch'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del kv_tensors_Greedy, previous_prob, topK

        # ══════════════════════════════════════════════════════════════════
        # Export: Apply Penalty
        # ══════════════════════════════════════════════════════════════════
        penalty_value  = torch.tensor([REPEAT_PENALTY], dtype=torch.float32)
        penalty_range = torch.tensor([PENALTY_RANGE], dtype=torch.int64)

        torch.onnx.export(
            APPLY_PENALTY(),
            (logits, save_id_in, penalty_value, penalty_range),
            onnx_model_Penalty,
            input_names=['logits_in', 'save_id_in', 'penalty_value', 'penalty_range'],
            output_names=['logits_out'],
            dynamic_axes={
                'logits_in':  {0: 'batch'},
                'save_id_in': {0: 'batch', 1: 'history_len'},
                'logits_out': {0: 'batch'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del save_id_in, penalty_value, penalty_range

        # ══════════════════════════════════════════════════════════════════
        # Export: Argmax
        # ══════════════════════════════════════════════════════════════════
        torch.onnx.export(
            ARGMAX(),
            (logits,),
            onnx_model_Argmax,
            input_names=['logits'],
            output_names=['max_logits_idx'],
            dynamic_axes={
                'logits':         {0: 'batch'},
                'max_logits_idx': {0: 'batch'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del logits
        gc.collect()
        
        # ══════════════════════════════════════════════════════════════════
        # Export: KV Slice
        # ══════════════════════════════════════════════════════════════════
        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors, batch_axis='batch_size', seq_axis='history_len', out_seq_axis='sliced_len')
        slice_start = torch.tensor([0], dtype=torch.int64)
        slice_end   = torch.tensor([5], dtype=torch.int64)  # 5 is a dummy value.

        torch.onnx.export(
            KV_SLICE(num_layers),
            tuple(kv_ins + [slice_start, slice_end]),
            onnx_model_KV_Slice,
            input_names=kv_in_names + ['slice_start', 'slice_end'],
            output_names=kv_out_names,
            dynamic_axes=kv_axes,
            opset_version=OPSET,
            dynamo=False
        )
        del slice_start, slice_end, kv_ins, kv_in_names, kv_out_names, kv_axes, kv_tensors
        gc.collect()

    print(
        '\nExport done!\n\n'
        'Start running the LLM by ONNXRuntime.\n'
        'Now loading . . . it could cost minutes.'
    )


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def bind_ort_in(binding, names, values):
    """Bind OrtValue inputs by name."""
    for name, val in zip(names, values):
        binding.bind_ortvalue_input(name, val)


def bind_ort_out(binding, names, device):
    """Bind outputs by name, letting ORT allocate on `device`."""
    for name in names:
        binding._iobinding.bind_output(name, device)


def create_ort_with_data(data, dtype, device, device_id):
    """Create an OrtValue from a Python list/scalar."""
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.array(data, dtype=dtype), device, device_id)


def create_ort_with_shape(shape, dtype, device, device_id):
    """Create a zero-filled OrtValue with the given shape."""
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape, dtype=dtype), device, device_id)


def create_session(model_path, _session_opts, _providers, _provider_options, _disabled_optimizers):
    """Create an ORT InferenceSession with standard options."""
    return onnxruntime.InferenceSession(
        model_path,
        sess_options=_session_opts,
        providers=_providers,
        provider_options=_provider_options,
        disabled_optimizers=_disabled_optimizers)


def get_in_names(session):
    return [x.name for x in session.get_inputs()]


def get_out_names(session):
    return [x.name for x in session.get_outputs()]


def run(session, binding):
    session.run_with_iobinding(binding, run_options=run_options)


# ══════════════════════════════════════════════════════════════════════════════
# ORT SESSION & RUNTIME OPTIONS
# ══════════════════════════════════════════════════════════════════════════════
session_opts = onnxruntime.SessionOptions()
run_options  = onnxruntime.RunOptions()

for opt in (session_opts, run_options):
    opt.log_severity_level  = 0 if ORT_LOG else 4
    opt.log_verbosity_level = 4

session_opts.inter_op_num_threads     = MAX_THREADS
session_opts.intra_op_num_threads     = MAX_THREADS
session_opts.execution_mode           = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

_session_configs = {
    'session.set_denormal_as_zero':                  '1',
    'session.intra_op.allow_spinning':               '1',
    'session.inter_op.allow_spinning':               '1',
    'session.enable_quant_qdq_cleanup':              '1',
    'session.qdq_matmulnbits_accuracy_level':        '2' if ORT_FP16 else '4',
    'session.use_device_allocator_for_initializers': '1',
    'session.graph_optimizations_loop_level':        '2',
    'optimization.enable_gelu_approximation':        '1',
    'optimization.minimal_build_optimizations':      '',
    'optimization.enable_cast_chain_elimination':    '1',
    'optimization.disable_specified_optimizers':
        'CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer' if ORT_FP16 else ''
}
for k, v in _session_configs.items():
    session_opts.add_session_config_entry(k, v)

run_options.add_run_config_entry('disable_synchronize_execution_providers', '0')

disabled_optimizers = ['CastFloat16Transformer', 'FuseFp16InitializerToFp32NodeTransformer'] if ORT_FP16 else None


# ══════════════════════════════════════════════════════════════════════════════
# EXECUTION PROVIDER CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_type':              'CPU',                 # [CPU, GPU, NPU, GPU.0, GPU.1]
        'precision':                'ACCURACY',            # [FP32, FP16, ACCURACY]
        'num_of_threads':           MAX_THREADS if MAX_THREADS != 0 else 8,
        'num_streams':              1,
        'enable_opencl_throttling': False,
        'enable_qdq_optimizer':     False,                 # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'disable_dynamic_shapes':   False
    }]
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                          DEVICE_ID,
        'gpu_mem_limit':                      24 * (1024 **3),    # 24GB
        'arena_extend_strategy':              'kNextPowerOfTwo',  # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
        'cudnn_conv_algo_search':             'EXHAUSTIVE',       # ["kNextPowerOfTwo", "kSameAsRequested"]
        'sdpa_kernel':                        '2',                # ["0", "1", "2"]
        'use_tf32':                           '1',
        'fuse_conv_bias':                     '0',          # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'cudnn_conv_use_max_workspace':       '1',
        'cudnn_conv1d_pad_to_nc1d':           '0',
        'tunable_op_enable':                  '0',
        'tunable_op_tuning_enable':           '0',
        'tunable_op_max_tuning_duration_ms':  10,
        'do_copy_in_default_stream':          '0',
        'enable_cuda_graph':                  '0',          # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'prefer_nhwc':                        '0',
        'enable_skip_layer_norm_strict_mode': '0',
        'use_ep_level_unified_stream':        '0'
    }]
    device_type      = 'cuda'
    _ort_device_type = C.OrtDevice.cuda()

elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                  DEVICE_ID,
        'performance_preference':     'high_performance',   # ["default", "high_performance", "minimum_power"] ; Default (Gpus first), HighPerformance (GPUs first), LowPower (NPUs first)
        'device_filter':              'gpu',                # [gpu, npu, any],
        'disable_metacommands':       'false',              # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'enable_graph_capture':       'false',              # Disable to avoid loading error with some models; can be re-enabled if not an issue
        'enable_graph_serialization': 'false'               # Disable to avoid loading error with some models; can be re-enabled if not an issue
    }]
    device_type      = 'dml'
    _ort_device_type = C.OrtDevice.dml()

else:
    provider_options = None
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

packed_settings = {
    "_session_opts":        session_opts,
    "_providers":           ORT_Accelerate_Providers,
    "_provider_options":    provider_options,
    "_disabled_optimizers": disabled_optimizers
}

_ort_device_type = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)
kv_device = 'cpu' if 'dml' in device_type else device_type


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ONNX SESSIONS
# ══════════════════════════════════════════════════════════════════════════════
# --- Embed ---
ort_session_Embed = create_session(onnx_model_Embed, **packed_settings)
binding_Embed     = ort_session_Embed.io_binding()
in_name_Embed     = get_in_names(ort_session_Embed)[0]
out_name_Embed    = get_out_names(ort_session_Embed)[0]

# --- Rotary + Mask (Prefill) ---
ort_session_Rotary_Mask_Prefill = create_session(onnx_model_Rotary_Mask_Prefill, **packed_settings)
binding_Rotary_Mask_Prefill     = ort_session_Rotary_Mask_Prefill.io_binding()
in_name_Rotary_Mask_Prefill     = get_in_names(ort_session_Rotary_Mask_Prefill)
out_name_Rotary_Mask_Prefill    = get_out_names(ort_session_Rotary_Mask_Prefill)

# --- Rotary + Mask (Decode) ---
ort_session_Rotary_Mask_Decode = create_session(onnx_model_Rotary_Mask_Decode, **packed_settings)
binding_Rotary_Mask_Decode     = ort_session_Rotary_Mask_Decode.io_binding()
in_name_Rotary_Mask_Decode     = get_in_names(ort_session_Rotary_Mask_Decode)[0]
out_name_Rotary_Mask_Decode    = get_out_names(ort_session_Rotary_Mask_Decode)

# --- Main ---
ort_session_Main = create_session(onnx_model_Main, **packed_settings)
binding_Main     = ort_session_Main.io_binding()
print(f"\nUsable Providers: {ort_session_Main.get_providers()}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN MODEL METADATA & INDEX OFFSETS
# ══════════════════════════════════════════════════════════════════════════════
in_name_Main           = get_in_names(ort_session_Main)
out_name_Main          = get_out_names(ort_session_Main)
amount_of_outputs_Main = len(out_name_Main)
num_keys_values        = amount_of_outputs_Main - 1

# Derived index offsets for accessing beam/greedy extra inputs
num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
num_keys_values_plus_3 = num_keys_values + 3
num_keys_values_plus_4 = num_keys_values + 4

# Partitioned name lists
in_name_Main_parts   = in_name_Main[:num_keys_values]
out_name_Main_kv     = out_name_Main[:num_keys_values]
out_name_Main_logits = out_name_Main[num_keys_values]

# Dtype introspection
kv_dtype_str             = ort_session_Main._inputs_meta[0].type
hidden_states_dtype_Main = np.float16 if 'float16' in ort_session_Main._inputs_meta[num_keys_values].type else np.float32
vocab_size               = ort_session_Main._outputs_meta[num_keys_values].shape[1]

_logits_out_meta  = ort_session_Main._outputs_meta[num_keys_values]
_logits_out_dtype = np.float16 if 'float16' in _logits_out_meta.type else np.float32


# ══════════════════════════════════════════════════════════════════════════════
# KV CACHE SETUP
# ══════════════════════════════════════════════════════════════════════════════
_meta = ort_session_Main._inputs_meta

if 'uint8' in kv_dtype_str or 'int32' in kv_dtype_str:
    kv_dtype_Main = np.int32 if 'int32' in kv_dtype_str else np.uint8
    num_layers    = num_keys_values // 6
    scale_dtype   = np.float16 if 'float16' in _meta[num_layers * 2].type else np.float32
    k_scales      = create_ort_with_shape((1, _meta[0].shape[1],          1, 1, 0), scale_dtype, kv_device, DEVICE_ID)
    k_biases      = create_ort_with_shape((1, _meta[0].shape[1],          1, 1, 0), scale_dtype, kv_device, DEVICE_ID)
    v_scales      = create_ort_with_shape((1, _meta[num_layers].shape[1], 1, 0, 1), scale_dtype, kv_device, DEVICE_ID)
    v_biases      = create_ort_with_shape((1, _meta[num_layers].shape[1], 1, 0, 1), scale_dtype, kv_device, DEVICE_ID)
else:
    kv_dtype_Main = np.float16 if 'float16' in kv_dtype_str else np.float32
    num_layers    = num_keys_values // 2
    k_scales      = None

past_keys_Main   = create_ort_with_shape((1, _meta[0].shape[1],          1, _meta[0].shape[3],          0), kv_dtype_Main, kv_device, DEVICE_ID)
past_values_Main = create_ort_with_shape((1, _meta[num_layers].shape[1], 1, 0, _meta[num_layers].shape[4]), kv_dtype_Main, kv_device, DEVICE_ID)


# ══════════════════════════════════════════════════════════════════════════════
# DECODING STRATEGY VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
if USE_BEAM_SEARCH and TOP_K < BEAM_SIZE:
    TOP_K = BEAM_SIZE

if TOP_K < 2 or BEAM_SIZE < 2:
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")

if not USE_BEAM_SEARCH:
    BEAM_SIZE = 1

USE_PENALTY = (REPEAT_PENALTY != 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZER & STOP TOKENS & PROMPT
# ══════════════════════════════════════════════════════════════════════════════
tokenizer = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)

STOP_TOKEN_SET = set(STOP_TOKEN)

prompt = (
    f'<|im_start|>user\n{TEST_QUERY}<|im_end|>\n<|im_start|>assistant\n'
    if TEST_THINK_MODE else
    f'<|im_start|>user\n{TEST_QUERY}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
)

tokens      = tokenizer(prompt, return_tensors='np')['input_ids'].astype(np.int32)
num_prefill = tokens.shape[-1]


# ══════════════════════════════════════════════════════════════════════════════
# SHARED ORTVALUE BUFFERS
# ══════════════════════════════════════════════════════════════════════════════
_rotary_meta = ort_session_Rotary_Mask_Decode._outputs_meta

# --- Input OrtValues ---
input_ids        = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, DEVICE_ID)
ids_len          = create_ort_with_data([num_prefill], np.int64,  device_type, DEVICE_ID)
init_history_len = create_ort_with_data([0],           np.int64,  device_type, DEVICE_ID)
topK             = create_ort_with_data([TOP_K],       np.int64,  device_type, DEVICE_ID)
beam_size        = create_ort_with_data([BEAM_SIZE],   np.int64,  device_type, DEVICE_ID)

# --- Decode-phase placeholder buffers (reused every step) ---
attention_mask_buf = create_ort_with_shape((1, 1, 1, 1, 1),                                 hidden_states_dtype_Main, device_type, DEVICE_ID)
rotary_cos_buf     = create_ort_with_shape((1, 1, 1, 1, _rotary_meta[0].shape[4]),          hidden_states_dtype_Main, device_type, DEVICE_ID)
rotary_sin_buf     = create_ort_with_shape((1, 1, 1, 1, _rotary_meta[1].shape[4]),          hidden_states_dtype_Main, device_type, DEVICE_ID)
hidden_states_buf  = create_ort_with_shape((BEAM_SIZE, 1, _meta[num_keys_values].shape[2]), hidden_states_dtype_Main, device_type, DEVICE_ID)
save_id_buf        = create_ort_with_shape((BEAM_SIZE, 0),                                  np.int32,                 device_type, DEVICE_ID)

# --- Logits & token-index buffers ---
prefill_logits_buf = create_ort_with_shape((1, vocab_size),         _logits_out_dtype, device_type, DEVICE_ID)
decode_logits_buf  = create_ort_with_shape((BEAM_SIZE, vocab_size), _logits_out_dtype, device_type, DEVICE_ID)
max_idx_buf        = create_ort_with_shape((1, 1),                  np.int32,          device_type, DEVICE_ID)


# ══════════════════════════════════════════════════════════════════════════════
# DECODE HEAD SESSIONS (Beam Search OR Greedy/Argmax)
# ══════════════════════════════════════════════════════════════════════════════
if USE_BEAM_SEARCH:
    print("\nBeam Search does not display immediate decoding results...")

    # --- First Beam ---
    ort_session_First_Beam    = create_session(onnx_model_First_Beam, **packed_settings)
    binding_First_Beam        = ort_session_First_Beam.io_binding()
    in_name_First_Beam        = get_in_names(ort_session_First_Beam)
    out_name_First_Beam       = get_out_names(ort_session_First_Beam)
    in_name_First_Beam_parts  = in_name_First_Beam[:num_keys_values_plus_1]
    out_name_First_Beam_parts = out_name_First_Beam[:num_keys_values_plus_1]

    # --- Second Beam ---
    ort_session_Second_Beam    = create_session(onnx_model_Second_Beam, **packed_settings)
    binding_Second_Beam        = ort_session_Second_Beam.io_binding()
    in_name_Second_Beam        = get_in_names(ort_session_Second_Beam)
    out_name_Second_Beam       = get_out_names(ort_session_Second_Beam)
    in_name_Second_Beam_parts  = in_name_Second_Beam[:num_keys_values_plus_1]
    out_name_Second_Beam_parts = out_name_Second_Beam[:num_keys_values_plus_1]

    # --- Beam-specific buffers ---
    beam_ids_buf    = create_ort_with_shape((BEAM_SIZE, 1), np.int32,                 device_type, DEVICE_ID)
    beam_score_buf  = create_ort_with_shape((BEAM_SIZE, 1), hidden_states_dtype_Main, device_type, DEVICE_ID)

    # --- Static beam bindings ---
    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[num_keys_values_plus_1], save_id_buf)
    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[num_keys_values_plus_2], beam_size)
    binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_plus_3], beam_size)
    binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_plus_4], topK)

else:
    # --- Greedy ---
    ort_session_Greedy = create_session(onnx_model_Greedy, **packed_settings)
    binding_Greedy     = ort_session_Greedy.io_binding()
    in_name_Greedy     = get_in_names(ort_session_Greedy)
    out_name_Greedy    = get_out_names(ort_session_Greedy)
    binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id_buf)

    # --- Argmax ---
    ort_session_Argmax = create_session(onnx_model_Argmax, **packed_settings)
    binding_Argmax     = ort_session_Argmax.io_binding()
    in_name_Argmax     = get_in_names(ort_session_Argmax)[0]
    out_name_Argmax    = get_out_names(ort_session_Argmax)[0]
    save_id_numpy      = np.zeros(MAX_SEQ_LEN, dtype=np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# PENALTY SESSION (optional)
# ══════════════════════════════════════════════════════════════════════════════
if USE_PENALTY:
    ort_session_Penalty = create_session(onnx_model_Penalty, **packed_settings)
    binding_Penalty     = ort_session_Penalty.io_binding()
    in_name_Penalty     = get_in_names(ort_session_Penalty)
    out_name_Penalty    = get_out_names(ort_session_Penalty)[0]

    penalty_dtype = np.float16 if 'float16' in ort_session_Penalty._inputs_meta[2].type else np.float32
    penalty_value = create_ort_with_data([REPEAT_PENALTY], penalty_dtype, device_type, DEVICE_ID)
    penalty_range = create_ort_with_data([PENALTY_RANGE],  np.int64,      device_type, DEVICE_ID)

    binding_Penalty.bind_ortvalue_input(in_name_Penalty[2], penalty_value)
    binding_Penalty.bind_ortvalue_input(in_name_Penalty[3], penalty_range)


# ══════════════════════════════════════════════════════════════════════════════
# PREFILL PHASE
# ══════════════════════════════════════════════════════════════════════════════
is_prefill_step = True
prefill_start_time = time.time()

# --- Step 1: Embed the input tokens ---
binding_Embed.bind_ortvalue_input(in_name_Embed, input_ids)
bind_ort_out(binding_Embed, [out_name_Embed], _ort_device_type)
run(ort_session_Embed, binding_Embed)
hidden_states = binding_Embed.get_outputs()[0]

# Pre-bind Embed input for decode phase (will read from max_idx_buf)
binding_Embed.bind_ortvalue_input(in_name_Embed, max_idx_buf)

# --- Step 2: Compute rotary embeddings & causal mask (prefill) ---
binding_Rotary_Mask_Prefill.bind_ortvalue_input(in_name_Rotary_Mask_Prefill[0], ids_len)
binding_Rotary_Mask_Prefill.bind_ortvalue_input(in_name_Rotary_Mask_Prefill[1], init_history_len)
bind_ort_out(binding_Rotary_Mask_Prefill, out_name_Rotary_Mask_Prefill, _ort_device_type)
run(ort_session_Rotary_Mask_Prefill, binding_Rotary_Mask_Prefill)
rotary_cos, rotary_sin, attention_mask, kv_seq_len = binding_Rotary_Mask_Prefill.get_outputs()

# --- Step 3: Pre-bind decode rotary outputs (reused every decode step) ---
binding_Rotary_Mask_Decode.bind_ortvalue_input(in_name_Rotary_Mask_Decode, kv_seq_len)
binding_Rotary_Mask_Decode.bind_ortvalue_output(out_name_Rotary_Mask_Decode[0], rotary_cos_buf)
binding_Rotary_Mask_Decode.bind_ortvalue_output(out_name_Rotary_Mask_Decode[1], rotary_sin_buf)
binding_Rotary_Mask_Decode.bind_ortvalue_output(out_name_Rotary_Mask_Decode[2], kv_seq_len)

# --- Step 4: Bind Main model inputs — non-KV (hidden_states, rotary, mask) ---
binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values],        hidden_states)
binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_1], rotary_cos)
binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_2], rotary_sin)
binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_3], attention_mask)

# --- Step 5: Bind Main model inputs — empty KV cache (keys, values, optional scales/biases) ---
i = 0
for _ in range(num_layers):
    binding_Main.bind_ortvalue_input(in_name_Main[i], past_keys_Main)
    i += 1
for _ in range(num_layers):
    binding_Main.bind_ortvalue_input(in_name_Main[i], past_values_Main)
    i += 1
if k_scales is not None:
    for ortval in (k_scales, k_biases, v_scales, v_biases):
        for _ in range(num_layers):
            binding_Main.bind_ortvalue_input(in_name_Main[i], ortval)
            i += 1

# --- Step 6: Bind Main model outputs ---
bind_ort_out(binding_Main, out_name_Main_kv, _ort_device_type)
binding_Main.bind_ortvalue_output(out_name_Main_logits, prefill_logits_buf)

# --- Step 7: Bind penalty inputs/outputs to prefill logits buffer ---
if USE_PENALTY:
    binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], prefill_logits_buf)
    binding_Penalty.bind_ortvalue_output(out_name_Penalty,  prefill_logits_buf)

# --- Step 8: Bind decode head inputs/outputs to prefill logits buffer ---
if USE_BEAM_SEARCH:
    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[num_keys_values], prefill_logits_buf)
elif USE_PENALTY:
    binding_Greedy.bind_ortvalue_input(in_name_Greedy[0],   prefill_logits_buf)
    binding_Greedy.bind_ortvalue_output(out_name_Greedy[0], max_idx_buf)
else:
    binding_Argmax.bind_ortvalue_input(in_name_Argmax,   prefill_logits_buf)
    binding_Argmax.bind_ortvalue_output(out_name_Argmax, max_idx_buf)


# ══════════════════════════════════════════════════════════════════════════════
# DECODE LOOP
# ══════════════════════════════════════════════════════════════════════════════
print(f'\nTest Question: {TEST_QUERY}\nLLM Answering:')

num_decode     = 0
generate_limit = MAX_SEQ_LEN - num_prefill

while num_decode < generate_limit:

    # ── 1. Run Main Model ────────────────────────────────────────────────
    run(ort_session_Main, binding_Main)
    outputs_Main = binding_Main.get_outputs()

    # ── 2. Apply Repetition Penalty (if enabled and enough tokens) ───────
    if USE_PENALTY and num_decode >= PENALTY_RANGE:
        binding_Penalty.bind_ortvalue_input(in_name_Penalty[1], save_id)
        run(ort_session_Penalty, binding_Penalty)

    # ── 3. Token Selection ───────────────────────────────────────────────
    if USE_BEAM_SEARCH:
        # ── 3a. Beam Search ─────────────────────────────────────────────
        if is_prefill_step:
            # First beam step: expand single-beam KV into BEAM_SIZE beams
            bind_ort_in(binding_First_Beam, in_name_First_Beam_parts, outputs_Main)
            bind_ort_out(binding_First_Beam, out_name_First_Beam_parts, _ort_device_type)
            binding_First_Beam.bind_ortvalue_output(out_name_First_Beam[num_keys_values_plus_1], beam_score_buf)
            binding_First_Beam.bind_ortvalue_output(out_name_First_Beam[num_keys_values_plus_2], beam_ids_buf)
            binding_First_Beam.bind_ortvalue_output(out_name_First_Beam[num_keys_values_plus_3], max_idx_buf)
            run(ort_session_First_Beam, binding_First_Beam)
            outputs_Beam = binding_First_Beam.get_outputs()
        else:
            # Subsequent beam steps: prune + expand
            bind_ort_in(binding_Second_Beam, in_name_Second_Beam_parts, outputs_Main)
            bind_ort_out(binding_Second_Beam, out_name_Second_Beam_parts, _ort_device_type)
            if num_decode < 2:
                binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_plus_2],   beam_score_buf)
                binding_Second_Beam.bind_ortvalue_output(out_name_Second_Beam[num_keys_values_plus_1], beam_score_buf)
                binding_Second_Beam.bind_ortvalue_output(out_name_Second_Beam[num_keys_values_plus_2], beam_ids_buf)
                binding_Second_Beam.bind_ortvalue_output(out_name_Second_Beam[num_keys_values_plus_3], max_idx_buf)
            run(ort_session_Second_Beam, binding_Second_Beam)
            outputs_Beam = binding_Second_Beam.get_outputs()

        # Stop-token check
        max_logits_idx = max_idx_buf.numpy().flat[0]
        if max_logits_idx in STOP_TOKEN_SET:
            break

        # Feed beam KV + save_id back into Main for next step
        save_id = outputs_Beam[num_keys_values]
        bind_ort_in(binding_Main, in_name_Main_parts, outputs_Beam)
        binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_plus_1], save_id)

    else:
        # ── 3b. Greedy / Argmax ─────────────────────────────────────────
        if USE_PENALTY:
            binding_Greedy._iobinding.bind_output(out_name_Greedy[1], _ort_device_type)
            run(ort_session_Greedy, binding_Greedy)
            save_id = binding_Greedy.get_outputs()[1]
        else:
            run(ort_session_Argmax, binding_Argmax)

        # Stop-token check
        max_logits_idx = max_idx_buf.numpy().flat[0]
        if max_logits_idx in STOP_TOKEN_SET:
            break

        # Track generated token IDs
        if USE_PENALTY:
            binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id)
        else:
            save_id_numpy[num_decode] = max_logits_idx

        # Feed greedy KV outputs back into Main
        bind_ort_in(binding_Main, in_name_Main_parts, outputs_Main)

        # Streaming print
        print(tokenizer.decode(max_logits_idx), end="", flush=True)

    # ── 4. Re-bind Main KV outputs (ORT allocates fresh each step) ───────
    bind_ort_out(binding_Main, out_name_Main_kv, _ort_device_type)

    # ── 5. Transition: prefill → decode (executes once) ──────────────────
    if is_prefill_step:

        # Switch Main to decode-sized non-KV inputs
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values],        hidden_states_buf)
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_1], rotary_cos_buf)
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_2], rotary_sin_buf)
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_3], attention_mask_buf)
        binding_Main.bind_ortvalue_output(out_name_Main_logits,                decode_logits_buf)

        # Switch Embed to write into decode hidden_states buffer
        binding_Embed.bind_ortvalue_output(out_name_Embed, hidden_states_buf)

        # Switch Penalty to decode logits buffer
        if USE_PENALTY:
            binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], decode_logits_buf)
            binding_Penalty.bind_ortvalue_output(out_name_Penalty, decode_logits_buf)

        # Switch decode head to decode logits buffer
        if USE_BEAM_SEARCH:
            binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values], decode_logits_buf)
            binding_Embed.bind_ortvalue_input(in_name_Embed, beam_ids_buf)
        elif USE_PENALTY:
            binding_Greedy.bind_ortvalue_input(in_name_Greedy[0], decode_logits_buf)
        else:
            binding_Argmax.bind_ortvalue_input(in_name_Argmax, decode_logits_buf)

        is_prefill_step = False
        
        # Record prefill time and start decode timer
        decode_start_time = time.time()
        prefill_elapsed = decode_start_time - prefill_start_time

    # ── 6. Prepare next step: Embed + Rotary ─────────────────────────────
    run(ort_session_Embed, binding_Embed)
    run(ort_session_Rotary_Mask_Decode, binding_Rotary_Mask_Decode)
    num_decode += 1


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS
# ══════════════════════════════════════════════════════════════════════════════
decode_end_time = time.time()

# Handle edge case where generation stopped at prefill (0 decode tokens after first)
if num_decode <= 1:
    # Only prefill happened (or single token generated during prefill step)
    prefill_elapsed = 0.0
    decode_elapsed = 0.0
else:
    decode_elapsed = decode_end_time - decode_start_time

total_elapsed = decode_end_time - prefill_start_time

# Prefill speed: tokens processed per second
prefill_tokens_per_second = num_prefill / prefill_elapsed if prefill_elapsed > 0 else 0.0

# Decode speed: tokens generated per second (excluding the first token from prefill)
decode_tokens_per_second = num_decode / decode_elapsed if decode_elapsed > 0 else 0.0

# Overall speed
overall_tokens_per_second = (num_decode + 1) / total_elapsed if total_elapsed > 0 else 0.0

if USE_PENALTY or USE_BEAM_SEARCH:
    result = tokenizer.decode(save_id.numpy()[0, :num_decode], skip_special_tokens=True)
else:
    result = tokenizer.decode(save_id_numpy[:num_decode], skip_special_tokens=True)

print(
    f"\n\n{'─' * 56}\n"
    f"  📝 Generated Output\n"
    f"{'─' * 56}\n"
    f"{result}\n"
    f"{'─' * 56}\n\n"
    f"  ⚡ Performance Summary\n"
    f"{'─' * 56}\n"
    f"  {'Phase':<12} {'Speed':>14} {'Tokens':>8} {'Time':>10}\n"
    f"  {'─' * 48}\n"
    f"  {'Prefill':<12} {prefill_tokens_per_second:>10.2f} t/s {num_prefill:>8d} {prefill_elapsed:>8.3f}s\n"
    f"  {'Decode':<12} {decode_tokens_per_second:>10.2f} t/s {num_decode:>8d} {decode_elapsed:>8.3f}s\n"
    f"  {'─' * 48}\n"
    f"  {'Overall':<12} {overall_tokens_per_second:>10.2f} t/s {num_decode:>8d} {total_elapsed:>8.3f}s\n"
    f"{'─' * 56}\n"
)


