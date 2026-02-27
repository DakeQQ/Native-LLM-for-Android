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
onnx_model_Rotary_Mask_Prefill  = r'/home/DakeQQ/Downloads/Qwen_ONNX/Rotary_Mask_Prefill.onnx'
onnx_model_Rotary_Mask_Decode   = r'/home/DakeQQ/Downloads/Qwen_ONNX/Rotary_Mask_Decode.onnx'
onnx_model_Greedy               = r'/home/DakeQQ/Downloads/Qwen_ONNX/Greedy_Search.onnx'
onnx_model_First_Beam           = r'/home/DakeQQ/Downloads/Qwen_ONNX/First_Beam_Search.onnx'
onnx_model_Second_Beam          = r'/home/DakeQQ/Downloads/Qwen_ONNX/Second_Beam_Search.onnx'
onnx_model_Penalty              = r'/home/DakeQQ/Downloads/Qwen_ONNX/Apply_Penalty.onnx'
onnx_model_Argmax               = r'/home/DakeQQ/Downloads/Qwen_ONNX/Argmax.onnx'


# Test input
TEST_THINK_MODE = True
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
ORT_FP16 = False                    # Set to True for FP16 ONNX Runtime settings. For CPUs, this requires ARM64-v8.2a or newer.
ORT_Accelerate_Providers = []       # ORT execution providers; ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
MAX_THREADS = 0                     # 0 = auto
DEVICE_ID = 0                       # Device ID for GPU
OPSET = 17                          # ONNX opset version


class GREEDY_SEARCH(torch.nn.Module):
    def __init__(self):
        super(GREEDY_SEARCH, self).__init__()

    def forward(self, logits, save_id):
        max_logits_idx = torch.argmax(logits, dim=-1, keepdim=True).int()
        save_id = torch.cat([save_id, max_logits_idx], dim=-1)
        return max_logits_idx, save_id


class FIRST_BEAM_SEARCH(torch.nn.Module):
    def __init__(self, total_layers):
        super(FIRST_BEAM_SEARCH, self).__init__()
        self.total_layers = total_layers
        self.save_keys_values = [None] * self.total_layers

    def forward(self, *all_inputs):
        logits = all_inputs[-3]
        save_id = all_inputs[-2]
        beam_size = all_inputs[-1]
        row_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_beam_logits, top_beam_indices = torch.topk(logits, dim=-1, k=beam_size, sorted=False, largest=True)
        top_beam_prob = top_beam_logits - row_logsumexp
        for i in range(self.total_layers):
            self.save_keys_values[i] = all_inputs[i].repeat(beam_size, *([1] * (all_inputs[i].dim() - 1)))
        top_beam_indices = top_beam_indices.transpose(0, 1)
        top_beam_indices = top_beam_indices.int()
        save_id = torch.cat([save_id, top_beam_indices], dim=-1)
        max_logits_idx = top_beam_indices[0]
        return *self.save_keys_values, top_beam_indices, save_id, top_beam_prob.transpose(0, 1), max_logits_idx


class SECOND_BEAM_SEARCH(torch.nn.Module):
    def __init__(self, total_layers):
        super(SECOND_BEAM_SEARCH, self).__init__()
        self.total_layers = total_layers
        self.save_keys_values = [None] * self.total_layers

    def forward(self, *all_inputs):
        logits = all_inputs[-5]
        save_id = all_inputs[-4]
        previous_prob = all_inputs[-3]
        beam_size = all_inputs[-2]
        topK = all_inputs[-1]
        row_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        top_k_logits, top_k_indices = torch.topk(logits, k=topK, dim=-1, largest=True, sorted=False)
        top_k_prob = top_k_logits - row_logsumexp
        current_prob = (top_k_prob + previous_prob).view(-1)
        top_beam_prob, top_beam_indices = torch.topk(current_prob, k=beam_size, dim=-1, largest=True, sorted=False)
        beam_index = top_beam_indices // topK
        top_beam_indices = top_k_indices.view(-1)[top_beam_indices]
        for i in range(self.total_layers):
            self.save_keys_values[i] = torch.index_select(all_inputs[i], dim=0, index=beam_index)
        gathered_save_id = torch.index_select(save_id, dim=0, index=beam_index)
        top_beam_indices = top_beam_indices.unsqueeze(-1)
        top_beam_indices = top_beam_indices.int()
        max_logits_idx = top_beam_indices[0]
        save_id = torch.cat([gathered_save_id, top_beam_indices], dim=-1)
        return *self.save_keys_values, top_beam_indices, save_id, top_beam_prob.unsqueeze(-1), max_logits_idx


class APPLY_PENALTY(torch.nn.Module):
    def __init__(self):
        super(APPLY_PENALTY, self).__init__()

    def forward(self, logits, save_id, penalty_value, penality_range):
        target_indices = save_id[:, -penality_range:].long()
        logits = logits.scatter(1, target_indices, logits.gather(1, target_indices) * penalty_value)
        return logits
    
    
class ARGMAX(torch.nn.Module):
    def __init__(self):
        super(ARGMAX, self).__init__()
        pass

    def forward(self, logits):
        max_logits_idx = torch.argmax(logits, dim=-1, keepdim=True)
        return max_logits_idx.int()


class LLM_EMBED(torch.nn.Module):
    def __init__(self, llm):
        super(LLM_EMBED, self).__init__()
        self.embed_tokens = llm.model.embed_tokens.float()

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)
    
    
class ROTARY_MASK_PREFILL(torch.nn.Module):
    def __init__(self, llm, max_seq_len):
        super(ROTARY_MASK_PREFILL, self).__init__()
        self.attention_mask = (1 - torch.tril(torch.ones([1, 1, 1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128

        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq = llm.model.rotary_emb.inv_freq
        idx_theta = (position_ids * inv_freq).unsqueeze(1).unsqueeze(1).unsqueeze(0)
        cos = torch.cos(idx_theta)
        sin = torch.sin(idx_theta)
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half(), persistent=False)

    def forward(self, ids_len, history_len):
        kv_seq_len = ids_len + history_len
        rotary_pos_emb_cos = self.cos_rotary_pos_emb[:, history_len:kv_seq_len].float()
        rotary_pos_emb_sin = self.sin_rotary_pos_emb[:, history_len:kv_seq_len].float()
        attention_mask = self.attention_mask[..., :ids_len, :kv_seq_len].float()
        return rotary_pos_emb_cos, rotary_pos_emb_sin, attention_mask, kv_seq_len
    

class ROTARY_MASK_DECODE(torch.nn.Module):
    def __init__(self, llm, max_seq_len):
        super(ROTARY_MASK_DECODE, self).__init__()
        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq = llm.model.rotary_emb.inv_freq
        idx_theta = (position_ids * inv_freq).unsqueeze(1).unsqueeze(1).unsqueeze(0)
        cos = torch.cos(idx_theta)
        sin = torch.sin(idx_theta)
        self.register_buffer("cos_rotary_pos_emb", torch.cat([cos, cos], dim=-1).half(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", torch.cat([-sin, sin], dim=-1).half(), persistent=False)

    def forward(self, kv_seq_len):
        kv_seq_len_next = kv_seq_len + 1
        rotary_pos_emb_cos = self.cos_rotary_pos_emb[:, kv_seq_len_next].float()
        rotary_pos_emb_sin = self.sin_rotary_pos_emb[:, kv_seq_len_next].float()
        return rotary_pos_emb_cos, rotary_pos_emb_sin, kv_seq_len_next


class KVQuantizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qmax = 255.0
        self.register_buffer("inv_qmax", torch.tensor([1.0 / self.qmax], dtype=torch.float32).view(1, 1, 1, 1, -1))
        self.register_buffer("_256", torch.tensor([256], dtype=torch.int32).view(1, 1, 1, 1, -1))
        self.register_buffer("_128", torch.tensor([128], dtype=torch.int32).view(1, 1, 1, 1, -1))
        self.register_buffer("_65536", torch.tensor([65536], dtype=torch.int32).view(1, 1, 1, 1, -1))
        self.register_buffer("_16777216", torch.tensor([16777216], dtype=torch.int32).view(1, 1, 1, 1, -1))

    def _quantize_block(self, x, dim):
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

    def pack_q8_cuda(self, x, dim, batch_size, num_key_value_heads, head_dim_quarter):
        x_i32 = x.to(torch.int32)
        if dim != -1:
            x_i32 = x_i32.reshape(batch_size, num_key_value_heads, 1, head_dim_quarter, 4, -1)
        else:
            x_i32 = x_i32.reshape(batch_size, num_key_value_heads, 1, -1, head_dim_quarter, 4)
        x0, x1, x2, x3 = torch.unbind(x_i32, dim=dim)
        packed = x0 + x1 * self._256 + x2 * self._65536 + (x3 - self._128) * self._16777216
        return packed

    def unpack_q8_cuda(self, x_i32, dim, batch_size, num_key_value_heads, head_dim):
        r3 = x_i32 % self._16777216
        x3 = ((x_i32 - r3) // self._16777216) + self._128
        x2 = r3 // self._65536
        r2 = r3 % self._65536
        x1 = r2 // self._256
        x0 = r2 % self._256
        unpacked = torch.stack([x0, x1, x2, x3], dim=dim)
        if dim != -1:
            return unpacked.reshape(batch_size, num_key_value_heads, 1, head_dim, -1)
        return unpacked.reshape(batch_size, num_key_value_heads, 1, -1, head_dim)

    def forward(self, keys, values, batch_size, num_key_value_heads, head_dim_quarter):
        k_packed, k_scale, k_bias = self._quantize_block(keys, -2)
        v_packed, v_scale, v_bias = self._quantize_block(values, -1)
        if KV_QUANT_DTYPE == "Q8_CUDA":
            k_packed = self.pack_q8_cuda(k_packed, -2, batch_size, num_key_value_heads, head_dim_quarter)
            v_packed = self.pack_q8_cuda(v_packed, -1, batch_size, num_key_value_heads, head_dim_quarter)
        return k_packed, k_scale, k_bias, v_packed, v_scale, v_bias


class LLM_MAIN(torch.nn.Module):
    def __init__(self, llm, num_heads, num_key_value_heads, head_dim, num_layers, hidden_size):
        super(LLM_MAIN, self).__init__()
        self.llm = llm
        self._replace_gelu_with_tanh_approximation(self.llm)
        self.head_dim = head_dim
        self.head_dim_half = head_dim // 2
        self.head_dim_quarter = head_dim // 4
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.qk_heads = self.num_heads + self.num_key_value_heads
        self.num_layers = num_layers
        self.num_layers_2 = num_layers * 2
        self.num_layers_3 = num_layers * 3
        self.num_layers_4 = num_layers * 4
        self.num_layers_5 = num_layers * 5
        self.kv_f16 = (KV_QUANT_DTYPE == "F16")
        self.kv_q8 = (KV_QUANT_DTYPE == "Q8")
        self.kv_q8_cuda = (KV_QUANT_DTYPE == "Q8_CUDA")
        self.quantizer = KVQuantizer().eval()
        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)
        scale_factor = head_dim ** -0.25
        norm_factor = hidden_size ** 0.5
        norm_factor_qk = head_dim ** 0.5
        
        self.save_key = [None] * num_layers
        self.save_value = [None] * num_layers

        if self.kv_q8 or self.kv_q8_cuda:
            self.save_k_scale = [None] * num_layers
            self.save_k_bias = [None] * num_layers
            self.save_v_scale = [None] * num_layers
            self.save_v_bias = [None] * num_layers

        with torch.no_grad():
            for layer in self.llm.model.layers:
                q_proj = layer.self_attn.q_proj
                k_proj = layer.self_attn.k_proj
                v_proj = layer.self_attn.v_proj
                in_features = int(q_proj.in_features)
                out_features = int(q_proj.out_features + k_proj.out_features + v_proj.out_features)
                has_bias = (q_proj.bias is not None) or (k_proj.bias is not None) or (v_proj.bias is not None)
                qkv = torch.nn.Linear(in_features, out_features, bias=has_bias)
                layer.self_attn.q_out_features = int(q_proj.out_features)
                layer.self_attn.k_out_features = int(k_proj.out_features)
                layer.self_attn.v_out_features = int(v_proj.out_features)
                layer.self_attn.qkv_in_features = int(in_features)
                qkv.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))
                if has_bias:
                    dtype = qkv.weight.dtype
                    qb = q_proj.bias if q_proj.bias is not None else torch.zeros(q_proj.out_features, dtype=dtype)
                    kb = k_proj.bias if k_proj.bias is not None else torch.zeros(k_proj.out_features, dtype=dtype)
                    vb = v_proj.bias if v_proj.bias is not None else torch.zeros(v_proj.out_features, dtype=dtype)
                    qkv.bias.copy_(torch.cat([qb, kb, vb], dim=0))
                del layer.self_attn.q_proj
                del layer.self_attn.k_proj
                del layer.self_attn.v_proj

                layer.self_attn.q_norm.weight.mul_(scale_factor * norm_factor_qk)
                layer.self_attn.k_norm.weight.mul_(scale_factor * norm_factor_qk)

                q_norm_w = layer.self_attn.q_norm.weight.repeat(self.num_heads)
                k_norm_w = layer.self_attn.k_norm.weight.repeat(self.num_key_value_heads)
                layer.self_attn.qk_norm_weight = torch.nn.Parameter(torch.cat([q_norm_w, k_norm_w], dim=0).view(1, 1, 1, -1, self.head_dim))
                del layer.self_attn.q_norm
                del layer.self_attn.k_norm

                w = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
                qkv.weight.mul_(w)
                layer.self_attn.qkv = qkv
                del layer.input_layernorm

                w = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
                gate = layer.mlp.gate_proj
                up = layer.mlp.up_proj

                in_feat = gate.in_features
                out_feat = gate.out_features + up.out_features
                gate_up = torch.nn.Linear(in_feat, out_feat, bias=False)

                gate_weight = gate.weight * w
                up_weight = up.weight * w
                gate_up.weight.copy_(torch.cat([gate_weight, up_weight], dim=0))

                layer.mlp.gate_up_proj = gate_up
                del layer.mlp.gate_proj
                del layer.mlp.up_proj
                del layer.post_attention_layernorm

            w = self.llm.model.norm.weight.unsqueeze(0) * norm_factor
            self.llm.lm_head.weight.mul_(w)
            del self.llm.model.norm

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
                print(f"Replaced GELU at: {name}")
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def rotate_half(self, x, batch_size):
        x = x.view(batch_size, -1, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(batch_size, -1, 1, self.qk_heads, self.head_dim)

    def forward(self, *all_inputs):
        hidden_states = all_inputs[-4]
        rotary_pos_emb_cos = all_inputs[-3]
        rotary_pos_emb_sin = all_inputs[-2]
        attention_mask = all_inputs[-1]
        batch_size = hidden_states.shape[0].unsqueeze(0)
        for i, layer in enumerate(self.llm.model.layers):
            residual = hidden_states
            if PREVENT_F16_OVERFLOW:
                hidden_states = hidden_states * self.overflow_scale
            hidden_states = hidden_states * torch.rsqrt(hidden_states.square().sum(-1, keepdim=True))
            qkv = layer.self_attn.qkv(hidden_states)
            qkv = qkv.reshape(batch_size, -1, 1, self.qk_heads + self.num_key_value_heads, self.head_dim)
            qk, v = torch.split(qkv, [self.qk_heads, self.num_key_value_heads], dim=-2)
            if PREVENT_F16_OVERFLOW:
                qk = qk * self.overflow_scale
            qk = qk * torch.rsqrt(qk.square().sum(dim=-1, keepdim=True)) * layer.self_attn.qk_norm_weight
            qk_rot = qk * rotary_pos_emb_cos + self.rotate_half(qk, batch_size) * rotary_pos_emb_sin
            q, k = torch.split(qk_rot, [self.num_heads, self.num_key_value_heads], dim=-2)
            q = q.reshape(batch_size, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q = q.permute(0, 2, 3, 1, 4)
            if self.kv_f16:
                k = k.half()
                v = v.half()
            k = k.permute(0, 3, 2, 4, 1)
            v = v.transpose(1, 3)
            if self.kv_q8 or self.kv_q8_cuda:
                packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(k, v, batch_size, self.num_key_value_heads, self.head_dim_quarter)
                k = torch.cat([all_inputs[i], packed_k], dim=-1)
                k_s = torch.cat([all_inputs[i + self.num_layers_2], scale_k], dim=-1)
                k_b = torch.cat([all_inputs[i + self.num_layers_3], bias_k], dim=-1)
                v = torch.cat([all_inputs[i + self.num_layers], packed_v], dim=-2)
                v_s = torch.cat([all_inputs[i + self.num_layers_4], scale_v], dim=-2)
                v_b = torch.cat([all_inputs[i + self.num_layers_5], bias_v], dim=-2)
                self.save_key[i] = k
                self.save_k_scale[i] = k_s
                self.save_k_bias[i] = k_b
                self.save_value[i] = v
                self.save_v_scale[i] = v_s
                self.save_v_bias[i] = v_b
                if USE_FLOAT16_SCALE_BIAS:
                    k_s = k_s.float()
                    k_b = k_b.float()
                    v_s = v_s.float()
                    v_b = v_b.float()
                if self.kv_q8_cuda:
                    k = self.quantizer.unpack_q8_cuda(k, -2, batch_size, self.num_key_value_heads, self.head_dim)
                    v = self.quantizer.unpack_q8_cuda(v, -1, batch_size, self.num_key_value_heads, self.head_dim)
                attn_Main = torch.matmul(q, k.float())
                q_sum = q.sum(dim=-1, keepdim=True)
                attn_bias = q_sum * k_b
                fused_bias_mask = attn_bias + attention_mask
                attn = torch.addcmul(fused_bias_mask, attn_Main, k_s)
                attn = torch.nn.functional.softmax(attn, dim=-1)
                v_dequant = torch.addcmul(v_b, v.float(), v_s)
                attn = torch.matmul(attn, v_dequant)
            else:
                k = torch.cat((all_inputs[i], k), dim=-1)
                v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)
                self.save_key[i] = k
                self.save_value[i] = v
                if self.kv_f16:
                    k = k.float()
                    v = v.float()
                attn = torch.matmul(q, k) + attention_mask
                attn = torch.nn.functional.softmax(attn, dim=-1)
                attn = torch.matmul(attn, v)
            attn = attn.permute(0, 3, 1, 2, 4).reshape(batch_size, -1, layer.self_attn.o_proj.in_features)
            attn_out = layer.self_attn.o_proj(attn)
            hidden_states = residual + attn_out
            residual = hidden_states
            if PREVENT_F16_OVERFLOW:
                hidden_states = hidden_states * self.overflow_scale
            hidden_states = hidden_states * torch.rsqrt(hidden_states.square().sum(-1, keepdim=True))
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)
            hidden_states = residual + hidden_states
        hidden_states = hidden_states[:, -1]
        if PREVENT_F16_OVERFLOW:
            hidden_states = hidden_states * self.overflow_scale
        hidden_states = hidden_states * torch.rsqrt(hidden_states.square().sum(-1, keepdim=True))
        logits = self.llm.lm_head(hidden_states)
        if self.kv_q8 or self.kv_q8_cuda:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_k_bias, *self.save_v_scale, *self.save_v_bias, logits
        return *self.save_key, *self.save_value, logits


if DO_EXPORT:
    print('Export start ...')
    with torch.inference_mode():
        model = AutoModelForCausalLM.from_pretrained(download_path, dtype=torch.float32, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
        num_layers, num_heads, num_kv_heads, head_dim = model.config.num_hidden_layers, model.config.num_attention_heads, model.config.num_key_value_heads, model.config.head_dim
        vocab_size, hidden_size = model.model.vocab_size, model.model.embed_tokens.embedding_dim
        if USE_FLOAT16_SCALE_BIAS:
            scale_dtype = torch.float16
        else:
            scale_dtype = torch.float32

        # Dummy Tensors
        batch_size = BEAM_SIZE
        ids_len = torch.tensor([10], dtype=torch.int64)
        history_len = torch.tensor([0], dtype=torch.int64)
        kv_seq_len = ids_len + history_len
        beam_size = torch.tensor([BEAM_SIZE], dtype=torch.int64)
        logits = torch.ones((BEAM_SIZE, vocab_size), dtype=torch.float32)
        kv_tensors = {}
        kv_specs = [('key', 4), ('value', 3)]
        if KV_QUANT_DTYPE == "F16":
            kv_dtype = torch.float16
        elif KV_QUANT_DTYPE == "Q8":
            kv_specs.extend([('key_scale', 4), ('key_bias', 4), ('value_scale', 3), ('value_bias', 3)])
            kv_dtype = torch.uint8
        elif KV_QUANT_DTYPE == "Q8_CUDA":
            kv_specs.extend([('key_scale', 4), ('key_bias', 4), ('value_scale', 3), ('value_bias', 3)])
            kv_dtype = torch.int32
        else:
            kv_dtype = torch.float32
        if KV_QUANT_DTYPE == "Q8_CUDA":
            kv_tensors['key'] = torch.zeros((batch_size, num_kv_heads, 1, head_dim // 4, history_len), dtype=kv_dtype)
            kv_tensors['value'] = torch.zeros((batch_size, num_kv_heads, 1, history_len, head_dim // 4), dtype=kv_dtype)
        else:
            kv_tensors['key'] = torch.zeros((batch_size, num_kv_heads, 1, head_dim, history_len), dtype=kv_dtype)
            kv_tensors['value'] = torch.zeros((batch_size, num_kv_heads, 1, history_len, head_dim), dtype=kv_dtype)
        if KV_QUANT_DTYPE in ["Q8", "Q8_CUDA"]:
            kv_tensors['key_scale'] = torch.ones([batch_size, num_kv_heads, 1, 1, history_len], dtype=scale_dtype)
            kv_tensors['key_bias'] = torch.ones([batch_size, num_kv_heads, 1, 1, history_len], dtype=scale_dtype)
            kv_tensors['value_scale'] = torch.ones([batch_size, num_kv_heads, 1, history_len, 1], dtype=scale_dtype)
            kv_tensors['value_bias'] = torch.ones([batch_size, num_kv_heads, 1, history_len, 1], dtype=scale_dtype)

        input_ids = torch.ones((1, ids_len), dtype=torch.int32)
        torch.onnx.export(
            LLM_EMBED(model),
            (input_ids,),
            onnx_model_Embed,
            input_names=['input_ids'],
            output_names=['hidden_states'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'ids_len'},
                'hidden_states': {0: 'batch', 1: 'ids_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del input_ids

        torch.onnx.export(
            ROTARY_MASK_PREFILL(model, MAX_SEQ_LEN),
            (ids_len, history_len),
            onnx_model_Rotary_Mask_Prefill,
            input_names=['ids_len', 'history_len'],
            output_names=['rotary_cos', 'rotary_sin', 'attention_mask', 'kv_seq_len'],
            dynamic_axes={
                'rotary_cos': {1: 'ids_len'},
                'rotary_sin': {1: 'ids_len'},
                'attention_mask': {3: 'ids_len', 4: 'kv_seq_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )

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

        def get_kv_io(tensors_dict, batch_axis='batch_size', seq_axis='history_len', out_seq_axis='kv_seq_len'):
            inputs, in_names, out_names, axes = [], [], [], {}
            for name, dim in kv_specs:
                t = tensors_dict[name]
                for i in range(num_layers):
                    in_n, out_n = f'in_{name}_{i}', f'out_{name}_{i}'
                    inputs.append(t)
                    in_names.append(in_n)
                    out_names.append(out_n)
                    axes[in_n] = {0: batch_axis, dim: seq_axis}
                    axes[out_n] = {0: batch_axis, dim: out_seq_axis}
            return inputs, in_names, out_names, axes

        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors, 'batch_size', 'history_len', 'kv_seq_len')
        hidden_states = torch.ones((batch_size, ids_len, hidden_size), dtype=torch.float32)
        rotary_cos = torch.zeros([1, ids_len, 1, 1, head_dim], dtype=torch.float32)
        rotary_sin = rotary_cos
        attention_mask = torch.zeros([1, 1, 1, ids_len, kv_seq_len], dtype=torch.float32)
        all_inputs = kv_ins + [hidden_states, rotary_cos, rotary_sin, attention_mask]
        input_names = kv_in_names + ['hidden_states', 'rotary_cos', 'rotary_sin', 'attention_mask']
        output_names = kv_out_names + ['logits']
        dynamic_axes = {**kv_axes, 'hidden_states': {0: 'batch', 1: 'ids_len'}, 'logits': {0: 'batch'}, 'rotary_cos': {1: 'ids_len'}, 'rotary_sin': {1: 'ids_len'}, 'attention_mask': {3: 'ids_len', 4: 'kv_seq_len'}}
        model_Main = LLM_MAIN(model, num_heads, num_kv_heads, head_dim, num_layers, hidden_size)
        del model
        torch.onnx.export(
            model_Main,
            tuple(all_inputs),
            onnx_model_Main,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=OPSET,
            dynamo=False
        )
        del model_Main, hidden_states, attention_mask, all_inputs
        gc.collect()

        save_id_in = torch.zeros((BEAM_SIZE, 10), dtype=torch.int32)
        torch.onnx.export(
            GREEDY_SEARCH(),
            (logits, save_id_in),
            onnx_model_Greedy,
            input_names=['logits', 'save_id_in'],
            output_names=['max_logits_idx'],
            dynamic_axes={
                'logits': {0: 'batch'},
                'save_id_in': {0: 'batch', 1: 'history_len'},
                'max_logits_idx': {0: 'batch'}
            },
            opset_version=OPSET, 
            dynamo=False
        )

        kv_tensors_Greedy = {k: v[[0]] for k, v in kv_tensors.items()}  # Slice batch dim for first beam
        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors_Greedy, 'batch_size', 'history_len', 'history_len')
        kv_axes = {k: v for k, v in kv_axes.items() if k not in kv_out_names}
        other_inputs = [logits[[0]], save_id_in, beam_size]
        other_names = ['logits', 'save_id_in', 'beam_size']
        dynamic_axes = {
            **kv_axes,
            'logits': {0: 'batch'},
            'save_id_in': {0: 'batch', 1: 'history_len'},
            'top_beam_prob': {0: 'batch'},
            'top_beam_indices': {0: 'batch'},
            'max_logits_idx': {0: 'batch'},
            'batch_indices': {0: 'batch'},
            'save_id_out': {0: 'batch', 1: 'history_len'}
        }
        num_layers_beam = num_layers * len(kv_specs)
        kv_ins_names = kv_in_names + other_names

        torch.onnx.export(
            FIRST_BEAM_SEARCH(num_layers_beam),
            tuple(kv_ins + other_inputs),
            onnx_model_First_Beam,
            input_names=kv_ins_names,
            output_names=['out_' + n[3:] for n in kv_in_names] + ['top_beam_indices', 'save_id_out', 'top_beam_prob', 'max_logits_idx'],  # Outputs mimic inputs for KV
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            dynamo=False
        )

        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors, 'batch_size', 'history_len', 'kv_seq_len')
        previous_prob = torch.zeros((BEAM_SIZE, 1), dtype=torch.float32)
        topK = torch.tensor([TOP_K], dtype=torch.int64)
        other_inputs = [logits, save_id_in, previous_prob, beam_size, topK]
        other_names = ['logits', 'save_id_in', 'previous_prob', 'beam_size', 'topK']
        dynamic_axes = {
            **kv_axes,
            'logits': {0: 'batch'},
            'save_id_in': {0: 'batch', 1: 'history_len'},
            'previous_prob': {0: 'batch'},
            'save_id_out': {0: 'batch', 1: 'history_len'},
            'top_beam_prob': {0: 'batch'},
            'top_beam_indices': {0: 'batch'},
            'max_logits_idx': {0: 'batch'}
        }

        torch.onnx.export(
            SECOND_BEAM_SEARCH(num_layers_beam),
            tuple(kv_ins + other_inputs),
            onnx_model_Second_Beam,
            input_names=kv_in_names + other_names,
            output_names=kv_out_names + ['top_beam_indices', 'save_id_out', 'top_beam_prob', 'max_logits_idx'],
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            dynamo=False
        )
        del kv_tensors, kv_tensors_Greedy, previous_prob, topK

        penalty_value = torch.tensor([REPEAT_PENALTY], dtype=torch.float32)
        penality_range = torch.tensor([PENALTY_RANGE], dtype=torch.int64)
        torch.onnx.export(
            APPLY_PENALTY(),
            (logits, save_id_in, penalty_value, penality_range),
            onnx_model_Penalty,
            input_names=['logits_in', 'save_id_in', 'penalty_value', 'penality_range'],
            output_names=['logits_out'],
            dynamic_axes={
                'logits_in': {0: 'batch'},
                'save_id_in': {0: 'batch', 1: 'history_len'},
                'logits_out': {0: 'batch'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del save_id_in, penalty_value, penality_range

        torch.onnx.export(
            ARGMAX(),
            (logits,),
            onnx_model_Argmax,
            input_names=['logits'],
            output_names=['max_logits_idx'],
            dynamic_axes={
                'logits': {0: 'batch'},
                'max_logits_idx': {0: 'batch'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del logits
        gc.collect()
    print('\nExport done!\n\nStart running the LLM by ONNXRuntime.\nNow loading . . . it could cost minutes.')


# ── Helpers ──────────────────────────────────────────────────────────────────

def bind_ort_in(binding, names, values, num=0):
    pairs = zip(names[:num], values[:num]) if num else zip(names, values)
    for name, val in pairs:
        binding.bind_ortvalue_input(name, val)

def bind_ort_out(binding, names, device):
    for name in names:
        binding._iobinding.bind_output(name, device)

def create_ortvalue(data, dtype, device, device_id):
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.array(data, dtype=dtype), device, device_id
    )

def make_ortvalue(shape, dtype, device, device_id):
    return onnxruntime.OrtValue.ortvalue_from_numpy(
        np.zeros(shape, dtype=dtype), device, device_id
    )

def make_session(model_path):
    return onnxruntime.InferenceSession(
        model_path,
        sess_options=session_opts,
        providers=ORT_Accelerate_Providers,
        provider_options=provider_options,
        run_options=run_options,
        disabled_optimizers=disabled_optimizers,
    )

def get_in_names(session):  return [x.name for x in session.get_inputs()]
    
def get_out_names(session): return [x.name for x in session.get_outputs()]

def run(session, binding): session.run_with_iobinding(binding, run_options=run_options)


# ── Session Options ───────────────────────────────────────────────────────────

session_opts = onnxruntime.SessionOptions()
run_options  = onnxruntime.RunOptions()

for obj in (session_opts, run_options):
    obj.log_severity_level  = 4
    obj.log_verbosity_level = 4

session_opts.inter_op_num_threads     = MAX_THREADS
session_opts.intra_op_num_threads     = MAX_THREADS
session_opts.enable_cpu_mem_arena     = True
session_opts.execution_mode           = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

_session_configs = {
    'session.set_denormal_as_zero':                   '1',
    'session.intra_op.allow_spinning':                '1',
    'session.inter_op.allow_spinning':                '1',
    'session.enable_quant_qdq_cleanup':               '1',
    'session.qdq_matmulnbits_accuracy_level':         '2' if ORT_FP16 else '4',
    'session.use_device_allocator_for_initializers':  '1',
    'session.graph_optimizations_loop_level':         '2',
    'optimization.enable_gelu_approximation':         '1',
    'optimization.minimal_build_optimizations':       '',
    'optimization.enable_cast_chain_elimination':     '1',
    'optimization.disable_specified_optimizers':
        'CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer'
        if ORT_FP16 else '',
}
for k, v in _session_configs.items():
    session_opts.add_session_config_entry(k, v)

run_options.add_run_config_entry('disable_synchronize_execution_providers', '0')  # Set to '0' to ensure proper synchronization when using multiple EPs.

disabled_optimizers = ['CastFloat16Transformer', 'FuseFp16InitializerToFp32NodeTransformer'] if ORT_FP16 else None


# ── Provider Config ───────────────────────────────────────────────────────────

if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_type':              'CPU',                            # [CPU, NPU, GPU, GPU.0, GPU.1]]
        'precision':                'ACCURACY',                       # [FP32, FP16, ACCURACY]
        'num_of_threads':           MAX_THREADS if MAX_THREADS != 0 else 8,  # The default value is 8. Edit freely.
        'num_streams':              1,
        'enable_opencl_throttling': False,
        'enable_qdq_optimizer':     False,                             # Enable it carefully
        'disable_dynamic_shapes':   False,
    }]
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                          DEVICE_ID,
        'gpu_mem_limit':                      24 * 1024 * 1024 * 1024,  # 24 GB
        'arena_extend_strategy':              'kNextPowerOfTwo',        # ["kNextPowerOfTwo", "kSameAsRequested"]
        'cudnn_conv_algo_search':             'EXHAUSTIVE',             # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
        'sdpa_kernel':                        '2',                      # ["0", "1", "2"]
        'use_tf32':                           '1',
        'fuse_conv_bias':                     '0',                      # Set to '0' to avoid potential errors when enabled.
        'cudnn_conv_use_max_workspace':       '1',
        'cudnn_conv1d_pad_to_nc1d':           '0',
        'tunable_op_enable':                  '0',
        'tunable_op_tuning_enable':           '0',
        'tunable_op_max_tuning_duration_ms':  10,
        'do_copy_in_default_stream':          '0',
        'enable_cuda_graph':                  '0',                      # Set to '0' to avoid potential errors when enabled.
        'prefer_nhwc':                        '0',
        'enable_skip_layer_norm_strict_mode': '0',
        'use_ep_level_unified_stream':        '0',
    }]
    device_type      = 'cuda'
    _ort_device_type = C.OrtDevice.cuda()

elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':               DEVICE_ID,
        'performance_preference': 'high_performance',                   # [high_performance, default, minimum_power]
        'device_filter':          'any',                                # [any, npu, gpu]
    }]
    device_type      = 'dml'
    _ort_device_type = C.OrtDevice.dml()

else:
    provider_options = None
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

_ort_device_type = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)
kv_device = 'cpu' if 'dml' in device_type else device_type


# ── Tokenizer ─────────────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)


# ── Session: Embed ────────────────────────────────────────────────────────────

ort_session_Embed               = make_session(onnx_model_Embed)
binding_Embed                   = ort_session_Embed.io_binding()
in_name_Embed                   = get_in_names(ort_session_Embed)[0]
out_name_Embed                  = get_out_names(ort_session_Embed)[0]


# ── Session: Rotary + Mask ────────────────────────────────────────────────────

ort_session_Rotary_Mask_Prefill = make_session(onnx_model_Rotary_Mask_Prefill)
binding_Rotary_Mask_Prefill     = ort_session_Rotary_Mask_Prefill.io_binding()
in_name_Rotary_Mask_Prefill     = get_in_names(ort_session_Rotary_Mask_Prefill)
out_name_Rotary_Mask_Prefill    = get_out_names(ort_session_Rotary_Mask_Prefill)

ort_session_Rotary_Mask_Decode  = make_session(onnx_model_Rotary_Mask_Decode)
binding_Rotary_Mask_Decode      = ort_session_Rotary_Mask_Decode.io_binding()
in_name_Rotary_Mask_Decode      = get_in_names(ort_session_Rotary_Mask_Decode)[0]
out_name_Rotary_Mask_Decode     = get_out_names(ort_session_Rotary_Mask_Decode)


# ── Session: Main ─────────────────────────────────────────────────────────────

ort_session_Main                = make_session(onnx_model_Main)
binding_Main                    = ort_session_Main.io_binding()
print(f"\nUsable Providers: {ort_session_Main.get_providers()}")

in_name_Main_objs               = ort_session_Main.get_inputs()
out_name_Main_objs              = ort_session_Main.get_outputs()
in_name_Main                    = [x.name for x in in_name_Main_objs]
out_name_Main                   = [x.name for x in out_name_Main_objs]
amount_of_outputs_Main          = len(out_name_Main_objs)
num_keys_values                 = amount_of_outputs_Main - 1

# Derived index offsets
num_keys_values_plus_1          = num_keys_values + 1
num_keys_values_plus_2          = num_keys_values + 2
num_keys_values_plus_3          = num_keys_values + 3
num_keys_values_plus_4          = num_keys_values + 4

in_name_Main_parts              = in_name_Main[:num_keys_values]
kv_dtype_str                    = ort_session_Main._inputs_meta[0].type
hidden_states_dtype_Main        = np.float16 if 'float16' in ort_session_Main._inputs_meta[num_keys_values].type else np.float32
vocab_size                      = ort_session_Main._outputs_meta[num_keys_values].shape[1]


# ── KV Cache Setup ────────────────────────────────────────────────────────────

_meta = ort_session_Main._inputs_meta

if 'uint8' in kv_dtype_str or 'int32' in kv_dtype_str:
    kv_dtype_Main   = np.int32  if 'int32'  in kv_dtype_str else np.uint8
    num_layers      = num_keys_values // 6
    scale_dtype     = np.float16 if 'float16' in _meta[num_layers * 2].type else np.float32
    k_scales        = make_ortvalue((1, _meta[0].shape[1],          1, 1, 0), scale_dtype, kv_device, DEVICE_ID)
    k_biases        = make_ortvalue((1, _meta[0].shape[1],          1, 1, 0), scale_dtype, kv_device, DEVICE_ID)
    v_scales        = make_ortvalue((1, _meta[num_layers].shape[1], 1, 0, 1), scale_dtype, kv_device, DEVICE_ID)
    v_biases        = make_ortvalue((1, _meta[num_layers].shape[1], 1, 0, 1), scale_dtype, kv_device, DEVICE_ID)
else:
    kv_dtype_Main   = np.float16 if 'float16' in kv_dtype_str else np.float32
    num_layers      = num_keys_values // 2
    k_scales        = None

past_keys_Main      = make_ortvalue((1, _meta[0].shape[1],          1, _meta[0].shape[3],          0), kv_dtype_Main, kv_device, DEVICE_ID)
past_values_Main    = make_ortvalue((1, _meta[num_layers].shape[1], 1, 0, _meta[num_layers].shape[4]), kv_dtype_Main, kv_device, DEVICE_ID)


# ── Beam Search Validation ────────────────────────────────────────────────────

if USE_BEAM_SEARCH and TOP_K < BEAM_SIZE:
    TOP_K = BEAM_SIZE
if TOP_K < 2 or BEAM_SIZE < 2:
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")
if not USE_BEAM_SEARCH:
    BEAM_SIZE = 1

USE_PENALTY = (REPEAT_PENALTY != 1.0)


# ── Prompt & Initial OrtValues ────────────────────────────────────────────────

prompt = (
    f'<|im_start|>user\n{TEST_QUERY}<|im_end|>\n<|im_start|>assistant\n'
    if TEST_THINK_MODE else
    f'<|im_start|>user\n{TEST_QUERY}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
)

tokens              = tokenizer(prompt, return_tensors='np')['input_ids'].astype(np.int32)
num_prefill         = tokens.shape[-1]

input_ids           = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, DEVICE_ID)
ids_len             = create_ortvalue([num_prefill],  np.int64,   device_type, DEVICE_ID)
init_ids_len_1      = create_ortvalue([1],            np.int64,   device_type, DEVICE_ID)
init_history_len    = create_ortvalue([0],            np.int64,   device_type, DEVICE_ID)
topK                = create_ortvalue([TOP_K],        np.int64,   device_type, DEVICE_ID)
beam_size           = create_ortvalue([BEAM_SIZE],    np.int64,   device_type, DEVICE_ID)

_dec_meta           = ort_session_Rotary_Mask_Decode._outputs_meta
init_save_id        = make_ortvalue((BEAM_SIZE, 0),                                   hidden_states_dtype_Main, device_type, DEVICE_ID)
init_attention_mask = make_ortvalue((1, 1, 1, 1, 1),                                  hidden_states_dtype_Main, device_type, DEVICE_ID)
init_rotary_cos     = make_ortvalue((1, 1, 1, 1, _dec_meta[0].shape[4]),              hidden_states_dtype_Main, device_type, DEVICE_ID)
init_rotary_sin     = make_ortvalue((1, 1, 1, 1, _dec_meta[1].shape[4]),              hidden_states_dtype_Main, device_type, DEVICE_ID)
init_hidden_states  = make_ortvalue((BEAM_SIZE, 1, _meta[num_keys_values].shape[2]),  hidden_states_dtype_Main, device_type, DEVICE_ID)
init_save_id        = make_ortvalue((BEAM_SIZE, 0),                                   np.int32,                 device_type, DEVICE_ID)


# ── Session: Decode (Beam or Greedy) ─────────────────────────────────────────

if USE_BEAM_SEARCH:
    print("\nBeam Search does not display immediate decoding results...")

    ort_session_First_Beam    = make_session(onnx_model_First_Beam)
    binding_First_Beam        = ort_session_First_Beam.io_binding()
    in_name_First_Beam        = get_in_names(ort_session_First_Beam)
    out_name_First_Beam       = get_out_names(ort_session_First_Beam)
    in_name_First_Beam_parts  = in_name_First_Beam[:num_keys_values]

    ort_session_Second_Beam   = make_session(onnx_model_Second_Beam)
    binding_Second_Beam       = ort_session_Second_Beam.io_binding()
    in_name_Second_Beam       = get_in_names(ort_session_Second_Beam)
    out_name_Second_Beam      = get_out_names(ort_session_Second_Beam)
    in_name_Second_Beam_parts = in_name_Second_Beam[:num_keys_values]

    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[num_keys_values_plus_1], init_save_id)
    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[num_keys_values_plus_2], beam_size)
    binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_plus_3], beam_size)
    binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_plus_4], topK)

else:
    ort_session_Greedy        = make_session(onnx_model_Greedy)
    binding_Greedy            = ort_session_Greedy.io_binding()
    in_name_Greedy            = get_in_names(ort_session_Greedy)
    out_name_Greedy           = get_out_names(ort_session_Greedy)
    binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], init_save_id)

    ort_session_Argmax        = make_session(onnx_model_Argmax)
    binding_Argmax            = ort_session_Argmax.io_binding()
    in_name_Argmax            = ort_session_Argmax.get_inputs()[0].name
    out_name_Argmax           = get_out_names(ort_session_Argmax)
    save_id_numpy             = np.zeros(MAX_SEQ_LEN, dtype=np.int32)


# ── Session: Penalty (optional) ───────────────────────────────────────────────

if USE_PENALTY:
    ort_session_Penalty       = make_session(onnx_model_Penalty)
    binding_Penalty           = ort_session_Penalty.io_binding()
    in_name_Penalty           = get_in_names(ort_session_Penalty)
    out_name_Penalty          = get_out_names(ort_session_Penalty)

    penalty_dtype             = np.float16 if 'float16' in ort_session_Penalty._inputs_meta[2].type else np.float32
    penalty_value             = create_ortvalue([REPEAT_PENALTY], penalty_dtype, device_type, DEVICE_ID)
    penalty_range             = create_ortvalue([PENALTY_RANGE],  np.int64,      device_type, DEVICE_ID)

    binding_Penalty.bind_ortvalue_input(in_name_Penalty[2], penalty_value)
    binding_Penalty.bind_ortvalue_input(in_name_Penalty[3], penalty_range)


# ── Prefill ───────────────────────────────────────────────────────────────────

# Embed
binding_Embed.bind_ortvalue_input(in_name_Embed, input_ids)
bind_ort_out(binding_Embed, [out_name_Embed], _ort_device_type)
run(ort_session_Embed, binding_Embed)
outputs_Embed = binding_Embed.get_outputs()[0]

# Rotary + Mask (prefill)
binding_Rotary_Mask_Prefill.bind_ortvalue_input(in_name_Rotary_Mask_Prefill[0], ids_len)
binding_Rotary_Mask_Prefill.bind_ortvalue_input(in_name_Rotary_Mask_Prefill[1], init_history_len)
bind_ort_out(binding_Rotary_Mask_Prefill, out_name_Rotary_Mask_Prefill, _ort_device_type)
run(ort_session_Rotary_Mask_Prefill, binding_Rotary_Mask_Prefill)
rotary_cos, rotary_sin, attention_mask, kv_seq_len = binding_Rotary_Mask_Prefill.get_outputs()

# Rotary + Mask (decode) — pre-bind outputs to init placeholders
binding_Rotary_Mask_Decode.bind_ortvalue_input(in_name_Rotary_Mask_Decode, kv_seq_len)
binding_Rotary_Mask_Decode.bind_ortvalue_output(out_name_Rotary_Mask_Decode[0], init_rotary_cos)
binding_Rotary_Mask_Decode.bind_ortvalue_output(out_name_Rotary_Mask_Decode[1], init_rotary_sin)
binding_Rotary_Mask_Decode.bind_ortvalue_output(out_name_Rotary_Mask_Decode[2], kv_seq_len)

# Main — bind static inputs
binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values],        outputs_Embed)
binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_1], rotary_cos)
binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_2], rotary_sin)
binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_3], attention_mask)

i = 0
j = num_layers
while i < j:
    binding_Main.bind_ortvalue_input(in_name_Main[i], past_keys_Main)
    i += 1
j = i + num_layers
while i < j:
    binding_Main.bind_ortvalue_input(in_name_Main[i], past_values_Main)
    i += 1

if k_scales:
    j = i + num_layers
    while i < j:
        binding_Main.bind_ortvalue_input(in_name_Main[i], k_scales)
        i += 1
    j = i + num_layers
    while i < j:
        binding_Main.bind_ortvalue_input(in_name_Main[i], k_biases)
        i += 1
    j = i + num_layers
    while i < j:
        binding_Main.bind_ortvalue_input(in_name_Main[i], v_scales)
        i += 1
    j = i + num_layers
    while i < j:
        binding_Main.bind_ortvalue_input(in_name_Main[i], v_biases)
        i += 1

# ── Decode Loop ───────────────────────────────────────────────────────────────

print(f'\n\nTest Question: {TEST_QUERY}\nLLM Answering:')
num_decode          = 0
generate_limit      = MAX_SEQ_LEN - num_prefill
start_time          = time.time()

while num_decode < generate_limit:
    # Run Main
    bind_ort_out(binding_Main, out_name_Main, _ort_device_type)
    run(ort_session_Main, binding_Main)
    outputs_Main    = binding_Main.get_outputs()
    logits          = outputs_Main[num_keys_values]

    # Optional penalty
    if USE_PENALTY and num_decode >= PENALTY_RANGE:
        binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], logits)
        binding_Penalty.bind_ortvalue_input(in_name_Penalty[1], save_id)
        bind_ort_out(binding_Penalty, out_name_Penalty, _ort_device_type)
        run(ort_session_Penalty, binding_Penalty)
        logits = binding_Penalty.get_outputs()[0]

    # Decode: Beam Search
    if USE_BEAM_SEARCH:
        if num_decode < 1:
            binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[num_keys_values], logits)
            bind_ort_in(binding_First_Beam, in_name_First_Beam_parts, outputs_Main)
            bind_ort_out(binding_First_Beam, out_name_First_Beam, _ort_device_type)
            run(ort_session_First_Beam, binding_First_Beam)
            outputs_Beam = binding_First_Beam.get_outputs()
        else:
            binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values], logits)
            bind_ort_in(binding_Second_Beam, in_name_Second_Beam_parts, outputs_Main)
            bind_ort_out(binding_Second_Beam, out_name_Second_Beam, _ort_device_type)
            run(ort_session_Second_Beam, binding_Second_Beam)
            outputs_Beam = binding_Second_Beam.get_outputs()

        max_logits_idx = outputs_Beam[num_keys_values_plus_3].numpy()
        if max_logits_idx in STOP_TOKEN:
            break

        save_id = outputs_Beam[num_keys_values_plus_1]
        bind_ort_in(binding_Main, in_name_Main_parts, outputs_Beam)
        binding_Embed.bind_ortvalue_input(in_name_Embed, outputs_Beam[num_keys_values])
        binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_plus_1], outputs_Beam[num_keys_values_plus_1])
        binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_plus_2], outputs_Beam[num_keys_values_plus_2])

    # Decode: Greedy / Argmax
    else:
        if USE_PENALTY:
            binding_Greedy.bind_ortvalue_input(in_name_Greedy[0], logits)
            bind_ort_out(binding_Greedy, out_name_Greedy, _ort_device_type)
            run(ort_session_Greedy, binding_Greedy)
            max_idx_ort, save_id = binding_Greedy.get_outputs()
        else:
            binding_Argmax.bind_ortvalue_input(in_name_Argmax, logits)
            bind_ort_out(binding_Argmax, out_name_Argmax, _ort_device_type)
            run(ort_session_Argmax, binding_Argmax)
            max_idx_ort = binding_Argmax.get_outputs()[0]

        max_logits_idx = max_idx_ort.numpy().flat[0]
        if max_logits_idx in STOP_TOKEN:
            break

        if USE_PENALTY:
            binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id)
        else:
            save_id_numpy[num_decode] = max_logits_idx

        binding_Embed.bind_ortvalue_input(in_name_Embed, max_idx_ort)
        bind_ort_in(binding_Main, in_name_Main_parts, outputs_Main)
        print(tokenizer.decode(max_logits_idx), end="", flush=True)

    # First-step: switch to decode-mode bindings
    if num_decode < 1:
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values],        init_hidden_states)
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_1], init_rotary_cos)
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_2], init_rotary_sin)
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_3], init_attention_mask)
        binding_Embed.bind_ortvalue_output(out_name_Embed,                     init_hidden_states)

    run(ort_session_Embed, binding_Embed)
    run(ort_session_Rotary_Mask_Decode, binding_Rotary_Mask_Decode)
    num_decode += 1


# ── Results ───────────────────────────────────────────────────────────────────

elapsed_time      = time.time() - start_time
tokens_per_second = (num_decode + 1) / elapsed_time

result = (
    tokenizer.decode(save_id.numpy()[0, :num_decode], skip_special_tokens=True)
    if USE_PENALTY or USE_BEAM_SEARCH else
    tokenizer.decode(save_id_numpy[:num_decode],      skip_special_tokens=True)
)

print(f"\n\nFinal:\n{result}\n\nDecode: {tokens_per_second:.3f} token/s\nTotal tokens generated: {num_decode}\nTotal time: {elapsed_time:.3f}s")

