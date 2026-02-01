import gc
import time
import torch
import numpy as np
import onnxruntime
from transformers import AutoModelForCausalLM, AutoTokenizer


path         = r'/home/DakeQQ/Downloads/Qwen3-1.7B'                             # Set the folder path where the Qwen whole project downloaded.
onnx_model_A = r'/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Embed.onnx'
onnx_model_B = r'/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Main.onnx'
onnx_model_C = r'/home/DakeQQ/Downloads/Qwen_ONNX/Greedy_Search.onnx'
onnx_model_D = r'/home/DakeQQ/Downloads/Qwen_ONNX/First_Beam_Search.onnx'
onnx_model_E = r'/home/DakeQQ/Downloads/Qwen_ONNX/Second_Beam_Search.onnx'
onnx_model_F = r'/home/DakeQQ/Downloads/Qwen_ONNX/Reset_Penality.onnx'
onnx_model_G = r'/home/DakeQQ/Downloads/Qwen_ONNX/Argmax.onnx'

# Test input
TEST_THINK_MODE = True
TEST_QUERY = "地球最高的山是哪座山？"

# Model Config
DO_EXPORT = True                    # Whether to export the ONNX models
PREVENT_F16_OVERFLOW = False        # Prevent float16 overflow. Set True for Q4F16 or Q8F16 or F16 quantization.
STOP_TOKEN = [151643, 151645]       # Qwen stop token ids
MAX_SEQ_LEN = 4096                  # Max context length. Can not edit after export.

# KV cache quantization
KV_QUANT_DTYPE = "F16"              # "Q8" | "F16" | "F32"
USE_FLOAT16_SCALE_BIAS = True       # If choose Q8, whether to use float16 for scale and bias.

# Decoding strategy
USE_BEAM_SEARCH = False             # Use beam search or greedy search
REPEAT_PENALTY = 1.0                # 0.0 ~ 1.0; No penalty = 1.0
PENALTY_RANGE = 30                  # Recent-token window to apply penalty
MAX_BEAM_SIZE = 10                  # Max beam size for beam search. Can not edit after export.
TOP_K = 3                           # Top-K for beam search
BEAM_SIZE = 3                       # Beam size for beam search. Must be <= MAX_BEAM_SIZE

# Runtime config
ORT_Accelerate_Providers = []       # ORT execution providers; ['CUDAExecutionProvider'', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
MAX_THREADS = 0                     # 0 = auto
DEVICE_ID = 0                       # Device ID for GPU
OPSET = 17                          # ONNX opset version


class ARGMAX(torch.nn.Module):
    def __init__(self):
        super(ARGMAX, self).__init__()
        pass

    def forward(self, logits):
        max_logits_idx = torch.argmax(logits, dim=-1, keepdim=True)
        return max_logits_idx.int()


class GREEDY_SEARCH(torch.nn.Module):
    def __init__(self):
        super(GREEDY_SEARCH, self).__init__()
        self.batch_indices = torch.arange(MAX_BEAM_SIZE, dtype=torch.int8)

    def forward(self, logits, repeat_penality, penality_value, batch_size):
        max_logits_idx = torch.argmax(logits * repeat_penality, dim=-1, keepdim=True)
        batch_indices = self.batch_indices[:batch_size].long()
        repeat_penality[batch_indices, max_logits_idx.squeeze(-1)] *= penality_value
        return max_logits_idx.int(), repeat_penality


class FIRST_BEAM_SEARCH(torch.nn.Module):
    def __init__(self, total_layers):
        super(FIRST_BEAM_SEARCH, self).__init__()
        self.total_layers = total_layers
        self.save_keys_values = [None] * self.total_layers
        self.batch_indices = torch.arange(MAX_BEAM_SIZE, dtype=torch.int8)

    def forward(self, *all_inputs):
        logits = all_inputs[-5]
        save_id = all_inputs[-4]
        repeat_penality = all_inputs[-3]
        penality_value = all_inputs[-2]
        beam_size = all_inputs[-1]
        logits = torch.log_softmax(logits, dim=-1)
        top_beam_prob, top_beam_indices = torch.topk(logits, dim=-1, k=beam_size, sorted=False, largest=True)
        for i in range(self.total_layers):
            self.save_keys_values[i] = all_inputs[i].repeat(beam_size, *([1] * (all_inputs[i].dim() - 1)))
        top_beam_indices = top_beam_indices.transpose(0, 1)
        batch_indices = self.batch_indices[:beam_size].long()
        repeat_penality[batch_indices, top_beam_indices] *= penality_value
        top_beam_indices = top_beam_indices.int()
        save_id = torch.cat([save_id, top_beam_indices], dim=-1)
        max_logits_idx = top_beam_indices[0]
        return *self.save_keys_values, top_beam_indices, save_id, repeat_penality, top_beam_prob.transpose(0, 1), batch_indices, max_logits_idx


class SECOND_BEAM_SEARCH(torch.nn.Module):
    def __init__(self, total_layers):
        super(SECOND_BEAM_SEARCH, self).__init__()
        self.total_layers = total_layers
        self.save_keys_values = [None] * self.total_layers
        self.batch_indices = torch.arange(MAX_BEAM_SIZE, dtype=torch.int8)

    def forward(self, *all_inputs):
        logits = all_inputs[-8]
        save_id = all_inputs[-7]
        repeat_penality = all_inputs[-6]
        previous_prob = all_inputs[-5]
        batch_indices = all_inputs[-4]
        penality_value = all_inputs[-3]
        beam_size = all_inputs[-2]
        topK = all_inputs[-1]
        logits = torch.log_softmax(logits * repeat_penality, dim=-1)
        top_k_prob, top_k_indices = torch.topk(logits, k=topK, dim=-1, largest=True, sorted=False)
        current_prob = (top_k_prob + previous_prob).view(-1)
        top_beam_prob, top_beam_indices = torch.topk(current_prob, k=beam_size, dim=-1, largest=True, sorted=False)
        beam_index = top_beam_indices // topK
        top_beam_indices = top_k_indices.view(-1)[top_beam_indices]
        for i in range(self.total_layers):
            self.save_keys_values[i] = all_inputs[i][beam_index]
        repeat_penality = repeat_penality[beam_index]
        repeat_penality[batch_indices, top_beam_indices] *= penality_value
        top_beam_indices = top_beam_indices.int()
        max_logits_idx = top_beam_indices[[0]]
        top_beam_indices = top_beam_indices.unsqueeze(-1)
        save_id = torch.cat([save_id[beam_index], top_beam_indices], dim=-1)
        return *self.save_keys_values, top_beam_indices, save_id, repeat_penality, top_beam_prob.unsqueeze(-1), max_logits_idx


class RESET_PENALITY(torch.nn.Module):
    def __init__(self):
        super(RESET_PENALITY, self).__init__()
        pass

    def forward(self, save_id, repeat_penality, penality_reset_count, batch_indices):
        repeat_penality[batch_indices, save_id[batch_indices, penality_reset_count[batch_indices]]] = 1.0
        penality_reset_count += 1
        return save_id, repeat_penality, penality_reset_count


class KVQuantizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qmax = 255.0
        self.register_buffer('inv_qmax', torch.tensor([1.0 / self.qmax], dtype=torch.float32).view(1, 1, 1, 1, -1))
        self.register_buffer('eps', torch.tensor([1e-7], dtype=torch.float32).view(1, 1, 1, 1, -1))

    def _quantize_block(self, x, dim):
        block_min = x.min(dim=dim, keepdim=True).values  # bias
        block_max = x.max(dim=dim, keepdim=True).values
        scale = (block_max - block_min) * self.inv_qmax + self.eps
        x_normalized = (x - block_min) / scale
        x_packed = torch.round(x_normalized).to(torch.uint8)
        if USE_FLOAT16_SCALE_BIAS:
            scale = scale.half()
            block_min = block_min.half()
        return x_packed, scale, block_min

    def forward(self, keys, values):
        k_packed, k_scale, k_bias = self._quantize_block(keys, 3)
        v_packed, v_scale, v_bias = self._quantize_block(values, 4)
        return k_packed, k_scale, k_bias, v_packed, v_scale, v_bias


class LLM_EMBED(torch.nn.Module):
    def __init__(self, llm):
        super(LLM_EMBED, self).__init__()
        self.embed_tokens = llm.model.embed_tokens.float()

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class LLM_MAIN(torch.nn.Module):
    def __init__(self, llm, max_seq_len, num_heads, num_key_value_heads, head_dim, num_layers, hidden_size):
        super(LLM_MAIN, self).__init__()
        self.llm = llm
        self._replace_gelu_with_tanh_approximation(self.llm)
        self.head_dim = head_dim
        self.head_dim_half = [head_dim // 2, head_dim // 2]
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.num_layers = num_layers
        self.num_layers_2 = num_layers * 2
        self.num_layers_3 = num_layers * 3
        self.num_layers_4 = num_layers * 4
        self.num_layers_5 = num_layers * 5
        self.kv_f16 = (KV_QUANT_DTYPE == "F16")
        self.kv_q8 = (KV_QUANT_DTYPE == "Q8")
        self.quantizer = KVQuantizer().eval()
        self.variance_epsilon = torch.tensor([1e-6], dtype=torch.float32)
        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)
        scale_factor = head_dim ** -0.25
        norm_factor = hidden_size ** 0.5
        position_ids = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(-1)
        inv_freq = self.llm.model.rotary_emb.inv_freq
        idx_theta = (position_ids * inv_freq).unsqueeze(0).unsqueeze(0)
        cos = torch.cos(idx_theta)
        sin = torch.sin(idx_theta)
        self.attention_mask = (1 - torch.tril(torch.ones([1, 1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128
        self.register_buffer("cos_rotary_pos_emb", torch.cat((cos, cos), dim=-1).half(), persistent=False)
        self.register_buffer("sin_rotary_pos_emb", torch.cat((sin, sin), dim=-1).half(), persistent=False)
        self.save_key = [None] * num_layers
        self.save_value = [None] * num_layers
        if self.kv_q8:
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

                layer.self_attn.q_norm.weight.mul_(scale_factor)
                layer.self_attn.k_norm.weight.mul_(scale_factor)

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

    def rotate_half(self, x, dim):
        x1, x2 = torch.split(x, self.head_dim_half, dim=dim)
        return torch.cat((-x2, x1), dim=dim)

    def repeat_k(self, kv_states, num_key_value_groups, head_dim, num_heads, batch_size):
        return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=2).view(batch_size, num_heads, head_dim, -1)

    def repeat_v(self, kv_states, num_key_value_groups, head_dim, num_heads, batch_size):
        return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=2).view(batch_size, num_heads, -1, head_dim)

    def forward(self, *all_inputs):
        hidden_states = all_inputs[-4]
        history_len = all_inputs[-3]
        ids_len = all_inputs[-2]
        mask = all_inputs[-1]
        kv_seq_len = history_len + ids_len
        rotary_pos_emb_cos_q = self.cos_rotary_pos_emb[..., history_len:kv_seq_len, :].float()
        rotary_pos_emb_sin_q = self.sin_rotary_pos_emb[..., history_len:kv_seq_len, :].float()
        rotary_pos_emb_cos_k = rotary_pos_emb_cos_q.transpose(-1, -2).unsqueeze(0)
        rotary_pos_emb_sin_k = rotary_pos_emb_sin_q.transpose(-1, -2).unsqueeze(0)
        attention_mask = (self.attention_mask[..., :ids_len, :kv_seq_len] * mask).float()
        batch_size = hidden_states.shape[0].unsqueeze(0)
        for i, layer in enumerate(self.llm.model.layers):
            residual = hidden_states
            if PREVENT_F16_OVERFLOW:
                hidden_states = hidden_states * self.overflow_scale
            hidden_states_norm = hidden_states * torch.rsqrt(hidden_states.pow(2).sum(-1, keepdim=True) + self.variance_epsilon)
            qkv = layer.self_attn.qkv(hidden_states_norm)
            q, k, v = torch.split(qkv, [layer.self_attn.q_out_features, layer.self_attn.k_out_features, layer.self_attn.v_out_features], dim=-1)
            q = q.view(batch_size, -1, self.num_heads, self.head_dim)
            k = k.view(batch_size, -1, 1, self.num_key_value_heads, self.head_dim)
            if self.kv_f16:
                v = v.half()
            v = v.view(batch_size, -1, 1, self.num_key_value_heads, self.head_dim).transpose(1, 3)
            q = (layer.self_attn.q_norm(q)).transpose(1, 2)
            k = (layer.self_attn.k_norm(k)).permute(0, 3, 2, 4, 1)
            q = q * rotary_pos_emb_cos_q + self.rotate_half(q, -1) * rotary_pos_emb_sin_q
            k = k * rotary_pos_emb_cos_k + self.rotate_half(k, -2) * rotary_pos_emb_sin_k
            if self.kv_f16:
                k = torch.cat((all_inputs[i], k.half()), dim=-1)
                v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)
                self.save_key[i] = k
                self.save_value[i] = v
                k = self.repeat_k(k, self.num_key_value_groups, self.head_dim, self.num_heads, batch_size)
                v = self.repeat_v(v, self.num_key_value_groups, self.head_dim, self.num_heads, batch_size)
                attn = torch.nn.functional.softmax(torch.matmul(q, k.float()) + attention_mask, dim=-1, dtype=torch.float32)
                attn = torch.matmul(attn, v.float())
            elif self.kv_q8:
                packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(k, v)
                self.save_key[i] = torch.cat([all_inputs[i], packed_k], dim=-1)
                self.save_k_scale[i] = torch.cat([all_inputs[i + self.num_layers_2], scale_k], dim=-1)
                self.save_k_bias[i] = torch.cat([all_inputs[i + self.num_layers_3], bias_k], dim=-1)
                self.save_value[i] = torch.cat([all_inputs[i + self.num_layers], packed_v], dim=-2)
                self.save_v_scale[i] = torch.cat([all_inputs[i + self.num_layers_4], scale_v], dim=-2)
                self.save_v_bias[i] = torch.cat([all_inputs[i + self.num_layers_5], bias_v], dim=-2)
                k_p = self.repeat_k(self.save_key[i], self.num_key_value_groups, self.head_dim, self.num_heads, batch_size)
                k_s = self.repeat_k(self.save_k_scale[i], self.num_key_value_groups, 1, self.num_heads, batch_size)
                k_b = self.repeat_k(self.save_k_bias[i], self.num_key_value_groups, 1, self.num_heads, batch_size)
                v_p = self.repeat_v(self.save_value[i], self.num_key_value_groups, self.head_dim, self.num_heads, batch_size)
                v_s = self.repeat_v(self.save_v_scale[i], self.num_key_value_groups, 1, self.num_heads, batch_size)
                v_b = self.repeat_v(self.save_v_bias[i], self.num_key_value_groups, 1, self.num_heads, batch_size)
                if USE_FLOAT16_SCALE_BIAS:
                    k_s = k_s.float()
                    k_b = k_b.float()
                    v_s = v_s.float()
                    v_b = v_b.float()
                attn_main = torch.matmul(q, k_p.float())
                q_sum = q.sum(dim=-1, keepdim=True)
                attn_bias = torch.matmul(q_sum, k_b)
                attn = attn_main * k_s + attn_bias
                attn = torch.nn.functional.softmax(attn + attention_mask, dim=-1, dtype=torch.float32)
                attn_scaled = attn * v_s.transpose(-2, -1)
                out_main = torch.matmul(attn_scaled, v_p.float())
                out_bias = torch.matmul(attn, v_b)
                attn = out_main + out_bias
            else:
                k = torch.cat((all_inputs[i], k), dim=-1)
                v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)
                self.save_key[i] = k
                self.save_value[i] = v
                k = self.repeat_k(k, self.num_key_value_groups, self.head_dim, self.num_heads, batch_size)
                v = self.repeat_v(v, self.num_key_value_groups, self.head_dim, self.num_heads, batch_size)
                attn = torch.nn.functional.softmax(torch.matmul(q, k) + attention_mask, dim=-1, dtype=torch.float32)
                attn = torch.matmul(attn, v)
            attn = attn.transpose(1, 2).reshape(batch_size, -1, layer.self_attn.o_proj.in_features)
            attn_out = layer.self_attn.o_proj(attn)
            hidden_states = residual + attn_out
            residual = hidden_states
            if PREVENT_F16_OVERFLOW:
                hidden_states = hidden_states * self.overflow_scale
            hidden_states = hidden_states * torch.rsqrt(hidden_states.pow(2).sum(-1, keepdim=True) + self.variance_epsilon)
            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(gate_up, [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features], dim=-1)
            hidden_states = layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)
            hidden_states += residual
        hidden_states = hidden_states[:, -1]
        if PREVENT_F16_OVERFLOW:
            hidden_states = hidden_states * self.overflow_scale
        hidden_states = hidden_states * torch.rsqrt(hidden_states.pow(2).sum(-1, keepdim=True) + self.variance_epsilon)
        logits = self.llm.lm_head(hidden_states)
        if self.kv_q8:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_k_bias, *self.save_v_scale, *self.save_v_bias, logits, kv_seq_len
        return *self.save_key, *self.save_value, logits, kv_seq_len


if DO_EXPORT:
    print('Export start ...')
    with torch.inference_mode():
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
        num_layers, num_heads, num_kv_heads, head_dim = model.config.num_hidden_layers, model.config.num_attention_heads, model.config.num_key_value_heads, model.config.head_dim
        vocab_size, hidden_size = model.model.vocab_size, model.model.embed_tokens.embedding_dim
        if USE_FLOAT16_SCALE_BIAS:
            scale_dtype = torch.float16
        else:
            scale_dtype = torch.float32

        # Dummy Tensors
        batch_size = 3
        ids_len = torch.tensor([10], dtype=torch.int64)
        history_len = torch.tensor([0], dtype=torch.int64)
        beam_size_t = torch.tensor([BEAM_SIZE], dtype=torch.int64)
        logits_t = torch.ones((BEAM_SIZE, vocab_size), dtype=torch.float32)
        kv_tensors = {}
        kv_specs = [('key', 4), ('value', 3)]
        if KV_QUANT_DTYPE == "F16":
            kv_dtype = torch.float16
        elif KV_QUANT_DTYPE == "Q8":
            kv_specs.extend([('key_scale', 4), ('key_bias', 4), ('value_scale', 3), ('value_bias', 3)])
            kv_dtype = torch.uint8
        else:
            kv_dtype = torch.float32
        kv_tensors['key'] = torch.zeros((batch_size, num_kv_heads, 1, head_dim, history_len), dtype=kv_dtype)
        kv_tensors['value'] = torch.zeros((batch_size, num_kv_heads, 1, history_len, head_dim), dtype=kv_dtype)
        if KV_QUANT_DTYPE == "Q8":
            kv_tensors['key_scale'] = torch.ones([batch_size, num_kv_heads, 1, 1, history_len], dtype=scale_dtype)
            kv_tensors['key_bias'] = torch.ones([batch_size, num_kv_heads, 1, 1, history_len], dtype=scale_dtype)
            kv_tensors['value_scale'] = torch.ones([batch_size, num_kv_heads, 1, history_len, 1], dtype=scale_dtype)
            kv_tensors['value_bias'] = torch.ones([batch_size, num_kv_heads, 1, history_len, 1], dtype=scale_dtype)

        print("Exporting LLM_EMBED...")
        input_ids = torch.ones((1, ids_len), dtype=torch.int32)
        model_A = LLM_EMBED(model)
        torch.onnx.export(
            model_A,
            (input_ids,),
            onnx_model_A,
            input_names=['input_ids'],
            output_names=['hidden_states'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'ids_len'},
                'hidden_states': {0: 'batch', 1: 'ids_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del model_A, input_ids

        # 5. Export LLM_MAIN
        print("Exporting LLM_MAIN...")

        def get_kv_io(tensors_dict, batch_axis='batch', seq_axis='history_len', out_seq_axis='kv_seq_len'):
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

        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors, 'batch', 'history_len', 'kv_seq_len')
        hidden_states = torch.ones((batch_size, ids_len, hidden_size), dtype=torch.float32)
        attention_mask = torch.tensor([1], dtype=torch.int8)
        all_inputs = kv_ins + [hidden_states, history_len, ids_len, attention_mask]
        input_names = kv_in_names + ['hidden_states', 'history_len', 'ids_len', 'attention_mask']
        output_names = kv_out_names + ['logits', 'kv_seq_len']
        dynamic_axes = {**kv_axes, 'hidden_states': {0: 'batch', 1: 'ids_len'}, 'logits': {0: 'batch'}}
        model_main = LLM_MAIN(model, MAX_SEQ_LEN, num_heads, num_kv_heads, head_dim, num_layers, hidden_size)
        del model
        torch.onnx.export(
            model_main,
            tuple(all_inputs),
            onnx_model_B,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=OPSET,
            dynamo=False
        )
        del model_main, hidden_states, attention_mask, all_inputs
        gc.collect()

        # 6. Export Helper Models (Greedy, Reset, Argmax)
        print("Exporting Helper Models...")
        repeat_penality_in = torch.ones((BEAM_SIZE, vocab_size), dtype=torch.float32)
        penality_value = torch.tensor([REPEAT_PENALTY], dtype=torch.float32)
        torch.onnx.export(
            GREEDY_SEARCH(),
            (logits_t, repeat_penality_in, penality_value, beam_size_t),
            onnx_model_C,
            input_names=['logits', 'repeat_penality_in', 'penality_value', 'batch_size'],
            output_names=['max_logits_idx', 'repeat_penality_out'],
            dynamic_axes={
                'logits': {0: 'batch'},
                'repeat_penality_in': {0: 'batch'},
                'repeat_penality_out': {0: 'batch'},
                'max_logits_idx': {0: 'batch'}
            },
            opset_version=OPSET, dynamo=False
        )

        save_id_in = torch.zeros((BEAM_SIZE, 10), dtype=torch.int32)
        penality_reset_count_in = torch.zeros(BEAM_SIZE, dtype=torch.int32)
        batch_indices = torch.arange(BEAM_SIZE, dtype=torch.int64)
        torch.onnx.export(
            RESET_PENALITY(),
            (save_id_in, repeat_penality_in, penality_reset_count_in, batch_indices),
            onnx_model_F,
            input_names=['save_id_in', 'repeat_penality_in', 'penality_reset_count_in', 'batch_indices'],
            output_names=['save_id_out', 'repeat_penality_out', 'penality_reset_count_out'],
            dynamic_axes={
                'save_id_in': {0: 'batch', 1: 'history_len'},
                'save_id_out': {0: 'batch', 1: 'history_len'},
                'repeat_penality_in': {0: 'batch'},
                'repeat_penality_out': {0: 'batch'},
                'penality_reset_count_in': {0: 'batch'},
                'penality_reset_count_out': {0: 'batch'},
                'batch_indices': {0: 'batch'}
            },
            opset_version=OPSET,
            dynamo=False
        )

        torch.onnx.export(
            ARGMAX(),
            (logits_t,),
            onnx_model_G,
            input_names=['logits'], output_names=['max_logits_idx'],
            dynamic_axes={
                'logits': {0: 'batch'},
                'max_logits_idx': {0: 'batch'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del penality_reset_count_in

        # 7. Export Beam Search Models
        print("Exporting Beam Search Models...")
        kv_tensors_greedy = {k: v[[0]] for k, v in kv_tensors.items()} # Slice batch dim for first beam
        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors_greedy, 'batch_size', 'history_len', 'history_len')
        kv_axes = {k: v for k, v in kv_axes.items() if k not in kv_out_names}
        other_inputs = [logits_t[[0]], save_id_in, repeat_penality_in, penality_value, beam_size_t]
        other_names = ['logits', 'save_id_in', 'repeat_penality_in', 'penality_value', 'beam_size']
        dynamic_axes = {
            **kv_axes,
            'logits': {0: 'batch'},
            'save_id_in': {0: 'batch', 1: 'history_len'},
            'repeat_penality_in': {0: 'batch'},
            'top_beam_prob': {0: 'batch'},
            'top_beam_indices': {0: 'batch'},
            'max_logits_idx': {0: 'batch'},
            'batch_indices': {0: 'batch'},
            'save_id_out': {0: 'batch', 1: 'history_len'},
            'repeat_penality_out': {0: 'batch'}
        }
        num_layers_beam = num_layers * len(kv_specs)
        kv_ins_names = kv_in_names + other_names

        torch.onnx.export(
            FIRST_BEAM_SEARCH(num_layers_beam),
            tuple(kv_ins + other_inputs),
            onnx_model_D,
            input_names=kv_ins_names,
            output_names=['out_' + n[3:] for n in kv_in_names] + ['top_beam_indices', 'save_id_out', 'repeat_penality_out', 'top_beam_prob', 'batch_indices', 'max_logits_idx'], # Outputs mimic inputs for KV
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            dynamo=False
        )

        # Second Beam Search
        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors, 'batch', 'history_len', 'kv_seq_len')
        previous_prob = torch.zeros((BEAM_SIZE, 1), dtype=torch.float32)
        topK = torch.tensor([TOP_K], dtype=torch.int64)
        other_inputs = [logits_t, save_id_in, repeat_penality_in, previous_prob, batch_indices, penality_value, beam_size_t, topK]
        other_names = ['logits', 'save_id_in', 'repeat_penality_in', 'previous_prob', 'batch_indices', 'penality_value', 'beam_size', 'topK']
        dynamic_axes = {
            **kv_axes,
            'logits': {0: 'batch'},
            'save_id_in': {0: 'batch', 1: 'history_len'},
            'repeat_penality_in': {0: 'batch'},
            'previous_prob': {0: 'batch'},
            'batch_indices': {0: 'batch'},
            'save_id_out': {0: 'batch', 1: 'history_len'},
            'repeat_penality_out': {0: 'batch'},
            'top_beam_prob': {0: 'batch'},
            'top_beam_indices': {0: 'batch'},
            'max_logits_idx': {0: 'batch'}
        }

        torch.onnx.export(
            SECOND_BEAM_SEARCH(num_layers_beam),
            tuple(kv_ins + other_inputs),
            onnx_model_E,
            input_names=kv_in_names + other_names,
            output_names=kv_out_names + ['top_beam_indices', 'save_id_out', 'repeat_penality_out', 'top_beam_prob', 'max_logits_idx'],
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            dynamo=False
        )

        del kv_tensors, kv_tensors_greedy, repeat_penality_in, penality_value, save_id_in, batch_indices, previous_prob, topK, logits_t
        gc.collect()
    print('\nExport done!\n\nStart running the LLM by ONNXRuntime.\nNow loading . . . it could cost minutes.')


# Inference with ONNXRuntime
# =======================================================#
def bind_ort_values(binding, names, values, num=0):
    if num != 0:
        for i in range(num):
            binding.bind_ortvalue_input(names[i], values[i])
    else:
        for name, val in zip(names, values):
            binding.bind_ortvalue_input(name, val)


def bind_outputs_generic(binding, output_names, device_type, device_id):
    for name in output_names:
        binding.bind_output(name, device_type=device_type, device_id=device_id)


def create_ortvalue(data, dtype, device_type, device_id):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.array(data, dtype=dtype), device_type, device_id)


session_opts = onnxruntime.SessionOptions()
run_options = onnxruntime.RunOptions()
session_opts.log_severity_level = 4
session_opts.log_verbosity_level = 4
run_options.log_severity_level = 4
run_options.log_verbosity_level = 4
session_opts.inter_op_num_threads = MAX_THREADS
session_opts.intra_op_num_threads = MAX_THREADS
session_opts.enable_cpu_mem_arena = True
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry('session.set_denormal_as_zero', '1')
session_opts.add_session_config_entry('session.intra_op.allow_spinning', '1')
session_opts.add_session_config_entry('session.inter_op.allow_spinning', '1')
session_opts.add_session_config_entry('session.enable_quant_qdq_cleanup', '1')
session_opts.add_session_config_entry('session.qdq_matmulnbits_accuracy_level', '4')
session_opts.add_session_config_entry('session.use_device_allocator_for_initializers', '1')
session_opts.add_session_config_entry('session.graph_optimizations_loop_level', '2')
session_opts.add_session_config_entry('optimization.enable_gelu_approximation', '1')
session_opts.add_session_config_entry('optimization.minimal_build_optimizations', '')
session_opts.add_session_config_entry('optimization.enable_cast_chain_elimination', '1')
run_options.add_run_config_entry('disable_synchronize_execution_providers', '0')

if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_type': 'CPU',                         # [CPU, NPU, GPU, GPU.0, GPU.1]]
            'precision': 'ACCURACY',                      # [FP32, FP16, ACCURACY]
            'num_of_threads': MAX_THREADS if MAX_THREADS != 0 else 8,  # The default value is 8. Edit freely.
            'num_streams': 1,
            'enable_opencl_throttling': False,
            'enable_qdq_optimizer': False,                # Enable it carefully
            'disable_dynamic_shapes': False
        }
    ]
    device_type = 'cpu'
elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'gpu_mem_limit': 24 * 1024 * 1024 * 1024,     # 24 GB
            'arena_extend_strategy': 'kNextPowerOfTwo',   # ["kNextPowerOfTwo", "kSameAsRequested"]
            'cudnn_conv_algo_search': 'EXHAUSTIVE',       # ["DEFAULT", "HEURISTIC", "EXHAUSTIVE"]
            'sdpa_kernel': '2',                           # ["0", "1", "2"]
            'use_tf32': '1',
            'fuse_conv_bias': '0',                        # Set to '0' to avoid potential errors when enabled.
            'cudnn_conv_use_max_workspace': '1',
            'cudnn_conv1d_pad_to_nc1d': '0',
            'tunable_op_enable': '0',
            'tunable_op_tuning_enable': '0',
            'tunable_op_max_tuning_duration_ms': 10,
            'do_copy_in_default_stream': '0',
            'enable_cuda_graph': '0',                     # Set to '0' to avoid potential errors when enabled.
            'prefer_nhwc': '0',
            'enable_skip_layer_norm_strict_mode': '0',
            'use_ep_level_unified_stream': '0',
        }
    ]
    device_type = 'cuda'
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'performance_preference': 'high_performance',  # [high_performance, default, minimum_power]
            'device_filter': 'npu'                         # [any, npu, gpu]
        }
    ]
    device_type = 'dml'
else:
    # Please config by yourself for others providers.
    device_type = 'cpu'
    provider_options = None


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
binding_A = ort_session_A.io_binding()
in_name_A = ort_session_A.get_inputs()[0].name
out_name_A = [ort_session_A.get_outputs()[0].name]

ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
binding_B = ort_session_B.io_binding()
print(f"\nUsable Providers: {ort_session_B.get_providers()}")

model_dtype_B_str = ort_session_B._inputs_meta[0].type
in_name_B_objs = ort_session_B.get_inputs()
out_name_B_objs = ort_session_B.get_outputs()
amount_of_outputs_B = len(out_name_B_objs)
in_name_B = [x.name for x in in_name_B_objs]
out_name_B = [x.name for x in out_name_B_objs]
in_name_B_parts = in_name_B[:-2]

device_type_copy = device_type
if 'dml' in device_type:
    device_type = 'cpu'

if 'uint8' in model_dtype_B_str:
    model_dtype_B = np.uint8
    num_keys_values = (amount_of_outputs_B - 2) // 3
    num_layers = num_keys_values // 2
    scale_dtype_B = np.float16 if 'float16' in ort_session_B._inputs_meta[num_keys_values].type else np.float32
    k_scales = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_B._inputs_meta[0].shape[1], 1, 1, 0), dtype=scale_dtype_B), device_type, DEVICE_ID)
    k_biases = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_B._inputs_meta[0].shape[1], 1, 1, 0), dtype=scale_dtype_B), device_type, DEVICE_ID)
    v_scales = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_B._inputs_meta[num_layers].shape[1], 1, 0, 1), dtype=scale_dtype_B), device_type, DEVICE_ID)
    v_biases = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_B._inputs_meta[num_layers].shape[1], 1, 0, 1), dtype=scale_dtype_B), device_type, DEVICE_ID)
    num_keys_values = amount_of_outputs_B - 2  # Revert to original for later use
else:
    model_dtype_B = np.float16 if 'float16' in model_dtype_B_str else np.float32
    num_keys_values = amount_of_outputs_B - 2
    num_layers = num_keys_values // 2
    k_scales = None

device_type = device_type_copy
past_keys_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_B._inputs_meta[0].shape[1], 1, ort_session_B._inputs_meta[0].shape[3], 0), dtype=model_dtype_B), device_type, DEVICE_ID)
past_values_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_B._inputs_meta[num_layers].shape[1], 1, 0, ort_session_B._inputs_meta[num_layers].shape[4]), dtype=model_dtype_B), device_type, DEVICE_ID)

num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
num_keys_values_plus_3 = num_keys_values + 3
num_keys_values_plus_4 = num_keys_values + 4
num_keys_values_plus_5 = num_keys_values + 5
num_keys_values_plus_6 = num_keys_values + 6
num_keys_values_plus_7 = num_keys_values + 7
vocab_size = ort_session_B._outputs_meta[num_keys_values].shape[1]

topK = create_ortvalue([TOP_K], np.int64, device_type, DEVICE_ID)
beam_size = create_ortvalue([BEAM_SIZE], np.int64, device_type, DEVICE_ID)
init_ids_len_1 = create_ortvalue([1], np.int64, device_type, DEVICE_ID)
init_history_len = create_ortvalue([0], np.int64, device_type, DEVICE_ID)
init_attention_mask_0 = create_ortvalue([0], np.int8, device_type, DEVICE_ID)
init_attention_mask_1 = create_ortvalue([1], np.int8, device_type, DEVICE_ID)
init_batch_size_greedy = create_ortvalue([1], np.int64, device_type, DEVICE_ID)
init_save_id_beam = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((BEAM_SIZE, 0), dtype=np.int32), device_type, DEVICE_ID)

if USE_BEAM_SEARCH and (TOP_K < BEAM_SIZE):
    print("\nBeam Search does not display immediate decoding results...")
    TOP_K = BEAM_SIZE

if (TOP_K < 2) or (BEAM_SIZE < 2):
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")

do_repeat_penalty = (REPEAT_PENALTY != 1.0)

if USE_BEAM_SEARCH:
    ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    binding_D = ort_session_D.io_binding()
    in_name_D = [x.name for x in ort_session_D.get_inputs()]
    out_name_D = [x.name for x in ort_session_D.get_outputs()]
    in_name_D_parts = in_name_D[:num_keys_values_plus_1]
    ort_session_E = onnxruntime.InferenceSession(onnx_model_E, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    binding_E = ort_session_E.io_binding()
    in_name_E = [x.name for x in ort_session_E.get_inputs()]
    out_name_E = [x.name for x in ort_session_E.get_outputs()]
    in_name_E_parts = in_name_E[:num_keys_values_plus_1]
    ort_session_F = onnxruntime.InferenceSession(onnx_model_F, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    binding_F = ort_session_F.io_binding()
    in_name_F = [x.name for x in ort_session_F.get_inputs()]
    out_name_F = [x.name for x in ort_session_F.get_outputs()]

    penality_dtype = np.float16 if 'float16' in ort_session_D._inputs_meta[num_keys_values_plus_3].type else np.float32
    penality_value = create_ortvalue([REPEAT_PENALTY], penality_dtype, device_type, DEVICE_ID)
    init_repeat_penality = onnxruntime.OrtValue.ortvalue_from_numpy(np.ones((BEAM_SIZE, vocab_size), dtype=penality_dtype), device_type, DEVICE_ID)
    init_penality_reset_count = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(BEAM_SIZE, dtype=np.int32), device_type, DEVICE_ID)

    binding_F.bind_ortvalue_input(in_name_F[2], init_penality_reset_count)
    binding_D.bind_ortvalue_input(in_name_D[num_keys_values_plus_1], init_save_id_beam)
    binding_D.bind_ortvalue_input(in_name_D[num_keys_values_plus_2], init_repeat_penality)
    binding_D.bind_ortvalue_input(in_name_D[num_keys_values_plus_3], penality_value)
    binding_D.bind_ortvalue_input(in_name_D[num_keys_values_plus_4], beam_size)
    binding_E.bind_ortvalue_input(in_name_E[num_keys_values_plus_5], penality_value)
    binding_E.bind_ortvalue_input(in_name_E[num_keys_values_plus_6], beam_size)
    binding_E.bind_ortvalue_input(in_name_E[num_keys_values_plus_7], topK)
else:
    BEAM_SIZE = 1
    save_id_greedy = np.zeros(MAX_SEQ_LEN, dtype=np.int32)
    if do_repeat_penalty:
        ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
        binding_C = ort_session_C.io_binding()
        in_name_C = [x.name for x in ort_session_C.get_inputs()]
        out_name_C = [x.name for x in ort_session_C.get_outputs()]
        penality_dtype = np.float16 if 'float16' in ort_session_C._inputs_meta[2].type else np.float32
        penality_value = create_ortvalue([REPEAT_PENALTY], penality_dtype, device_type, DEVICE_ID)
        current_penalty = onnxruntime.OrtValue.ortvalue_from_numpy(np.ones((BEAM_SIZE, vocab_size), dtype=penality_dtype),device_type, DEVICE_ID)
        next_penalty = onnxruntime.OrtValue.ortvalue_from_numpy(np.ones((BEAM_SIZE, vocab_size), dtype=penality_dtype), device_type, DEVICE_ID)
        binding_C.bind_ortvalue_input(in_name_C[2], penality_value)
        binding_C.bind_ortvalue_input(in_name_C[3], init_batch_size_greedy)
        penalty_shape = (BEAM_SIZE, vocab_size)
        binding_C.bind_output(out_name_C[0], device_type=device_type, device_id=DEVICE_ID)
        binding_C.bind_output(name=out_name_C[1], device_type=device_type, device_id=DEVICE_ID, element_type=penality_dtype, shape=penalty_shape, buffer_ptr=next_penalty.data_ptr())
        init_penality_reset_count = 0
    else:
        ort_session_G = onnxruntime.InferenceSession(onnx_model_G, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
        binding_G = ort_session_G.io_binding()
        in_name_G = ort_session_G.get_inputs()[0].name
        out_name_G = [ort_session_G.get_outputs()[0].name]
        penality_dtype = np.float32

if TEST_THINK_MODE:
    prompt = f'<|im_start|>user\n{TEST_QUERY}<|im_end|>\n<|im_start|>assistant\n'
else:
    prompt = f'<|im_start|>user\n{TEST_QUERY}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
tokens = tokenizer(prompt, return_tensors='np')['input_ids'].astype(np.int32)
num_prefill = tokens.shape[-1]
input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, DEVICE_ID)
ids_len = create_ortvalue([num_prefill], np.int64, device_type, DEVICE_ID)

binding_A.bind_ortvalue_input(in_name_A, input_ids)
bind_outputs_generic(binding_A, out_name_A, device_type, DEVICE_ID)
ort_session_A.run_with_iobinding(binding_A, run_options=run_options)
out_A = binding_A.get_outputs()[0]

binding_B.bind_ortvalue_input(in_name_B[num_keys_values], out_A)
binding_B.bind_ortvalue_input(in_name_B[num_keys_values_plus_1], init_history_len)
binding_B.bind_ortvalue_input(in_name_B[num_keys_values_plus_2], ids_len)
binding_B.bind_ortvalue_input(in_name_B[num_keys_values_plus_3], init_attention_mask_1)

i = 0
j = num_layers
while i < j:
    binding_B.bind_ortvalue_input(in_name_B[i], past_keys_B)
    i += 1
j = i + num_layers
while i < j:
    binding_B.bind_ortvalue_input(in_name_B[i], past_values_B)
    i += 1

if k_scales:
    j = i + num_layers
    while i < j:
        binding_B.bind_ortvalue_input(in_name_B[i], k_scales)
        i += 1
    j = i + num_layers
    while i < j:
        binding_B.bind_ortvalue_input(in_name_B[i], k_biases)
        i += 1
    j = i + num_layers
    while i < j:
        binding_B.bind_ortvalue_input(in_name_B[i], v_scales)
        i += 1
    j = i + num_layers
    while i < j:
        binding_B.bind_ortvalue_input(in_name_B[i], v_biases)
        i += 1

print(f'\n\nTest Question: {TEST_QUERY}\nLLM Answering:')
num_decode = 0
generate_limit = MAX_SEQ_LEN - num_prefill
start_time = time.time()
while num_decode < generate_limit:
    bind_outputs_generic(binding_B, out_name_B, device_type, DEVICE_ID)
    ort_session_B.run_with_iobinding(binding_B, run_options=run_options)
    all_outputs_B = binding_B.get_outputs()
    if USE_BEAM_SEARCH:
        if num_decode < 1:
            bind_ort_values(binding_D, in_name_D_parts, all_outputs_B)
            bind_outputs_generic(binding_D, out_name_D, device_type, DEVICE_ID)
            ort_session_D.run_with_iobinding(binding_D, run_options=run_options)
            all_outputs_D = binding_D.get_outputs()
            max_logits_idx = all_outputs_D[num_keys_values_plus_5].numpy()
            if max_logits_idx in STOP_TOKEN:
                break
            binding_E.bind_ortvalue_input(in_name_E[num_keys_values_plus_4], all_outputs_D[num_keys_values_plus_4])
            if do_repeat_penalty:
                binding_F.bind_ortvalue_input(in_name_F[3], all_outputs_D[num_keys_values_plus_4])
        else:
            bind_ort_values(binding_E, in_name_E_parts, all_outputs_B)
            bind_outputs_generic(binding_E, out_name_E, device_type, DEVICE_ID)
            ort_session_E.run_with_iobinding(binding_E, run_options=run_options)
            all_outputs_E = binding_E.get_outputs()
            max_logits_idx = all_outputs_E[num_keys_values_plus_4].numpy()
            if max_logits_idx in STOP_TOKEN:
                break
        if do_repeat_penalty and (num_decode >= PENALTY_RANGE):
            binding_F.bind_ortvalue_input(in_name_F[0], all_outputs_E[num_keys_values_plus_1])
            binding_F.bind_ortvalue_input(in_name_F[1], all_outputs_E[num_keys_values_plus_2])
            bind_outputs_generic(binding_F, out_name_F, device_type, DEVICE_ID)
            ort_session_F.run_with_iobinding(binding_F, run_options=run_options)
            all_outputs_F = binding_F.get_outputs()
            binding_F.bind_ortvalue_input(in_name_F[2], all_outputs_F[2])
            binding_E.bind_ortvalue_input(in_name_E[num_keys_values_plus_1], all_outputs_F[0])
            binding_E.bind_ortvalue_input(in_name_E[num_keys_values_plus_2], all_outputs_F[1])
        if num_decode < 1:
            bind_ort_values(binding_B, in_name_B_parts, all_outputs_D)
            binding_A.bind_ortvalue_input(in_name_A, all_outputs_D[num_keys_values])
            binding_E.bind_ortvalue_input(in_name_E[num_keys_values_plus_1], all_outputs_D[num_keys_values_plus_1])
            binding_E.bind_ortvalue_input(in_name_E[num_keys_values_plus_2], all_outputs_D[num_keys_values_plus_2])
            binding_E.bind_ortvalue_input(in_name_E[num_keys_values_plus_3], all_outputs_D[num_keys_values_plus_3])
        else:
            bind_ort_values(binding_B, in_name_B_parts, all_outputs_E)
            binding_A.bind_ortvalue_input(in_name_A, all_outputs_E[num_keys_values])
            binding_E.bind_ortvalue_input(in_name_E[num_keys_values_plus_1], all_outputs_E[num_keys_values_plus_1])
            binding_E.bind_ortvalue_input(in_name_E[num_keys_values_plus_2], all_outputs_E[num_keys_values_plus_2])
            binding_E.bind_ortvalue_input(in_name_E[num_keys_values_plus_3], all_outputs_E[num_keys_values_plus_3])
    else:
        if do_repeat_penalty:
            binding_C.bind_ortvalue_input(in_name_C[0], all_outputs_B[num_keys_values])
            binding_C.bind_ortvalue_input(in_name_C[1], current_penalty)
            ort_session_C.run_with_iobinding(binding_C, run_options=run_options)
            all_outputs_C = binding_C.get_outputs()
            max_logits_idx = all_outputs_C[0].numpy().flat[0]
            if max_logits_idx in STOP_TOKEN:
                break
            if num_decode >= PENALTY_RANGE:
                reset_ids = save_id_greedy[init_penality_reset_count]
                if reset_ids != max_logits_idx:
                    tmp = next_penalty.numpy()
                    tmp[:, reset_ids] = 1.0
                    next_penalty.update_inplace(tmp)
                init_penality_reset_count += 1
            current_penalty, next_penalty = next_penalty, current_penalty
            binding_C.bind_output(
                name=out_name_C[1],
                device_type=device_type,
                device_id=DEVICE_ID,
                element_type=penality_dtype,
                shape=penalty_shape,
                buffer_ptr=next_penalty.data_ptr()
            )
            binding_A.bind_ortvalue_input(in_name_A, all_outputs_C[0])
        else:
            binding_G.bind_ortvalue_input(in_name_G, all_outputs_B[num_keys_values])
            bind_outputs_generic(binding_G, out_name_G, device_type, DEVICE_ID)
            ort_session_G.run_with_iobinding(binding_G)
            all_outputs_G = binding_G.get_outputs()
            binding_A.bind_ortvalue_input(in_name_A, all_outputs_G[0])
            max_logits_idx = all_outputs_G[0].numpy().flat[0]
            if max_logits_idx in STOP_TOKEN:
                break
        bind_ort_values(binding_B, in_name_B_parts, all_outputs_B)
        save_id_greedy[num_decode] = max_logits_idx
        print(tokenizer.decode(max_logits_idx), end="", flush=True)
    bind_outputs_generic(binding_A, out_name_A, device_type, DEVICE_ID)
    ort_session_A.run_with_iobinding(binding_A)
    binding_B.bind_ortvalue_input(in_name_B[num_keys_values], binding_A.get_outputs()[0])
    binding_B.bind_ortvalue_input(in_name_B[num_keys_values_plus_1], all_outputs_B[num_keys_values_plus_1])
    if num_decode < 1:
        binding_B.bind_ortvalue_input(in_name_B[num_keys_values_plus_2], init_ids_len_1)
        binding_B.bind_ortvalue_input(in_name_B[num_keys_values_plus_3], init_attention_mask_0)
    num_decode += 1

if USE_BEAM_SEARCH:
    result = tokenizer.decode(all_outputs_E[num_keys_values_plus_1].numpy()[0, :num_decode], skip_special_tokens=True)
else:
    result = tokenizer.decode(save_id_greedy[:num_decode], skip_special_tokens=True)

elapsed_time = time.time() - start_time
tokens_per_second = (num_decode + 1) / elapsed_time

print(f"\n\nFinal:\n{result}\n\nDecode: {tokens_per_second:.3f} token/s")
print(f"Total tokens generated: {num_decode}")
print(f"Total time: {elapsed_time:.3f}s")
