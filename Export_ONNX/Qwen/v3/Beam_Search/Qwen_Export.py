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

# KV cache quantization
KV_QUANT_DTYPE = "Q8"               # "Q4" | "Q8" | "F16" | "F32"
KV_BLOCK_SIZE = 1024                # Block size for KV quantization. block_size <= (num_key_value_heads * head_dim). If use Q4, block_size recommended to <= 8.
USE_FLOAT16_SCALE_BIAS = False      # Set to True for ARM64 devices or GPU that perform better with float16 arithmetic.

# Test input
TEST_THINK_MODE = False
TEST_QUERY = "地球最高的山是哪座山？"

# Decoding strategy
STOP_TOKEN = (151643, 151645)       # Qwen stop token ids
USE_BEAM_SEARCH = False             # Use beam search or greedy search
REPEAT_PENALTY = 1.0                # 0.0 ~ 1.0; No penalty = 1.0
PENALTY_RANGE = 20                  # Recent-token window to apply penalty
MAX_BEAM_SIZE = 10                  # Max beam size for beam search. Can not edit after export.
TOP_K = 3
BEAM_SIZE = 3

# Runtime config
MAX_SEQ_LEN = 4096                  # Max context length. Can not edit after export.
MAX_THREADS = 0                     # 0 = auto
OPSET = 17
DEVICE_ID = 0


if USE_FLOAT16_SCALE_BIAS:
    scale_dtype = torch.float16
else:
    scale_dtype = torch.float32


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
    def __init__(self, quant_dtype, block_size, scale_dtype):
        super().__init__()
        self.dtype = quant_dtype
        self.block_size = block_size
        self.block_size_half = block_size // 2

        if self.dtype == "Q4":
            assert block_size % 2 == 0, "Block size must be even for Q4 packing"
            self.qmax = 15.0
            self.register_buffer('pack_multiplier', torch.tensor([1, 16], dtype=torch.uint8))
        else:
            self.qmax = 255.0

        self.register_buffer('inv_qmax', torch.tensor([1.0 / self.qmax], dtype=scale_dtype).view(1, 1, -1))
        self.register_buffer('eps', torch.tensor([1e-6], dtype=scale_dtype).view(1, 1, -1))

    def _quantize_block(self, x, batch_size):
        x_blocks = x.view(batch_size, -1, self.block_size)

        block_min = x_blocks.min(dim=-1, keepdim=True).values
        block_max = x_blocks.max(dim=-1, keepdim=True).values
        scale = (block_max - block_min) * self.inv_qmax

        x_normalized = (x_blocks - block_min) / (scale + self.eps)
        x_packed = torch.clamp(torch.round(x_normalized), max=self.qmax).to(torch.uint8)

        if self.dtype == "Q4":
            x_pairs = x_packed.view(batch_size, -1, self.block_size_half, 2)
            x_packed = (x_pairs * self.pack_multiplier).sum(dim=-1, dtype=torch.uint8)

        return x_packed, scale, block_min

    def forward(self, keys, values, batch_size):
        k_packed, k_scale, k_bias = self._quantize_block(keys, batch_size)
        v_packed, v_scale, v_bias = self._quantize_block(values, batch_size)
        return k_packed, k_scale, k_bias, v_packed, v_scale, v_bias


class KVDequantizer(torch.nn.Module):
    def __init__(self, quant_dtype, block_size, num_key_value_heads, head_dim, scale_dtype):
        super().__init__()
        self.dtype = quant_dtype
        self.block_size = block_size
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.scale_dtype = scale_dtype

    def _dequantize_block(self, x_packed, scale, bias, is_k, batch_size):
        if self.dtype == "Q4":
            low = x_packed % 16
            high = x_packed // 16
            x_packed = torch.stack([low, high], dim=3).view(batch_size, -1, self.block_size)

        x_flat = x_packed.to(self.scale_dtype) * scale + bias

        if is_k:
            return x_flat.view(batch_size, self.num_key_value_heads, 1, self.head_dim, -1)
        else:
            return x_flat.view(batch_size, self.num_key_value_heads, 1, -1, self.head_dim)

    def forward(
            self,
            k_packed, k_scale, k_bias,
            v_packed, v_scale, v_bias,
            batch_size
    ):
        k_rec = self._dequantize_block(k_packed, k_scale, k_bias, True, batch_size)
        v_rec = self._dequantize_block(v_packed, v_scale, v_bias, False, batch_size)
        return k_rec, v_rec


class LLM_EMBED(torch.nn.Module):
    def __init__(self, llm):
        super(LLM_EMBED, self).__init__()
        self.embed_tokens = llm.model.embed_tokens.float()

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


class LLM_MAIN(torch.nn.Module):
    def __init__(self, llm, max_seq_len, num_heads, num_key_value_heads, head_dim, num_layers, scale_dtype):
        super(LLM_MAIN, self).__init__()

        # --- Core config ---
        self.llm = llm
        self.head_dim = int(head_dim)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.num_layers_2 = self.num_layers * 2
        self.num_layers_3 = self.num_layers * 3
        self.num_layers_4 = self.num_layers * 4
        self.num_layers_5 = self.num_layers * 5

        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim_half = self.head_dim // 2
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.variance_epsilon = torch.tensor([1e-6], dtype=torch.float32)

        # --- KV cache quant flags ---
        self.kv_f16 = (KV_QUANT_DTYPE == "F16")
        self.kv_f32 = (KV_QUANT_DTYPE == "F32")
        self.kv_q4 = (KV_QUANT_DTYPE == "Q4")
        self.kv_q8 = (KV_QUANT_DTYPE == "Q8")

        # --- Quantizer / Dequantizer ---
        self.quantizer = KVQuantizer(KV_QUANT_DTYPE, KV_BLOCK_SIZE, scale_dtype).eval()
        self.dequantizer = KVDequantizer(
            KV_QUANT_DTYPE, KV_BLOCK_SIZE, self.num_key_value_heads, self.head_dim, scale_dtype
        ).eval()

        # --- RoPE buffers (precompute once) ---
        scale_factor = float(self.head_dim ** -0.25)
        position_ids = torch.arange(int(max_seq_len), dtype=torch.float32).unsqueeze(-1)            # [S, 1]
        idx_theta = (position_ids * self.llm.model.rotary_emb.inv_freq).unsqueeze(0).unsqueeze(0)   # [1, 1, S, D/2]
        cos = torch.cos(idx_theta) * scale_factor
        sin = torch.sin(idx_theta) * scale_factor
        cos = torch.cat((cos, cos), dim=-1).half()  # [1, 1, S, D]
        sin = torch.cat((sin, sin), dim=-1).half()

        self.register_buffer("cos_rotary_pos_emb", cos, persistent=False)
        self.register_buffer("sin_rotary_pos_emb", sin, persistent=False)

        # --- Causal mask buffer ---
        mask = (1 - torch.tril(torch.ones([1, 1, int(max_seq_len), int(max_seq_len)], dtype=torch.int8))) * -128
        self.register_buffer("attention_mask", mask, persistent=False)

        # --- KV output holders ---
        self.save_key = [None] * self.num_layers
        self.save_value = [None] * self.num_layers
        if self.kv_q4 or self.kv_q8:
            self.save_k_scale = [None] * self.num_layers
            self.save_k_bias = [None] * self.num_layers
            self.save_v_scale = [None] * self.num_layers
            self.save_v_bias = [None] * self.num_layers

        # --- Fuse / rearrange weights (all surgery under no_grad) ---
        with torch.no_grad():
            # 1) Fuse q/k/v into qkv
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
                    device = qkv.weight.device
                    dtype = qkv.weight.dtype
                    qb = q_proj.bias if q_proj.bias is not None else torch.zeros(q_proj.out_features, dtype=dtype)
                    kb = k_proj.bias if k_proj.bias is not None else torch.zeros(k_proj.out_features, dtype=dtype)
                    vb = v_proj.bias if v_proj.bias is not None else torch.zeros(v_proj.out_features, dtype=dtype)
                    qkv.bias.copy_(torch.cat([qb, kb, vb], dim=0))

                layer.self_attn.qkv = qkv
                del layer.self_attn.q_proj
                del layer.self_attn.k_proj
                del layer.self_attn.v_proj

            # 2) Fuse input rmsnorm weight into qkv input columns
            for layer in self.llm.model.layers:
                w = layer.input_layernorm.weight   # [hidden]
                qkv = layer.self_attn.qkv          # weight: [out, in]
                qkv.weight.mul_(w.to(dtype=qkv.weight.dtype).unsqueeze(0))
                del layer.input_layernorm

            # 3) Fuse post-attention rmsnorm weight into MLP gate/up input columns
            for layer in self.llm.model.layers:
                w = layer.post_attention_layernorm.weight  # [hidden]
                gate = layer.mlp.gate_proj
                up = layer.mlp.up_proj
                gate.weight.mul_(w.to(dtype=gate.weight.dtype).unsqueeze(0))
                up.weight.mul_(w.to(dtype=up.weight.dtype).unsqueeze(0))
                del layer.post_attention_layernorm

            # 4) Fuse final norm weight into lm_head
            norm_w = self.llm.model.norm.weight.to(dtype=self.llm.lm_head.weight.dtype)  # [hidden]
            self.llm.lm_head.weight.mul_(norm_w.unsqueeze(0))                            # [vocab, hidden] *= [1, hidden]
            del self.llm.model.norm


    def rotate_half(self, x, head_dim_half, dim):
        x1, x2 = torch.split(x, [head_dim_half, head_dim_half], dim=dim)
        return torch.cat((-x2, x1), dim=dim)

    def repeat_k(self, kv_states, num_key_value_groups, head_dim, num_heads, batch_size):
        return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=2).view(batch_size, num_heads, head_dim, -1)

    def repeat_v(self, kv_states, num_key_value_groups, head_dim, num_heads, batch_size):
        return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=2).view(batch_size, num_heads, -1, head_dim)

    def forward(self, *all_inputs):
        hidden_states = all_inputs[-4]
        history_len = all_inputs[-3]
        ids_len = all_inputs[-2]
        kv_seq_len = history_len + ids_len
        rotary_pos_emb_cos_q = self.cos_rotary_pos_emb[..., history_len:kv_seq_len, :].float()
        rotary_pos_emb_sin_q = self.sin_rotary_pos_emb[..., history_len:kv_seq_len, :].float()
        rotary_pos_emb_cos_k = rotary_pos_emb_cos_q.transpose(-1, -2).unsqueeze(0)
        rotary_pos_emb_sin_k = rotary_pos_emb_sin_q.transpose(-1, -2).unsqueeze(0)
        attention_mask = (self.attention_mask[..., :ids_len, :kv_seq_len] * all_inputs[-1]).float()
        batch_size = hidden_states.shape[0].unsqueeze(0)
        for i, layer in enumerate(self.llm.model.layers):
            hidden_states_norm = hidden_states * torch.rsqrt(hidden_states.square().mean(-1, keepdim=True) + self.variance_epsilon)
            qkv = layer.self_attn.qkv(hidden_states_norm)
            q, k, v = torch.split(qkv, [layer.self_attn.q_out_features, layer.self_attn.k_out_features, layer.self_attn.v_out_features], dim=-1,)
            q = q.view(batch_size, -1, self.num_heads, self.head_dim)
            k = k.view(batch_size, -1, 1, self.num_key_value_heads, self.head_dim)
            v = v.view(batch_size, -1, 1, self.num_key_value_heads, self.head_dim).transpose(1, 3)
            q = layer.self_attn.q_norm(q).transpose(1, 2)
            k = layer.self_attn.k_norm(k).permute(0, 3, 2, 4, 1)
            q = q * rotary_pos_emb_cos_q + self.rotate_half(q, self.head_dim_half, -1) * rotary_pos_emb_sin_q
            k = k * rotary_pos_emb_cos_k + self.rotate_half(k, self.head_dim_half, -2) * rotary_pos_emb_sin_k
            if self.kv_f32:
                k = torch.cat((all_inputs[i], k), dim=-1)
                v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)
                self.save_key[i] = k
                self.save_value[i] = v
                k = self.repeat_k(k, self.num_key_value_groups, self.head_dim, self.num_heads, batch_size)
                v = self.repeat_v(v, self.num_key_value_groups, self.head_dim, self.num_heads, batch_size)
            elif self.kv_f16:
                k = torch.cat((all_inputs[i], k.half()), dim=-1)
                v = torch.cat((all_inputs[i + self.num_layers], v.half()), dim=-2)
                self.save_key[i] = k
                self.save_value[i] = v
                k = self.repeat_k(k, self.num_key_value_groups, self.head_dim, self.num_heads, batch_size).float()
                v = self.repeat_v(v, self.num_key_value_groups, self.head_dim, self.num_heads, batch_size).float()
            elif self.kv_q4 or self.kv_q8:
                unpacked_k, unpacked_v = self.dequantizer(
                    all_inputs[i],
                    all_inputs[i + self.num_layers_2],
                    all_inputs[i + self.num_layers_3],
                    all_inputs[i + self.num_layers],
                    all_inputs[i + self.num_layers_4],
                    all_inputs[i + self.num_layers_5],
                    batch_size,
                )
                if USE_FLOAT16_SCALE_BIAS:
                    k = k.half()
                    v = v.half()
                k = torch.cat((unpacked_k, k), dim=-1)
                v = torch.cat((unpacked_v, v), dim=-2)
                self.save_key[i], self.save_k_scale[i], self.save_k_bias[i], self.save_value[i], self.save_v_scale[i], self.save_v_bias[i] = self.quantizer(k, v, batch_size)
                k = self.repeat_k(k, self.num_key_value_groups, self.head_dim, self.num_heads, batch_size)
                v = self.repeat_v(v, self.num_key_value_groups, self.head_dim, self.num_heads, batch_size)
                if USE_FLOAT16_SCALE_BIAS:
                    k = k.float()
                    v = v.float()
            attn = torch.nn.functional.softmax(torch.matmul(q, k) + attention_mask, dim=-1, dtype=torch.float32)
            attn = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, -1, layer.self_attn.o_proj.in_features)
            attn_out = layer.self_attn.o_proj(attn)
            hidden_states += attn_out
            residual = hidden_states
            hidden_states = hidden_states * torch.rsqrt(hidden_states.square().mean(-1, keepdim=True) + self.variance_epsilon)
            hidden_states = layer.mlp(hidden_states)
            hidden_states += residual
        hidden_states = hidden_states[:, -1]
        hidden_states = hidden_states * torch.rsqrt(hidden_states.square().mean(-1, keepdim=True) + self.variance_epsilon)
        logits = self.llm.lm_head(hidden_states)
        if self.kv_q4 or self.kv_q8:
            return *self.save_key, *self.save_value, self.save_k_scale, self.save_k_bias, self.save_v_scale, self.save_v_bias, logits, kv_seq_len
        return *self.save_key, *self.save_value, logits, kv_seq_len


print('Export start ...')
with torch.inference_mode():
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.float32,
        device_map='cpu',
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).eval()

    head_dim = model.config.head_dim
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads
    vocab_size = model.model.vocab_size
    hidden_size = model.model.embed_tokens.embedding_dim

    batch_size = 3
    ids_len = torch.tensor([10], dtype=torch.int64)
    history_len = torch.tensor([0], dtype=torch.int64)
    input_ids = torch.ones((1, ids_len), dtype=torch.int32)
    hidden_states = torch.ones((batch_size, ids_len, hidden_size), dtype=torch.float32)
    attention_mask = torch.tensor([1], dtype=torch.int8)

    # 1. Export Embedding Model (Model A)
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
    del model_A
    del input_ids

    # 2. Prepare Inputs for Main Model (Model B)
    all_inputs = []
    input_names = []
    output_names = []
    dynamic_axes = {'hidden_states': {0: 'batch', 1: 'ids_len'}}

    if KV_QUANT_DTYPE == "Q4" or KV_QUANT_DTYPE == "Q8":
        total_elements = num_key_value_heads * head_dim * history_len
        num_blocks = total_elements // KV_BLOCK_SIZE
        k_scale = torch.ones([batch_size, num_blocks, 1], dtype=scale_dtype)
        k_bias = torch.ones([batch_size, num_blocks, 1], dtype=scale_dtype)
        v_scale = torch.ones([batch_size, num_blocks, 1], dtype=scale_dtype)
        v_bias = torch.ones([batch_size, num_blocks, 1], dtype=scale_dtype)
        if KV_QUANT_DTYPE == "Q4":
            past_keys = torch.randint(0, 256, (batch_size, num_blocks, KV_BLOCK_SIZE // 2), dtype=torch.uint8)
            past_values = torch.randint(0, 256, (batch_size, num_blocks, KV_BLOCK_SIZE // 2), dtype=torch.uint8)
        elif KV_QUANT_DTYPE == "Q8":
            past_keys = torch.randint(0, 256, (batch_size, num_blocks, KV_BLOCK_SIZE), dtype=torch.uint8)
            past_values = torch.randint(0, 256, (batch_size, num_blocks, KV_BLOCK_SIZE), dtype=torch.uint8)

        for i in range(num_layers):
            name = f'in_key_{i}'
            input_names.append(name)
            all_inputs.append(past_keys)
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_blocks'}
            name = f'out_key_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_blocks_plus'}
        for i in range(num_layers):
            name = f'in_value_{i}'
            input_names.append(name)
            all_inputs.append(past_values)
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_blocks'}
            name = f'out_value_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_blocks_plus'}
        for i in range(num_layers):
            name = f'in_key_scale_{i}'
            input_names.append(name)
            all_inputs.append(k_scale)
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_blocks'}
            name = f'out_key_scale_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_blocks_plus'}
        for i in range(num_layers):
            name = f'in_key_bias_{i}'
            input_names.append(name)
            all_inputs.append(k_bias)
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_blocks'}
            name = f'out_key_bias_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_blocks_plus'}
        for i in range(num_layers):
            name = f'in_value_scale_{i}'
            input_names.append(name)
            all_inputs.append(v_scale)
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_blocks'}
            name = f'out_value_scale_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_blocks_plus'}
        for i in range(num_layers):
            name = f'in_value_bias_{i}'
            input_names.append(name)
            all_inputs.append(v_bias)
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_blocks'}
            name = f'out_value_bias_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_blocks_plus'}
    else:
        if KV_QUANT_DTYPE == "F16":
            kv_dtype = torch.float16
        else:
            kv_dtype = torch.float32
        past_keys = torch.zeros((batch_size, num_key_value_heads, 1, head_dim, history_len), dtype=kv_dtype)
        past_values = torch.zeros((batch_size, num_key_value_heads, 1, history_len, head_dim), dtype=kv_dtype)
        for i in range(num_layers):
            name = f'in_key_{i}'
            input_names.append(name)
            all_inputs.append(past_keys)
            dynamic_axes[name] = {0: 'batch', 4: 'history_len'}
            name = f'out_key_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch', 4: 'ks_seq_len'}
        for i in range(num_layers):
            name = f'in_value_{i}'
            input_names.append(name)
            all_inputs.append(past_values)
            dynamic_axes[name] = {0: 'batch', 3: 'history_len'}
            name = f'out_value_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch', 3: 'ks_seq_len'}

    input_names.append('hidden_states')
    all_inputs.append(hidden_states)
    input_names.append('history_len')
    all_inputs.append(history_len)
    input_names.append('ids_len')
    all_inputs.append(ids_len)
    input_names.append('attention_mask')
    all_inputs.append(attention_mask)
    output_names.append('logits')
    output_names.append('kv_seq_len')
    dynamic_axes['logits'] = {0: 'batch'}

    # Export Main Model (Model B)
    model = LLM_MAIN(model, MAX_SEQ_LEN, num_heads, num_key_value_heads, head_dim, num_layers, scale_dtype)
    torch.onnx.export(
        model,
        tuple(all_inputs),
        onnx_model_B,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=OPSET,
        dynamo=False
    )
    del model
    del input_names
    del output_names
    del dynamic_axes
    del all_inputs
    del num_heads
    del num_key_value_heads
    del head_dim
    del hidden_size
    del ids_len
    del history_len
    del batch_size
    del hidden_states
    del attention_mask
    gc.collect()

    # 3. Export Greedy Search (Model C)
    greedy = GREEDY_SEARCH()
    beam_size = torch.tensor([BEAM_SIZE], dtype=torch.int64)
    repeat_penality = torch.ones((beam_size, vocab_size), dtype=torch.float32)
    penality_reset_count = torch.zeros(beam_size, dtype=torch.int32)
    logits = torch.ones((beam_size, vocab_size), dtype=torch.float32)
    penality_value = torch.tensor([REPEAT_PENALTY], dtype=torch.float32)
    batch_indices = torch.arange(BEAM_SIZE, dtype=torch.int64)

    torch.onnx.export(
        greedy,
        (logits, repeat_penality, penality_value, beam_size),
        # Reuse the beam_size tensor as batch_size during export process.
        onnx_model_C,
        input_names=['logits', 'repeat_penality_in', 'penality_value', 'batch_size'],
        output_names=['max_logits_idx', 'repeat_penality_out'],
        dynamic_axes={
            'logits': {0: 'batch'},
            'repeat_penality_in': {0: 'batch'},
            'repeat_penality_out': {0: 'batch'},
            'max_logits_idx': {0: 'batch'}
        },
        opset_version=OPSET,
        dynamo=False
    )
    del greedy

    # 4. Export First Beam Search (Model D)
    first_beam_search = FIRST_BEAM_SEARCH(num_layers * 2 if (KV_QUANT_DTYPE == "F16" or KV_QUANT_DTYPE == "F32") else num_layers * 6)
    topK = torch.tensor([TOP_K], dtype=torch.int64)
    save_id = torch.zeros((beam_size, 10), dtype=torch.int32)
    previous_prob = torch.zeros((beam_size, 1), dtype=torch.float32)
    past_keys_greedy = past_keys[[0]]
    past_values_greedy = past_values[[0]]

    all_inputs = []
    input_names = []
    output_names = []
    dynamic_axes = {}
    if KV_QUANT_DTYPE == "Q4" or KV_QUANT_DTYPE == "Q8":
        k_scale_greedy = k_scale[[0]]
        k_bias_greedy = k_bias[[0]]
        v_scale_greedy = v_scale[[0]]
        v_bias_greedy = v_bias[[0]]
        for i in range(num_layers):
            name = f'in_key_{i}'
            input_names.append(name)
            all_inputs.append(past_keys_greedy)
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_blocks'}
            name = f'out_key_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_blocks_plus'}
        for i in range(num_layers):
            name = f'in_value_{i}'
            input_names.append(name)
            all_inputs.append(past_values_greedy)
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_blocks'}
            name = f'out_value_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_blocks_plus'}
        for i in range(num_layers):
            name = f'in_key_scale_{i}'
            input_names.append(name)
            all_inputs.append(k_scale_greedy)
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_blocks'}
            name = f'out_key_scale_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_blocks_plus'}
        for i in range(num_layers):
            name = f'in_key_bias_{i}'
            input_names.append(name)
            all_inputs.append(k_bias_greedy)
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_blocks'}
            name = f'out_key_bias_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_blocks_plus'}
        for i in range(num_layers):
            name = f'in_value_scale_{i}'
            input_names.append(name)
            all_inputs.append(v_scale_greedy)
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_blocks'}
            name = f'out_value_scale_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_blocks_plus'}
        for i in range(num_layers):
            name = f'in_value_bias_{i}'
            input_names.append(name)
            all_inputs.append(v_bias_greedy)
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_blocks'}
            name = f'out_value_bias_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_blocks_plus'}
    else:
        for i in range(num_layers):
            name = f'in_key_{i}'
            input_names.append(name)
            all_inputs.append(past_keys_greedy)
            dynamic_axes[name] = {0: 'batch', 4: 'history_len'}
            name = f'out_key_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch', 4: 'kv_seq_len'}
        for i in range(num_layers):
            name = f'in_value_{i}'
            input_names.append(name)
            all_inputs.append(past_values_greedy)
            dynamic_axes[name] = {0: 'batch', 3: 'history_len'}
            name = f'out_value_{i}'
            output_names.append(name)
            dynamic_axes[name] = {0: 'batch', 3: 'kv_seq_len'}

    input_names.append('logits')
    all_inputs.append(logits[[0]])
    input_names.append('save_id_in')
    all_inputs.append(save_id)
    input_names.append('repeat_penality_in')
    all_inputs.append(repeat_penality)
    input_names.append('penality_value')
    all_inputs.append(penality_value)
    input_names.append('beam_size')
    all_inputs.append(beam_size)
    output_names.append('top_beam_indices')
    output_names.append('save_id_out')
    output_names.append('repeat_penality_out')
    output_names.append('top_beam_prob')
    output_names.append('batch_indices')
    output_names.append('max_logits_idx')
    dynamic_axes['save_id_in'] = {0: 'batch', 1: 'history_len'}
    dynamic_axes['save_id_out'] = {0: 'batch', 1: 'history_len'}
    dynamic_axes['repeat_penality_in'] = {0: 'batch'}
    dynamic_axes['repeat_penality_out'] = {0: 'batch'}
    dynamic_axes['logits'] = {0: 'batch'}
    dynamic_axes['top_beam_prob'] = {0: 'batch'}
    dynamic_axes['top_beam_indices'] = {0: 'batch'}
    dynamic_axes['max_logits_idx'] = {0: 'batch'}
    dynamic_axes['batch_indices'] = {0: 'batch'}

    torch.onnx.export(
        first_beam_search,
        tuple(all_inputs),
        onnx_model_D,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False
    )
    del first_beam_search

    # 5. Export Second Beam Search (Model E)
    all_inputs = []
    input_names = []
    for i in range(num_layers):
        name = f'in_key_{i}'
        input_names.append(name)
        all_inputs.append(past_keys)
    for i in range(num_layers):
        name = f'in_value_{i}'
        input_names.append(name)
        all_inputs.append(past_values)
    if KV_QUANT_DTYPE == "Q4" or KV_QUANT_DTYPE == "Q8":
        for i in range(num_layers):
            name = f'in_key_scale_{i}'
            input_names.append(name)
            all_inputs.append(k_scale)
        for i in range(num_layers):
            name = f'in_key_bias_{i}'
            input_names.append(name)
            all_inputs.append(k_bias)
        for i in range(num_layers):
            name = f'in_value_scale_{i}'
            input_names.append(name)
            all_inputs.append(v_scale)
        for i in range(num_layers):
            name = f'in_value_bias_{i}'
            input_names.append(name)
            all_inputs.append(v_bias)

    input_names.append('logits')
    all_inputs.append(logits)
    input_names.append('save_id_in')
    all_inputs.append(save_id)
    input_names.append('repeat_penality_in')
    all_inputs.append(repeat_penality)
    input_names.append('previous_prob')
    all_inputs.append(previous_prob)
    input_names.append('batch_indices')
    all_inputs.append(batch_indices)
    input_names.append('penality_value')
    all_inputs.append(penality_value)
    input_names.append('beam_size')
    all_inputs.append(beam_size)
    input_names.append('topK')
    all_inputs.append(topK)
    dynamic_axes['previous_prob'] = {0: 'batch'}
    output_names.remove("batch_indices")

    second_beam_search = SECOND_BEAM_SEARCH(num_layers * 2 if (KV_QUANT_DTYPE == "F16" or KV_QUANT_DTYPE == "F32") else num_layers * 6)
    torch.onnx.export(
        second_beam_search,
        tuple(all_inputs),
        onnx_model_E,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False
    )
    del second_beam_search
    del num_layers
    del past_keys
    del past_values
    del input_names
    del output_names
    del dynamic_axes
    del all_inputs
    del beam_size
    del penality_value

    # 6. Export Reset Penalty (Model F)
    reset_penality = RESET_PENALITY()
    torch.onnx.export(
        reset_penality,
        (save_id, repeat_penality, penality_reset_count, batch_indices),
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
    del reset_penality
    del save_id
    del repeat_penality
    del penality_reset_count
    del batch_indices

    # 7. Export Argmax (Model G)
    argmax = ARGMAX()
    torch.onnx.export(
        argmax,
        (logits,),
        onnx_model_G,
        input_names=['logits'],
        output_names=['max_logits_idx'],
        dynamic_axes={
            'logits': {0: 'batch'},
            'max_logits_idx': {0: 'batch'}
        },
        opset_version=OPSET,
        dynamo=False
    )
print('\nExport done!\n\nStart running the LLM by ONNXRuntime.\nNow loading . . . it could cost minutes.')


# Run the exported model by ONNX Runtime
max_single_chat_length = MAX_SEQ_LEN                # It is an adjustable value, but must be less than max_seq_len.
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
run_options = onnxruntime.RunOptions()
session_opts.log_severity_level = 4                 # Fatal level, it is an adjustable value.
session_opts.log_verbosity_level = 4                # Fatal level, it is an adjustable value.
run_options.log_severity_level = 4                  # Fatal level, it is an adjustable value.
run_options.log_verbosity_level = 4                 # Fatal level, it is an adjustable value.
session_opts.inter_op_num_threads = MAX_THREADS     # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = MAX_THREADS     # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True            # True for execute speed; False for less memory usage.
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
run_options.add_run_config_entry('disable_synchronize_execution_providers', '1')

ORT_Accelerate_Providers = ['CPUExecutionProvider']
provider_options = None
device_type = 'cpu'

# --- Load Models ---
ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A = in_name_A[0].name
out_name_A = [out_name_A[0].name]

ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options,  run_options=run_options)
print(f"\nUsable Providers: {ort_session_B.get_providers()}")
model_dtype_B = ort_session_B._inputs_meta[0].type
in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
amount_of_outputs_B = len(out_name_B)
in_name_B = [in_name_B[i].name for i in range(len(in_name_B))]
out_name_B = [out_name_B[i].name for i in range(amount_of_outputs_B)]

device_type_copy = device_type
if 'dml' in device_type:
    device_type = 'cpu'

# --- Key-Value Cache Setup ---
if 'uint8' in model_dtype_B:
    model_dtype_B = np.uint8
    num_keys_values = (amount_of_outputs_B - 2) // 3
    num_layers = num_keys_values // 2
    past_keys_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, 0, ort_session_B._inputs_meta[0].shape[2]), dtype=model_dtype_B), device_type, DEVICE_ID)
    past_values_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, 0, ort_session_B._inputs_meta[num_layers].shape[2]), dtype=model_dtype_B), device_type, DEVICE_ID)
    if 'float16' in ort_session_B._inputs_meta[num_keys_values].type:
        scale_dtype_B = np.float16
    else:
        scale_dtype_B = np.float32
    k_scales = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, 0, 1), dtype=scale_dtype_B), device_type, DEVICE_ID)
    k_biases = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, 0, 1), dtype=scale_dtype_B), device_type, DEVICE_ID)
    v_scales = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, 0, 1), dtype=scale_dtype_B), device_type, DEVICE_ID)
    v_biases = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, 0, 1), dtype=scale_dtype_B), device_type, DEVICE_ID)
    num_keys_values = amount_of_outputs_B - 2  # Revert to original for later use
else:
    if 'float16' in model_dtype_B:
        model_dtype_B = np.float16
    else:
        model_dtype_B = np.float32
    num_keys_values = amount_of_outputs_B - 2
    num_layers = num_keys_values // 2
    past_keys_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_B._inputs_meta[0].shape[1], 1, ort_session_B._inputs_meta[0].shape[3], 0), dtype=model_dtype_B), device_type, DEVICE_ID)
    past_values_B = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_B._inputs_meta[num_layers].shape[1], 1, 0, ort_session_B._inputs_meta[num_layers].shape[4]), dtype=model_dtype_B), device_type, DEVICE_ID)
    k_scales = None

device_type = device_type_copy

generate_limit = MAX_SEQ_LEN - 10                   # 10 = length of basic ids
num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
num_keys_values_plus_3 = num_keys_values + 3
num_keys_values_plus_4 = num_keys_values + 4
num_keys_values_plus_5 = num_keys_values + 5
num_keys_values_plus_6 = num_keys_values + 6
num_keys_values_plus_7 = num_keys_values + 7
vocab_size = ort_session_B._outputs_meta[num_keys_values].shape[1]
topK = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([TOP_K], dtype=np.int64), device_type, DEVICE_ID)
beam_size = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([BEAM_SIZE], dtype=np.int64), device_type, DEVICE_ID)

# --- Search Strategy Setup ---
if USE_BEAM_SEARCH and (TOP_K < BEAM_SIZE):
    print("\nBeam Search does not display the immediate decoding results; the best result is shown only after the entire decoding process is complete.\n")
    TOP_K = BEAM_SIZE

if (TOP_K < 2) or (BEAM_SIZE < 2):
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")

if REPEAT_PENALTY != 1.0:
    do_repeat_penality = True
else:
    do_repeat_penality = False

if USE_BEAM_SEARCH:
    ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    in_name_D = ort_session_D.get_inputs()
    out_name_D = ort_session_D.get_outputs()
    in_name_D = [in_name_D[i].name for i in range(len(in_name_D))]
    out_name_D = [out_name_D[i].name for i in range(len(out_name_D))]

    ort_session_E = onnxruntime.InferenceSession(onnx_model_E, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    in_name_E = ort_session_E.get_inputs()
    out_name_E = ort_session_E.get_outputs()
    amount_of_outputs_E = len(out_name_E)
    in_name_E = [in_name_E[i].name for i in range(len(in_name_E))]
    out_name_E = [out_name_E[i].name for i in range(amount_of_outputs_E)]

    ort_session_F = onnxruntime.InferenceSession(onnx_model_F, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    in_name_F = ort_session_F.get_inputs()
    out_name_F = ort_session_F.get_outputs()
    in_name_F = [in_name_F[i].name for i in range(len(in_name_F))]
    out_name_F = [out_name_F[i].name for i in range(len(out_name_F))]

    penality_dtype = ort_session_D._inputs_meta[num_keys_values_plus_3].type
    if 'float16' in penality_dtype:
        penality_dtype = np.float16
    else:
        penality_dtype = np.float32
    penality_value = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([REPEAT_PENALTY], dtype=penality_dtype), device_type, DEVICE_ID)

    input_feed_D = {
        in_name_D[num_keys_values_plus_3]: penality_value,
        in_name_D[num_keys_values_plus_4]: beam_size
    }

    input_feed_E = {
        in_name_E[num_keys_values_plus_5]: penality_value,
        in_name_E[num_keys_values_plus_6]: beam_size,
        in_name_E[num_keys_values_plus_7]: topK
    }
    
    penality_reset_count_beam_init = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(BEAM_SIZE, dtype=np.int32), device_type, DEVICE_ID)
else:
    BEAM_SIZE = 1
    save_id_greedy = np.zeros(MAX_SEQ_LEN, dtype=np.int32)
    if do_repeat_penality:
        ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
        in_name_C = ort_session_C.get_inputs()
        out_name_C = ort_session_C.get_outputs()
        in_name_C = [in_name_C[i].name for i in range(len(in_name_C))]
        out_name_C = [out_name_C[i].name for i in range(len(out_name_C))]

        penality_dtype = ort_session_C._inputs_meta[2].type
        if 'float16' in penality_dtype:
            penality_dtype = np.float16
        else:
            penality_dtype = np.float32
        penality_value = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([REPEAT_PENALTY], dtype=penality_dtype), device_type, DEVICE_ID)
        input_feed_C = {in_name_C[2]: penality_value}
    else:
        ort_session_G = onnxruntime.InferenceSession(onnx_model_G, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
        in_name_G = ort_session_G.get_inputs()
        out_name_G = ort_session_G.get_outputs()
        in_name_G = in_name_G[0].name
        out_name_G = [out_name_G[0].name]
        penality_dtype = np.float32
        
init_ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
init_history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, DEVICE_ID)
init_attention_mask_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), device_type, DEVICE_ID)
init_attention_mask_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), device_type, DEVICE_ID)
init_repeat_penality = onnxruntime.OrtValue.ortvalue_from_numpy(np.ones((BEAM_SIZE, vocab_size), dtype=penality_dtype), device_type, DEVICE_ID)
init_batch_size_greedy = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
init_save_id_beam = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((BEAM_SIZE, 0), dtype=np.int32), device_type, DEVICE_ID)

if USE_BEAM_SEARCH:
    input_feed_D[in_name_D[num_keys_values_plus_1]] = init_save_id_beam
    input_feed_D[in_name_D[num_keys_values_plus_2]] = init_repeat_penality
    if do_repeat_penality:
        input_feed_F = {in_name_F[2]: penality_reset_count_beam_init}
elif do_repeat_penality:
    penality_reset_count_greedy = 0
    input_feed_C[in_name_C[1]] = init_repeat_penality
    input_feed_C[in_name_C[3]] = init_batch_size_greedy

# --- Pre-process Inputs ---
if TEST_THINK_MODE:
    prompt = f'<|im_start|>user\n{TEST_QUERY}<|im_end|>\n<|im_start|>assistant\n'
else:
    prompt = f'<|im_start|>user\n{TEST_QUERY}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
tokens = tokenizer(prompt, return_tensors='np')['input_ids'].astype(np.int32)
input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, DEVICE_ID)
ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([tokens.shape[-1]], dtype=np.int64), device_type, DEVICE_ID)
history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, DEVICE_ID)
attention_mask_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), device_type, DEVICE_ID)
attention_mask_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), device_type, DEVICE_ID)


input_feed_A = {in_name_A: input_ids}
all_outputs_A = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)[0]

input_feed_B = {
    in_name_B[num_keys_values]: all_outputs_A,
    in_name_B[num_keys_values_plus_1]: history_len,
    in_name_B[num_keys_values_plus_2]: ids_len,
    in_name_B[num_keys_values_plus_3]: attention_mask_1
}

i = 0
j = num_layers
while i < j:
    input_feed_B[in_name_B[i]] = past_keys_B
    i += 1
j = i + num_layers
while i < j:
    input_feed_B[in_name_B[i]] = past_values_B
    i += 1

if k_scales:
    j = i + num_layers
    while i < j:
        input_feed_B[in_name_B[i]] = k_scales
        i += 1
    j = i + num_layers
    while i < j:
        input_feed_B[in_name_B[i]] = k_biases
        i += 1
    j = i + num_layers
    while i < j:
        input_feed_B[in_name_B[i]] = v_scales
        i += 1
    j = i + num_layers
    while i < j:
        input_feed_B[in_name_B[i]] = v_biases
        i += 1

# --- Run Inference ---
print(f'\n\nTest Question: {TEST_QUERY}\nLLM Answering:')
num_decode = 0
generate_limit = MAX_SEQ_LEN - tokens.shape[-1]
start_time = time.time()
while num_decode < generate_limit:
    all_outputs_B = ort_session_B.run_with_ort_values(out_name_B, input_feed_B)
    if USE_BEAM_SEARCH:
        if num_decode < 1:
            input_feed_D.update(zip(in_name_D[:num_keys_values_plus_1], all_outputs_B))
            all_outputs_D = ort_session_D.run_with_ort_values(out_name_D, input_feed_D)
            max_logits_idx = all_outputs_D[num_keys_values_plus_5].numpy()
            input_feed_E[in_name_E[num_keys_values_plus_4]] = all_outputs_D[num_keys_values_plus_4]
            if do_repeat_penality:
                input_feed_F[in_name_F[3]] = all_outputs_D[num_keys_values_plus_4]
        else:
            input_feed_E.update(zip(in_name_E[:num_keys_values_plus_1], all_outputs_B))
            all_outputs_E = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)
            max_logits_idx = all_outputs_E[num_keys_values_plus_4].numpy()
        if max_logits_idx in STOP_TOKEN:
            break
        if do_repeat_penality and (num_decode >= PENALTY_RANGE):
            input_feed_F[in_name_F[0]] = all_outputs_E[num_keys_values_plus_1]
            input_feed_F[in_name_F[1]] = all_outputs_E[num_keys_values_plus_2]
            all_outputs_F = ort_session_F.run_with_ort_values(out_name_F, input_feed_F)
            input_feed_F[in_name_F[2]] = all_outputs_F[2]
            input_feed_E[in_name_E[num_keys_values_plus_1]] = all_outputs_F[0]
            input_feed_E[in_name_E[num_keys_values_plus_2]] = all_outputs_F[1]
        if num_decode < 1:
            input_feed_B.update(zip(in_name_B[:num_keys_values], all_outputs_D))
            input_feed_A[in_name_A] = all_outputs_D[num_keys_values]
            input_feed_E[in_name_E[num_keys_values_plus_1]] = all_outputs_D[num_keys_values_plus_1]
            input_feed_E[in_name_E[num_keys_values_plus_2]] = all_outputs_D[num_keys_values_plus_2]
            input_feed_E[in_name_E[num_keys_values_plus_3]] = all_outputs_D[num_keys_values_plus_3]
        else:
            input_feed_B.update(zip(in_name_B[:num_keys_values], all_outputs_E))
            input_feed_A[in_name_A] = all_outputs_E[num_keys_values]
            input_feed_E[in_name_E[num_keys_values_plus_1]] = all_outputs_E[num_keys_values_plus_1]
            input_feed_E[in_name_E[num_keys_values_plus_2]] = all_outputs_E[num_keys_values_plus_2]
            input_feed_E[in_name_E[num_keys_values_plus_3]] = all_outputs_E[num_keys_values_plus_3]
    else:
        if do_repeat_penality:
            input_feed_C[in_name_C[0]] = all_outputs_B[num_keys_values]
            all_outputs_C = ort_session_C.run_with_ort_values(out_name_C, input_feed_C)
            max_logits_idx = all_outputs_C[0].numpy()[0, 0]
            if num_decode >= PENALTY_RANGE:
                reset_ids = save_id_greedy[penality_reset_count_greedy]
                if reset_ids != max_logits_idx:
                    repeat_penality = all_outputs_C[1].numpy()
                    repeat_penality[:, reset_ids] = 1.0
                    input_feed_C[in_name_C[1]].update_inplace(repeat_penality)
                penality_reset_count_greedy += 1
            else:
                input_feed_C[in_name_C[1]] = all_outputs_C[1]
            input_feed_A[in_name_A] = all_outputs_C[0]
        else:
            input_feed_G = {in_name_G: all_outputs_B[num_keys_values]}
            all_outputs_G = ort_session_G.run_with_ort_values(out_name_G, input_feed_G)
            input_feed_A[in_name_A] = all_outputs_G[0]
            max_logits_idx = all_outputs_G[0].numpy()[0, 0]
        if max_logits_idx in STOP_TOKEN:
            break
        input_feed_B.update(zip(in_name_B[:num_keys_values], all_outputs_B))
        save_id_greedy[num_decode] = max_logits_idx
        print(tokenizer.decode(max_logits_idx), end="", flush=True)
    input_feed_B[in_name_B[num_keys_values]] = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)[0]
    input_feed_B[in_name_B[num_keys_values_plus_1]] = all_outputs_B[num_keys_values_plus_1]
    if num_decode < 1:
        input_feed_B[in_name_B[num_keys_values_plus_2]] = init_ids_len_1
        input_feed_B[in_name_B[num_keys_values_plus_3]] = init_attention_mask_0
    num_decode += 1

if USE_BEAM_SEARCH:
    result = tokenizer.decode(all_outputs_E[num_keys_values_plus_1].numpy()[0, :num_decode], skip_special_tokens=True)  # 0 is the Top_1
else:
    result = tokenizer.decode(save_id_greedy[:num_decode], skip_special_tokens=True)

print(f"\n\nFinal:\n{result}\n\nDecode: {((num_decode + 1) / (time.time() - start_time)):.3f} token/s")
