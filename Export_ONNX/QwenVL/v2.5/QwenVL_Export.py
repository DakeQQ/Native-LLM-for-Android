import os
import gc
import time
import torch
import onnxruntime
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer


path = r'/home/DakeQQ/Downloads/Qwen2.5-VL-3B-Instruct'             # Set the folder path where the Qwen2.5-VL whole project downloaded.
onnx_model_A = r'/home/DakeQQ/Downloads/Qwen_ONNX/QwenVL_A.onnx'    # Assign a path where the exported QwenVL model stored.
onnx_model_B = r'/home/DakeQQ/Downloads/Qwen_ONNX/QwenVL_B.onnx'
onnx_model_C = r'/home/DakeQQ/Downloads/Qwen_ONNX/QwenVL_C.onnx'
onnx_model_D = r'/home/DakeQQ/Downloads/Qwen_ONNX/QwenVL_D.onnx'
onnx_model_E = r'/home/DakeQQ/Downloads/Qwen_ONNX/QwenVL_E.onnx'
onnx_model_F = r'/home/DakeQQ/Downloads/Qwen_ONNX_2/QwenVL_F.onnx'  # You must assign a different folder path to avoid duplicate weight file names, which would cause loading failures.

image_path = r"../psyduck.png"                                      # Test image for the exported onnx model.
query = "Describe this image."                                      # Test query for the exported onnx model.

INPUT_IMAGE_SIZE = [960, 960]                                       # Input image shape. Should be a multiple of GPU group (e.g., 16) for optimal efficiency.
HEIGHT_FACTOR = 15                                                  # Adjust this value to determine the resize shape and vision resolution.
WIDTH_FACTOR = 15                                                   # Adjust this value to determine the resize shape and vision resolution.
MAX_SEQ_LENGTH = 4096                                               # The max token length. Note, this value include the 10 tokens for system prompt and (HEIGHT_FACTOR * WIDTH_FACTOR) tokens for image prompt. Hence, only (MAX_SEQ_LENGTH - (HEIGHT_FACTOR * WIDTH_FACTOR) - 10) tokens for query + response.
IMAGE_RESIZE = [HEIGHT_FACTOR * 28, WIDTH_FACTOR * 28]              # 28 = self.patch_size * self.merge_size
STOP_TOKEN = [151643, 151645]                                       # The stop_id in Qwen is "151643" & "151645"


def is_valid_image_path(image_path):
    if not os.path.exists(image_path):
        return False
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    _, ext = os.path.splitext(image_path)
    return ext.lower() in valid_extensions


def quantize_to_uint8(tensor, scale, zero_point):
    return ((tensor - zero_point) * scale).round().clamp(0, 255).to(torch.uint8)


def repeat_k(kv_states, num_key_value_groups, head_dim, num_heads):
    return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=1).view(num_heads, head_dim, -1)


def repeat_v(kv_states, num_key_value_groups, head_dim, num_heads):
    return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=1).view(num_heads, -1, head_dim)


def rotate_half(x, head_dim_half, dim):
    x1, x2 = torch.split(x, [head_dim_half, head_dim_half], dim=dim)
    return torch.cat((-x2, x1), dim=dim)


class QwenVL_PartA(torch.nn.Module):
    def __init__(self, qwenvl):
        super(QwenVL_PartA, self).__init__()
        self.qwenvl = qwenvl

    def forward(self, input_ids):
        # text_hidden_states
        return self.qwenvl.model.language_model.embed_tokens(input_ids)


class QwenVL_PartB(torch.nn.Module):
    def __init__(self, qwenvl):
        super(QwenVL_PartB, self).__init__()
        self.qwenvl = qwenvl
        self.num_heads = self.qwenvl.config.vision_config.num_heads
        self.head_dim = self.qwenvl.config.vision_config.hidden_size // self.num_heads
        self.head_dim_half = self.head_dim // 2
        self.variance_epsilon = float(1e-6)
        self.patch_size = self.qwenvl.visual.patch_size
        self.merge_size = self.qwenvl.visual.spatial_merge_size
        self.means = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1)
        self.inv_std = torch.tensor([1.0 / 0.26862954, 1.0 / 0.26130258, 1.0 / 0.27577711], dtype=torch.float32).view(1, 3, 1, 1)
        self.means_inv_std = self.means * self.inv_std
        self.inv_255_std = self.inv_std / 255.0
        self.factor_size = WIDTH_FACTOR * HEIGHT_FACTOR * self.merge_size * self.merge_size
        grid_thw = torch.tensor([[1, HEIGHT_FACTOR * 2, WIDTH_FACTOR * 2]], dtype=torch.int32)

        rotary_pos_emb = self.qwenvl.visual.rot_pos_emb(grid_thw).float().unsqueeze(0)
        cos = rotary_pos_emb.cos()
        sin = rotary_pos_emb.sin()
        self.rotary_pos_emb_cos = torch.cat([cos, cos], dim=-1).transpose(0, 1)
        self.rotary_pos_emb_sin = torch.cat([sin, sin], dim=-1).transpose(0, 1)

        scale_factor = float(self.head_dim ** -0.25)
        init_attention_mask = torch.ones([1, self.factor_size, self.factor_size], dtype=torch.int8)
        _, cu_window_seqlens = self.qwenvl.visual.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(cu_window_seqlens, dtype=torch.int32)
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        cu_seqlens = torch.tensor([0, self.factor_size], dtype=torch.int32)
        self.attention_mask = []
        for layer_num, blk in enumerate(self.qwenvl.visual.blocks):
            blk.attn.qkv.weight.data[:-self.qwenvl.visual.patch_embed.embed_dim] *= scale_factor
            blk.attn.qkv.bias.data[:-self.qwenvl.visual.patch_embed.embed_dim] *= scale_factor
            if layer_num in self.qwenvl.visual.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens
            attention_mask = init_attention_mask
            for i in range(1, len(cu_seqlens_now)):
                attention_mask[..., cu_seqlens_now[i - 1]: cu_seqlens_now[i], cu_seqlens_now[i - 1]: cu_seqlens_now[i]] = 0
            self.attention_mask.append((attention_mask * -128).float())

    def forward(self, pixel_values):
        pixel_values = torch.nn.functional.interpolate(
            pixel_values.float(),
            IMAGE_RESIZE,
            mode='bilinear',
            align_corners=True)
        pixel_values = pixel_values * self.inv_255_std - self.means_inv_std
        pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
        pixel_values = pixel_values.reshape(
            self.qwenvl.visual.patch_embed.temporal_patch_size,
            3,
            HEIGHT_FACTOR,
            self.merge_size,
            self.patch_size,
            WIDTH_FACTOR,
            self.merge_size,
            self.patch_size
        )
        pixel_values = pixel_values.permute(2, 5, 3, 6, 1, 0, 4, 7)
        pixel_values = pixel_values.reshape(
            self.factor_size,
            3,
            self.qwenvl.visual.patch_embed.temporal_patch_size,
            self.patch_size,
            self.patch_size
        )
        vision_hidden_states = self.qwenvl.visual.patch_embed.proj(pixel_values).view(1, -1, self.qwenvl.visual.patch_embed.embed_dim)
        for layer_num, blk in enumerate(self.qwenvl.visual.blocks):
            hidden_states_norm = blk.norm1.weight * (vision_hidden_states / torch.sqrt(vision_hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
            q, k, v = blk.attn.qkv(hidden_states_norm).reshape(self.factor_size, -1, self.head_dim).split([self.num_heads, self.num_heads, self.num_heads], dim=1)
            q = q * self.rotary_pos_emb_cos + rotate_half(q, self.head_dim_half, -1) * self.rotary_pos_emb_sin
            k = k * self.rotary_pos_emb_cos + rotate_half(k, self.head_dim_half, -1) * self.rotary_pos_emb_sin
            q = q.transpose(0, 1)
            k = k.permute(1, 2, 0)
            v = v.transpose(0, 1)
            attn = torch.nn.functional.softmax(torch.matmul(q, k) + self.attention_mask[layer_num], dim=-1, dtype=torch.float32)
            attn_out = blk.attn.proj(torch.matmul(attn, v).transpose(0, 1).contiguous().view(1, -1, blk.attn.proj.in_features))
            vision_hidden_states += attn_out
            vision_hidden_states += blk.mlp(blk.norm2.weight * (vision_hidden_states / torch.sqrt(vision_hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)))
        vision_hidden_states = self.qwenvl.visual.merger.ln_q.weight * (vision_hidden_states / torch.sqrt(vision_hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
        return self.qwenvl.visual.merger.mlp(vision_hidden_states.view(1, -1, self.qwenvl.visual.merger.hidden_size))


class QwenVL_PartC(torch.nn.Module):
    def __init__(self, prompt_head_len):
        super(QwenVL_PartC, self).__init__()
        self.prompt_head_len = prompt_head_len

    def forward(self, text_hidden_states, vision_hidden_states):
        # Concat hidden_states
        return torch.cat([text_hidden_states[:, :self.prompt_head_len], vision_hidden_states, text_hidden_states[:, self.prompt_head_len:]], dim=1)


class QwenVL_PartD(torch.nn.Module):
    def __init__(self, qwenvl, width_factor, height_factor, prompt_head_len, max_seq_len):
        super(QwenVL_PartD, self).__init__()
        self.image_embed_size = width_factor * height_factor
        self.prompt_head_len = prompt_head_len
        self.max_seq_len = max_seq_len
        self.image_factor_plus = self.image_embed_size + self.prompt_head_len
        position_ids = torch.arange(self.image_factor_plus, dtype=torch.float32).repeat(3, 1, 1)
        position_ids[0, :, self.prompt_head_len: self.image_factor_plus] = self.prompt_head_len
        j = self.prompt_head_len
        for i in range(self.prompt_head_len, self.image_factor_plus, width_factor):
            position_ids[1, :, i: i + width_factor] = j
            j += 1
        self.start_id = self.prompt_head_len + width_factor
        fill_id = torch.arange(self.prompt_head_len, self.start_id, dtype=torch.float32)
        for i in range(self.start_id, self.image_factor_plus, width_factor):
            position_ids[2, :, i: i + width_factor] = fill_id
        fill_tail_position = torch.arange(self.start_id + 1, self.max_seq_len + self.start_id + 1, dtype=torch.float32).repeat(3, 1, 1)
        position_ids = torch.cat([position_ids[:, :, :self.image_factor_plus], fill_tail_position], dim=-1)
        self.inv_freq_expanded = qwenvl.model.language_model.rotary_emb.inv_freq[None, :, None].float().expand(3, -1, 1)
        self.mrope_section = qwenvl.model.language_model.layers._modules['0'].self_attn.rope_scaling["mrope_section"] * 2
        freqs = (self.inv_freq_expanded @ position_ids).unsqueeze(0)
        emb = torch.cat((freqs, freqs), dim=-2)
        cos = emb.cos()
        sin = emb.sin()
        self.rotary_pos_emb_cos = torch.cat([m[:, i % 3] for i, m in enumerate(cos.split(self.mrope_section, dim=-2))], dim=-2).half()
        self.rotary_pos_emb_sin = torch.cat([m[:, i % 3] for i, m in enumerate(sin.split(self.mrope_section, dim=-2))], dim=-2).half()
        
    def forward(self, history_len, kv_seq_len):
        rotary_pos_emb_cos_k = self.rotary_pos_emb_cos[:, :, history_len:kv_seq_len].float()
        rotary_pos_emb_sin_k = self.rotary_pos_emb_sin[:, :, history_len:kv_seq_len].float()
        rotary_pos_emb_cos_q = rotary_pos_emb_cos_k.transpose(-1, -2)
        rotary_pos_emb_sin_q = rotary_pos_emb_sin_k.transpose(-1, -2)
        return rotary_pos_emb_cos_q, rotary_pos_emb_sin_q, rotary_pos_emb_cos_k, rotary_pos_emb_sin_k
    

class QwenVL_PartE(torch.nn.Module):
    def __init__(self, qwenvl, max_seq_len):
        super(QwenVL_PartE, self).__init__()
        position_ids = torch.arange(max_seq_len, dtype=torch.float32).repeat(3, 1, 1)
        self.inv_freq_expanded = qwenvl.model.language_model.rotary_emb.inv_freq[None, :, None].float().expand(3, -1, 1)
        self.mrope_section = qwenvl.model.language_model.layers._modules['0'].self_attn.rope_scaling["mrope_section"] * 2
        freqs = (self.inv_freq_expanded @ position_ids).unsqueeze(0)
        emb = torch.cat((freqs, freqs), dim=-2)
        cos = emb.cos()
        sin = emb.sin()
        self.rotary_pos_emb_cos = torch.cat([m[:, i % 3] for i, m in enumerate(cos.split(self.mrope_section, dim=-2))], dim=-2).half()
        self.rotary_pos_emb_sin = torch.cat([m[:, i % 3] for i, m in enumerate(sin.split(self.mrope_section, dim=-2))], dim=-2).half()
        
    def forward(self, history_len, kv_seq_len):
        rotary_pos_emb_cos_k = self.rotary_pos_emb_cos[:, :, history_len:kv_seq_len].float()
        rotary_pos_emb_sin_k = self.rotary_pos_emb_sin[:, :, history_len:kv_seq_len].float()
        rotary_pos_emb_cos_q = rotary_pos_emb_cos_k.transpose(-1, -2)
        rotary_pos_emb_sin_q = rotary_pos_emb_sin_k.transpose(-1, -2)
        return rotary_pos_emb_cos_q, rotary_pos_emb_sin_q, rotary_pos_emb_cos_k, rotary_pos_emb_sin_k


class QwenVL_PartF(torch.nn.Module):
    def __init__(self, qwenvl, head_dim, num_heads, num_layers, max_seq_len):
        super(QwenVL_PartF, self).__init__()
        self.qwenvl = qwenvl
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim_half = head_dim // 2
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.variance_epsilon = float(1e-6)
        self.save_key = [None] * num_layers
        self.save_value = [None] * num_layers
        self.attention_mask = (1 - torch.tril(torch.ones([1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128
        scale_factor = float(head_dim ** -0.25)
        for i in range(num_layers):
            self.qwenvl.model.language_model.layers._modules[f'{i}'].self_attn.q_proj.weight.data *= scale_factor
            self.qwenvl.model.language_model.layers._modules[f'{i}'].self_attn.q_proj.bias.data *= scale_factor
            self.qwenvl.model.language_model.layers._modules[f'{i}'].self_attn.k_proj.weight.data *= scale_factor
            self.qwenvl.model.language_model.layers._modules[f'{i}'].self_attn.k_proj.bias.data *= scale_factor

    def forward(self, *all_inputs):
        kv_seq_len = all_inputs[-8]
        hidden_states = all_inputs[-7]
        rotary_pos_emb_cos_q = all_inputs[-6]
        rotary_pos_emb_sin_q = all_inputs[-5]
        rotary_pos_emb_cos_k = all_inputs[-4]
        rotary_pos_emb_sin_k = all_inputs[-3]
        ids_len = all_inputs[-2]
        attention_mask = (self.attention_mask[:, :ids_len, :kv_seq_len] * all_inputs[-1]).float()
        for i, layer in enumerate(self.qwenvl.model.language_model.layers):
            hidden_states_norm = layer.input_layernorm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
            q = layer.self_attn.q_proj(hidden_states_norm).view(-1, self.num_heads, self.head_dim).transpose(0, 1)
            k = layer.self_attn.k_proj(hidden_states_norm).view(-1, 1, self.num_key_value_heads, self.head_dim).permute(2, 1, 3, 0)
            v = layer.self_attn.v_proj(hidden_states_norm).view(-1, 1, self.num_key_value_heads, self.head_dim).transpose(0, 2)
            q = q * rotary_pos_emb_cos_q + rotate_half(q, self.head_dim_half, -1) * rotary_pos_emb_sin_q
            k = k * rotary_pos_emb_cos_k + rotate_half(k, self.head_dim_half, -2) * rotary_pos_emb_sin_k
            k = torch.cat((all_inputs[i], k), dim=-1)
            v = torch.cat((all_inputs[i + self.num_layers], v), dim=2)
            self.save_key[i] = k
            self.save_value[i] = v
            k = repeat_k(k, self.num_key_value_groups, self.head_dim, self.num_heads)
            v = repeat_v(v, self.num_key_value_groups, self.head_dim, self.num_heads)
            attn = torch.nn.functional.softmax(torch.matmul(q, k) + attention_mask, dim=-1, dtype=torch.float32)
            attn_out = layer.self_attn.o_proj(torch.matmul(attn, v).transpose(0, 1).contiguous().view(1, -1, layer.self_attn.o_proj.in_features))
            hidden_states += attn_out
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
            hidden_states = layer.mlp(hidden_states)
            hidden_states += residual
        hidden_states = hidden_states[:, -1]
        hidden_states = self.qwenvl.model.language_model.norm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
        max_logit_ids = torch.argmax(self.qwenvl.lm_head(hidden_states), dim=-1, keepdim=True).int()
        return *self.save_key, *self.save_value, kv_seq_len + 1, max_logit_ids


# Load the model
with torch.inference_mode():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True).eval()
    max_seq_len = MAX_SEQ_LENGTH
    num_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // num_heads
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    prompt_head_len = 4  # <|im_start|>user\n<|vision_start|>
    ids_len = torch.tensor([10], dtype=torch.long)      # "10" is just a dummy value.
    history_len = torch.tensor([0], dtype=torch.long)
    image_embed_size = WIDTH_FACTOR * HEIGHT_FACTOR
    image_pad_len = torch.tensor([image_embed_size], dtype=torch.long)
    vision_hidden_states = torch.ones((1, image_embed_size, hidden_size), dtype=torch.float32)
    pixel_values = torch.ones([1, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]]).to(torch.uint8)
    input_ids = torch.ones((1, ids_len), dtype=torch.int32)
    text_hidden_states = torch.ones((1, ids_len, hidden_size), dtype=torch.float32)
    rotary_pos_emb_cos_q = torch.ones((1, ids_len, head_dim), dtype=torch.float32)
    rotary_pos_emb_sin_q = rotary_pos_emb_cos_q
    rotary_pos_emb_cos_k = rotary_pos_emb_cos_q.transpose(-1, -2)
    rotary_pos_emb_sin_k = rotary_pos_emb_cos_k
    attention_mask = torch.tensor([0], dtype=torch.int8)
    past_keys = torch.zeros((num_key_value_heads, 1, head_dim, 0), dtype=torch.float32)
    past_values = torch.zeros((num_key_value_heads, 1, 0, head_dim), dtype=torch.float32)
    hidden_states = torch.ones((1, ids_len, hidden_size), dtype=torch.float32)
    kv_seq_len = history_len + ids_len

    print('\nExport Part_A Start...')
    model_A = QwenVL_PartA(model)
    torch.onnx.export(
        model_A,
        (input_ids,),
        onnx_model_A,
        input_names=['input_ids'],
        output_names=['text_hidden_states'],
        dynamic_axes={
            'input_ids': {1: 'ids_len'},
            'text_hidden_states': {1: 'ids_len'}
        },
        do_constant_folding=True,
        opset_version=17
    )
    del model_A
    del input_ids
    gc.collect()

    print('\nExport Part_A Done!  \n\nExport Part_B Start...')
    model_B = QwenVL_PartB(model)
    torch.onnx.export(
        model_B,
        (pixel_values,),
        onnx_model_B,
        input_names=['pixel_values'],
        output_names=['vision_hidden_states'],
        do_constant_folding=True,
        opset_version=17
    )
    del pixel_values
    gc.collect()
    print('\nExport Part_B Done! \n\nExport Part_C Start...')

    model_C = QwenVL_PartC(prompt_head_len)
    torch.onnx.export(
        model_C,
        (text_hidden_states, vision_hidden_states),
        onnx_model_C,
        input_names=['text_hidden_states', 'vision_hidden_states'],
        output_names=['hidden_states'],
        dynamic_axes={
            'text_hidden_states': {1: 'ids_len'},
            'hidden_states': {1: 'ids_len_plus_image'}
        },
        do_constant_folding=True,
        opset_version=17
    )
    del model_C
    del text_hidden_states
    del vision_hidden_states
    print('\nExport Part_C Done! \n\nExport Part_D Start...')

    model_D = QwenVL_PartD(model, WIDTH_FACTOR, HEIGHT_FACTOR, prompt_head_len, max_seq_len)
    torch.onnx.export(
        model_D,
        (history_len, kv_seq_len),
        onnx_model_D,
        input_names=['history_len', 'kv_seq_len'],
        output_names=['rotary_pos_emb_cos_q', 'rotary_pos_emb_sin_q', 'rotary_pos_emb_cos_k', 'rotary_pos_emb_sin_k'],
        dynamic_axes={
            'rotary_pos_emb_cos_q': {1: 'ids_len'},
            'rotary_pos_emb_sin_q': {1: 'ids_len'},
            'rotary_pos_emb_cos_k': {2: 'ids_len'},
            'rotary_pos_emb_sin_k': {2: 'ids_len'}
        },
        do_constant_folding=True,
        opset_version=17
    )
    del model_D
    print('\nExport Part_D Done! \n\nExport Part_E Start...')

    model_E = QwenVL_PartE(model, max_seq_len)
    torch.onnx.export(
        model_E,
        (history_len, kv_seq_len),
        onnx_model_E,
        input_names=['history_len', 'kv_seq_len'],
        output_names=['rotary_pos_emb_cos_q', 'rotary_pos_emb_sin_q', 'rotary_pos_emb_cos_k', 'rotary_pos_emb_sin_k'],
        dynamic_axes={
            'rotary_pos_emb_cos_q': {1: 'ids_len'},
            'rotary_pos_emb_sin_q': {1: 'ids_len'},
            'rotary_pos_emb_cos_k': {2: 'ids_len'},
            'rotary_pos_emb_sin_k': {2: 'ids_len'}
        },
        do_constant_folding=True,
        opset_version=17
    )
    del model_E
    print('\nExport Part_E Done! \n\nExport Part_F Start...')

    model_F = QwenVL_PartF(model, head_dim, num_heads, num_layers, max_seq_len)
    # Prepare input and output names
    all_inputs = []
    input_names = []
    output_names = []
    dynamic_axes = {
        'hidden_states': {1: 'ids_len_plus_image'},
        'rotary_pos_emb_cos_q': {1: 'ids_len_plus_image'},
        'rotary_pos_emb_sin_q': {1: 'ids_len_plus_image'},
        'rotary_pos_emb_cos_k': {2: 'ids_len_plus_image'},
        'rotary_pos_emb_sin_k': {2: 'ids_len_plus_image'}
    }
    for i in range(num_layers):
        name = f'in_key_{i}'
        input_names.append(name)
        all_inputs.append(past_keys)
        dynamic_axes[name] = {3: 'history_len'}
        name = f'out_key_{i}'
        output_names.append(name)
        dynamic_axes[name] = {3: 'history_len_plus_ids_len'}
    for i in range(num_layers):
        name = f'in_value_{i}'
        input_names.append(name)
        all_inputs.append(past_values)
        dynamic_axes[name] = {2: 'history_len'}
        name = f'out_value_{i}'
        output_names.append(name)
        dynamic_axes[name] = {2: 'history_len_plus_ids_len'}
    input_names.append('kv_seq_len')
    all_inputs.append(kv_seq_len)
    input_names.append('hidden_states')
    all_inputs.append(hidden_states)
    input_names.append('rotary_pos_emb_cos_q')
    all_inputs.append(rotary_pos_emb_cos_q)
    input_names.append('rotary_pos_emb_sin_q')
    all_inputs.append(rotary_pos_emb_sin_q)
    input_names.append('rotary_pos_emb_cos_k')
    all_inputs.append(rotary_pos_emb_cos_k)
    input_names.append('rotary_pos_emb_sin_k')
    all_inputs.append(rotary_pos_emb_sin_k)
    input_names.append('ids_len')
    all_inputs.append(ids_len)
    input_names.append('attention_mask')
    all_inputs.append(attention_mask)
    output_names.append('next_kv_seq_len')
    output_names.append('max_logit_id')

    torch.onnx.export(
        model_F,
        tuple(all_inputs),
        onnx_model_F,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=17
    )
    del model
    del model_F
    del ids_len
    del history_len
    del attention_mask
    del past_keys
    del past_values
    del input_names
    del output_names
    del dynamic_axes
    del all_inputs
    gc.collect()
print('\nExport Part_F done!\n\nStart running the Qwen by ONNXRuntime.\nNow loading . . . it could cost minutes.')


# Run the exported model by ONNX Runtime
max_single_chat_length = 4096                         # It an adjustable value, but must less than max_seq_len.
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4                   # fatal level = 4, it an adjustable value.
session_opts.log_verbosity_level = 4                  # fatal level = 4, it an adjustable value.
session_opts.inter_op_num_threads = 0                 # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0                 # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True              # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")
session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")

device_type = 'cpu'
device_id = 0


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
out_name_A0 = out_name_A[0].name


ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
in_name_B0 = in_name_B[0].name
out_name_B0 = out_name_B[0].name


ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_C = ort_session_C.get_inputs()
out_name_C = ort_session_C.get_outputs()
in_name_C0 = in_name_C[0].name
in_name_C1 = in_name_C[1].name
out_name_C0 = out_name_C[0].name


ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_D = ort_session_D.get_inputs()
out_name_D = ort_session_D.get_outputs()
in_name_D0 = in_name_D[0].name
in_name_D1 = in_name_D[1].name
rotary_outputs_len = len(out_name_D)
out_name_D = [out_name_D[i].name for i in range(rotary_outputs_len)]


ort_session_E = onnxruntime.InferenceSession(onnx_model_E, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_E = ort_session_E.get_inputs()
out_name_E = ort_session_E.get_outputs()
in_name_E0 = in_name_E[0].name
in_name_E1 = in_name_E[1].name
out_name_E = [out_name_E[i].name for i in range(rotary_outputs_len)]


ort_session_F = onnxruntime.InferenceSession(onnx_model_F, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_F = ort_session_F.get_inputs()
out_name_F = ort_session_F.get_outputs()
amount_of_outputs_F = len(out_name_F)
in_name_F = [in_name_F[i].name for i in range(len(in_name_F))]
out_name_F = [out_name_F[i].name for i in range(amount_of_outputs_F)]


num_layers = (amount_of_outputs_F - 2) // 2
num_keys_values = num_layers + num_layers
num_keys_values_plus = num_keys_values + 1
rotary_indices = np.arange(num_keys_values, num_keys_values + rotary_outputs_len, dtype=np.int32) + 2
amount_of_outputs_F -= 1


num_decode = 0
prompt = f"<|im_start|>user\n<|vision_start|><|vision_end|>{query}<|im_end|>\n<|im_start|>assistant\n"
prompt_head_len = np.array([4], dtype=np.int64)
image_embed_size = WIDTH_FACTOR * HEIGHT_FACTOR
tokens = tokenizer(prompt, return_tensors='np')['input_ids'].astype(np.int32)
input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, device_id)
ids_len = np.array([tokens.shape[-1]], dtype=np.int64)
history_len = np.array([0], dtype=np.int64)
kv_seq_len = onnxruntime.OrtValue.ortvalue_from_numpy(ids_len + history_len, device_type, device_id)
ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(ids_len, device_type, device_id)
history_len = onnxruntime.OrtValue.ortvalue_from_numpy(history_len, device_type, device_id)
attention_mask = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), device_type, device_id)
max_single_chat_length -= tokens.shape[-1]
if device_type != 'dml':
    past_keys_F = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_F._inputs_meta[0].shape[0], 1, ort_session_F._inputs_meta[0].shape[2], 0), dtype=np.float32), device_type, device_id)
    past_values_F = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_F._inputs_meta[num_layers].shape[0], 1, 0, ort_session_F._inputs_meta[num_layers].shape[3]), dtype=np.float32), device_type, device_id)
else:
    # Crash with unknown reason.
    past_keys_F = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_F._inputs_meta[0].shape[0], 1, ort_session_F._inputs_meta[0].shape[2], 0), dtype=np.float32), 'cpu', device_id)
    past_values_F = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_F._inputs_meta[num_layers].shape[0], 1, 0, ort_session_F._inputs_meta[num_layers].shape[3]), dtype=np.float32), 'cpu', device_id)


# Load input image
if is_valid_image_path(image_path):
    image = Image.open(image_path)
    image = image.resize((INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0]))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    pixel_values = np.transpose(np.array(image).astype(np.uint8), (2, 0, 1))
    pixel_values = np.expand_dims(pixel_values, axis=0)
    use_vision = True
    print('\nChat with image.')
else:
    use_vision = False
    print('\nChat without image.')


# Start to run LLM
hidden_states = ort_session_A.run_with_ort_values([out_name_A0], {in_name_A0: input_ids})[0]


if use_vision:
    ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(onnxruntime.OrtValue.numpy(ids_len) + image_embed_size, device_type, device_id)
    kv_seq_len = onnxruntime.OrtValue.ortvalue_from_numpy(onnxruntime.OrtValue.numpy(kv_seq_len) + image_embed_size, device_type, device_id)

    max_single_chat_length -= image_embed_size

    print('\nStart to Process the Image...')
    start_time = time.time()
    vision_hidden_states = ort_session_B.run_with_ort_values([out_name_B0], {in_name_B0: onnxruntime.OrtValue.ortvalue_from_numpy(pixel_values, device_type, device_id)})[0]
    print(f'\nImage Process Complete. Time Cost: {time.time() - start_time:.3f} Seconds')

    hidden_states = ort_session_C.run_with_ort_values([out_name_C0], {in_name_C0: hidden_states, in_name_C1: vision_hidden_states})[0]

    rotary_outputs = ort_session_D.run_with_ort_values(out_name_D, {in_name_D0: history_len, in_name_D1: kv_seq_len})
else:
    rotary_outputs = ort_session_E.run_with_ort_values(out_name_E, {in_name_E0: history_len, in_name_E1: kv_seq_len})


input_feed_F = {
    in_name_F[num_keys_values]: kv_seq_len,
    in_name_F[num_keys_values_plus]: hidden_states,
    in_name_F[-2]: ids_len,
    in_name_F[-1]: attention_mask
}

for i in range(num_layers):
    input_feed_F[in_name_F[i]] = past_keys_F
for i in range(num_layers, num_keys_values):
    input_feed_F[in_name_F[i]] = past_values_F
for i in range(rotary_outputs_len):
    input_feed_F[in_name_F[rotary_indices[i]]] = rotary_outputs[i]


print(f'\nTest Question: {query}\n\nQwenVL Answering:\n')
start_time = time.time()
while num_decode < max_single_chat_length:
    
    all_outputs_F = ort_session_F.run_with_ort_values(out_name_F, input_feed_F)
    
    max_logit_ids = onnxruntime.OrtValue.numpy(all_outputs_F[-1])[0]
    num_decode += 1
    
    if max_logit_ids in STOP_TOKEN:
        break    
        
    if num_decode < 2:
        input_feed_F[in_name_F[-2]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, device_id)
        input_feed_F[in_name_F[-1]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), device_type, device_id)

    input_feed_F[in_name_F[num_keys_values_plus]] = ort_session_A.run_with_ort_values([out_name_A0], {in_name_A0: all_outputs_F[-1]})[0]
    
    if use_vision:
        rotary_outputs = ort_session_D.run_with_ort_values(out_name_D, {in_name_D0: input_feed_F[in_name_F[num_keys_values]], in_name_D1: all_outputs_F[-2]})
    else:
        rotary_outputs = ort_session_E.run_with_ort_values(out_name_E, {in_name_E0: input_feed_F[in_name_F[num_keys_values]], in_name_E1: all_outputs_F[-2]})

    for i in range(amount_of_outputs_F):
        input_feed_F[in_name_F[i]] = all_outputs_F[i]
        
    for i in range(rotary_outputs_len):
        input_feed_F[in_name_F[rotary_indices[i]]] = rotary_outputs[i]
        
    print(tokenizer.decode(max_logit_ids), end="", flush=True)
    
print(f"\n\nDecode: {(num_decode / (time.time() - start_time)):.3f} token/s")



