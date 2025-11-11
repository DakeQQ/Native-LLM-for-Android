import os
import gc
import time
import torch
import onnxruntime
import numpy as np
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLVisionModel, AutoTokenizer
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

path = r'/home/DakeQQ/Downloads/Qwen3-VL-2B-Instruct'               # Set the folder path where the Qwen3-VL whole project downloaded.

# For QwenVL
onnx_model_A = r'/home/DakeQQ/Downloads/Qwen_ONNX/QwenVL_A.onnx'    # Assign a path where the exported QwenVL model stored.
onnx_model_B = r'/home/DakeQQ/Downloads/Qwen_ONNX/QwenVL_B.onnx'
onnx_model_C = r'/home/DakeQQ/Downloads/Qwen_ONNX/QwenVL_C.onnx'
onnx_model_D = r'/home/DakeQQ/Downloads/Qwen_ONNX/QwenVL_D.onnx'
onnx_model_E = r'/home/DakeQQ/Downloads/Qwen_ONNX/QwenVL_E.onnx'

# You must assign a different folder path to avoid duplicate weight file names, which would cause loading failures.
onnx_model_F = r'/home/DakeQQ/Downloads/Qwen_ONNX_2/QwenVL_F.onnx'

# For Beam_Search & Greedy_Search
onnx_model_G = r'/home/DakeQQ/Downloads/Qwen_ONNX/QwenVL_G.onnx'
onnx_model_H = r'/home/DakeQQ/Downloads/Qwen_ONNX/QwenVL_H.onnx'
onnx_model_I = r'/home/DakeQQ/Downloads/Qwen_ONNX/QwenVL_I.onnx'
onnx_model_J = r'/home/DakeQQ/Downloads/Qwen_ONNX/QwenVL_J.onnx'


image_path = r"../psyduck.png"  # Test image for the exported onnx model.
query = "Describe this image."                                      # Test query for the exported onnx model.

DYNAMIC_IMAGE_SHAPE = False                                         # Allow for a dynamic number of image inputs.
VISION_BATCH_SIZE = 1                                               # Set the number of images for the vision LLM, whether DYNAMIC_IMAGE_SHAPE is True or False.
INPUT_IMAGE_SIZE = [960, 960]                                       # Input image shape. Should be a multiple of GPU group (e.g., 16) for optimal efficiency.
HEIGHT_FACTOR = 15                                                  # Adjust this value to determine the resize shape and vision resolution.
WIDTH_FACTOR = 15                                                   # Adjust this value to determine the resize shape and vision resolution.
MAX_SEQ_LENGTH = 4096                                               # The max token length. Note, this value include inputs and generated.
IMAGE_RESIZE = [HEIGHT_FACTOR * 32, WIDTH_FACTOR * 32]              # 32 = self.patch_size * self.merge_size
STOP_TOKEN = [151643, 151645]                                       # The stop_id in Qwen is "151643" & "151645"
REPEAT_PENALITY = 0.9                                               # Range from 0.0 to 1.0; "1.0" means no penality.
PENALITY_RANGE = 10                                                 # Penalizes the most recent output. "10" means the last 10 tokens.
USE_BEAM_SEARCH = True                                              # Use beam search or greedy search.
TOP_K = 3                                                           # The top k candidate in decoding.
BEAM_SIZE = 3                                                       # Number of beams in searching.
MAX_BEAM_SIZE = 10                                                  # Max beams for exported model.
MAX_THREADS = 0                                                     # Parllel CPU threads. Set 0 for auto.
DEVICE_ID = 0                                                       # Default to zero.


def is_valid_image_path(image_path):
    if image_path is None:
        return False
    elif image_path is "":
        return False
    elif not os.path.exists(image_path):
        return False
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    _, ext = os.path.splitext(image_path)
    return ext.lower() in valid_extensions


def rotate_half(x, head_dim_half, dim):
    x1, x2 = torch.split(x, [head_dim_half, head_dim_half], dim=dim)
    return torch.cat((-x2, x1), dim=dim)


def repeat_k(kv_states, num_key_value_groups, head_dim, num_heads, batch_size):
    return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=2).view(batch_size, num_heads, head_dim, -1)


def repeat_v(kv_states, num_key_value_groups, head_dim, num_heads, batch_size):
    return torch.cat([kv_states for _ in range(num_key_value_groups)], dim=2).view(batch_size, num_heads, -1, head_dim)


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
    def __init__(self, num_layers):
        super(FIRST_BEAM_SEARCH, self).__init__()
        self.num_keys_values = num_layers + num_layers
        self.save_keys_values = [None] * self.num_keys_values
        self.batch_indices = torch.arange(MAX_BEAM_SIZE, dtype=torch.int8)

    def forward(self, *all_inputs):
        logits = all_inputs[-5]
        save_id = all_inputs[-4]
        repeat_penality = all_inputs[-3]
        penality_value = all_inputs[-2]
        beam_size = all_inputs[-1]
        logits = torch.log_softmax(logits, dim=-1)
        top_beam_prob, top_beam_indices = torch.topk(logits, dim=-1, k=beam_size, sorted=False, largest=True)
        for i in range(self.num_keys_values):
            self.save_keys_values[i] = all_inputs[i].repeat(beam_size, *([1] * (all_inputs[i].dim() - 1)))
        top_beam_indices = top_beam_indices.transpose(0, 1)
        batch_indices = self.batch_indices[:beam_size].long()
        repeat_penality[batch_indices, top_beam_indices] *= penality_value
        top_beam_indices = top_beam_indices.int()
        save_id = torch.cat([save_id, top_beam_indices], dim=-1)
        max_logits_idx = top_beam_indices[0]
        return *self.save_keys_values, top_beam_indices, save_id, repeat_penality, top_beam_prob.transpose(0, 1), batch_indices, max_logits_idx


class SECOND_BEAM_SEARCH(torch.nn.Module):
    def __init__(self, num_layers):
        super(SECOND_BEAM_SEARCH, self).__init__()
        self.num_keys_values = num_layers + num_layers
        self.save_keys_values = [None] * self.num_keys_values
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
        for i in range(self.num_keys_values):
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
        self.patch_size = self.qwenvl.visual.patch_size
        self.merge_size = self.qwenvl.visual.spatial_merge_size
        self.means = torch.full((1, 1, 1, 1, 1), 127.5, dtype=torch.float32)
        self.inv_std = torch.full((1, 1, 1, 1, 1), 1.0 / 127.5, dtype=torch.float32)
        grid_thw = torch.tensor([[1, HEIGHT_FACTOR * 2, WIDTH_FACTOR * 2]], dtype=torch.int32)
        self.pos_embeds = Qwen3VLVisionModel.fast_pos_embed_interpolate(self.qwenvl.visual, grid_thw).unsqueeze(0)
        self.image_hidden_len = grid_thw[0, 1] * grid_thw[0, 2]
        rotary_pos_emb = Qwen3VLVisionModel.rot_pos_emb(self.qwenvl.visual, grid_thw).float().unsqueeze(0).unsqueeze(0)
        cos = rotary_pos_emb.cos()
        sin = rotary_pos_emb.sin()
        self.rotary_pos_emb_cos = torch.cat([cos, cos], dim=-1)
        self.rotary_pos_emb_sin = torch.cat([sin, sin], dim=-1)
        scaling = float(self.head_dim ** -0.25)
        for blk in self.qwenvl.visual.blocks:
            blk.attn.qkv.weight.data[:-self.qwenvl.visual.patch_embed.embed_dim] *= scaling
            blk.attn.qkv.bias.data[:-self.qwenvl.visual.patch_embed.embed_dim] *= scaling

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0] # [batch, 1, 3, width, height]
        pixel_values = pixel_values.float()
        if DYNAMIC_IMAGE_SHAPE or list(pixel_values.shape[-2:]) != IMAGE_RESIZE:
            pixel_values = pixel_values.squeeze(1)
            pixel_values = torch.nn.functional.interpolate(
                pixel_values,
                IMAGE_RESIZE,
                mode='bilinear',
                align_corners=False)
            pixel_values = pixel_values.unsqueeze(1)
        pixel_values = (pixel_values - self.means) * self.inv_std
        pixel_values = torch.cat([pixel_values, pixel_values], dim=1)
        pixel_values = pixel_values.reshape(
            batch_size,
            1,
            self.qwenvl.visual.patch_embed.temporal_patch_size,
            3,
            HEIGHT_FACTOR,
            self.merge_size,
            self.patch_size,
            WIDTH_FACTOR,
            self.merge_size,
            self.patch_size
        )
        pixel_values = pixel_values.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        pixel_values = pixel_values.reshape(
            -1,
            3,
            self.qwenvl.visual.patch_embed.temporal_patch_size,
            self.patch_size,
            self.patch_size
        )
        vision_hidden_states = self.qwenvl.visual.patch_embed.proj(pixel_values).view(1, -1, self.qwenvl.visual.patch_embed.embed_dim)
        if batch_size != 1:
            if DYNAMIC_IMAGE_SHAPE:
                self.pos_embeds = self.pos_embeds.repeat(1, batch_size, 1)                        # For batch size != 1 & dynamic shape inputs
                self.rotary_pos_emb_cos = self.rotary_pos_emb_cos.repeat(1, 1, batch_size, 1)
                self.rotary_pos_emb_sin = self.rotary_pos_emb_sin.repeat(1, 1, batch_size, 1)
            else:
                self.pos_embeds = torch.cat([self.pos_embeds for _ in range(batch_size)], dim=1)  # For batch size != 1 & static shape inputs.
                self.rotary_pos_emb_cos = torch.cat([self.rotary_pos_emb_cos for _ in range(batch_size)], dim=2)
                self.rotary_pos_emb_sin = torch.cat([self.rotary_pos_emb_sin for _ in range(batch_size)], dim=2)
        vision_hidden_states += self.pos_embeds
        deepstack_features = []
        for layer_num, blk in enumerate(self.qwenvl.visual.blocks):
            hidden_states_norm = blk.norm1(vision_hidden_states)
            q, k, v = blk.attn.qkv(hidden_states_norm).reshape(-1, 3, self.num_heads, self.head_dim).permute(1, 2, 0, 3).split([1, 1, 1], dim=0)
            q = q * self.rotary_pos_emb_cos + rotate_half(q, self.head_dim_half, -1) * self.rotary_pos_emb_sin
            k = k * self.rotary_pos_emb_cos + rotate_half(k, self.head_dim_half, -1) * self.rotary_pos_emb_sin
            q = torch.cat(q.split(self.image_hidden_len, dim=2), dim=0)
            k = torch.cat(k.split(self.image_hidden_len, dim=2), dim=0)
            v = torch.cat(v.split(self.image_hidden_len, dim=2), dim=0)
            attn = torch.nn.functional.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1, dtype=torch.float32)
            attn = torch.matmul(attn, v).transpose(1, 2)
            attn_out = blk.attn.proj(attn.reshape(1, -1, blk.attn.proj.in_features))
            vision_hidden_states += attn_out
            vision_hidden_states = vision_hidden_states + blk.mlp(blk.norm2(vision_hidden_states))
            if layer_num in self.qwenvl.visual.deepstack_visual_indexes:
                deepstack_layer = self.qwenvl.visual.deepstack_merger_list[self.qwenvl.visual.deepstack_visual_indexes.index(layer_num)]
                x = vision_hidden_states.view(1, -1, deepstack_layer.hidden_size)
                x = deepstack_layer.norm(x)
                x = deepstack_layer.linear_fc2(deepstack_layer.act_fn(deepstack_layer.linear_fc1(x)))
                deepstack_features.append(x)
        vision_hidden_states = self.qwenvl.visual.merger.norm(vision_hidden_states)
        vision_hidden_states = self.qwenvl.visual.merger.linear_fc2(self.qwenvl.visual.merger.act_fn(self.qwenvl.visual.merger.linear_fc1(vision_hidden_states.view(1, -1, self.qwenvl.visual.merger.hidden_size))))
        return deepstack_features, vision_hidden_states


class QwenVL_PartC(torch.nn.Module):
    def __init__(self, max_seq_len, prompt_head_len, hidden_size, deepstack_features_len):
        super(QwenVL_PartC, self).__init__()
        self.prompt_head_len = prompt_head_len
        self.zeros_A = torch.zeros([1, prompt_head_len, hidden_size], dtype=torch.float32)
        self.zeros_B = torch.zeros([1, max_seq_len, hidden_size], dtype=torch.int8)
        self.deepstack_features_len = deepstack_features_len

    def forward(self, *all_inputs):
        text_hidden_states = all_inputs[-2]
        vision_hidden_states = all_inputs[-1]
        concate_hidden_states = torch.cat([text_hidden_states[:, :self.prompt_head_len], vision_hidden_states, text_hidden_states[:, self.prompt_head_len:]], dim=1)
        zeros_B = self.zeros_B[:, :text_hidden_states.shape[1] - self.prompt_head_len].float()
        deepstack_features = [torch.cat([self.zeros_A, all_inputs[i], zeros_B], dim=1) for i in range(self.deepstack_features_len)]
        return deepstack_features, concate_hidden_states


class QwenVL_PartD(torch.nn.Module):
    def __init__(self, qwenvl, width_factor, height_factor, prompt_head_len, max_seq_len):
        super(QwenVL_PartD, self).__init__()
        self.vision_embed_size = width_factor * height_factor
        self.prompt_head_len = prompt_head_len
        self.image_factor_plus = self.vision_embed_size + self.prompt_head_len
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
        fill_tail_position = torch.arange(self.start_id, max_seq_len + self.start_id, dtype=torch.float32).repeat(3, 1, 1)
        position_ids = torch.cat([position_ids[:, :, :self.image_factor_plus], fill_tail_position], dim=-1)
        self.inv_freq_expanded = qwenvl.model.language_model.rotary_emb.inv_freq[None, :, None].float().expand(3, -1, 1)
        freqs = (self.inv_freq_expanded @ position_ids)
        freqs = freqs.transpose(-1, -2).unsqueeze(1)
        freqs = Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope(qwenvl.model.language_model.rotary_emb, freqs, qwenvl.model.language_model.rotary_emb.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0)
        self.rotary_pos_emb_cos = emb.cos().half()
        self.rotary_pos_emb_sin = emb.sin().half()
        
    def forward(self, history_len, kv_seq_len):
        rotary_pos_emb_cos_q = self.rotary_pos_emb_cos[..., history_len:kv_seq_len, :].float()
        rotary_pos_emb_sin_q = self.rotary_pos_emb_sin[..., history_len:kv_seq_len, :].float()
        rotary_pos_emb_cos_k = rotary_pos_emb_cos_q.transpose(-1, -2).unsqueeze(0)
        rotary_pos_emb_sin_k = rotary_pos_emb_sin_q.transpose(-1, -2).unsqueeze(0)
        return rotary_pos_emb_cos_q, rotary_pos_emb_sin_q, rotary_pos_emb_cos_k, rotary_pos_emb_sin_k
    

class QwenVL_PartE(torch.nn.Module):
    def __init__(self, qwenvl, max_seq_len):
        super(QwenVL_PartE, self).__init__()
        position_ids = torch.arange(max_seq_len, dtype=torch.float32).repeat(3, 1, 1)
        self.inv_freq_expanded = qwenvl.model.language_model.rotary_emb.inv_freq[None, :, None].float().expand(3, -1, 1)
        freqs = (self.inv_freq_expanded @ position_ids)
        freqs = freqs.transpose(-1, -2).unsqueeze(1)
        freqs = Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope(qwenvl.model.language_model.rotary_emb, freqs, qwenvl.model.language_model.rotary_emb.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0)
        self.rotary_pos_emb_cos = emb.cos().half()
        self.rotary_pos_emb_sin = emb.sin().half()
        
    def forward(self, history_len, kv_seq_len):
        rotary_pos_emb_cos_q = self.rotary_pos_emb_cos[..., history_len:kv_seq_len, :].float()
        rotary_pos_emb_sin_q = self.rotary_pos_emb_sin[..., history_len:kv_seq_len, :].float()
        rotary_pos_emb_cos_k = rotary_pos_emb_cos_q.transpose(-1, -2).unsqueeze(0)
        rotary_pos_emb_sin_k = rotary_pos_emb_sin_q.transpose(-1, -2).unsqueeze(0)
        return rotary_pos_emb_cos_q, rotary_pos_emb_sin_q, rotary_pos_emb_cos_k, rotary_pos_emb_sin_k


class QwenVL_PartF(torch.nn.Module):
    def __init__(self, qwenvl, head_dim, num_heads, num_layers, num_key_value_heads, max_seq_len, prompt_head_len, deepstack_features_len, height_factor, width_factor):
        super(QwenVL_PartF, self).__init__()
        self.qwenvl = qwenvl
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_key_value = num_layers + num_layers
        self.num_key_value_plus_1 = self.num_key_value + 1
        self.num_key_value_heads = num_key_value_heads
        self.head_dim_half = head_dim // 2
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.save_key = [None] * num_layers
        self.save_value = [None] * num_layers
        max_seq_len += height_factor * width_factor
        self.attention_mask = (1 - torch.tril(torch.ones([1, 1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128
        self.deepstack_features_len = deepstack_features_len
        scale_factor = float(head_dim ** -0.25)
        for i in range(num_layers):
            self.qwenvl.model.language_model.layers._modules[f'{i}'].self_attn.q_norm.weight.data *= scale_factor
            self.qwenvl.model.language_model.layers._modules[f'{i}'].self_attn.k_norm.weight.data *= scale_factor

    def forward(self, *all_inputs):
        hidden_states = all_inputs[self.num_key_value]
        rotary_pos_emb_cos_q = all_inputs[-7]
        rotary_pos_emb_sin_q = all_inputs[-6]
        rotary_pos_emb_cos_k = all_inputs[-5]
        rotary_pos_emb_sin_k = all_inputs[-4]
        kv_seq_len = all_inputs[-3]
        ids_len = all_inputs[-2]
        attention_mask = all_inputs[-1]
        attention_mask = (self.attention_mask[..., :ids_len, :kv_seq_len] * attention_mask).float()
        batch_size = hidden_states.shape[0].unsqueeze(0)
        for i, layer in enumerate(self.qwenvl.model.language_model.layers):
            hidden_states_norm = layer.input_layernorm(hidden_states)
            q = layer.self_attn.q_proj(hidden_states_norm).view(batch_size, -1, self.num_heads, self.head_dim)
            k = layer.self_attn.k_proj(hidden_states_norm).view(batch_size, -1, 1, self.num_key_value_heads, self.head_dim)
            v = layer.self_attn.v_proj(hidden_states_norm).view(batch_size, -1, 1, self.num_key_value_heads, self.head_dim).transpose(1, 3)
            q = (layer.self_attn.q_norm(q)).transpose(1, 2)
            k = (layer.self_attn.k_norm(k)).permute(0, 3, 2, 4, 1)
            q = q * rotary_pos_emb_cos_q + rotate_half(q, self.head_dim_half, -1) * rotary_pos_emb_sin_q
            k = k * rotary_pos_emb_cos_k + rotate_half(k, self.head_dim_half, -2) * rotary_pos_emb_sin_k
            k = torch.cat((all_inputs[i], k), dim=-1)
            v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)
            self.save_key[i] = k
            self.save_value[i] = v
            k = repeat_k(k, self.num_key_value_groups, self.head_dim, self.num_heads, batch_size)
            v = repeat_v(v, self.num_key_value_groups, self.head_dim, self.num_heads, batch_size)
            attn = torch.nn.functional.softmax(torch.matmul(q, k) + attention_mask, dim=-1, dtype=torch.float32)
            attn_out = layer.self_attn.o_proj(torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, -1, layer.self_attn.o_proj.in_features))
            hidden_states += attn_out
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp.down_proj(layer.mlp.act_fn(layer.mlp.gate_proj(hidden_states)) * layer.mlp.up_proj(hidden_states))
            hidden_states += residual
            if i < self.deepstack_features_len:
                hidden_states += all_inputs[self.num_key_value_plus_1 + i]
        hidden_states = hidden_states[:, -1]
        hidden_states = self.qwenvl.model.language_model.norm(hidden_states)
        logits = self.qwenvl.lm_head(hidden_states)
        return *self.save_key, *self.save_value, logits, kv_seq_len + 1


# Load the model
with torch.inference_mode():
    model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True).eval()
    num_heads = model.config.text_config.num_attention_heads
    num_key_value_heads = model.config.text_config.num_key_value_heads
    head_dim = model.config.text_config.head_dim
    num_layers = model.config.text_config.num_hidden_layers
    hidden_size = model.config.text_config.hidden_size
    vocab_size = model.model.language_model.vocab_size
    deepstack_features_len = len(model.visual.deepstack_visual_indexes)

    pixel_values = torch.randint(low=0, high=255, size=[VISION_BATCH_SIZE, 1, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]]).to(torch.uint8)
    vision_embed_size = WIDTH_FACTOR * HEIGHT_FACTOR * VISION_BATCH_SIZE
    vision_hidden_states = torch.ones((1, vision_embed_size, hidden_size), dtype=torch.float32)
    deepstack_features = torch.ones((1, vision_embed_size, hidden_size), dtype=torch.float32)
    prompt_head_len = 4                                      # <|im_start|>user\n<|vision_start|>
    ids_len = torch.tensor([10], dtype=torch.long)      # "10" is just a dummy value.
    history_len = torch.tensor([0], dtype=torch.long)
    input_ids = torch.ones((1, ids_len), dtype=torch.int32)
    text_hidden_states = torch.ones((1, ids_len, hidden_size), dtype=torch.float32)
    batch_size = 3
    ids_len = ids_len + vision_embed_size
    hidden_states = torch.ones((batch_size, ids_len, hidden_size), dtype=torch.float32)
    rotary_pos_emb_cos_q = torch.ones((1, 1, ids_len, head_dim), dtype=torch.float32)
    rotary_pos_emb_sin_q = rotary_pos_emb_cos_q
    rotary_pos_emb_cos_k = rotary_pos_emb_cos_q.transpose(-1, -2).unsqueeze(0)
    rotary_pos_emb_sin_k = rotary_pos_emb_cos_k
    attention_mask = torch.tensor([1], dtype=torch.int8)
    past_keys = torch.zeros((batch_size, num_key_value_heads, 1, head_dim, 0), dtype=torch.float32)
    past_values = torch.zeros((batch_size, num_key_value_heads, 1, 0, head_dim), dtype=torch.float32)
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
            'input_ids': {0: 'batch', 1: 'ids_len'},
            'text_hidden_states': {0: 'batch', 1: 'ids_len'}
        },
        do_constant_folding=True,
        opset_version=17,
        dynamo = False
    )
    del model_A
    gc.collect()
    print('\nExport Part_A Done!  \n\nExport Part_B Start...')


    model_B = QwenVL_PartB(model)
    dynamic_axes = {
        'pixel_values': {0: 'batch_size', 3: 'width', 4: 'height'},
        'vision_hidden_states': {1: 'vision_embed_len'}
    }
    output_names = []
    for i in range(deepstack_features_len):
        name = f'deepstack_feature_{i}'
        output_names.append(name)
        dynamic_axes[name] = {1: 'vision_embed_len'}
    output_names.append('vision_hidden_states')

    torch.onnx.export(
        model_B,
        (pixel_values,),
        onnx_model_B,
        input_names=['pixel_values'],
        output_names=output_names,
        dynamic_axes=dynamic_axes if DYNAMIC_IMAGE_SHAPE else None,
        do_constant_folding=True,
        opset_version=17,
        dynamo=False
    )
    del model_B
    del pixel_values
    gc.collect()
    print('\nExport Part_B Done! \n\nExport Part_C Start...')

    all_inputs = []
    input_names = []
    output_names = []
    dynamic_axes = {
        'text_hidden_states': {1: 'ids_len'},
        'vision_hidden_states': {1: 'vision_embed_len'},
        'concate_hidden_states': {1: 'total_len'}
    }
    for i in range(deepstack_features_len):
        name = f'in_deepstack_feature_{i}'
        input_names.append(name)
        all_inputs.append(deepstack_features)
        dynamic_axes[name] = {1: 'vision_embed_len'}
        name = f'out_deepstack_feature_{i}'
        dynamic_axes[name] = {1: 'total_len'}
        output_names.append(name)
    input_names.append("text_hidden_states")
    input_names.append("vision_hidden_states")
    all_inputs.append(text_hidden_states)
    all_inputs.append(vision_hidden_states)
    output_names.append("concate_hidden_states")

    model_C = QwenVL_PartC(MAX_SEQ_LENGTH, prompt_head_len, hidden_size, deepstack_features_len)
    torch.onnx.export(
        model_C,
        tuple(all_inputs),
        onnx_model_C,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=17,
        dynamo=False
    )
    del model_C
    del text_hidden_states
    del vision_hidden_states
    print('\nExport Part_C Done! \n\nExport Part_D Start...')

    model_D = QwenVL_PartD(model, WIDTH_FACTOR, HEIGHT_FACTOR, prompt_head_len, MAX_SEQ_LENGTH)
    torch.onnx.export(
        model_D,
        (history_len, kv_seq_len),
        onnx_model_D,
        input_names=['history_len', 'kv_seq_len'],
        output_names=['rotary_pos_emb_cos_q', 'rotary_pos_emb_sin_q', 'rotary_pos_emb_cos_k', 'rotary_pos_emb_sin_k'],
        dynamic_axes={
            'rotary_pos_emb_cos_q': {2: 'hidden_state_len'},
            'rotary_pos_emb_sin_q': {2: 'hidden_state_len'},
            'rotary_pos_emb_cos_k': {4: 'hidden_state_len'},
            'rotary_pos_emb_sin_k': {4: 'hidden_state_len'}
        },
        do_constant_folding=True,
        opset_version=17,
        dynamo=False
    )
    del model_D
    print('\nExport Part_D Done! \n\nExport Part_E Start...')

    model_E = QwenVL_PartE(model, MAX_SEQ_LENGTH)
    torch.onnx.export(
        model_E,
        (history_len, kv_seq_len),
        onnx_model_E,
        input_names=['history_len', 'kv_seq_len'],
        output_names=['rotary_pos_emb_cos_q', 'rotary_pos_emb_sin_q', 'rotary_pos_emb_cos_k', 'rotary_pos_emb_sin_k'],
        dynamic_axes={
            'rotary_pos_emb_cos_q': {2: 'hidden_state_len'},
            'rotary_pos_emb_sin_q': {2: 'hidden_state_len'},
            'rotary_pos_emb_cos_k': {4: 'hidden_state_len'},
            'rotary_pos_emb_sin_k': {4: 'hidden_state_len'}
        },
        do_constant_folding=True,
        opset_version=17,
        dynamo=False
    )
    del model_E
    print('\nExport Part_E Done! \n\nExport Part_F Start...')

    model_F = QwenVL_PartF(model, head_dim, num_heads, num_layers, num_key_value_heads, MAX_SEQ_LENGTH, prompt_head_len, deepstack_features_len, HEIGHT_FACTOR, WIDTH_FACTOR)
    del model
    gc.collect()
    all_inputs = []
    input_names = []
    output_names = []
    dynamic_axes = {
        'hidden_states': {0: 'batch', 1: 'hidden_state_len'},
        'rotary_pos_emb_cos_q': {2: 'hidden_state_len'},
        'rotary_pos_emb_sin_q': {2: 'hidden_state_len'},
        'rotary_pos_emb_cos_k': {4: 'hidden_state_len'},
        'rotary_pos_emb_sin_k': {4: 'hidden_state_len'}
    }
    for i in range(num_layers):
        name = f'in_key_{i}'
        input_names.append(name)
        all_inputs.append(past_keys)
        dynamic_axes[name] = {0: 'batch', 4: 'history_len'}
        name = f'out_key_{i}'
        output_names.append(name)
        dynamic_axes[name] = {0: 'batch', 4: 'history_len_plus'}
    for i in range(num_layers):
        name = f'in_value_{i}'
        input_names.append(name)
        all_inputs.append(past_values)
        dynamic_axes[name] = {0: 'batch', 3: 'history_len'}
        name = f'out_value_{i}'
        output_names.append(name)
        dynamic_axes[name] = {0: 'batch', 3: 'history_len_plus_ids_len'}
    input_names.append('hidden_states')
    all_inputs.append(hidden_states)
    deepstack_features = torch.ones((1, ids_len, hidden_size), dtype=torch.float32)
    for i in range(deepstack_features_len):
        name = f'deepstack_features_{i}'
        input_names.append(name)
        dynamic_axes[name] = {1: 'total_len'}
        all_inputs.append(deepstack_features)
    input_names.append('rotary_pos_emb_cos_q')
    all_inputs.append(rotary_pos_emb_cos_q)
    input_names.append('rotary_pos_emb_sin_q')
    all_inputs.append(rotary_pos_emb_sin_q)
    input_names.append('rotary_pos_emb_cos_k')
    all_inputs.append(rotary_pos_emb_cos_k)
    input_names.append('rotary_pos_emb_sin_k')
    all_inputs.append(rotary_pos_emb_sin_k)
    input_names.append('kv_seq_len')
    all_inputs.append(kv_seq_len)
    input_names.append('ids_len')
    all_inputs.append(ids_len)
    input_names.append('attention_mask')
    all_inputs.append(attention_mask)
    output_names.append('logits')
    output_names.append('kv_seq_len_plus')
    dynamic_axes['logits'] = {0: 'batch'}

    torch.onnx.export(
        model_F,
        tuple(all_inputs),
        onnx_model_F,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=17,
        dynamo=False
    )
    del model_F
    del input_names
    del output_names
    del dynamic_axes
    del all_inputs
    del num_heads
    del num_key_value_heads
    del head_dim
    del hidden_size
    del deepstack_features_len
    del vision_embed_size
    del deepstack_features
    del prompt_head_len
    del ids_len
    del history_len
    del input_ids
    del batch_size
    del hidden_states
    del rotary_pos_emb_cos_q
    del rotary_pos_emb_sin_q
    del rotary_pos_emb_cos_k
    del rotary_pos_emb_sin_k
    del attention_mask
    del kv_seq_len
    gc.collect()
print('\nExport Part_F Done! \n\nExport Part_G Start...')

greedy = GREEDY_SEARCH()
beam_size = torch.tensor([BEAM_SIZE], dtype=torch.int64)
repeat_penality = torch.ones((beam_size, vocab_size), dtype=torch.float32)
penality_reset_count = torch.zeros(beam_size, dtype=torch.int32)
logits = torch.ones((beam_size, vocab_size), dtype=torch.float32)
penality_value = torch.tensor(REPEAT_PENALITY, dtype=torch.float32)
batch_indices = torch.arange(BEAM_SIZE, dtype=torch.int64)

torch.onnx.export(
    greedy,
    (logits, repeat_penality, penality_value, beam_size),
    # Reuse the beam_size tensor as batch_size during export process.
    onnx_model_G,
    input_names=['logits', 'repeat_penality_in', 'penality_value', 'batch_size'],
    output_names=['max_logits_idx', 'repeat_penality_out'],
    dynamic_axes={
        'logits': {0: 'batch'},
        'repeat_penality_in': {0: 'batch'},
        'repeat_penality_out': {0: 'batch'},
        'max_logits_idx': {0: 'batch'}
    },
    do_constant_folding=True,
    opset_version=17,
    dynamo=False
)
del greedy
print('\nExport Part_G Done! \n\nExport Part_H Start...')

first_beam_search = FIRST_BEAM_SEARCH(num_layers)
topK = torch.tensor([TOP_K], dtype=torch.int64)
save_id = torch.zeros((beam_size, 10), dtype=torch.int32)
previous_prob = torch.zeros((beam_size, 1), dtype=torch.float32)
past_keys_greedy = past_keys[[0]]
past_values_greedy = past_values[[0]]

all_inputs = []
input_names = []
output_names = []
dynamic_axes = {}
for i in range(num_layers):
    name = f'in_key_{i}'
    input_names.append(name)
    all_inputs.append(past_keys_greedy)
    dynamic_axes[name] = {0: 'batch', 4: 'history_len'}
    name = f'out_key_{i}'
    output_names.append(name)
    dynamic_axes[name] = {0: 'batch', 4: 'history_len_plus_ids_len'}
for i in range(num_layers):
    name = f'in_value_{i}'
    input_names.append(name)
    all_inputs.append(past_values_greedy)
    dynamic_axes[name] = {0: 'batch', 3: 'history_len'}
    name = f'out_value_{i}'
    output_names.append(name)
    dynamic_axes[name] = {0: 'batch', 3: 'history_len_plus_ids_len'}
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
    onnx_model_H,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    do_constant_folding=True,
    opset_version=17,
    dynamo=False
)
del first_beam_search
print('\nExport Part_H Done! \n\nExport Part_I Start...')

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

second_beam_search = SECOND_BEAM_SEARCH(num_layers)
torch.onnx.export(
    second_beam_search,
    tuple(all_inputs),
    onnx_model_I,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    do_constant_folding=True,
    opset_version=17,
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
del logits
del beam_size
del penality_value
print('\nExport Part_I Done! \n\nExport Part_J Start...')

reset_penality = RESET_PENALITY()
torch.onnx.export(
    reset_penality,
    (save_id, repeat_penality, penality_reset_count, batch_indices),
    onnx_model_J,
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
    do_constant_folding=True,
    opset_version=17,
    dynamo=False
)
del reset_penality
del save_id
del repeat_penality
del penality_reset_count
del batch_indices
print('\nExport Part_J Done!\n\nStrat to run QwenVL by ONNX Runtime.')


# Run the exported model by ONNX Runtime
# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
run_options = onnxruntime.RunOptions()
session_opts.log_severity_level = 4                 # Fatal level, it an adjustable value.
session_opts.log_verbosity_level = 4                # Fatal level, it an adjustable value.
run_options.log_severity_level = 4                  # Fatal level, it an adjustable value.
run_options.log_verbosity_level = 4                 # Fatal level, it an adjustable value.
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
session_opts.add_session_config_entry('optimization.enable_gelu_approximation', '1')
session_opts.add_session_config_entry('optimization.minimal_build_optimizations', '')
session_opts.add_session_config_entry('session.use_device_allocator_for_initializers', '1')
session_opts.add_session_config_entry('optimization.enable_cast_chain_elimination', '1')
session_opts.add_session_config_entry('session.graph_optimizations_loop_level', '2')

run_options.add_run_config_entry('disable_synchronize_execution_providers', '1')

ORT_Accelerate_Providers = ['CPUExecutionProvider']
provider_options = None
device_type = 'cpu'
DEVICE_ID = 0


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A = in_name_A[0].name
out_name_A = [out_name_A[0].name]


ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
in_name_B = [in_name_B[i].name for i in range(len(in_name_B))]
out_name_B = [out_name_B[i].name for i in range(len(out_name_B))]


ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
in_name_C = ort_session_C.get_inputs()
out_name_C = ort_session_C.get_outputs()
amount_of_outputs_C = len(out_name_C)
in_name_C = [in_name_C[i].name for i in range(len(in_name_C))]
out_name_C = [out_name_C[i].name for i in range(amount_of_outputs_C)]


ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
in_name_D = ort_session_D.get_inputs()
out_name_D = ort_session_D.get_outputs()
amount_of_outputs_D = len(out_name_D)
in_name_D = [in_name_D[i].name for i in range(len(in_name_D))]
out_name_D = [out_name_D[i].name for i in range(amount_of_outputs_D)]


ort_session_E = onnxruntime.InferenceSession(onnx_model_E, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
in_name_E = ort_session_E.get_inputs()
out_name_E = ort_session_E.get_outputs()
in_name_E = [in_name_E[i].name for i in range(len(in_name_E))]
out_name_E = [out_name_E[i].name for i in range(len(out_name_E))]


ort_session_F = onnxruntime.InferenceSession(onnx_model_F, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
print(f"\nUsable Providers: {ort_session_F.get_providers()}")
model_dtype = ort_session_F._inputs_meta[0].type
if 'float16' in model_dtype:
    model_dtype = np.float16
else:
    model_dtype = np.float32
in_name_F = ort_session_F.get_inputs()
out_name_F = ort_session_F.get_outputs()
amount_of_outputs_F = len(out_name_F)
in_name_F = [in_name_F[i].name for i in range(len(in_name_F))]
out_name_F = [out_name_F[i].name for i in range(amount_of_outputs_F)]


generate_limit = MAX_SEQ_LENGTH - 10                   # 10 = length of basic ids
deepstack_features_len = amount_of_outputs_C - 1
rotary_outputs_len = amount_of_outputs_D
vision_embed_size = VISION_BATCH_SIZE * WIDTH_FACTOR * HEIGHT_FACTOR
num_layers = (amount_of_outputs_F - 2) // 2
num_keys_values = num_layers + num_layers
num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
num_keys_values_plus_3 = num_keys_values + 3
num_keys_values_plus_4 = num_keys_values + 4
num_keys_values_plus_5 = num_keys_values + 5
num_keys_values_plus_6 = num_keys_values + 6
num_keys_values_plus_7 = num_keys_values + 7
num_keys_values_plus_deepstack_plus_rotary = num_keys_values + deepstack_features_len + rotary_outputs_len + 1
num_keys_values_plus_deepstack_plus_rotary_plus_1 = num_keys_values_plus_deepstack_plus_rotary + 1
num_keys_values_plus_deepstack_plus_rotary_plus_2 = num_keys_values_plus_deepstack_plus_rotary + 2
rotary_indices = np.arange(rotary_outputs_len, dtype=np.int32) + num_keys_values + deepstack_features_len + 1
deepstack_indices = np.arange(deepstack_features_len, dtype=np.int32) + num_keys_values + 1
rotary_in_name_F = in_name_F[rotary_indices[0]:rotary_indices[rotary_outputs_len - 1] + 1]
deepstack_in_name_F = in_name_F[deepstack_indices[0]:deepstack_indices[deepstack_features_len - 1] + 1]
vocab_size = ort_session_F._outputs_meta[num_keys_values].shape[1]
topK = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([TOP_K], dtype=np.int64), device_type, DEVICE_ID)
beam_size = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([BEAM_SIZE], dtype=np.int64), device_type, DEVICE_ID)
penality_value = onnxruntime.OrtValue.ortvalue_from_numpy(np.array(REPEAT_PENALITY, dtype=model_dtype), device_type, DEVICE_ID)
prompt = f"<|im_start|>user\n<|vision_start|><|vision_end|>{query}<|im_end|>\n<|im_start|>assistant\n"
prompt_head_len = np.array([4], dtype=np.int64)
tokenizer = AutoTokenizer.from_pretrained(path)


# Pre-process inputs
if USE_BEAM_SEARCH and (TOP_K < BEAM_SIZE):
    print("\nBeam Search does not display the immediate decoding results; the best result is shown only after the entire decoding process is complete.\n")
    TOP_K = BEAM_SIZE


if (TOP_K < 2) or (BEAM_SIZE < 2):
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")


if USE_BEAM_SEARCH:
    ort_session_H = onnxruntime.InferenceSession(onnx_model_H, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    in_name_H = ort_session_H.get_inputs()
    out_name_H = ort_session_H.get_outputs()
    in_name_H = [in_name_H[i].name for i in range(len(in_name_H))]
    out_name_H = [out_name_H[i].name for i in range(len(out_name_H))]
    
    ort_session_I = onnxruntime.InferenceSession(onnx_model_I, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    in_name_I = ort_session_I.get_inputs()
    out_name_I = ort_session_I.get_outputs()
    in_name_I = [in_name_I[i].name for i in range(len(in_name_I))]
    out_name_I = [out_name_I[i].name for i in range(len(out_name_I))]
    
    ort_session_J = onnxruntime.InferenceSession(onnx_model_J, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    in_name_J = ort_session_J.get_inputs()
    out_name_J = ort_session_J.get_outputs()
    in_name_J = [in_name_J[i].name for i in range(len(in_name_J))]
    out_name_J = [out_name_J[i].name for i in range(len(out_name_J))]

    input_feed_H = {
        in_name_H[num_keys_values_plus_3]: penality_value,
        in_name_H[num_keys_values_plus_4]: beam_size
    }

    input_feed_I = {
        in_name_I[num_keys_values_plus_5]: penality_value,
        in_name_I[num_keys_values_plus_6]: beam_size,
        in_name_I[num_keys_values_plus_7]: topK
    }

else:
    BEAM_SIZE = 1
    ort_session_G = onnxruntime.InferenceSession(onnx_model_G, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    in_name_G = ort_session_G.get_inputs()
    out_name_G = ort_session_G.get_outputs()
    in_name_G = [in_name_G[i].name for i in range(len(in_name_G))]
    out_name_G = [out_name_G[i].name for i in range(len(out_name_G))]
    input_feed_G = {in_name_G[2]: penality_value}


if USE_BEAM_SEARCH:
    penality_reset_count_beam_init = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(BEAM_SIZE, dtype=np.int32), device_type, DEVICE_ID)
else:
    save_id_greedy = np.zeros(MAX_SEQ_LENGTH, dtype=np.int32)


if REPEAT_PENALITY != 1.0:
    do_repeat_penality = True
else:
    do_repeat_penality = False


num_decode = 0
tokens = tokenizer(prompt, return_tensors='np')['input_ids'].astype(np.int32)
input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, DEVICE_ID)
ids_len = np.array([tokens.shape[-1]], dtype=np.int64)
kv_seq_len = onnxruntime.OrtValue.ortvalue_from_numpy(ids_len, device_type, DEVICE_ID)
ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(ids_len, device_type, DEVICE_ID)
init_ids_len_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
init_history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), device_type, DEVICE_ID)
init_attention_mask_0 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), device_type, DEVICE_ID)
init_attention_mask_1 = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), device_type, DEVICE_ID)
if device_type != 'dml':
    init_past_keys_F = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_F._inputs_meta[0].shape[1], 1, ort_session_F._inputs_meta[0].shape[3], 0), dtype=model_dtype), device_type, DEVICE_ID)
    init_past_values_F = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_F._inputs_meta[num_layers].shape[1], 1, 0, ort_session_F._inputs_meta[num_layers].shape[4]), dtype=model_dtype), device_type, DEVICE_ID)
else:
    init_past_keys_F = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_F._inputs_meta[0].shape[1], 1, ort_session_F._inputs_meta[0].shape[3], 0), dtype=model_dtype), 'cpu', 0)
    init_past_values_F = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_F._inputs_meta[num_layers].shape[1], 1, 0, ort_session_F._inputs_meta[num_layers].shape[4]), dtype=model_dtype), 'cpu', 0)
init_deepstack_features = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, 1, ort_session_C._inputs_meta[0].shape[2]), dtype=model_dtype), device_type, DEVICE_ID)
init_save_id = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((BEAM_SIZE, 0), dtype=np.int32), device_type, DEVICE_ID)
init_repeat_penality = onnxruntime.OrtValue.ortvalue_from_numpy(np.ones((BEAM_SIZE, vocab_size), dtype=model_dtype), device_type, DEVICE_ID)
init_batch_size_greedy = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, DEVICE_ID)
init_beam_size = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((BEAM_SIZE, 0), dtype=np.int32), device_type, DEVICE_ID)


generate_limit -= tokens.shape[-1]


# Load input image
if is_valid_image_path(image_path):
    image = Image.open(image_path)
    image = image.resize((INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0]))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    pixel_values = np.transpose(np.array(image).astype(np.uint8), (2, 0, 1))
    pixel_values = np.expand_dims(pixel_values, axis=(0, 1))
    use_vision = True
    print('\nChat with image.')
else:
    use_vision = False
    print('\nChat without image.')


# Start to run LLM
input_feed_A = {in_name_A: input_ids}
all_outputs_A = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)


if use_vision:
    ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(ids_len.numpy() + vision_embed_size, device_type, DEVICE_ID)
    kv_seq_len = onnxruntime.OrtValue.ortvalue_from_numpy(ids_len.numpy(), device_type, DEVICE_ID)

    generate_limit -= vision_embed_size

    print('\nStart to Process the Image...')
    input_feed_B = {in_name_B[0]: onnxruntime.OrtValue.ortvalue_from_numpy(pixel_values, device_type, DEVICE_ID)}
    start_time = time.time()
    all_outputs_B = ort_session_B.run_with_ort_values(out_name_B, input_feed_B)
    print(f'\nImage Process Complete. Time Cost: {time.time() - start_time:.3f} Seconds')
    input_feed_C = {
        in_name_C[deepstack_features_len]: all_outputs_A[0],
        in_name_C[amount_of_outputs_C]: all_outputs_B[deepstack_features_len]
    }
    for i in range(deepstack_features_len):
        input_feed_C[in_name_C[i]] = all_outputs_B[i]
    all_outputs_C = ort_session_C.run_with_ort_values(out_name_C, input_feed_C)
    input_feed_D = {
        in_name_D[0]: init_history_len,
        in_name_D[1]: kv_seq_len
    }
    rotary_outputs = ort_session_D.run_with_ort_values(out_name_D, input_feed_D)
    input_feed_F = {in_name_F[num_keys_values]: all_outputs_C[deepstack_features_len]}
    for i in range(deepstack_features_len):
        input_feed_F[deepstack_in_name_F[i]] = all_outputs_C[i]
else:
    input_feed_E = {
        in_name_E[0]: init_history_len,
        in_name_E[1]: kv_seq_len
    }
    rotary_outputs = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)
    input_feed_F = {in_name_F[num_keys_values]: all_outputs_A[0]}
    for i in range(deepstack_features_len):
        input_feed_F[deepstack_in_name_F[i]] = init_deepstack_features


input_feed_F[in_name_F[num_keys_values_plus_deepstack_plus_rotary]] = kv_seq_len
input_feed_F[in_name_F[num_keys_values_plus_deepstack_plus_rotary_plus_1]] = ids_len
input_feed_F[in_name_F[num_keys_values_plus_deepstack_plus_rotary_plus_2]] = init_attention_mask_1

for i in range(num_layers):
    input_feed_F[in_name_F[i]] = init_past_keys_F
for i in range(num_layers, num_keys_values):
    input_feed_F[in_name_F[i]] = init_past_values_F
for i in range(rotary_outputs_len):
    input_feed_F[in_name_F[rotary_indices[i]]] = rotary_outputs[i]


if USE_BEAM_SEARCH:
    input_feed_H[in_name_H[num_keys_values_plus_1]] = init_beam_size
    input_feed_H[in_name_H[num_keys_values_plus_2]] = init_repeat_penality
else:
    input_feed_G[in_name_G[1]] = init_repeat_penality
    input_feed_G[in_name_G[3]] = init_batch_size_greedy


if do_repeat_penality:
    if USE_BEAM_SEARCH:
        input_feed_J = {in_name_J[2]: penality_reset_count_beam_init}
    else:
        penality_reset_count_greedy = 0


print(f'\nTest Question: {query}\n\nQwenVL Answering:\n')
num_decode = 0
start_time = time.time()
while num_decode < generate_limit:
    all_outputs_F = ort_session_F.run_with_ort_values(out_name_F, input_feed_F)
    if USE_BEAM_SEARCH:
        if num_decode < 1:
            input_feed_H.update(zip(in_name_H[:num_keys_values_plus_1], all_outputs_F))
            all_outputs_H = ort_session_H.run_with_ort_values(out_name_H, input_feed_H)
            max_logits_idx = all_outputs_H[num_keys_values_plus_5].numpy()
            input_feed_I[in_name_I[num_keys_values_plus_4]] = all_outputs_H[num_keys_values_plus_4]
            if do_repeat_penality:
                input_feed_J[in_name_J[3]] = all_outputs_H[num_keys_values_plus_4]
        else:
            input_feed_I.update(zip(in_name_I[:num_keys_values_plus_1], all_outputs_F))
            all_outputs_I = ort_session_I.run_with_ort_values(out_name_I, input_feed_I)
            max_logits_idx = all_outputs_I[num_keys_values_plus_4].numpy()
        if max_logits_idx in STOP_TOKEN:
            save_id = all_outputs_I[num_keys_values_plus_1].numpy()[0, :num_decode]  # 0 is the Top_1
            sentence = tokenizer.decode(save_id)
            print(f"\nBest: {sentence}")
            break
        if do_repeat_penality and (num_decode >= PENALITY_RANGE):
            input_feed_J[in_name_J[0]] = all_outputs_I[num_keys_values_plus_1]
            input_feed_J[in_name_J[1]] = all_outputs_I[num_keys_values_plus_2]
            all_outputs_J = ort_session_J.run_with_ort_values(out_name_J, input_feed_J)
            input_feed_J[in_name_J[2]] = all_outputs_J[2]
            input_feed_I[in_name_I[num_keys_values_plus_1]] = all_outputs_J[0]
            input_feed_I[in_name_I[num_keys_values_plus_2]] = all_outputs_J[1]
        if num_decode < 1:
            input_feed_F.update(zip(in_name_F[:num_keys_values], all_outputs_H))
            input_feed_A[in_name_A] = all_outputs_H[num_keys_values]
            input_feed_I[in_name_I[num_keys_values_plus_1]] = all_outputs_H[num_keys_values_plus_1]
            input_feed_I[in_name_I[num_keys_values_plus_2]] = all_outputs_H[num_keys_values_plus_2]
            input_feed_I[in_name_I[num_keys_values_plus_3]] = all_outputs_H[num_keys_values_plus_3]
        else:
            input_feed_F.update(zip(in_name_F[:num_keys_values], all_outputs_I))
            input_feed_A[in_name_A] = all_outputs_I[num_keys_values]
            input_feed_I[in_name_I[num_keys_values_plus_1]] = all_outputs_I[num_keys_values_plus_1]
            input_feed_I[in_name_I[num_keys_values_plus_2]] = all_outputs_I[num_keys_values_plus_2]
            input_feed_I[in_name_I[num_keys_values_plus_3]] = all_outputs_I[num_keys_values_plus_3]
        input_feed_F[in_name_F[num_keys_values]] = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)[0]
    else:
        input_feed_G[in_name_G[0]] = all_outputs_F[num_keys_values]
        all_outputs_G = ort_session_G.run_with_ort_values(out_name_G, input_feed_G)
        max_logits_idx = all_outputs_G[0].numpy()[0, 0]
        if max_logits_idx in STOP_TOKEN:
            break
        print(tokenizer.decode(max_logits_idx), end="", flush=True)
        if do_repeat_penality and (num_decode >= PENALITY_RANGE):
            reset_ids = save_id_greedy[penality_reset_count_greedy]
            if reset_ids != max_logits_idx:
                repeat_penality = all_outputs_G[1].numpy()
                repeat_penality[:, reset_ids] = 1.0
                input_feed_G[in_name_G[1]].update_inplace(repeat_penality)
            penality_reset_count_greedy += 1
        else:
            input_feed_G[in_name_G[1]] = all_outputs_G[1]
        input_feed_G[in_name_G[0]] = all_outputs_G[0]
        input_feed_F.update(zip(in_name_F[:num_keys_values_plus_1], all_outputs_F))
        input_feed_A[in_name_A] = all_outputs_G[0]
        input_feed_F[in_name_F[num_keys_values]] = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)[0]
        save_id_greedy[num_decode] = max_logits_idx
    if use_vision:
        input_feed_D = {
            in_name_D[0]: input_feed_F[in_name_F[num_keys_values_plus_deepstack_plus_rotary]],
            in_name_D[1]: all_outputs_F[num_keys_values_plus_1]
        }
        rotary_outputs = ort_session_D.run_with_ort_values(out_name_D, input_feed_D)
    else:
        input_feed_E = {
            in_name_E[0]: input_feed_F[in_name_F[num_keys_values_plus_deepstack_plus_rotary]],
            in_name_E[1]: all_outputs_F[num_keys_values_plus_1]
        }
        rotary_outputs = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)
    input_feed_F[in_name_F[num_keys_values_plus_deepstack_plus_rotary]] = all_outputs_F[num_keys_values_plus_1]
    input_feed_F.update(zip(rotary_in_name_F, rotary_outputs))
    if num_decode < 1:
        input_feed_F[in_name_F[num_keys_values_plus_deepstack_plus_rotary_plus_1]] = init_ids_len_1
        input_feed_F[in_name_F[num_keys_values_plus_deepstack_plus_rotary_plus_2]] = init_attention_mask_0
        for i in range(deepstack_features_len):
            input_feed_F[deepstack_in_name_F[i]] = init_deepstack_features
    num_decode += 1
print(f"\n\nDecode: {((num_decode + 1) / (time.time() - start_time)):.3f} token/s")
