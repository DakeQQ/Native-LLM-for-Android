import os
import gc
import time
import torch
import numpy as np
from PIL import Image
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLVisionModel, AutoTokenizer
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding


path = r'/home/DakeQQ/Downloads/Qwen3-VL-2B-Instruct'                  # Set the folder path where the Qwen3-VL whole project downloaded.
onnx_model_A = r'/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Embed.onnx'      # Assign a path where the exported QwenVL model stored.
onnx_model_B = r'/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Vision.onnx'
onnx_model_C = r'/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Concat.onnx'
onnx_model_D = r'/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Rotary_Vision.onnx'
onnx_model_E = r'/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Rotary_Text.onnx'
onnx_model_F = r'/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Main.onnx'
onnx_model_G = r'/home/DakeQQ/Downloads/Qwen_ONNX/Greedy_Search.onnx'
onnx_model_H = r'/home/DakeQQ/Downloads/Qwen_ONNX/First_Beam_Search.onnx'
onnx_model_I = r'/home/DakeQQ/Downloads/Qwen_ONNX/Second_Beam_Search.onnx'
onnx_model_J = r'/home/DakeQQ/Downloads/Qwen_ONNX/Reset_Penality.onnx'
onnx_model_K = r'/home/DakeQQ/Downloads/Qwen_ONNX/Argmax.onnx'

# Test Input
image_path = r"../psyduck.png"                                      # Test image for the exported onnx model.
query = "Describe this image."                                      # Test query for the exported onnx model.

DO_EXPORT = True                                                    # Whether to export the ONNX models

KV_QUANT_DTYPE = "F16"                                              # "Q8" | "F16" | "F32"
USE_FLOAT16_SCALE_BIAS = True                                       # If choose Q8, whether to use float16 for scale and bias.
PREVENT_F16_OVERFLOW = False                                        # Prevent float16 overflow. Set True for Q4F16 or Q8F16 or F16 quantization.

MAX_SEQ_LEN = 4096                                                  # The max token length. Note, this value include inputs and generated.
STOP_TOKEN = [151643, 151645]                                       # The stop_id in Qwen is "151643" & "151645"

HEIGHT_FACTOR = 15                                                  # Adjust this value to determine the resize shape and vision resolution.
WIDTH_FACTOR = 15                                                   # Adjust this value to determine the resize shape and vision resolution.
IMAGE_RESIZE = [HEIGHT_FACTOR * 32, WIDTH_FACTOR * 32]              # 32 = self.patch_size * self.merge_size
INPUT_IMAGE_SIZE = [960, 960]                                       # Input image shape. Should be a multiple of GPU group (e.g., 16) for optimal efficiency.

VISION_BATCH_SIZE = 1                                               # Set the number of images for the vision LLM, whether DYNAMIC_IMAGE_SHAPE is True or False.
DYNAMIC_IMAGE_SHAPE = False                                         # Allow for a dynamic number of image inputs.
INPUT_IMAGE_DIM = 5                                                 # 4 for [batch, 3, height, width]; 5 for [batch, 1, 3, height, width]

USE_BEAM_SEARCH = False                                             # Use beam search or greedy search.
TOP_K = 3                                                           # The top k candidate in decoding.
BEAM_SIZE = 3                                                       # Number of beams in searching.
MAX_BEAM_SIZE = 10                                                  # Max beams for exported model.
REPEAT_PENALITY = 1.0                                               # Range from 0.0 to 1.0; "1.0" means no penality.
PENALITY_RANGE = 30                                                 # Penalizes the most recent output. "10" means the last 10 tokens.

ORT_Accelerate_Providers = []                                       # ORT execution providers; ['CUDAExecutionProvider'', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
MAX_THREADS = 0                                                     # 0 = auto
DEVICE_ID = 0                                                       # Device ID for GPU
OPSET = 17                                                          # ONNX opset version


def is_valid_image_path(image_path):
    if image_path is None:
        return False
    elif image_path == "":
        return False
    elif not os.path.exists(image_path):
        return False
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    _, ext = os.path.splitext(image_path)
    return ext.lower() in valid_extensions


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

    def forward(self, logits, repeat_penality, penality_value):
        max_logits_idx = torch.argmax(logits * repeat_penality, dim=-1, keepdim=True)
        repeat_penality.scatter_(1, max_logits_idx, repeat_penality.gather(1, max_logits_idx) * penality_value)
        return max_logits_idx.int(), repeat_penality


class FIRST_BEAM_SEARCH(torch.nn.Module):
    def __init__(self, total_layers):
        super(FIRST_BEAM_SEARCH, self).__init__()
        self.total_layers = total_layers
        self.save_keys_values = [None] * self.total_layers

    def forward(self, *all_inputs):
        logits = all_inputs[-5]
        save_id = all_inputs[-4]
        repeat_penality = all_inputs[-3]
        penality_value = all_inputs[-2]
        beam_size = all_inputs[-1]
        logits = torch.log_softmax(logits, dim=-1)
        top_beam_prob, top_beam_indices = torch.topk(logits, dim=-1, k=beam_size, sorted=True, largest=True)
        for i in range(self.total_layers):
            self.save_keys_values[i] = all_inputs[i].repeat(beam_size, *([1] * (all_inputs[i].dim() - 1)))
        top_beam_indices = top_beam_indices.transpose(0, 1)
        repeat_penality.scatter_(1, top_beam_indices, repeat_penality.gather(1, top_beam_indices) * penality_value)
        top_beam_indices = top_beam_indices.int()
        save_id = torch.cat([save_id, top_beam_indices], dim=-1)
        max_logits_idx = top_beam_indices[0]
        return *self.save_keys_values, top_beam_indices, save_id, repeat_penality, top_beam_prob.transpose(0, 1), max_logits_idx


class SECOND_BEAM_SEARCH(torch.nn.Module):
    def __init__(self, total_layers):
        super(SECOND_BEAM_SEARCH, self).__init__()
        self.total_layers = total_layers
        self.save_keys_values = [None] * self.total_layers

    def forward(self, *all_inputs):
        logits = all_inputs[-7]
        save_id = all_inputs[-6]
        repeat_penality = all_inputs[-5]
        previous_prob = all_inputs[-4]
        penality_value = all_inputs[-3]
        beam_size = all_inputs[-2]
        topK = all_inputs[-1]
        logits = torch.log_softmax(logits * repeat_penality, dim=-1)
        top_k_prob, top_k_indices = torch.topk(logits, k=topK, dim=-1, largest=True, sorted=True)
        current_prob = (top_k_prob + previous_prob).view(-1)
        top_beam_prob, top_beam_indices = torch.topk(current_prob, k=beam_size, dim=-1, largest=True, sorted=True)
        beam_index = top_beam_indices // topK
        top_beam_indices = top_k_indices.view(-1)[top_beam_indices]
        for i in range(self.total_layers):
            self.save_keys_values[i] = all_inputs[i][beam_index]
        repeat_penality = repeat_penality[beam_index]
        top_beam_indices = top_beam_indices.unsqueeze(-1)
        repeat_penality.scatter_(1, top_beam_indices, repeat_penality.gather(1, top_beam_indices) * penality_value)
        top_beam_indices = top_beam_indices.int()
        max_logits_idx = top_beam_indices[0]
        save_id = torch.cat([save_id[beam_index], top_beam_indices], dim=-1)
        return *self.save_keys_values, top_beam_indices, save_id, repeat_penality, top_beam_prob.unsqueeze(-1), max_logits_idx


class RESET_PENALITY(torch.nn.Module):
    def __init__(self):
        super(RESET_PENALITY, self).__init__()
        pass

    def forward(self, save_id, repeat_penality, penality_reset_count):
        token_indices = save_id.gather(1, penality_reset_count).long()
        repeat_penality.scatter_(1, token_indices, 1.0)
        penality_reset_count += 1
        return repeat_penality, penality_reset_count


class KVQuantizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qmax = 255.0
        self.register_buffer('inv_qmax', torch.tensor([1.0 / self.qmax], dtype=torch.float32).view(1, 1, 1, 1, -1))

    def _quantize_block(self, x, dim):
        block_min = x.min(dim=dim, keepdim=True).values  # bias
        block_max = x.max(dim=dim, keepdim=True).values
        scale = (block_max - block_min) * self.inv_qmax
        x_normalized = (x - block_min) / scale
        x_packed = torch.round(x_normalized).to(torch.uint8)
        if USE_FLOAT16_SCALE_BIAS:
            scale = scale.half()
            block_min = block_min.half()
        return x_packed, scale, block_min

    def forward(self, keys, values):
        k_packed, k_scale, k_bias = self._quantize_block(keys, 3)
        v_packed, v_scale, v_bias = self._quantize_block(values, 4)
        return k_packed, k_scale, k_bias, v_packed, v_scale.transpose(-1, -2), v_bias


class LLM_EMBED(torch.nn.Module):
    def __init__(self, llm):
        super(LLM_EMBED, self).__init__()
        self.llm = llm

    def forward(self, input_ids):
        return self.llm.model.language_model.embed_tokens(input_ids)


class LLM_VISION(torch.nn.Module):
    def __init__(self, llm):
        super(LLM_VISION, self).__init__()
        self.llm = llm
        self._replace_gelu_with_tanh_approximation(self.llm)
        visual_model = self.llm.model.visual
        vision_config = self.llm.config.vision_config
        self.num_heads = vision_config.num_heads
        self.qk_heads = self.num_heads // 3 * 2
        self.head_dim = vision_config.hidden_size // self.num_heads
        self.head_dim_half = self.head_dim // 2
        self.patch_size = visual_model.patch_size
        self.merge_size = visual_model.spatial_merge_size
        self.register_buffer("means", torch.tensor([127.5], dtype=torch.float32).view(1, 1, 1, 1, 1))
        self.register_buffer("inv_std", torch.tensor([1.0 / 127.5], dtype=torch.float32))
        grid_thw = torch.tensor([[1, HEIGHT_FACTOR * 2, WIDTH_FACTOR * 2]], dtype=torch.int32)
        self.image_hidden_len = int(grid_thw[0, 1] * grid_thw[0, 2])
        pos_embeds = Qwen3VLVisionModel.fast_pos_embed_interpolate(visual_model, grid_thw).unsqueeze(0)
        self.register_buffer("pos_embeds", pos_embeds)
        rotary_pos_emb = Qwen3VLVisionModel.rot_pos_emb(visual_model, grid_thw).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
        cos = rotary_pos_emb.cos()
        sin = rotary_pos_emb.sin()
        self.register_buffer("rotary_pos_emb_cos", torch.cat([cos, cos], dim=-1))
        self.register_buffer("rotary_pos_emb_sin", torch.cat([-sin, sin], dim=-1))
        scaling = self.head_dim ** -0.25
        embed_dim_patch = visual_model.patch_embed.embed_dim
        for blk in visual_model.blocks:
            blk.attn.qkv.weight.data[:-embed_dim_patch] *= scaling
            blk.attn.qkv.bias.data[:-embed_dim_patch] *= scaling
            self.fuse_norm(blk.norm1, blk.attn.qkv)
            self.fuse_norm(blk.norm2, blk.mlp.linear_fc1)
        for deepstack_layer in visual_model.deepstack_merger_list:
            self.fuse_norm(deepstack_layer.norm, deepstack_layer.linear_fc1)
        self.fuse_norm(visual_model.merger.norm, visual_model.merger.linear_fc1)
        patch_embed = self.llm.model.visual.patch_embed
        if patch_embed.temporal_patch_size > 1:
            fused_weight = patch_embed.proj.weight.data.sum(dim=2, keepdim=True)
            patch_embed.proj.weight.data = fused_weight * self.inv_std
            patch_embed.proj.kernel_size = (1, self.patch_size, self.patch_size)

    def fuse_norm(self, norm, linear):
        norm_bias = norm.bias.data
        norm_weight = norm.weight.data
        if linear.weight.shape[1] != norm_bias.shape[0]:
            repeat_factor = linear.weight.shape[1] // norm_bias.shape[0]
            norm_bias = norm_bias.repeat(repeat_factor)
            norm_weight = norm_weight.repeat(repeat_factor)
        linear.bias.data.add_(torch.matmul(linear.weight.data, norm_bias))
        linear.weight.data.mul_(norm_weight.unsqueeze(0))
        norm.elementwise_affine = False
        norm.weight = None
        norm.bias = None

    def rotate_half(self, x, batch_size):
        x = x.view(2, batch_size, self.num_heads, -1, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(2, batch_size, self.num_heads, -1, self.head_dim)

    def _replace_gelu_with_tanh_approximation(self, module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate='tanh'))
            else:
                self._replace_gelu_with_tanh_approximation(child)

    def forward(self, pixel_values):
        if INPUT_IMAGE_DIM != 5:
            pixel_values = pixel_values.unsqueeze(1)
        if DYNAMIC_IMAGE_SHAPE:
            batch_size = pixel_values.shape[0]
        else:
            batch_size = VISION_BATCH_SIZE
        pixel_values = pixel_values.float()
        if DYNAMIC_IMAGE_SHAPE or list(pixel_values.shape[-2:]) != IMAGE_RESIZE:
            pixel_values = pixel_values.squeeze(1)
            pixel_values = torch.nn.functional.interpolate(
                pixel_values,
                size=IMAGE_RESIZE,  # Ensure IMAGE_RESIZE is a tuple of ints
                mode='bilinear',
                align_corners=False
            )
            pixel_values = pixel_values.unsqueeze(1)
        pixel_values = (pixel_values - self.means)
        pixel_values = pixel_values.reshape(
            batch_size, 1, 1, 3,
            HEIGHT_FACTOR, self.merge_size, self.patch_size,
            WIDTH_FACTOR, self.merge_size, self.patch_size
        )
        pixel_values = pixel_values.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        pixel_values = pixel_values.reshape(-1, 3, 1, self.patch_size, self.patch_size)
        vision_hidden_states = self.llm.model.visual.patch_embed.proj(pixel_values)
        vision_hidden_states = vision_hidden_states.view(1, -1, self.llm.model.visual.patch_embed.embed_dim)
        if batch_size != 1:
             vision_hidden_states = vision_hidden_states + self.pos_embeds.expand(batch_size, -1, -1)
        else:
             vision_hidden_states = vision_hidden_states + self.pos_embeds
        deepstack_features = []
        deepstack_indices = self.llm.model.visual.deepstack_visual_indexes
        deepstack_modules = self.llm.model.visual.deepstack_merger_list
        for layer_num, blk in enumerate(self.llm.model.visual.blocks):
            hidden_states_norm = blk.norm1(vision_hidden_states)
            qkv = blk.attn.qkv(hidden_states_norm)
            qkv = qkv.view(batch_size, -1, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            qk, v = qkv.split([2, 1], dim=0)
            qk_rot = qk * self.rotary_pos_emb_cos + self.rotate_half(qk, batch_size) * self.rotary_pos_emb_sin
            q_rot, k_rot = qk_rot.split([1, 1], dim=0)
            attn_weights = torch.matmul(q_rot, k_rot.transpose(-1, -2))
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(2, 3).reshape(batch_size, -1, blk.attn.proj.in_features)
            vision_hidden_states = vision_hidden_states + blk.attn.proj(attn_output)
            mlp_out = blk.mlp.linear_fc1(blk.norm2(vision_hidden_states))
            mlp_out = blk.mlp.act_fn(mlp_out)
            mlp_out = blk.mlp.linear_fc2(mlp_out)
            vision_hidden_states = vision_hidden_states + mlp_out
            if layer_num in deepstack_indices:
                idx = deepstack_indices.index(layer_num)
                ds_layer = deepstack_modules[idx]
                x_ds = vision_hidden_states.view(batch_size, -1, ds_layer.hidden_size)
                x_ds = ds_layer.norm(x_ds)
                x_ds = ds_layer.linear_fc1(x_ds)
                x_ds = ds_layer.act_fn(x_ds)
                x_ds = ds_layer.linear_fc2(x_ds)
                deepstack_features.append(x_ds)
        vision_hidden_states = self.llm.model.visual.merger.norm(vision_hidden_states)
        vision_hidden_states = vision_hidden_states.view(batch_size, -1, self.llm.model.visual.merger.hidden_size)
        vision_hidden_states = self.llm.model.visual.merger.linear_fc1(vision_hidden_states)
        vision_hidden_states = self.llm.model.visual.merger.act_fn(vision_hidden_states)
        vision_hidden_states = self.llm.model.visual.merger.linear_fc2(vision_hidden_states)
        return tuple(deepstack_features), vision_hidden_states


class LLM_CONCAT(torch.nn.Module):
    def __init__(self, max_seq_len, prompt_head_len, hidden_size, deepstack_features_len):
        super(LLM_CONCAT, self).__init__()
        self.prompt_head_len = prompt_head_len
        self.zeros_A = torch.zeros([1, prompt_head_len, hidden_size], dtype=torch.float32)
        self.zeros_B = torch.zeros([1, max_seq_len, hidden_size], dtype=torch.int8)
        self.deepstack_features_len = deepstack_features_len
        self.hidden_size = hidden_size

    def forward(self, *all_inputs):
        text_hidden_states = all_inputs[-2]
        vision_hidden_states = all_inputs[-1]
        if text_hidden_states.shape[0] != 1:
            vision_hidden_states = vision_hidden_states.expand(text_hidden_states.shape[0], -1, self.hidden_size)
        concate_hidden_states = torch.cat([text_hidden_states[:, :self.prompt_head_len], vision_hidden_states, text_hidden_states[:, self.prompt_head_len:]], dim=1)
        zeros_B = self.zeros_B[:, :text_hidden_states.shape[1] - self.prompt_head_len].float()
        deepstack_features = [torch.cat([self.zeros_A, all_inputs[i], zeros_B], dim=1) for i in range(self.deepstack_features_len)]
        return deepstack_features, concate_hidden_states


class LLM_ROTARY_VISION(torch.nn.Module):
    def __init__(self, llm, width_factor, height_factor, prompt_head_len, max_seq_len):
        super(LLM_ROTARY_VISION, self).__init__()
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
        self.inv_freq_expanded = llm.model.language_model.rotary_emb.inv_freq[None, :, None].float().expand(3, -1, 1)
        freqs = (self.inv_freq_expanded @ position_ids)
        freqs = freqs.transpose(-1, -2).unsqueeze(1)
        freqs = Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope(llm.model.language_model.rotary_emb, freqs, llm.model.language_model.rotary_emb.mrope_section)
        cos = freqs.cos()
        sin = freqs.sin()
        self.rotary_pos_emb_cos = torch.cat([cos, cos], dim=-1).half().unsqueeze(2).unsqueeze(2)
        self.rotary_pos_emb_sin = torch.cat([-sin, sin], dim=-1).half().unsqueeze(2).unsqueeze(2)
        
    def forward(self, ids_len, history_len):
        kv_seq_len = ids_len + history_len
        rotary_pos_emb_cos = self.rotary_pos_emb_cos[:, history_len:kv_seq_len].float()
        rotary_pos_emb_sin = self.rotary_pos_emb_sin[:, history_len:kv_seq_len].float()
        return rotary_pos_emb_cos, rotary_pos_emb_sin, kv_seq_len
    

class LLM_ROTARY_TEXT(torch.nn.Module):
    def __init__(self, llm, max_seq_len):
        super(LLM_ROTARY_TEXT, self).__init__()
        position_ids = torch.arange(max_seq_len, dtype=torch.float32).repeat(3, 1, 1)
        self.inv_freq_expanded = llm.model.language_model.rotary_emb.inv_freq[None, :, None].float().expand(3, -1, 1)
        freqs = (self.inv_freq_expanded @ position_ids)
        freqs = freqs.transpose(-1, -2).unsqueeze(1)
        freqs = Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope(llm.model.language_model.rotary_emb, freqs, llm.model.language_model.rotary_emb.mrope_section)
        cos = freqs.cos()
        sin = freqs.sin()
        self.rotary_pos_emb_cos = torch.cat([cos, cos], dim=-1).half().unsqueeze(2).unsqueeze(2)
        self.rotary_pos_emb_sin = torch.cat([-sin, sin], dim=-1).half().unsqueeze(2).unsqueeze(2)
        
    def forward(self, ids_len, history_len):
        kv_seq_len = ids_len + history_len
        rotary_pos_emb_cos = self.rotary_pos_emb_cos[:, history_len:kv_seq_len].float()
        rotary_pos_emb_sin = self.rotary_pos_emb_sin[:, history_len:kv_seq_len].float()
        return rotary_pos_emb_cos, rotary_pos_emb_sin, kv_seq_len


class LLM_MAIN(torch.nn.Module):
    def __init__(self, llm, head_dim, num_heads, num_layers, num_key_value_heads, hidden_size, max_seq_len, deepstack_features_len, height_factor, width_factor):
        super(LLM_MAIN, self).__init__()
        self.llm = llm
        self._replace_gelu_with_tanh_approximation(self.llm)
        self.head_dim = head_dim
        self.head_dim_half = head_dim // 2
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_layers_2 = num_layers * 2
        self.num_layers_3 = num_layers * 3
        self.num_layers_4 = num_layers * 4
        self.num_layers_5 = num_layers * 5
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_heads = self.num_heads + self.num_key_value_heads
        max_seq_len += height_factor * width_factor
        self.attention_mask = (1 - torch.tril(torch.ones([1, 1, 1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128
        self.deepstack_features_len = deepstack_features_len
        self.kv_f16 = (KV_QUANT_DTYPE == "F16")
        self.kv_q8 = (KV_QUANT_DTYPE == "Q8")
        self.quantizer = KVQuantizer().eval()
        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)
        norm_factor = hidden_size ** 0.5
        norm_factor_qk = head_dim ** 0.5
        scale_factor = head_dim ** -0.25
        self.save_key = [None] * num_layers
        self.save_value = [None] * num_layers
        if self.kv_q8:
            self.save_k_scale = [None] * num_layers
            self.save_k_bias = [None] * num_layers
            self.save_v_scale = [None] * num_layers
            self.save_v_bias = [None] * num_layers

        with torch.no_grad():
            for layer in self.llm.model.language_model.layers:
                q_proj = layer.self_attn.q_proj
                k_proj = layer.self_attn.k_proj
                v_proj = layer.self_attn.v_proj
                layer.self_attn.q_out_features = int(q_proj.out_features)
                layer.self_attn.k_out_features = int(k_proj.out_features)
                layer.self_attn.v_out_features = int(v_proj.out_features)
                has_bias = (q_proj.bias is not None)
                qkv = torch.nn.Linear(q_proj.in_features, q_proj.out_features + k_proj.out_features + v_proj.out_features, bias=has_bias)
                qkv.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))
                if has_bias:
                    qkv.bias.copy_(torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0))
                w = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
                qkv.weight.mul_(w)
                layer.self_attn.qkv = qkv
                del layer.self_attn.q_proj
                del layer.self_attn.k_proj
                del layer.self_attn.v_proj
                del layer.input_layernorm

                layer.self_attn.q_norm.weight.mul_(scale_factor * norm_factor_qk)
                layer.self_attn.k_norm.weight.mul_(scale_factor * norm_factor_qk)
                q_norm_w = layer.self_attn.q_norm.weight.repeat(self.num_heads)
                k_norm_w = layer.self_attn.k_norm.weight.repeat(self.num_key_value_heads)
                layer.self_attn.qk_norm_weight = torch.nn.Parameter(torch.cat([q_norm_w, k_norm_w], dim=0).view(1, 1, 1, -1, self.head_dim))
                del layer.self_attn.q_norm
                del layer.self_attn.k_norm

                gate = layer.mlp.gate_proj
                up = layer.mlp.up_proj
                gate_up = torch.nn.Linear(gate.in_features, gate.out_features + up.out_features, bias=False)
                w = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
                gate_weight = gate.weight * w
                up_weight = up.weight * w
                gate_up.weight.copy_(torch.cat([gate_weight, up_weight], dim=0))
                layer.mlp.gate_up_proj = gate_up
                del layer.mlp.gate_proj
                del layer.mlp.up_proj
                del layer.post_attention_layernorm

            w = self.llm.model.language_model.norm.weight.unsqueeze(0) * norm_factor
            self.llm.lm_head.weight.mul_(w)
            del self.llm.model.language_model.norm

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
        hidden_states = all_inputs[-9]
        rotary_pos_emb_cos = all_inputs[-5]
        rotary_pos_emb_sin = all_inputs[-4]
        kv_seq_len = all_inputs[-3]
        ids_len = all_inputs[-2]
        mask = all_inputs[-1]
        attention_mask = (self.attention_mask[..., :ids_len, :kv_seq_len] * mask).float()
        batch_size = hidden_states.shape[0].unsqueeze(0)
        for i, layer in enumerate(self.llm.model.language_model.layers):
            residual = hidden_states
            if PREVENT_F16_OVERFLOW:
                hidden_states = hidden_states * self.overflow_scale
            hidden_states = hidden_states * torch.rsqrt(hidden_states.square().sum(-1, keepdim=True))
            qkv = layer.self_attn.qkv(hidden_states)
            qkv = qkv.view(batch_size, -1, 1, self.qk_heads + self.num_key_value_heads, self.head_dim)
            qk, v = torch.split(qkv, [self.qk_heads, self.num_key_value_heads], dim=-2)
            if PREVENT_F16_OVERFLOW:
                qk = qk * self.overflow_scale
            qk = qk * torch.rsqrt(qk.square().sum(dim=-1, keepdim=True)) * layer.self_attn.qk_norm_weight
            qk_rot = qk * rotary_pos_emb_cos + self.rotate_half(qk, batch_size) * rotary_pos_emb_sin
            q, k = torch.split(qk_rot, [self.num_heads, self.num_key_value_heads], dim=-2)
            q = q.view(batch_size, -1, self.num_key_value_heads, self.num_key_value_groups, self.head_dim)
            q = q.permute(0, 2, 3, 1, 4)
            if self.kv_f16:
                v = v.half()
                k = k.half()
            k = k.permute(0, 3, 2, 4, 1)
            v = v.transpose(1, 3)
            if self.kv_f16:
                k = torch.cat((all_inputs[i], k), dim=-1)
                v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)
                self.save_key[i] = k
                self.save_value[i] = v
                attn = torch.matmul(q, k.float())
                attn = torch.nn.functional.softmax(attn + attention_mask, dim=-1, dtype=torch.float32)
                attn = torch.matmul(attn, v.float())
            elif self.kv_q8:
                packed_k, scale_k, bias_k, packed_v, scale_v, bias_v = self.quantizer(k, v)
                self.save_key[i] = torch.cat([all_inputs[i], packed_k], dim=-1)
                self.save_k_scale[i] = torch.cat([all_inputs[i + self.num_layers_2], scale_k], dim=-1)
                self.save_k_bias[i] = torch.cat([all_inputs[i + self.num_layers_3], bias_k], dim=-1)
                self.save_value[i] = torch.cat([all_inputs[i + self.num_layers], packed_v], dim=-2)
                self.save_v_scale[i] = torch.cat([all_inputs[i + self.num_layers_4], scale_v], dim=-1)
                self.save_v_bias[i] = torch.cat([all_inputs[i + self.num_layers_5], bias_v], dim=-2)
                if USE_FLOAT16_SCALE_BIAS:
                    k_s = self.save_k_scale[i].float()
                    k_b = self.save_k_bias[i].float()
                    v_s = self.save_v_scale[i].float()
                    v_b = self.save_v_bias[i].float()
                else:
                    k_s = self.save_k_scale[i]
                    k_b = self.save_k_bias[i]
                    v_s = self.save_v_scale[i]
                    v_b = self.save_v_bias[i]
                attn_main = torch.matmul(q, self.save_key[i].float())
                q_sum = q.sum(dim=-1, keepdim=True)
                attn_bias = torch.matmul(q_sum, k_b)
                attn = attn_main * k_s + attn_bias
                attn = torch.nn.functional.softmax(attn + attention_mask, dim=-1, dtype=torch.float32)
                attn_scaled = attn * v_s
                out_main = torch.matmul(attn_scaled, self.save_value[i].float())
                out_bias = torch.matmul(attn, v_b)
                attn = out_main + out_bias
            else:
                k = torch.cat((all_inputs[i], k), dim=-1)
                v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)
                self.save_key[i] = k
                self.save_value[i] = v
                attn = torch.matmul(q, k)
                attn = torch.nn.functional.softmax(attn + attention_mask, dim=-1, dtype=torch.float32)
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
            hidden_states += residual
            if i < self.deepstack_features_len:
                hidden_states += all_inputs[i - 8]
        hidden_states = hidden_states[:, -1]
        if PREVENT_F16_OVERFLOW:
            hidden_states = hidden_states * self.overflow_scale
        hidden_states = hidden_states * torch.rsqrt(hidden_states.square().sum(-1, keepdim=True))
        logits = self.llm.lm_head(hidden_states)
        if self.kv_q8:
            return *self.save_key, *self.save_value, *self.save_k_scale, *self.save_k_bias, *self.save_v_scale, *self.save_v_bias, logits
        return *self.save_key, *self.save_value, logits


if DO_EXPORT:
    with torch.inference_mode():
        model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True).eval()
        num_heads = model.config.text_config.num_attention_heads
        num_key_value_heads = model.config.text_config.num_key_value_heads
        head_dim = model.config.text_config.head_dim
        num_layers = model.config.text_config.num_hidden_layers
        hidden_size = model.config.text_config.hidden_size
        vocab_size = model.model.language_model.vocab_size
        deepstack_features_len = len(model.model.visual.deepstack_visual_indexes)

        if USE_FLOAT16_SCALE_BIAS:
            scale_dtype = torch.float16
        else:
            scale_dtype = torch.float32

        pixel_values = torch.randint(low=0, high=255, size=[VISION_BATCH_SIZE, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]]).to(torch.uint8)
        if INPUT_IMAGE_DIM != 4:
            pixel_values = pixel_values.unsqueeze(1)
        vision_embed_size = WIDTH_FACTOR * HEIGHT_FACTOR * VISION_BATCH_SIZE
        vision_hidden_states = torch.ones((1, vision_embed_size, hidden_size), dtype=torch.float32)
        deepstack_features = torch.ones((1, vision_embed_size, hidden_size), dtype=torch.float32)
        prompt_head_len = 4                                       # <|im_start|>user\n<|vision_start|>
        ids_len = torch.tensor([10], dtype=torch.int64)      # "10" is just a dummy value.
        history_len = torch.tensor([0], dtype=torch.int64)
        input_ids = torch.ones((1, ids_len), dtype=torch.int32)
        text_hidden_states = torch.ones((1, ids_len, hidden_size), dtype=torch.float32)
        batch_size = 3
        ids_len = ids_len + vision_embed_size
        hidden_states = torch.ones((batch_size, ids_len, hidden_size), dtype=torch.float32)
        rotary_pos_emb_cos = torch.ones((1, ids_len, 1, 1, head_dim), dtype=torch.float32)
        rotary_pos_emb_sin = rotary_pos_emb_cos
        attention_mask = torch.tensor([1], dtype=torch.int8)

        kv_tensors = {}
        kv_specs = [('key', 4), ('value', 3)]
        if KV_QUANT_DTYPE == "F16":
            kv_dtype = torch.float16
        elif KV_QUANT_DTYPE == "Q8":
            kv_specs.extend([('key_scale', 4), ('key_bias', 4), ('value_scale', 4), ('value_bias', 3)])
            kv_dtype = torch.uint8
        else:
            kv_dtype = torch.float32

        kv_tensors['key'] = torch.zeros((batch_size, num_key_value_heads, 1, head_dim, history_len), dtype=kv_dtype)
        kv_tensors['value'] = torch.zeros((batch_size, num_key_value_heads, 1, history_len, head_dim), dtype=kv_dtype)

        if KV_QUANT_DTYPE == "Q8":
            kv_tensors['key_scale'] = torch.ones([batch_size, num_key_value_heads, 1, 1, history_len], dtype=scale_dtype)
            kv_tensors['key_bias'] = torch.ones([batch_size, num_key_value_heads, 1, 1, history_len], dtype=scale_dtype)
            kv_tensors['value_scale'] = torch.ones([batch_size, num_key_value_heads, 1, 1, history_len], dtype=scale_dtype)
            kv_tensors['value_bias'] = torch.ones([batch_size, num_key_value_heads, 1, history_len, 1], dtype=scale_dtype)

        kv_seq_len = history_len.long() + ids_len

        torch.onnx.export(
            LLM_EMBED(model),
            (input_ids,),
            onnx_model_A,
            input_names=['input_ids'],
            output_names=['text_hidden_states'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'ids_len'},
                'text_hidden_states': {0: 'batch', 1: 'ids_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        gc.collect()

        dynamic_axes = {
            'pixel_values': {0: 'batch_size', -2: 'width', -1: 'height'},
            'vision_hidden_states': {1: 'vision_embed_len'}
        }
        output_names = []
        for i in range(deepstack_features_len):
            name = f'deepstack_feature_{i}'
            output_names.append(name)
            dynamic_axes[name] = {1: 'vision_embed_len'}
        output_names.append('vision_hidden_states')

        torch.onnx.export(
            LLM_VISION(model),
            (pixel_values,),
            onnx_model_B,
            input_names=['pixel_values'],
            output_names=output_names,
            dynamic_axes=dynamic_axes if DYNAMIC_IMAGE_SHAPE else None,
            opset_version=OPSET,
            dynamo=False
        )
        del pixel_values
        gc.collect()

        all_inputs = []
        input_names = []
        output_names = []
        dynamic_axes = {
            'text_hidden_states': {0: 'batch_size', 1: 'ids_len'},
            'concate_hidden_states': {0: 'batch_size', 1: 'total_len'}
        }
        if DYNAMIC_IMAGE_SHAPE:
            dynamic_axes['vision_hidden_states'] = {1: 'vision_embed_len'}
        for i in range(deepstack_features_len):
            name = f'in_deepstack_feature_{i}'
            input_names.append(name)
            all_inputs.append(deepstack_features)
            if DYNAMIC_IMAGE_SHAPE:
                dynamic_axes[name] = {1: 'vision_embed_len'}
            name = f'out_deepstack_feature_{i}'
            dynamic_axes[name] = {1: 'total_len'}
            output_names.append(name)
        input_names.append("text_hidden_states")
        input_names.append("vision_hidden_states")
        all_inputs.append(text_hidden_states)
        all_inputs.append(vision_hidden_states)
        output_names.append("concate_hidden_states")

        torch.onnx.export(
            LLM_CONCAT(MAX_SEQ_LEN, prompt_head_len, hidden_size, deepstack_features_len),
            tuple(all_inputs),
            onnx_model_C,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            dynamo=False
        )
        del text_hidden_states
        del vision_hidden_states

        torch.onnx.export(
            LLM_ROTARY_VISION(model, WIDTH_FACTOR, HEIGHT_FACTOR, prompt_head_len, MAX_SEQ_LEN),
            (ids_len, history_len),
            onnx_model_D,
            input_names=['ids_len', 'history_len'],
            output_names=['rotary_pos_emb_cos', 'rotary_pos_emb_sin', 'kv_seq_len'],
            dynamic_axes={
                'rotary_pos_emb_cos': {1: 'ids_len'},
                'rotary_pos_emb_sin': {1: 'ids_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )

        torch.onnx.export(
            LLM_ROTARY_TEXT(model, MAX_SEQ_LEN),
            (ids_len, history_len),
            onnx_model_E,
            input_names=['ids_len', 'history_len'],
            output_names=['rotary_pos_emb_cos', 'rotary_pos_emb_sin', 'kv_seq_len'],
            dynamic_axes={
                'rotary_pos_emb_cos': {1: 'ids_len'},
                'rotary_pos_emb_sin': {1: 'ids_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )


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


        model_F = LLM_MAIN(model, head_dim, num_heads, num_layers, num_key_value_heads, hidden_size, MAX_SEQ_LEN, deepstack_features_len, HEIGHT_FACTOR, WIDTH_FACTOR)
        del model
        gc.collect()
        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors, 'batch', 'history_len', 'kv_seq_len')
        all_inputs = kv_ins
        input_names = kv_in_names
        output_names = kv_out_names
        dynamic_axes = {
            **kv_axes,
            'hidden_states': {0: 'batch', 1: 'ids_len'},
            'rotary_pos_emb_cos': {1: 'ids_len'},
            'rotary_pos_emb_sin': {1: 'ids_len'},
            'logits': {0: 'batch'}
        }
        input_names.append('hidden_states')
        all_inputs.append(hidden_states)
        deepstack_features = torch.ones((1, ids_len, hidden_size), dtype=torch.float32)
        for i in range(deepstack_features_len):
            name = f'deepstack_features_{i}'
            input_names.append(name)
            dynamic_axes[name] = {1: 'total_len'}
            all_inputs.append(deepstack_features)
        input_names.append('rotary_pos_emb_cos')
        all_inputs.append(rotary_pos_emb_cos)
        input_names.append('rotary_pos_emb_sin')
        all_inputs.append(rotary_pos_emb_sin)
        input_names.append('kv_seq_len')
        all_inputs.append(kv_seq_len)
        input_names.append('ids_len')
        all_inputs.append(ids_len)
        input_names.append('attention_mask')
        all_inputs.append(attention_mask)
        output_names.append('logits')

        torch.onnx.export(
            model_F,
            tuple(all_inputs),
            onnx_model_F,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
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
        del rotary_pos_emb_cos
        del rotary_pos_emb_sin
        del attention_mask
        del kv_seq_len
        gc.collect()

        beam_size = torch.tensor([BEAM_SIZE], dtype=torch.int64)
        repeat_penality = torch.ones((beam_size, vocab_size), dtype=torch.float32)
        penality_reset_count = torch.zeros([beam_size, 1], dtype=torch.int32)
        logits = torch.ones((beam_size, vocab_size), dtype=torch.float32)
        penality_value = torch.tensor([REPEAT_PENALITY], dtype=torch.float32)
        torch.onnx.export(
            GREEDY_SEARCH(),
            (logits, repeat_penality, penality_value),
            onnx_model_G,
            input_names=['logits', 'repeat_penality_in', 'penality_value'],
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

        num_layers_beam = num_layers * len(kv_specs)
        kv_tensors_greedy = {k: v[[0]] for k, v in kv_tensors.items()}
        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors_greedy, 'batch_size', 'history_len', 'history_len')
        kv_axes = {k: v for k, v in kv_axes.items() if k not in kv_out_names} # Inputs only
        topK = torch.tensor([TOP_K], dtype=torch.int64)
        save_id = torch.zeros((beam_size, 10), dtype=torch.int32)
        previous_prob = torch.zeros((beam_size, 1), dtype=torch.float32)
        other_inputs = [logits[[0]], save_id, repeat_penality, penality_value, beam_size]
        other_input_names = ['logits', 'save_id_in', 'repeat_penality_in', 'penality_value', 'beam_size']
        other_output_names = ['top_beam_indices', 'save_id_out', 'repeat_penality_out', 'top_beam_prob', 'max_logits_idx']
        dynamic_axes = {
            **kv_axes,
            'save_id_in': {0: 'batch', 1: 'history_len'},
            'save_id_out': {0: 'batch', 1: 'history_len'},
            'repeat_penality_in': {0: 'batch'},
            'repeat_penality_out': {0: 'batch'},
            'logits': {0: 'batch'},
            'top_beam_prob': {0: 'batch'},
            'top_beam_indices': {0: 'batch'},
            'max_logits_idx': {0: 'batch'}
        }

        torch.onnx.export(
            FIRST_BEAM_SEARCH(num_layers_beam),
            tuple(kv_ins + other_inputs),
            onnx_model_H,
            input_names=kv_in_names + other_input_names,
            output_names=kv_out_names + other_output_names,
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            dynamo=False
        )
        del kv_tensors_greedy
        del kv_ins
        del kv_in_names
        del kv_out_names
        del kv_axes
        del other_inputs
        del other_input_names
        del other_output_names
        del dynamic_axes

        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors, 'batch', 'history_len', 'kv_seq_len')
        other_inputs = [logits, save_id, repeat_penality, previous_prob, penality_value, beam_size, topK]
        other_input_names = ['logits', 'save_id_in', 'repeat_penality_in', 'previous_prob', 'penality_value', 'beam_size', 'topK']
        other_output_names = ['top_beam_indices', 'save_id_out', 'repeat_penality_out', 'top_beam_prob', 'max_logits_idx']
        dynamic_axes = {
            **kv_axes,
            'logits': {0: 'batch'},
            'save_id_in': {0: 'batch', 1: 'history_len'},
            'repeat_penality_in': {0: 'batch'},
            'previous_prob': {0: 'batch'},
            'save_id_out': {0: 'batch', 1: 'history_len'},
            'repeat_penality_out': {0: 'batch'},
            'top_beam_prob': {0: 'batch'},
            'top_beam_indices': {0: 'batch'},
            'max_logits_idx': {0: 'batch'}
        }

        torch.onnx.export(
            SECOND_BEAM_SEARCH(num_layers_beam),
            tuple(kv_ins + other_inputs),
            onnx_model_I,
            input_names=kv_in_names + other_input_names,
            output_names=kv_out_names + other_output_names,
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            dynamo=False
        )
        del num_layers
        del kv_tensors
        del kv_ins
        del kv_in_names
        del kv_out_names
        del kv_axes
        del other_inputs
        del other_input_names
        del other_output_names
        del dynamic_axes
        del logits
        del beam_size
        del penality_value
        del previous_prob
        del topK

        torch.onnx.export(
            RESET_PENALITY(),
            (save_id, repeat_penality, penality_reset_count),
            onnx_model_J,
            input_names=['save_id', 'repeat_penality_in', 'penality_reset_count_in'],
            output_names=['repeat_penality_out', 'penality_reset_count_out'],
            dynamic_axes={
                'save_id': {0: 'batch', 1: 'history_len'},
                'repeat_penality_in': {0: 'batch'},
                'repeat_penality_out': {0: 'batch'},
                'penality_reset_count_in': {0: 'batch'},
                'penality_reset_count_out': {0: 'batch'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del save_id
        del repeat_penality
        del penality_reset_count

        logits_t = torch.ones((BEAM_SIZE, vocab_size), dtype=torch.float32)
        torch.onnx.export(
            ARGMAX(),
            (logits_t,),
            onnx_model_K,
            input_names=['logits'], output_names=['max_logits_idx'],
            dynamic_axes={
                'logits': {0: 'batch'},
                'max_logits_idx': {0: 'batch'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del logits_t
        del vocab_size
        print('\nExport Done!\n\nStrat to run QwenVL by ONNX Runtime.')


# Inference with ONNXRuntime
# =======================================================#
def bind_ort_in(binding, names, values, num=0):
    if num != 0:
        for i in range(num):
            binding.bind_ortvalue_input(names[i], values[i])
    else:
        for name, val in zip(names, values):
            binding.bind_ortvalue_input(name, val)


def bind_ort_out(binding, output_names, device_type):
    for name in output_names:
        binding._iobinding.bind_output(name, device_type)


def create_ortvalue(data, dtype, device_type, device_id):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.array(data, dtype=dtype), device_type, device_id)


tokenizer = AutoTokenizer.from_pretrained(path)

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
session_opts.add_session_config_entry('session.use_device_allocator_for_initializers', '1')
session_opts.add_session_config_entry('session.graph_optimizations_loop_level', '2')
session_opts.add_session_config_entry('optimization.enable_gelu_approximation', '1')
session_opts.add_session_config_entry('optimization.minimal_build_optimizations', '')
session_opts.add_session_config_entry('optimization.enable_cast_chain_elimination', '1')
session_opts.add_session_config_entry('optimization.disable_specified_optimizers', '')
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
    _ort_device_type = C.OrtDevice.cpu()
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
    _ort_device_type = C.OrtDevice.cuda()
elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [
        {
            'device_id': DEVICE_ID,
            'performance_preference': 'high_performance',  # [high_performance, default, minimum_power]
            'device_filter': 'npu'                         # [any, npu, gpu]
        }
    ]
    device_type = 'dml'
    _ort_device_type = C.OrtDevice.dml()
else:
    # Please config by yourself for others providers.
    device_type = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()
    provider_options = None


_ort_device_type = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
binding_A = ort_session_A.io_binding()
in_name_A = ort_session_A.get_inputs()[0].name
out_name_A = [ort_session_A.get_outputs()[0].name]

ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
binding_B = ort_session_B.io_binding()
in_name_B_objs = ort_session_B.get_inputs()
out_name_B_objs = ort_session_B.get_outputs()
in_name_B = [x.name for x in in_name_B_objs]
out_name_B = [x.name for x in out_name_B_objs]
deepstack_features_len = len(out_name_B) - 1
in_name_B_parts = in_name_B[:deepstack_features_len+1]
vision_dtype = np.float16 if 'float16' in ort_session_B._outputs_meta[0].type else np.float32

ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
binding_C = ort_session_C.io_binding()
in_name_C_objs = ort_session_C.get_inputs()
out_name_C_objs = ort_session_C.get_outputs()
amount_of_outputs_C = len(out_name_C_objs)
in_name_C = [x.name for x in in_name_C_objs]
out_name_C = [x.name for x in out_name_C_objs]

ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
binding_D = ort_session_D.io_binding()
in_name_D = [x.name for x in ort_session_D.get_inputs()]
out_name_D = [x.name for x in ort_session_D.get_outputs()]
rotary_outputs_len = len(out_name_D)
rotary_outputs_len_minus = rotary_outputs_len - 1

ort_session_E = onnxruntime.InferenceSession(onnx_model_E, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
binding_E = ort_session_E.io_binding()
in_name_E = [x.name for x in ort_session_E.get_inputs()]
out_name_E = [x.name for x in ort_session_E.get_outputs()]

ort_session_F = onnxruntime.InferenceSession(onnx_model_F, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
binding_F = ort_session_F.io_binding()
print(f"\nUsable Providers: {ort_session_F.get_providers()}")

model_dtype_F_str = ort_session_F._inputs_meta[0].type
in_name_F_objs = ort_session_F.get_inputs()
out_name_F_objs = ort_session_F.get_outputs()
amount_of_outputs_F = len(out_name_F_objs)
in_name_F = [x.name for x in in_name_F_objs]
out_name_F = [x.name for x in out_name_F_objs]

if 'dml' in device_type:
    kv_device = 'cpu'
else:
    kv_device = device_type

if 'uint8' in model_dtype_F_str:
    kv_dtype = np.uint8
    num_keys_values_temp = (amount_of_outputs_F - 1) // 3
    num_layers = num_keys_values_temp // 2
    num_keys_values = amount_of_outputs_F - 1  # Including scales and biases, but excluding logits.
    kv_scale_dtype = np.float16 if 'float16' in ort_session_F._inputs_meta[num_keys_values_temp].type else np.float32
    k_scales = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_F._inputs_meta[0].shape[1], 1, 1, 0), dtype=kv_scale_dtype), kv_device, DEVICE_ID)
    k_biases = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_F._inputs_meta[0].shape[1], 1, 1, 0), dtype=kv_scale_dtype), kv_device, DEVICE_ID)
    v_scales = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_F._inputs_meta[num_layers].shape[1], 1, 1, 0), dtype=kv_scale_dtype), kv_device, DEVICE_ID)
    v_biases = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_F._inputs_meta[num_layers].shape[1], 1, 0, 1), dtype=kv_scale_dtype), kv_device, DEVICE_ID)
else:
    kv_dtype = np.float16 if 'float16' in model_dtype_F_str else np.float32
    num_keys_values = amount_of_outputs_F - 1
    num_layers = num_keys_values // 2
    k_scales = None

vision_embed_size = VISION_BATCH_SIZE * WIDTH_FACTOR * HEIGHT_FACTOR
num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
num_keys_values_plus_3 = num_keys_values + 3
num_keys_values_plus_4 = num_keys_values + 4
num_keys_values_plus_5 = num_keys_values + 5
num_keys_values_plus_6 = num_keys_values + 6
num_keys_values_plus_deepstack_plus_rotary = num_keys_values + deepstack_features_len + rotary_outputs_len
num_keys_values_plus_deepstack_plus_rotary_plus_1 = num_keys_values_plus_deepstack_plus_rotary + 1
num_keys_values_plus_deepstack_plus_rotary_plus_2 = num_keys_values_plus_deepstack_plus_rotary + 2
rotary_indices = np.arange(rotary_outputs_len, dtype=np.int32) + num_keys_values + deepstack_features_len + 1
deepstack_indices = np.arange(deepstack_features_len, dtype=np.int32) + num_keys_values + 1
rotary_in_name_F = in_name_F[rotary_indices[0]:rotary_indices[rotary_outputs_len-1]+1]
deepstack_in_name_F = in_name_F[deepstack_indices[0]:deepstack_indices[deepstack_features_len-1]+1]
in_name_F_parts = in_name_F[:num_keys_values]
vocab_size = ort_session_F._outputs_meta[num_keys_values].shape[1]
topK = create_ortvalue([TOP_K], np.int64, device_type, DEVICE_ID)
beam_size = create_ortvalue([BEAM_SIZE], np.int64, device_type, DEVICE_ID)

prompt = f"<|im_start|>user\n<|vision_start|><|vision_end|>{query}<|im_end|>\n<|im_start|>assistant\n"
prompt_head_len = np.array([4], dtype=np.int64)
tokens = tokenizer(prompt, return_tensors='np')['input_ids'].astype(np.int32)
input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, DEVICE_ID)
ids_len_val = tokens.shape[-1]
ids_len = create_ortvalue([ids_len_val], np.int64, device_type, DEVICE_ID)
init_ids_len_1 = create_ortvalue([1], np.int64, device_type, DEVICE_ID)
init_history_len = create_ortvalue([0], np.int64, device_type, DEVICE_ID)
init_attention_mask_0 = create_ortvalue([0], np.int8, device_type, DEVICE_ID)
init_attention_mask_1 = create_ortvalue([1], np.int8, device_type, DEVICE_ID)
init_deepstack_features = [onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, 1, ort_session_C._inputs_meta[0].shape[2]), dtype=vision_dtype), device_type, DEVICE_ID)] * deepstack_features_len
init_past_keys_F = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_F._inputs_meta[0].shape[1], 1, ort_session_F._inputs_meta[0].shape[3], 0), dtype=kv_dtype), kv_device, DEVICE_ID)
init_past_values_F = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_F._inputs_meta[num_layers].shape[1], 1, 0, ort_session_F._inputs_meta[num_layers].shape[4]), dtype=kv_dtype), kv_device, DEVICE_ID)
generate_limit = MAX_SEQ_LEN - tokens.shape[-1]

USE_PENALTY = (REPEAT_PENALITY != 1.0)

if USE_BEAM_SEARCH and (TOP_K < BEAM_SIZE):
    TOP_K = BEAM_SIZE

if (TOP_K < 2) or (BEAM_SIZE < 2):
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting. Falling back to Greedy Search.")

if USE_BEAM_SEARCH:
    print("\nBeam Search does not display immediate decoding results...")

    ort_session_H = onnxruntime.InferenceSession(onnx_model_H, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    binding_H = ort_session_H.io_binding()
    in_name_H = [x.name for x in ort_session_H.get_inputs()]
    out_name_H = [x.name for x in ort_session_H.get_outputs()]
    in_name_H_parts = in_name_H[:num_keys_values_plus_1]
    
    ort_session_I = onnxruntime.InferenceSession(onnx_model_I, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    binding_I = ort_session_I.io_binding()
    in_name_I = [x.name for x in ort_session_I.get_inputs()]
    out_name_I = [x.name for x in ort_session_I.get_outputs()]
    in_name_I_parts = in_name_I[:num_keys_values_plus_1]
    
    ort_session_J = onnxruntime.InferenceSession(onnx_model_J, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    binding_J = ort_session_J.io_binding()
    in_name_J = [x.name for x in ort_session_J.get_inputs()]
    out_name_J = [x.name for x in ort_session_J.get_outputs()]
    
    penality_dtype = np.float16 if 'float16' in ort_session_H._inputs_meta[num_keys_values_plus_3].type else np.float32
    penality_value = create_ortvalue([REPEAT_PENALITY], penality_dtype, device_type, DEVICE_ID)
    init_repeat_penality = onnxruntime.OrtValue.ortvalue_from_numpy(np.ones((BEAM_SIZE, vocab_size), dtype=penality_dtype), device_type, DEVICE_ID)
    init_penality_reset_count = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(BEAM_SIZE, dtype=np.int32), device_type, DEVICE_ID)
    init_save_id_beam = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((BEAM_SIZE, 0), dtype=np.int32), device_type, DEVICE_ID)
    
    binding_J.bind_ortvalue_input(in_name_J[2], init_penality_reset_count)
    binding_H.bind_ortvalue_input(in_name_H[num_keys_values_plus_1], init_save_id_beam)
    binding_H.bind_ortvalue_input(in_name_H[num_keys_values_plus_2], init_repeat_penality)
    binding_H.bind_ortvalue_input(in_name_H[num_keys_values_plus_3], penality_value)
    binding_H.bind_ortvalue_input(in_name_H[num_keys_values_plus_4], beam_size)
    binding_I.bind_ortvalue_input(in_name_I[num_keys_values_plus_4], penality_value)
    binding_I.bind_ortvalue_input(in_name_I[num_keys_values_plus_5], beam_size)
    binding_I.bind_ortvalue_input(in_name_I[num_keys_values_plus_6], topK)
else:
    BEAM_SIZE = 1
    save_id_greedy = np.zeros(MAX_SEQ_LEN, dtype=np.int32)
    if USE_PENALTY:
        ort_session_G = onnxruntime.InferenceSession(onnx_model_G, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
        binding_G = ort_session_G.io_binding()
        in_name_G = [x.name for x in ort_session_G.get_inputs()]
        out_name_G = [x.name for x in ort_session_G.get_outputs()]
        penality_dtype = np.float16 if 'float16' in ort_session_G._inputs_meta[2].type else np.float32
        penality_value = create_ortvalue([REPEAT_PENALITY], penality_dtype, device_type, DEVICE_ID)
        current_penalty = onnxruntime.OrtValue.ortvalue_from_numpy(np.ones((BEAM_SIZE, vocab_size), dtype=penality_dtype), device_type, DEVICE_ID)
        next_penalty = onnxruntime.OrtValue.ortvalue_from_numpy(np.ones((BEAM_SIZE, vocab_size), dtype=penality_dtype), device_type, DEVICE_ID)
        binding_G.bind_ortvalue_input(in_name_G[2], penality_value)
        penalty_shape = (BEAM_SIZE, vocab_size)
        binding_G.bind_output(out_name_G[0], device_type=device_type, device_id=DEVICE_ID)
        binding_G.bind_output(name=out_name_G[1], device_type=device_type, device_id=DEVICE_ID, element_type=penality_dtype, shape=penalty_shape, buffer_ptr=next_penalty.data_ptr())
        init_penality_reset_count = 0
    else:
        ort_session_K = onnxruntime.InferenceSession(onnx_model_K, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
        binding_K = ort_session_K.io_binding()
        in_name_K = ort_session_K.get_inputs()[0].name
        out_name_K = [x.name for x in ort_session_K.get_outputs()]

if is_valid_image_path(image_path):
    image = Image.open(image_path)
    image = image.resize((INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0]))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    pixel_values = np.transpose(np.array(image).astype(np.uint8), (2, 0, 1))
    if len(ort_session_B._inputs_meta[0].shape) != 4:
        axis = (0, 1)
    else:
        axis = 0
    pixel_values = np.expand_dims(pixel_values, axis=axis)
    use_vision = True
    print('\nChat with image.')
else:
    use_vision = False
    print('\nChat without image.')

binding_A.bind_ortvalue_input(in_name_A, input_ids)
bind_ort_out(binding_A, out_name_A, _ort_device_type)
ort_session_A.run_with_iobinding(binding_A, run_options=run_options)
out_A = binding_A.get_outputs()[0]

if use_vision:
    ids_len_val += vision_embed_size
    ids_len = create_ortvalue([ids_len_val], np.int64, device_type, DEVICE_ID)
    generate_limit -= vision_embed_size
    
    print('\nStart to Process the Image...')
    start_time = time.time()
    binding_B.bind_ortvalue_input(in_name_B[0], onnxruntime.OrtValue.ortvalue_from_numpy(pixel_values, device_type, DEVICE_ID))
    bind_ort_out(binding_B, out_name_B, _ort_device_type)
    ort_session_B.run_with_iobinding(binding_B, run_options=run_options)
    all_outputs_B = binding_B.get_outputs()
    print(f'\nImage Process Complete. Time Cost: {time.time() - start_time:.3f} Seconds')
    
    binding_C.bind_ortvalue_input(in_name_C[deepstack_features_len], out_A)
    binding_C.bind_ortvalue_input(in_name_C[amount_of_outputs_C], all_outputs_B[deepstack_features_len])
    bind_ort_in(binding_C, in_name_C, all_outputs_B, deepstack_features_len)
    bind_ort_out(binding_C, out_name_C, _ort_device_type)
    ort_session_C.run_with_iobinding(binding_C, run_options=run_options)
    all_outputs_C = binding_C.get_outputs()
    
    binding_D.bind_ortvalue_input(in_name_D[0], ids_len)
    binding_D.bind_ortvalue_input(in_name_D[1], init_history_len)
    bind_ort_out(binding_D, out_name_D, _ort_device_type)
    ort_session_D.run_with_iobinding(binding_D, run_options=run_options)
    rotary_outputs = binding_D.get_outputs()
    
    binding_F.bind_ortvalue_input(in_name_F[num_keys_values], all_outputs_C[deepstack_features_len])
    bind_ort_in(binding_F, deepstack_in_name_F, all_outputs_C)
else:
    binding_E.bind_ortvalue_input(in_name_E[0], ids_len)
    binding_E.bind_ortvalue_input(in_name_E[1], init_history_len)
    bind_ort_out(binding_E, out_name_E, _ort_device_type)
    ort_session_E.run_with_iobinding(binding_E, run_options=run_options)
    rotary_outputs = binding_E.get_outputs()
    
    binding_F.bind_ortvalue_input(in_name_F[num_keys_values], out_A)
    bind_ort_in(binding_F, deepstack_in_name_F, init_deepstack_features)

binding_F.bind_ortvalue_input(in_name_F[num_keys_values_plus_deepstack_plus_rotary_plus_1], ids_len)
binding_F.bind_ortvalue_input(in_name_F[num_keys_values_plus_deepstack_plus_rotary_plus_2], init_attention_mask_1)

i = 0
j = num_layers
while i < j:
    binding_F.bind_ortvalue_input(in_name_F[i], init_past_keys_F)
    i += 1
j = i + num_layers
while i < j:
    binding_F.bind_ortvalue_input(in_name_F[i], init_past_values_F)
    i += 1

if k_scales:
    j = i + num_layers
    while i < j:
        binding_F.bind_ortvalue_input(in_name_F[i], k_scales)
        i += 1
    j = i + num_layers
    while i < j:
        binding_F.bind_ortvalue_input(in_name_F[i], k_biases)
        i += 1
    j = i + num_layers
    while i < j:
        binding_F.bind_ortvalue_input(in_name_F[i], v_scales)
        i += 1
    j = i + num_layers
    while i < j:
        binding_F.bind_ortvalue_input(in_name_F[i], v_biases)
        i += 1

for i in range(rotary_outputs_len):
    binding_F.bind_ortvalue_input(in_name_F[rotary_indices[i]], rotary_outputs[i])

print(f'\nTest Question: {query}\nLLM Answering:\n')
num_decode = 0
start_time = time.time()
while num_decode < generate_limit:
    bind_ort_out(binding_F, out_name_F, _ort_device_type)
    ort_session_F.run_with_iobinding(binding_F, run_options=run_options)
    all_outputs_F = binding_F.get_outputs()
    if USE_BEAM_SEARCH:
        if num_decode < 1:
            bind_ort_in(binding_H, in_name_H_parts, all_outputs_F)
            bind_ort_out(binding_H, out_name_H, _ort_device_type)
            ort_session_H.run_with_iobinding(binding_H, run_options=run_options)
            all_outputs_H = binding_H.get_outputs()
            max_logits_idx = all_outputs_H[num_keys_values_plus_4].numpy()
            if max_logits_idx in STOP_TOKEN:
                print("\n\nBad Generation. Please Retry.")
                break
        else:
            bind_ort_in(binding_I, in_name_I_parts, all_outputs_F)
            bind_ort_out(binding_I, out_name_I, _ort_device_type)
            ort_session_I.run_with_iobinding(binding_I, run_options=run_options)
            all_outputs_I = binding_I.get_outputs()
            max_logits_idx = all_outputs_I[num_keys_values_plus_4].numpy()
            if max_logits_idx in STOP_TOKEN:
                break
        if USE_PENALTY and (num_decode >= PENALITY_RANGE):
            binding_J.bind_ortvalue_input(in_name_J[0], all_outputs_I[num_keys_values_plus_1])
            binding_J.bind_ortvalue_input(in_name_J[1], all_outputs_I[num_keys_values_plus_2])
            bind_ort_out(binding_J, out_name_J, _ort_device_type)
            ort_session_J.run_with_iobinding(binding_J, run_options=run_options)
            all_outputs_J = binding_J.get_outputs()
            binding_I.bind_ortvalue_input(in_name_I[num_keys_values_plus_2], all_outputs_J[0])
            binding_J.bind_ortvalue_input(in_name_J[2], all_outputs_J[1])
        if num_decode < 1:
            bind_ort_in(binding_F, in_name_F_parts, all_outputs_H)
            binding_A.bind_ortvalue_input(in_name_A, all_outputs_H[num_keys_values])
            binding_I.bind_ortvalue_input(in_name_I[num_keys_values_plus_1], all_outputs_H[num_keys_values_plus_1])
            binding_I.bind_ortvalue_input(in_name_I[num_keys_values_plus_2], all_outputs_H[num_keys_values_plus_2])
            binding_I.bind_ortvalue_input(in_name_I[num_keys_values_plus_3], all_outputs_H[num_keys_values_plus_3])
        else:
            bind_ort_in(binding_F, in_name_F_parts, all_outputs_I)
            binding_A.bind_ortvalue_input(in_name_A, all_outputs_I[num_keys_values])
            binding_I.bind_ortvalue_input(in_name_I[num_keys_values_plus_1], all_outputs_I[num_keys_values_plus_1])
            binding_I.bind_ortvalue_input(in_name_I[num_keys_values_plus_2], all_outputs_I[num_keys_values_plus_2])
            binding_I.bind_ortvalue_input(in_name_I[num_keys_values_plus_3], all_outputs_I[num_keys_values_plus_3])
    else:
        if USE_PENALTY:
            binding_G.bind_ortvalue_input(in_name_G[0], all_outputs_F[num_keys_values])
            binding_G.bind_ortvalue_input(in_name_G[1], current_penalty)
            ort_session_G.run_with_iobinding(binding_G, run_options=run_options)
            all_outputs_G = binding_G.get_outputs()
            max_logits_idx = all_outputs_G[0].numpy().flat[0]
            if max_logits_idx in STOP_TOKEN:
                break
            if num_decode >= PENALITY_RANGE:
                reset_ids = save_id_greedy[init_penality_reset_count]
                if reset_ids != max_logits_idx:
                    tmp = next_penalty.numpy()
                    tmp[:, reset_ids] = 1.0
                    next_penalty.update_inplace(tmp)
                init_penality_reset_count += 1
            current_penalty, next_penalty = next_penalty, current_penalty
            binding_G.bind_output(name=out_name_G[1], device_type=device_type, device_id=DEVICE_ID, element_type=penality_dtype, shape=penalty_shape, buffer_ptr=next_penalty.data_ptr())
            binding_A.bind_ortvalue_input(in_name_A, all_outputs_G[0])
        else:
            binding_K.bind_ortvalue_input(in_name_K, all_outputs_F[num_keys_values])
            bind_ort_out(binding_K, out_name_K, _ort_device_type)
            ort_session_K.run_with_iobinding(binding_K, run_options=run_options)
            all_outputs_K = binding_K.get_outputs()[0]
            max_logits_idx = all_outputs_K.numpy().flat[0]
            if max_logits_idx in STOP_TOKEN:
                break
            binding_A.bind_ortvalue_input(in_name_A, all_outputs_K)
        print(tokenizer.decode(max_logits_idx), end="", flush=True)
        bind_ort_in(binding_F, in_name_F_parts, all_outputs_F)
        save_id_greedy[num_decode] = max_logits_idx
    bind_ort_out(binding_A, out_name_A, _ort_device_type)
    ort_session_A.run_with_iobinding(binding_A, run_options=run_options)
    binding_F.bind_ortvalue_input(in_name_F[num_keys_values], binding_A.get_outputs()[0])
    if use_vision:
        binding_D.bind_ortvalue_input(in_name_D[0], init_ids_len_1)
        binding_D.bind_ortvalue_input(in_name_D[1], rotary_outputs[rotary_outputs_len_minus])
        bind_ort_out(binding_D, out_name_D, _ort_device_type)
        ort_session_D.run_with_iobinding(binding_D, run_options=run_options)
        rotary_outputs = binding_D.get_outputs()
    else:
        binding_E.bind_ortvalue_input(in_name_E[0], init_ids_len_1)
        binding_E.bind_ortvalue_input(in_name_E[1], rotary_outputs[rotary_outputs_len_minus])
        bind_ort_out(binding_E, out_name_E, _ort_device_type)
        ort_session_E.run_with_iobinding(binding_E, run_options=run_options)
        rotary_outputs = binding_E.get_outputs()
    bind_ort_in(binding_F, rotary_in_name_F, rotary_outputs)
    if num_decode < 1:
        binding_F.bind_ortvalue_input(in_name_F[num_keys_values_plus_deepstack_plus_rotary_plus_1], init_ids_len_1)
        binding_F.bind_ortvalue_input(in_name_F[num_keys_values_plus_deepstack_plus_rotary_plus_2], init_attention_mask_0)
        bind_ort_in(binding_F, deepstack_in_name_F, init_deepstack_features)
    num_decode += 1

elapsed_time = time.time() - start_time
tokens_per_second = (num_decode + 1) / elapsed_time

if USE_BEAM_SEARCH:
    result = tokenizer.decode(all_outputs_I[num_keys_values_plus_1].numpy()[0, :num_decode], skip_special_tokens=True)
else:
    result = tokenizer.decode(save_id_greedy[:num_decode], skip_special_tokens=True)

print(f"\n\nFinal:\n{result}\n\nDecode: {tokens_per_second:.3f} token/s")
print(f"Total tokens generated: {num_decode}")
print(f"Total time: {elapsed_time:.3f}s")
