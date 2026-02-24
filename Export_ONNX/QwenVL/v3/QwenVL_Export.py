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


download_path               = r'/home/DakeQQ/Downloads/Qwen3-VL-2B-Instruct'          # Set the folder path where the Qwen3-VL whole project downloaded.
onnx_model_Embed            = r'/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Embed.onnx'      # Assign a path where the exported QwenVL model stored.
onnx_model_Vision           = r'/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Vision.onnx'
onnx_model_Concat           = r'/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Concat.onnx'
onnx_model_Rotary_Vision    = r'/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Rotary_Vision.onnx'
onnx_model_Rotary_Text      = r'/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Rotary_Text.onnx'
onnx_model_Main             = r'/home/DakeQQ/Downloads/Qwen_ONNX/LLM_Main.onnx'
onnx_model_Greedy           = r'/home/DakeQQ/Downloads/Qwen_ONNX/Greedy_Search.onnx'
onnx_model_First_Beam       = r'/home/DakeQQ/Downloads/Qwen_ONNX/First_Beam_Search.onnx'
onnx_model_Second_Beam      = r'/home/DakeQQ/Downloads/Qwen_ONNX/Second_Beam_Search.onnx'
onnx_model_Penalty          = r'/home/DakeQQ/Downloads/Qwen_ONNX/Apply_Penalty.onnx'
onnx_model_Argmax           = r'/home/DakeQQ/Downloads/Qwen_ONNX/Argmax.onnx'

# Test Input
TEST_IMAGE = r"../psyduck.png"                                      # Test image for the exported onnx model.
TEST_QUERY = "Describe this image."                                 # Test query for the exported onnx model.

DO_EXPORT = True                                                    # Whether to export the ONNX models

KV_QUANT_DTYPE = "F16"                                              # "Q8" | "Q8_CUDA" | "F16" | "F32"
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
PENALTY_RANGE = 20                                                  # Penalizes the most recent output. "10" means the last 10 tokens.

ORT_Accelerate_Providers = []                                       # ORT execution providers; ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
MAX_THREADS = 0                                                     # 0 = auto
DEVICE_ID = 0                                                       # Device ID for GPU
OPSET = 17                                                          # ONNX opset version


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
            attn = torch.matmul(q_rot, k_rot.transpose(-1, -2))
            attn = torch.nn.functional.softmax(attn, dim=-1)
            attn = torch.matmul(attn, v)
            attn_out = attn.transpose(2, 3).reshape(batch_size, -1, blk.attn.proj.in_features)
            vision_hidden_states = vision_hidden_states + blk.attn.proj(attn_out)
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
        concat_hidden_states = torch.cat([text_hidden_states[:, :self.prompt_head_len], vision_hidden_states, text_hidden_states[:, self.prompt_head_len:]], dim=1)
        zeros_B = self.zeros_B[:, :text_hidden_states.shape[1] - self.prompt_head_len].float()
        deepstack_features = [torch.cat([self.zeros_A, all_inputs[i], zeros_B], dim=1) for i in range(self.deepstack_features_len)]
        return deepstack_features, concat_hidden_states


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


class KVQuantizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qmax = 255.0
        self.register_buffer("inv_qmax", torch.tensor([1.0 / self.qmax], dtype=torch.float32).view(1, 1, 1, 1, -1))
        self.register_buffer("_256", torch.tensor([256], dtype=torch.int32))
        self.register_buffer("_128", torch.tensor([128], dtype=torch.int32))
        self.register_buffer("_65536", torch.tensor([65536], dtype=torch.int32))
        self.register_buffer("_16777216", torch.tensor([16777216], dtype=torch.int32))

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
        return k_packed, k_scale, k_bias, v_packed, v_scale.transpose(-1, -2), v_bias


class LLM_MAIN(torch.nn.Module):
    def __init__(self, llm, head_dim, num_heads, num_layers, num_key_value_heads, hidden_size, max_seq_len, deepstack_features_len, height_factor, width_factor):
        super(LLM_MAIN, self).__init__()
        self.llm = llm
        self._replace_gelu_with_tanh_approximation(self.llm)
        self.head_dim = head_dim
        self.head_dim_half = head_dim // 2
        self.head_dim_quarter = head_dim // 4
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
        self.kv_q8_cuda = (KV_QUANT_DTYPE == "Q8_CUDA")
        self.quantizer = KVQuantizer().eval()
        self.overflow_scale = torch.tensor([0.01], dtype=torch.float32)
        norm_factor = hidden_size ** 0.5
        norm_factor_qk = head_dim ** 0.5
        scale_factor = head_dim ** -0.25
        self.save_key = [None] * num_layers
        self.save_value = [None] * num_layers
        if self.kv_q8 or self.kv_q8_cuda:
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
                v_s = torch.cat([all_inputs[i + self.num_layers_4], scale_v], dim=-1)
                v_b = torch.cat([all_inputs[i + self.num_layers_5], bias_v], dim=-2)
                self.save_k_scale[i] = k_s
                self.save_k_bias[i] = k_b
                self.save_v_scale[i] = v_s
                self.save_v_bias[i] = v_b
                self.save_key[i] = k
                self.save_value[i] = v
                if USE_FLOAT16_SCALE_BIAS:
                    k_s = k_s.float()
                    k_b = k_b.float()
                    v_s = v_s.float()
                    v_b = v_b.float()
                if self.kv_q8_cuda:
                    k = self.quantizer.unpack_q8_cuda(k, -2, batch_size, self.num_key_value_heads, self.head_dim)
                    v = self.quantizer.unpack_q8_cuda(v, -1, batch_size, self.num_key_value_heads, self.head_dim)
                attn_main = torch.matmul(q, k.float())
                q_sum = q.sum(dim=-1, keepdim=True)
                attn_bias = torch.matmul(q_sum, k_b)
                attn = attn_main * k_s + attn_bias
                attn = torch.nn.functional.softmax(attn + attention_mask, dim=-1)
                attn_scaled = attn * v_s
                out_main = torch.matmul(attn_scaled, v.float())
                out_bias = torch.matmul(attn, v_b)
                attn = out_main + out_bias
            else:
                k = torch.cat((all_inputs[i], k), dim=-1)
                v = torch.cat((all_inputs[i + self.num_layers], v), dim=-2)
                self.save_key[i] = k
                self.save_value[i] = v
                if self.kv_f16:
                    k = k.float()
                    v = v.float()
                attn = torch.matmul(q, k)
                attn = torch.nn.functional.softmax(attn + attention_mask, dim=-1)
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
            if i < self.deepstack_features_len:
                hidden_states = all_inputs[i - 8] + hidden_states
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
        model = Qwen3VLForConditionalGeneration.from_pretrained(download_path, dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True).eval()
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

        ids_len = torch.tensor([10], dtype=torch.int64)
        input_ids = torch.ones((1, ids_len), dtype=torch.int32)
        torch.onnx.export(
            LLM_EMBED(model),
            (input_ids,),
            onnx_model_Embed,
            input_names=['input_ids'],
            output_names=['text_hidden_states'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'ids_len'},
                'text_hidden_states': {0: 'batch', 1: 'ids_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        del input_ids
        gc.collect()

        pixel_values = torch.randint(low=0, high=255, size=[VISION_BATCH_SIZE, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]]).to(torch.uint8)
        if INPUT_IMAGE_DIM != 4:
            pixel_values = pixel_values.unsqueeze(1)
        
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
            onnx_model_Vision,
            input_names=['pixel_values'],
            output_names=output_names,
            dynamic_axes=dynamic_axes if DYNAMIC_IMAGE_SHAPE else None,
            opset_version=OPSET,
            dynamo=False
        )
        del pixel_values
        gc.collect()

        prompt_head_len = 4
        text_hidden_states = torch.ones((1, ids_len, hidden_size), dtype=torch.float32)
        vision_embed_size = WIDTH_FACTOR * HEIGHT_FACTOR * VISION_BATCH_SIZE
        vision_hidden_states = torch.ones((1, vision_embed_size, hidden_size), dtype=torch.float32)
        deepstack_features = torch.ones((1, vision_embed_size, hidden_size), dtype=torch.float32)
        
        all_inputs = []
        input_names = []
        output_names = []
        dynamic_axes = {
            'text_hidden_states': {0: 'batch_size', 1: 'ids_len'},
            'concat_hidden_states': {0: 'batch_size', 1: 'total_len'}
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
        output_names.append("concat_hidden_states")

        torch.onnx.export(
            LLM_CONCAT(MAX_SEQ_LEN, prompt_head_len, hidden_size, deepstack_features_len),
            tuple(all_inputs),
            onnx_model_Concat,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            dynamo=False
        )
        del text_hidden_states, vision_hidden_states, deepstack_features, all_inputs
        gc.collect()

        history_len = torch.tensor([0], dtype=torch.int64)
        torch.onnx.export(
            LLM_ROTARY_VISION(model, WIDTH_FACTOR, HEIGHT_FACTOR, prompt_head_len, MAX_SEQ_LEN),
            (ids_len, history_len),
            onnx_model_Rotary_Vision,
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
            onnx_model_Rotary_Text,
            input_names=['ids_len', 'history_len'],
            output_names=['rotary_pos_emb_cos', 'rotary_pos_emb_sin', 'kv_seq_len'],
            dynamic_axes={
                'rotary_pos_emb_cos': {1: 'ids_len'},
                'rotary_pos_emb_sin': {1: 'ids_len'}
            },
            opset_version=OPSET,
            dynamo=False
        )
        
        batch_size = 3
        ids_len = ids_len + vision_embed_size
        kv_tensors = {}
        kv_specs = [('key', 4), ('value', 3)]
        if KV_QUANT_DTYPE == "F16":
            kv_dtype = torch.float16
        elif KV_QUANT_DTYPE == "Q8":
            kv_specs.extend([('key_scale', 4), ('key_bias', 4), ('value_scale', 4), ('value_bias', 3)])
            kv_dtype = torch.uint8
        elif KV_QUANT_DTYPE == "Q8_CUDA":
            kv_specs.extend([('key_scale', 4), ('key_bias', 4), ('value_scale', 4), ('value_bias', 3)])
            kv_dtype = torch.int32
        else:
            kv_dtype = torch.float32

        if KV_QUANT_DTYPE == "Q8_CUDA":
            kv_tensors['key'] = torch.zeros((batch_size, num_key_value_heads, 1, head_dim // 4, history_len), dtype=kv_dtype)
            kv_tensors['value'] = torch.zeros((batch_size, num_key_value_heads, 1, history_len, head_dim // 4), dtype=kv_dtype)
        else:
            kv_tensors['key'] = torch.zeros((batch_size, num_key_value_heads, 1, head_dim, history_len), dtype=kv_dtype)
            kv_tensors['value'] = torch.zeros((batch_size, num_key_value_heads, 1, history_len, head_dim), dtype=kv_dtype)

        if KV_QUANT_DTYPE in ["Q8", "Q8_CUDA"]:
            kv_tensors['key_scale'] = torch.ones([batch_size, num_key_value_heads, 1, 1, history_len], dtype=scale_dtype)
            kv_tensors['key_bias'] = torch.ones([batch_size, num_key_value_heads, 1, 1, history_len], dtype=scale_dtype)
            kv_tensors['value_scale'] = torch.ones([batch_size, num_key_value_heads, 1, 1, history_len], dtype=scale_dtype)
            kv_tensors['value_bias'] = torch.ones([batch_size, num_key_value_heads, 1, history_len, 1], dtype=scale_dtype)

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
        deepstack_features = torch.ones((1, ids_len, hidden_size), dtype=torch.float32)
        rotary_pos_emb_cos = torch.ones((1, ids_len, 1, 1, head_dim), dtype=torch.float32)
        rotary_pos_emb_sin = rotary_pos_emb_cos
        kv_seq_len = history_len.long() + ids_len
        attention_mask = torch.tensor([1], dtype=torch.int8)

        all_inputs = kv_ins + [hidden_states]
        input_names = kv_in_names + ['hidden_states']
        dynamic_axes = {
            **kv_axes, 'hidden_states': {0: 'batch', 1: 'ids_len'},
            'rotary_pos_emb_cos': {1: 'ids_len'},
            'rotary_pos_emb_sin': {1: 'ids_len'},
            'logits': {0: 'batch'}
        }
        
        for i in range(deepstack_features_len):
            name = f'deepstack_features_{i}'
            input_names.append(name)
            all_inputs.append(deepstack_features)
            dynamic_axes[name] = {1: 'total_len'}
        
        all_inputs.extend([rotary_pos_emb_cos, rotary_pos_emb_sin, kv_seq_len, ids_len, attention_mask])
        input_names.extend(['rotary_pos_emb_cos', 'rotary_pos_emb_sin', 'kv_seq_len', 'ids_len', 'attention_mask'])
        output_names = kv_out_names + ['logits']

        model_Main = LLM_MAIN(model, head_dim, num_heads, num_layers, num_key_value_heads, hidden_size, MAX_SEQ_LEN, deepstack_features_len, HEIGHT_FACTOR, WIDTH_FACTOR)
        del model
        gc.collect()

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
        del model_Main, hidden_states, deepstack_features, rotary_pos_emb_cos, rotary_pos_emb_sin, attention_mask, all_inputs
        gc.collect()

        save_id_in = torch.zeros((BEAM_SIZE, 10), dtype=torch.int32)
        logits = torch.ones((BEAM_SIZE, vocab_size), dtype=torch.float32)
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
        beam_size_t = torch.tensor([BEAM_SIZE], dtype=torch.int64)
        other_inputs = [logits[[0]], save_id_in, beam_size_t]
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
            output_names=['out_' + n[3:] for n in kv_in_names] + ['top_beam_indices', 'save_id_out', 'top_beam_prob', 'max_logits_idx'],
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            dynamo=False
        )
        del kv_tensors_Greedy

        kv_ins, kv_in_names, kv_out_names, kv_axes = get_kv_io(kv_tensors, 'batch_size', 'history_len', 'kv_seq_len')
        previous_prob = torch.zeros((BEAM_SIZE, 1), dtype=torch.float32)
        topK = torch.tensor([TOP_K], dtype=torch.int64)
        other_inputs = [logits, save_id_in, previous_prob, beam_size_t, topK]
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
        del kv_tensors, previous_prob, topK, num_layers

        penalty_value = torch.tensor([REPEAT_PENALITY], dtype=torch.float32)
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
        del beam_size_t
        del vocab_size

        print('\nExport Done!\n\nStrat to run QwenVL by ONNX Runtime.')


# Inference with ONNXRuntime
# =======================================================#
def is_valid_image_path(TEST_IMAGE):
    if TEST_IMAGE is None:
        return False
    elif TEST_IMAGE == "":
        return False
    elif not os.path.exists(TEST_IMAGE):
        return False
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', 'raw'}
    _, ext = os.path.splitext(TEST_IMAGE)
    return ext.lower() in valid_extensions


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


tokenizer = AutoTokenizer.from_pretrained(download_path)

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

ort_session_Embed = onnxruntime.InferenceSession(onnx_model_Embed, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
binding_Embed = ort_session_Embed.io_binding()
in_name_Embed = ort_session_Embed.get_inputs()[0].name
out_name_Embed = [ort_session_Embed.get_outputs()[0].name]

ort_session_Vision = onnxruntime.InferenceSession(onnx_model_Vision, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
binding_Vision = ort_session_Vision.io_binding()
in_name_Vision_objs = ort_session_Vision.get_inputs()
out_name_Vision_objs = ort_session_Vision.get_outputs()
in_name_Vision = [x.name for x in in_name_Vision_objs]
out_name_Vision = [x.name for x in out_name_Vision_objs]
deepstack_features_len = len(out_name_Vision) - 1
in_name_Vision_parts = in_name_Vision[:deepstack_features_len+1]
vision_dtype = np.float16 if 'float16' in ort_session_Vision._outputs_meta[0].type else np.float32

ort_session_Concat = onnxruntime.InferenceSession(onnx_model_Concat, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
binding_Concat = ort_session_Concat.io_binding()
in_name_Concat_objs = ort_session_Concat.get_inputs()
out_name_Concat_objs = ort_session_Concat.get_outputs()
amount_of_outputs_Concat = len(out_name_Concat_objs)
in_name_Concat = [x.name for x in in_name_Concat_objs]
out_name_Concat = [x.name for x in out_name_Concat_objs]

ort_session_Rotary_Vision = onnxruntime.InferenceSession(onnx_model_Rotary_Vision, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
binding_Rotary_Vision = ort_session_Rotary_Vision.io_binding()
in_name_Rotary_Vision = [x.name for x in ort_session_Rotary_Vision.get_inputs()]
out_name_Rotary_Vision = [x.name for x in ort_session_Rotary_Vision.get_outputs()]
rotary_outputs_len = len(out_name_Rotary_Vision)
rotary_outputs_len_minus = rotary_outputs_len - 1

ort_session_Rotary_Text = onnxruntime.InferenceSession(onnx_model_Rotary_Text, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
binding_Rotary_Text = ort_session_Rotary_Text.io_binding()
in_name_Rotary_Text = [x.name for x in ort_session_Rotary_Text.get_inputs()]
out_name_Rotary_Text = [x.name for x in ort_session_Rotary_Text.get_outputs()]

ort_session_Main = onnxruntime.InferenceSession(onnx_model_Main, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
binding_Main = ort_session_Main.io_binding()
print(f"\nUsable Providers: {ort_session_Main.get_providers()}")

model_dtype_Main_str = ort_session_Main._inputs_meta[0].type
in_name_Main_objs = ort_session_Main.get_inputs()
out_name_Main_objs = ort_session_Main.get_outputs()
amount_of_outputs_Main = len(out_name_Main_objs)
in_name_Main = [x.name for x in in_name_Main_objs]
out_name_Main = [x.name for x in out_name_Main_objs]

if 'dml' in device_type:
    kv_device = 'cpu'
else:
    kv_device = device_type

num_keys_values = amount_of_outputs_Main - 1
if 'uint8' in model_dtype_Main_str or 'int32' in model_dtype_Main_str:
    kv_dtype = np.int32 if 'int32' in model_dtype_Main_str else np.uint8
    num_layers = num_keys_values // 6
    kv_scale_dtype = np.float16 if 'float16' in ort_session_Main._inputs_meta[num_layers + num_layers].type else np.float32
    k_scales = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_Main._inputs_meta[0].shape[1], 1, 1, 0), dtype=kv_scale_dtype), kv_device, DEVICE_ID)
    k_biases = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_Main._inputs_meta[0].shape[1], 1, 1, 0), dtype=kv_scale_dtype), kv_device, DEVICE_ID)
    v_scales = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_Main._inputs_meta[num_layers].shape[1], 1, 1, 0), dtype=kv_scale_dtype), kv_device, DEVICE_ID)
    v_biases = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_Main._inputs_meta[num_layers].shape[1], 1, 0, 1), dtype=kv_scale_dtype), kv_device, DEVICE_ID)
else:
    kv_dtype = np.float16 if 'float16' in model_dtype_Main_str else np.float32
    num_layers = num_keys_values // 2
    k_scales = None

vision_embed_size = VISION_BATCH_SIZE * WIDTH_FACTOR * HEIGHT_FACTOR
num_keys_values_plus_1 = num_keys_values + 1
num_keys_values_plus_2 = num_keys_values + 2
num_keys_values_plus_3 = num_keys_values + 3
num_keys_values_plus_4 = num_keys_values + 4
num_keys_values_plus_deepstack_plus_rotary = num_keys_values + deepstack_features_len + rotary_outputs_len
num_keys_values_plus_deepstack_plus_rotary_plus_1 = num_keys_values_plus_deepstack_plus_rotary + 1
num_keys_values_plus_deepstack_plus_rotary_plus_2 = num_keys_values_plus_deepstack_plus_rotary + 2
rotary_indices = np.arange(rotary_outputs_len, dtype=np.int32) + num_keys_values + deepstack_features_len + 1
deepstack_indices = np.arange(deepstack_features_len, dtype=np.int32) + num_keys_values + 1
rotary_in_name_Main = in_name_Main[rotary_indices[0]:rotary_indices[rotary_outputs_len-1]+1]
deepstack_in_name_Main = in_name_Main[deepstack_indices[0]:deepstack_indices[deepstack_features_len-1]+1]
in_name_Main_parts = in_name_Main[:num_keys_values]
vocab_size = ort_session_Main._outputs_meta[num_keys_values].shape[1]

if USE_BEAM_SEARCH and (TOP_K < BEAM_SIZE):
    TOP_K = BEAM_SIZE

if (TOP_K < 2) or (BEAM_SIZE < 2):
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting. Falling back to Greedy Search.")

if not USE_BEAM_SEARCH:
    BEAM_SIZE = 1


prompt = f"<|im_start|>user\n<|vision_start|><|vision_end|>{TEST_QUERY}<|im_end|>\n<|im_start|>assistant\n"
tokens = tokenizer(prompt, return_tensors='np')['input_ids'].astype(np.int32)
input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, DEVICE_ID)
num_prefill = tokens.shape[-1]
ids_len = create_ortvalue([num_prefill], np.int64, device_type, DEVICE_ID)
init_ids_len_1 = create_ortvalue([1], np.int64, device_type, DEVICE_ID)
init_history_len = create_ortvalue([0], np.int64, device_type, DEVICE_ID)
init_attention_mask_0 = create_ortvalue([0], np.int8, device_type, DEVICE_ID)
init_attention_mask_1 = create_ortvalue([1], np.int8, device_type, DEVICE_ID)
init_deepstack_features = [onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, 1, ort_session_Concat._inputs_meta[0].shape[2]), dtype=vision_dtype), device_type, DEVICE_ID)] * deepstack_features_len
init_past_keys_Main = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_Main._inputs_meta[0].shape[1], 1, ort_session_Main._inputs_meta[0].shape[3], 0), dtype=kv_dtype), kv_device, DEVICE_ID)
init_past_values_Main = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((1, ort_session_Main._inputs_meta[num_layers].shape[1], 1, 0, ort_session_Main._inputs_meta[num_layers].shape[4]), dtype=kv_dtype), kv_device, DEVICE_ID)
init_save_id = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((BEAM_SIZE, 0), dtype=np.int32), device_type, DEVICE_ID)
topK = create_ortvalue([TOP_K], np.int64, device_type, DEVICE_ID)
beam_size = create_ortvalue([BEAM_SIZE], np.int64, device_type, DEVICE_ID)

USE_PENALTY = (REPEAT_PENALITY != 1.0)

if USE_BEAM_SEARCH:
    print("\nBeam Search does not display immediate decoding results...")
    ort_session_First_Beam = onnxruntime.InferenceSession(onnx_model_First_Beam, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    binding_First_Beam = ort_session_First_Beam.io_binding()
    in_name_First_Beam = [x.name for x in ort_session_First_Beam.get_inputs()]
    out_name_First_Beam = [x.name for x in ort_session_First_Beam.get_outputs()]
    in_name_First_Beam_parts = in_name_First_Beam[:num_keys_values]
    ort_session_Second_Beam = onnxruntime.InferenceSession(onnx_model_Second_Beam, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    binding_Second_Beam = ort_session_Second_Beam.io_binding()
    in_name_Second_Beam = [x.name for x in ort_session_Second_Beam.get_inputs()]
    out_name_Second_Beam = [x.name for x in ort_session_Second_Beam.get_outputs()]
    in_name_Second_Beam_parts = in_name_Second_Beam[:num_keys_values]
    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[num_keys_values_plus_1], init_save_id)
    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[num_keys_values_plus_2], beam_size)
    binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_plus_3], beam_size)
    binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_plus_4], topK)
else:
    ort_session_Greedy = onnxruntime.InferenceSession(onnx_model_Greedy, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    binding_Greedy = ort_session_Greedy.io_binding()
    in_name_Greedy = [x.name for x in ort_session_Greedy.get_inputs()]
    out_name_Greedy = [x.name for x in ort_session_Greedy.get_outputs()]
    binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], init_save_id)
    ort_session_Argmax = onnxruntime.InferenceSession(onnx_model_Argmax, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    binding_Argmax = ort_session_Argmax.io_binding()
    in_name_Argmax = ort_session_Argmax.get_inputs()[0].name
    out_name_Argmax = [x.name for x in ort_session_Argmax.get_outputs()]
    save_id_numpy = np.zeros(MAX_SEQ_LEN, dtype=np.int32)

if USE_PENALTY:
    ort_session_Penalty = onnxruntime.InferenceSession(onnx_model_Penalty, sess_options=session_opts, providers=ORT_Accelerate_Providers, provider_options=provider_options, run_options=run_options)
    binding_Penalty = ort_session_Penalty.io_binding()
    in_name_Penalty = [x.name for x in ort_session_Penalty.get_inputs()]
    out_name_Penalty = [x.name for x in ort_session_Penalty.get_outputs()]
    penality_dtype = np.float16 if 'float16' in ort_session_Penalty._inputs_meta[2].type else np.float32
    penalty_value = create_ortvalue([REPEAT_PENALITY], penality_dtype, device_type, DEVICE_ID)
    penality_range = create_ortvalue([PENALTY_RANGE], np.int64, device_type, DEVICE_ID)
    binding_Penalty.bind_ortvalue_input(in_name_Penalty[2], penalty_value)
    binding_Penalty.bind_ortvalue_input(in_name_Penalty[3], penality_range)

if is_valid_image_path(TEST_IMAGE):
    image = Image.open(TEST_IMAGE)
    image = image.resize((INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0]))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    pixel_values = np.transpose(np.array(image).astype(np.uint8), (2, 0, 1))
    if len(ort_session_Vision._inputs_meta[0].shape) != 4:
        axis = (0, 1)
    else:
        axis = 0
    pixel_values = np.expand_dims(pixel_values, axis=axis)
    use_vision = True
    print('\nChat with image.')
else:
    use_vision = False
    print('\nChat without image.')

binding_Embed.bind_ortvalue_input(in_name_Embed, input_ids)
bind_ort_out(binding_Embed, out_name_Embed, _ort_device_type)
ort_session_Embed.run_with_iobinding(binding_Embed, run_options=run_options)
outputs_Embed = binding_Embed.get_outputs()[0]

generate_limit = MAX_SEQ_LEN - num_prefill

if use_vision:
    num_prefill += vision_embed_size
    ids_len = create_ortvalue([num_prefill], np.int64, device_type, DEVICE_ID)
    generate_limit -= vision_embed_size
    
    print('\nStart to Process the Image...')
    start_time = time.time()
    binding_Vision.bind_ortvalue_input(in_name_Vision[0], onnxruntime.OrtValue.ortvalue_from_numpy(pixel_values, device_type, DEVICE_ID))
    bind_ort_out(binding_Vision, out_name_Vision, _ort_device_type)
    ort_session_Vision.run_with_iobinding(binding_Vision, run_options=run_options)
    outputs_Vision = binding_Vision.get_outputs()
    print(f'\nImage Process Complete. Time Cost: {time.time() - start_time:.3f} Seconds')
    
    binding_Concat.bind_ortvalue_input(in_name_Concat[deepstack_features_len], outputs_Embed)
    binding_Concat.bind_ortvalue_input(in_name_Concat[amount_of_outputs_Concat], outputs_Vision[deepstack_features_len])
    bind_ort_in(binding_Concat, in_name_Concat, outputs_Vision, deepstack_features_len)
    bind_ort_out(binding_Concat, out_name_Concat, _ort_device_type)
    ort_session_Concat.run_with_iobinding(binding_Concat, run_options=run_options)
    outputs_Concat = binding_Concat.get_outputs()
    
    binding_Rotary_Vision.bind_ortvalue_input(in_name_Rotary_Vision[0], ids_len)
    binding_Rotary_Vision.bind_ortvalue_input(in_name_Rotary_Vision[1], init_history_len)
    bind_ort_out(binding_Rotary_Vision, out_name_Rotary_Vision, _ort_device_type)
    ort_session_Rotary_Vision.run_with_iobinding(binding_Rotary_Vision, run_options=run_options)
    rotary_outputs = binding_Rotary_Vision.get_outputs()
    
    binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values], outputs_Concat[deepstack_features_len])
    bind_ort_in(binding_Main, deepstack_in_name_Main, outputs_Concat)
else:
    binding_Rotary_Text.bind_ortvalue_input(in_name_Rotary_Text[0], ids_len)
    binding_Rotary_Text.bind_ortvalue_input(in_name_Rotary_Text[1], init_history_len)
    bind_ort_out(binding_Rotary_Text, out_name_Rotary_Text, _ort_device_type)
    ort_session_Rotary_Text.run_with_iobinding(binding_Rotary_Text, run_options=run_options)
    rotary_outputs = binding_Rotary_Text.get_outputs()
    
    binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values], outputs_Embed)
    bind_ort_in(binding_Main, deepstack_in_name_Main, init_deepstack_features)

binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_deepstack_plus_rotary_plus_1], ids_len)
binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_deepstack_plus_rotary_plus_2], init_attention_mask_1)

i = 0
j = num_layers
while i < j:
    binding_Main.bind_ortvalue_input(in_name_Main[i], init_past_keys_Main)
    i += 1
j = i + num_layers
while i < j:
    binding_Main.bind_ortvalue_input(in_name_Main[i], init_past_values_Main)
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

for i in range(rotary_outputs_len):
    binding_Main.bind_ortvalue_input(in_name_Main[rotary_indices[i]], rotary_outputs[i])

print(f'\nTest Question: {TEST_QUERY}\nLLM Answering:\n')
num_decode = 0
start_time = time.time()
while num_decode < generate_limit:
    bind_ort_out(binding_Main, out_name_Main, _ort_device_type)
    ort_session_Main.run_with_iobinding(binding_Main, run_options=run_options)
    outputs_Main = binding_Main.get_outputs()
    logits = outputs_Main[num_keys_values]
    if USE_PENALTY and (num_decode >= PENALTY_RANGE):
        binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], logits)
        binding_Penalty.bind_ortvalue_input(in_name_Penalty[1], save_id)
        bind_ort_out(binding_Penalty, out_name_Penalty, _ort_device_type)
        ort_session_Penalty.run_with_iobinding(binding_Penalty, run_options=run_options)
        logits = binding_Penalty.get_outputs()[0]
    if USE_BEAM_SEARCH:
        if num_decode < 1:
            binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[num_keys_values], logits)
            bind_ort_in(binding_First_Beam, in_name_First_Beam_parts, outputs_Main)
            bind_ort_out(binding_First_Beam, out_name_First_Beam, _ort_device_type)
            ort_session_First_Beam.run_with_iobinding(binding_First_Beam, run_options=run_options)
            outputs_Beam = binding_First_Beam.get_outputs()
        else:
            binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values], logits)
            bind_ort_in(binding_Second_Beam, in_name_Second_Beam_parts, outputs_Main)
            bind_ort_out(binding_Second_Beam, out_name_Second_Beam, _ort_device_type)
            ort_session_Second_Beam.run_with_iobinding(binding_Second_Beam, run_options=run_options)
            outputs_Beam = binding_Second_Beam.get_outputs()
        max_logits_idx = outputs_Beam[num_keys_values_plus_3].numpy()
        if max_logits_idx in STOP_TOKEN:
            break
        save_id = outputs_Beam[num_keys_values_plus_1]
        bind_ort_in(binding_Main, in_name_Main_parts, outputs_Beam)
        binding_Embed.bind_ortvalue_input(in_name_Embed, outputs_Beam[num_keys_values])
        binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_plus_1], outputs_Beam[num_keys_values_plus_1])
        binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_plus_2], outputs_Beam[num_keys_values_plus_2])
    else:
        if USE_PENALTY:
            binding_Greedy.bind_ortvalue_input(in_name_Greedy[0], logits)
            bind_ort_out(binding_Greedy, out_name_Greedy, _ort_device_type)
            ort_session_Greedy.run_with_iobinding(binding_Greedy, run_options=run_options)
            max_logits_ort, save_id = binding_Greedy.get_outputs()
            max_logits_idx = max_logits_ort.numpy().flat[0]
        else:
            binding_Argmax.bind_ortvalue_input(in_name_Argmax, logits)
            bind_ort_out(binding_Argmax, out_name_Argmax, _ort_device_type)
            ort_session_Argmax.run_with_iobinding(binding_Argmax, run_options=run_options)
            max_logits_ort = binding_Argmax.get_outputs()[0]
            max_logits_idx = max_logits_ort.numpy().flat[0]
        if max_logits_idx in STOP_TOKEN:
            break
        if USE_PENALTY:
            binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id)
        else:
            save_id_numpy[num_decode] = max_logits_idx
        binding_Embed.bind_ortvalue_input(in_name_Embed, max_logits_ort)
        bind_ort_in(binding_Main, in_name_Main_parts, outputs_Main)
        print(tokenizer.decode(max_logits_idx), end="", flush=True)
    bind_ort_out(binding_Embed, out_name_Embed, _ort_device_type)
    ort_session_Embed.run_with_iobinding(binding_Embed, run_options=run_options)
    binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values], binding_Embed.get_outputs()[0])
    if use_vision:
        binding_Rotary_Vision.bind_ortvalue_input(in_name_Rotary_Vision[0], init_ids_len_1)
        binding_Rotary_Vision.bind_ortvalue_input(in_name_Rotary_Vision[1], rotary_outputs[rotary_outputs_len_minus])
        bind_ort_out(binding_Rotary_Vision, out_name_Rotary_Vision, _ort_device_type)
        ort_session_Rotary_Vision.run_with_iobinding(binding_Rotary_Vision, run_options=run_options)
        rotary_outputs = binding_Rotary_Vision.get_outputs()
    else:
        binding_Rotary_Text.bind_ortvalue_input(in_name_Rotary_Text[0], init_ids_len_1)
        binding_Rotary_Text.bind_ortvalue_input(in_name_Rotary_Text[1], rotary_outputs[rotary_outputs_len_minus])
        bind_ort_out(binding_Rotary_Text, out_name_Rotary_Text, _ort_device_type)
        ort_session_Rotary_Text.run_with_iobinding(binding_Rotary_Text, run_options=run_options)
        rotary_outputs = binding_Rotary_Text.get_outputs()
    bind_ort_in(binding_Main, rotary_in_name_Main, rotary_outputs)
    if num_decode < 1:
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_deepstack_plus_rotary_plus_1], init_ids_len_1)
        binding_Main.bind_ortvalue_input(in_name_Main[num_keys_values_plus_deepstack_plus_rotary_plus_2], init_attention_mask_0)
        bind_ort_in(binding_Main, deepstack_in_name_Main, init_deepstack_features)
    num_decode += 1

elapsed_time = time.time() - start_time
tokens_per_second = (num_decode + 1) / elapsed_time

if USE_PENALTY or USE_BEAM_SEARCH:
    result = tokenizer.decode(save_id.numpy()[0, :num_decode], skip_special_tokens=True)
else:
    result = tokenizer.decode(save_id_numpy[:num_decode], skip_special_tokens=True)

print(f"\n\nFinal:\n{result}\n\nDecode: {tokens_per_second:.3f} token/s")
print(f"Total tokens generated: {num_decode}")
print(f"Total time: {elapsed_time:.3f}s")

