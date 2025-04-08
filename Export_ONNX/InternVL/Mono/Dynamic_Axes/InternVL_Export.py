import gc
import os
import time
import shutil
import torch
import numpy as np
import onnxruntime
from PIL import Image

try:
    from export_config import MAX_SEQ_LENGTH, PROMPT_HEAD_LENGTH, INPUT_IMAGE_SIZE
except:
    # Default Values if import failed
    INPUT_IMAGE_SIZE = [960, 960]
    MAX_SEQ_LENGTH = 4096                                                           # The max token length. Note, this value include the 10 tokens for system prompt and 256 tokens for image prompt. Hence, only (MAX_SEQ_LENGTH - 256 - 10) tokens for query + response.
    PROMPT_HEAD_LENGTH = 5


use_dynamic_input_image_size = False                                                # False for fixed image size as input.
use_center_crop = True                                                              # Set true for focus on the center object.
path = r'/home/DakeQQ/Downloads/Mono-InternVL-2B-S1-3'                              # Set the folder path where the Mono-InternVL whole project downloaded.
onnx_model_A = r'/home/DakeQQ/Downloads/Intern_ONNX/InternVL_A.onnx'                # Assign a path where the exported InternVL model stored.
onnx_model_B = r'/home/DakeQQ/Downloads/Intern_ONNX/InternVL_B.onnx'
onnx_model_C = r'/home/DakeQQ/Downloads/Intern_ONNX/InternVL_C.onnx'
onnx_model_D = r'/home/DakeQQ/Downloads/Intern_ONNX/InternVL_D.onnx'
image_path = r"./psyduck.png"                                                       # Test image for the exported onnx model.
query = "Provide a detailed description of the image."                              # Test query for the exported onnx model.


shutil.copyfile("./modeling_modified/modeling_intern_patch.py", path + "/modeling_intern_patch.py")
shutil.copyfile("./modeling_modified/modeling_internlm2_ve.py", path + "/modeling_internlm2_ve.py")
shutil.copyfile("./export_config.py", path + "/export_config.py")
from transformers import AutoModel, AutoTokenizer


def quantize_to_uint8(tensor, scale, zero_point):
    return ((tensor - zero_point) * scale).round().clamp(0, 255).to(torch.uint8)


def is_valid_image_path(image_path):
    if not os.path.exists(image_path):
        return False
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    _, ext = os.path.splitext(image_path)
    return ext.lower() in valid_extensions


class InternVL_PartA(torch.nn.Module):
    def __init__(self, intern_chat_model, h_w_factor, image_size, num_image_token):
        super(InternVL_PartA, self).__init__()
        self.vision_model = intern_chat_model.vision_model.embeddings
        self.position_embedding = torch.cat([self.vision_model.position_embedding[:, :1, :], self.vision_model._get_pos_embed(self.vision_model.position_embedding[:, 1:, :], h_w_factor, h_w_factor)], dim=1)[:, 1:]
        self.mlp1 = intern_chat_model.mlp1
        self.num_image_token = num_image_token
        self.image_size = image_size
        self.image_size_2 = image_size + image_size
        self.center_crop = [image_size // 2, image_size // 2 + image_size]
        self.h_w_half = h_w_factor // 2
        means = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).expand(1, 3, image_size, image_size)
        inv_std = torch.tensor([1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225], dtype=torch.float32).view(1, 3, 1, 1).expand(1, 3, image_size, image_size)
        self.means_inv_std = means * inv_std
        self.inv_255_std = inv_std / 255.0
        
    def forward(self, pixel_values):
        # The original repository uses 9 grid-cropped images along with thumbnails. However, we are using only the thumbnail, which is expected to result in lower OCR performance. It takes time to process multi-cropped images as input.
        pixel_values = pixel_values.float()
        if use_center_crop:
            pixel_values = torch.nn.functional.interpolate(
                pixel_values,
                (self.image_size_2, self.image_size_2),
                mode='bilinear',     # bilinear for speed / bicubic for accuracy
                align_corners=True)[:, :, self.center_crop[0]:self.center_crop[1], self.center_crop[0]:self.center_crop[1]]
        else:
            pixel_values = torch.nn.functional.interpolate(
                pixel_values,
                (self.image_size, self.image_size),
                mode='bicubic',      # bilinear for speed / bicubic for accuracy
                align_corners=True)
        pixel_values = pixel_values * self.inv_255_std - self.means_inv_std
        vision_embed = self.vision_model(pixel_values=pixel_values) + self.position_embedding
        return self.mlp1(vision_embed.reshape(1, self.h_w_half, 2, self.h_w_half, -1).transpose(2, 3).contiguous().reshape(1, self.num_image_token, -1))


class InternVL_PartB(torch.nn.Module):
    def __init__(self, prompt_head_len):
        super(InternVL_PartB, self).__init__()
        self.prompt_head_len = torch.tensor([prompt_head_len], dtype=torch.int64)

    def forward(self, hidden_states, vision_embed):
        return torch.cat((hidden_states[:, :self.prompt_head_len], vision_embed, hidden_states[:, self.prompt_head_len:]), dim=1)


class InternVL_PartC(torch.nn.Module):
    def __init__(self, embed_data, scale, zero_point, hidden_size):
        super(InternVL_PartC, self).__init__()
        self.embed_data = embed_data
        self.scale = scale
        self.zero_point = zero_point
        self.hidden_size = hidden_size

    def forward(self,
                input_ids
                ):
        hidden_states = self.embed_data[input_ids] * self.scale[input_ids] + self.zero_point[input_ids]
        return hidden_states


# Load the model
with torch.inference_mode():
    model = AutoModel.from_pretrained(path, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True, trust_remote_code=True).eval()
    num_heads = model.config.llm_config.num_attention_heads
    num_key_value_heads = model.config.llm_config.num_key_value_heads
    head_dim = model.config.llm_config.hidden_size // num_heads
    num_layers = model.config.llm_config.num_hidden_layers
    hidden_size = model.config.llm_config.hidden_size
    image_size = model.config.force_image_size
    num_image_token = model.num_image_token
    height_factor = int(model.vision_model.embeddings.embed_dim ** 0.5)
    width_factor = height_factor

    pixel_values = torch.ones((1, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]), dtype=torch.uint8)
    vision_embed = torch.ones((1, num_image_token, hidden_size), dtype=torch.float32)
    attention_mask = torch.tensor([1], dtype=torch.int8)
    split_factor = torch.tensor([PROMPT_HEAD_LENGTH], dtype=torch.int64)
    input_ids = torch.ones((1, MAX_SEQ_LENGTH), dtype=torch.int32)
    hidden_states = torch.ones((1, MAX_SEQ_LENGTH, hidden_size), dtype=torch.float32)
    past_keys = torch.zeros((num_key_value_heads, head_dim, 0), dtype=torch.float16)
    past_values = torch.zeros((num_key_value_heads, 0, head_dim), dtype=torch.float16)
    position_ids = torch.arange(MAX_SEQ_LENGTH, dtype=torch.float32).unsqueeze(-1)
    theta = 10000.0 ** -(torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    idx_theta = position_ids * theta
    cos_rotary_pos_emb = torch.cos(idx_theta)
    sin_rotary_pos_emb = torch.sin(idx_theta)
    cos_rotary_pos_emb = torch.cat((cos_rotary_pos_emb, cos_rotary_pos_emb), dim=-1).unsqueeze(0).half()
    sin_rotary_pos_emb = torch.cat((sin_rotary_pos_emb, sin_rotary_pos_emb), dim=-1).unsqueeze(0).half()
    model.base_model.language_model.register_buffer('cos_rotary_pos_emb', cos_rotary_pos_emb)
    model.base_model.language_model.register_buffer('sin_rotary_pos_emb', sin_rotary_pos_emb)
    del theta
    del idx_theta
    del position_ids
    del cos_rotary_pos_emb
    del sin_rotary_pos_emb

    model.base_model.language_model.model.tok_embeddings.weight.requires_grad = False
    data = model.base_model.language_model.model.tok_embeddings.weight.data
    zero_point = (torch.min(data, dim=1)[0]).unsqueeze(1)
    scale = ((torch.max(data, dim=1)[0] - zero_point[:, 0]) / 255.0).unsqueeze(1)
    embed_data = quantize_to_uint8(data, 1.0 / scale, zero_point)

    model_A = InternVL_PartA(model, height_factor, image_size, num_image_token)
    print('\nExport Part_A Start...')
    torch.onnx.export(
        model_A,
        (pixel_values,),
        onnx_model_A,
        input_names=[
            'pixel_values'
        ],
        output_names=['vision_embed'],
        dynamic_axes={
            'pixel_values': {2: 'height', 3: 'width'}
        } if use_dynamic_input_image_size else None,
        do_constant_folding=True,
        opset_version=17
    )
    del model_A
    del pixel_values
    gc.collect()
    print('\nExport Part_A Done!  \n\nExport Part_B Start...')

    model_B = InternVL_PartB(PROMPT_HEAD_LENGTH)
    torch.onnx.export(
        model_B,
        (hidden_states, vision_embed),
        onnx_model_B,
        input_names=[
            'hidden_states',
            'vision_embed'
        ],
        output_names=['hidden_states_with_vision'],
        dynamic_axes={
            'hidden_states': {1: 'ids_len'},
            'hidden_states_with_vision': {1: 'ids_len_vision_embed_len'}
        },
        do_constant_folding=True,
        opset_version=17
    )
    del model_B
    del vision_embed
    print('\nExport Part_B Done! \n\nExport Part_C Start...')

    model_C = InternVL_PartC(embed_data, scale, zero_point, hidden_size)
    torch.onnx.export(
        model_C,
        (input_ids,),
        onnx_model_C,
        input_names=[
            'input_ids'
        ],
        output_names=['hidden_states'],
        dynamic_axes={
            'input_ids': {1: 'ids_len'},
            'hidden_states': {1: 'ids_len'}
        },
        do_constant_folding=True,
        opset_version=17
    )
    del model_C
    del embed_data
    del data
    del scale
    del zero_point
    del input_ids
    gc.collect()
    print('\nExport Part_C Done! \n\nExport Part_D Start...')

    model = model.base_model.language_model
    input_names = []
    keys_values = []
    output_names = ['max_logit_id']
    dynamic_axes = {'hidden_states': {1: 'ids_len'}}
    for i in range(num_layers):
        key_name = f'in_key_{i}'
        input_names.append(key_name)
        keys_values.append(past_keys)
        dynamic_axes[key_name] = {2: 'history_len'}
        key_name = f'out_key_{i}'
        output_names.append(key_name)
        dynamic_axes[key_name] = {2: 'history_len_plus_ids_len'}

    for i in range(num_layers):
        value_name = f'in_value_{i}'
        input_names.append(value_name)
        keys_values.append(past_values)
        dynamic_axes[value_name] = {1: 'history_len'}
        value_name = f'out_value_{i}'
        output_names.append(value_name)
        dynamic_axes[value_name] = {1: 'history_len_plus_ids_len'}

    input_names.append('attention_mask')
    input_names.append('split_factor')
    input_names.append('hidden_states')

    print('Export start ...')
    torch.onnx.export(
        model,
        tuple(keys_values + [attention_mask, split_factor, hidden_states]),
        onnx_model_D,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=17
    )
    del model
    del hidden_states
    del attention_mask
    del past_keys
    del past_values
    del split_factor
    print('\nExport Part_D Done!\n\nStart running the InternVL by ONNXRuntime.\nNow loading . . . it could cost minutes.')


# Run the exported model by ONNX Runtime
max_single_chat_length = 512                        # It an adjustable value, but must less than MAX_SEQ_LENGTH.
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4                 # fatal level, it an adjustable value.
session_opts.inter_op_num_threads = 0               # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0               # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True            # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
out_name_A0 = out_name_A[0].name

ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
in_name_B0 = in_name_B[0].name
in_name_B1 = in_name_B[1].name
out_name_B0 = out_name_B[0].name

ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_C = ort_session_C.get_inputs()
out_name_C = ort_session_C.get_outputs()
in_name_C0 = in_name_C[0].name
out_name_C0 = out_name_C[0].name

ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_D = ort_session_D.get_inputs()
out_name_D = ort_session_D.get_outputs()

# Pre-process inputs
if is_valid_image_path(image_path):
    image = Image.open(image_path)
    if not use_dynamic_input_image_size:
        image = image.resize((INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0]))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    pixel_values = np.transpose(np.array(image).astype(np.uint8), (2, 0, 1))
    pixel_values = np.expand_dims(pixel_values, axis=0)
    pixel_values = onnxruntime.OrtValue.ortvalue_from_numpy(pixel_values, 'cpu', 0)
    use_vision = True
else:
    print("\nImage not found.")
    use_vision = False

prompt = f"<|im_start|>user\n<img></img>\n{query}<|im_end|><|im_start|>assistant\n"
tokens = tokenizer(prompt, return_tensors='pt')['input_ids']
input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens.int().numpy(), 'cpu', 0)
attention_mask = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), 'cpu', 0)
past_keys_D = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((num_key_value_heads, head_dim, 0), dtype=np.float16), 'cpu', 0)
past_values_D = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((num_key_value_heads, 0, head_dim), dtype=np.float16), 'cpu', 0)
split_factor = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([PROMPT_HEAD_LENGTH], dtype=np.int64), 'cpu', 0)
num_keys_values = num_layers + num_layers
num_image_token_plus = num_image_token + PROMPT_HEAD_LENGTH
num_decode = 0

# Start to run LLM
hidden_states = ort_session_C.run_with_ort_values(
    [out_name_C0],
    {
        in_name_C0: input_ids
    })[0]

if use_vision:
    print('\nStart to Process the Image...')
    start_time = time.time()
    vision_embed = ort_session_A.run_with_ort_values(
        [out_name_A0],
        {in_name_A0: pixel_values})[0]
    hidden_states = ort_session_B.run_with_ort_values(
        [out_name_B0],
        {
            in_name_B0: hidden_states,
            in_name_B1: vision_embed
        })[0]
    split_factor = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([num_image_token_plus], dtype=np.int64), 'cpu', 0)
    print(f'\nImage Process Complete. Time Cost: {(time.time() - start_time):.3f} seconds')

output_names = []
input_feed = {
    in_name_D[-1].name: hidden_states,
    in_name_D[-2].name: split_factor,
    in_name_D[-3].name: attention_mask
}
for i in range(num_layers):
    input_feed[in_name_D[i].name] = past_keys_D
    output_names.append(out_name_D[i].name)
for i in range(num_layers, num_keys_values):
    input_feed[in_name_D[i].name] = past_values_D
    output_names.append(out_name_D[i].name)
output_names.append(out_name_D[num_keys_values].name)

print('\nTest Question: ' + query + "\n\nInternVL Answering:\n")
start_time = time.time()
while num_decode < max_single_chat_length:
    max_logit_ids, *keys_values = ort_session_D.run_with_ort_values(
        output_names,
        input_feed
    )
    token_id = onnxruntime.OrtValue.numpy(max_logit_ids)
    if token_id in [2, 92542]:  # the stop_id in Intern is "2" & "92542"
        break
    else:
        for i in range(num_keys_values):
            input_feed[in_name_D[i].name] = keys_values[i]
        if num_decode < 1:
            input_feed[in_name_D[-2].name] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([PROMPT_HEAD_LENGTH], dtype=np.int64), 'cpu', 0)
            input_feed[in_name_D[-3].name] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), 'cpu', 0)
        num_decode += 1
        print(tokenizer.decode(token_id[0]), end="", flush=True)
        hidden_states = ort_session_C.run_with_ort_values(
            [out_name_C0],
            {
                in_name_C0: max_logit_ids
            })[0]
        input_feed[in_name_D[-1].name] = hidden_states
print(f"\n\nDecode: {(num_decode / (time.time() - start_time)):.3f} token/s")
