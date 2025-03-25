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
    MAX_SEQ_LENGTH = 1024                                                           # The max token length. Note, this value include the 10 tokens for system prompt and 256 tokens for image prompt. Hence, only (MAX_SEQ_LENGTH - 256 - 10) tokens for query + response.
    PROMPT_HEAD_LENGTH = 5


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
        self.position_embedding = torch.cat([self.vision_model.position_embedding[:, :1, :], self.vision_model._get_pos_embed(self.vision_model.position_embedding[:, 1:, :], 32, 32)], dim=1)
        self.mlp1 = intern_chat_model.mlp1
        self.num_image_token = num_image_token
        self.image_size = image_size
        self.h_w_half = h_w_factor // 2
        means = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).expand(1, 3, image_size, image_size)
        inv_std = torch.tensor([1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225], dtype=torch.float32).view(1, 3, 1, 1).expand(1, 3, image_size, image_size)
        self.means_inv_std = means * inv_std
        self.inv_255_std = inv_std / 255.0

    def forward(self, pixel_values):
        pixel_values = torch.nn.functional.interpolate(
            pixel_values.float(),
            (self.image_size, self.image_size),
            mode='bicubic',
            align_corners=True) * self.inv_255_std - self.means_inv_std
        vision_embed = (self.vision_model(pixel_values=pixel_values) + self.position_embedding)[:, 1:]
        return self.mlp1(vision_embed.reshape(1, self.h_w_half, 2, self.h_w_half, -1).transpose(2, 3).contiguous().reshape(1, self.num_image_token, -1)).half()


class InternVL_PartB(torch.nn.Module):
    def __init__(self, prompt_head_len, max_seq_len, num_image_token):
        super(InternVL_PartB, self).__init__()
        self.prompt_head_len = torch.tensor([prompt_head_len], dtype=torch.int64)
        self.max_seq_len_minus = torch.tensor([max_seq_len - num_image_token], dtype=torch.int64)

    def forward(self, hidden_states, vision_embed, ids_len):
        return torch.cat((hidden_states[:, :self.prompt_head_len], vision_embed, hidden_states[:, self.prompt_head_len:ids_len], hidden_states[:, (ids_len - self.max_seq_len_minus):]), dim=1)


class InternVL_PartC(torch.nn.Module):
    def __init__(self, embed_data, scale, zero_point, hidden_size, max_seq_len):
        super(InternVL_PartC, self).__init__()
        self.embed_data = embed_data
        self.scale = scale
        self.zero_point = zero_point
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

    def forward(self,
                input_ids,
                ids_len
                ):
        ids = input_ids[:, :ids_len]
        hidden_states = self.embed_data[ids] * self.scale[ids] + self.zero_point[ids]
        return torch.cat((hidden_states.half(), torch.zeros(1, self.max_seq_len - ids_len, self.hidden_size, dtype=torch.float16)), dim=1)


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
    vision_embed = torch.ones((1, num_image_token, hidden_size), dtype=torch.float16)
    ids_len = torch.tensor([10], dtype=torch.long)             # "10" is just a dummy value.
    history_len = torch.tensor([10], dtype=torch.long)         # "10" is just a dummy value.
    attention_mask = torch.tensor([-65504.0], dtype=torch.float32)
    split_factor = torch.tensor([PROMPT_HEAD_LENGTH], dtype=torch.int64)
    kv_seq_len = ids_len + history_len
    input_ids = torch.ones((1, MAX_SEQ_LENGTH), dtype=torch.int32)
    hidden_states = torch.ones((1, MAX_SEQ_LENGTH, hidden_size), dtype=torch.float16)
    past_key_states = torch.zeros((num_layers, num_key_value_heads, MAX_SEQ_LENGTH, head_dim), dtype=torch.float16)
    past_value_states = past_key_states
    position_ids = torch.zeros((MAX_SEQ_LENGTH, 1), dtype=torch.float32)
    for i in range(MAX_SEQ_LENGTH):
        position_ids[i, 0] = float(i)
    theta = 10000.0 ** -(torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    idx_theta = position_ids * theta
    cos_rotary_pos_emb = torch.cos(idx_theta)
    sin_rotary_pos_emb = torch.sin(idx_theta)
    cos_rotary_pos_emb = torch.cat((cos_rotary_pos_emb, cos_rotary_pos_emb), dim=-1).unsqueeze(0).half()
    sin_rotary_pos_emb = torch.cat((sin_rotary_pos_emb, sin_rotary_pos_emb), dim=-1).unsqueeze(0).half()
    model.base_model.language_model.register_buffer('cos_rotary_pos_emb', cos_rotary_pos_emb)
    model.base_model.language_model.register_buffer('sin_rotary_pos_emb', sin_rotary_pos_emb)

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
        do_constant_folding=True,
        opset_version=17
    )
    del model_A
    del pixel_values
    gc.collect()
    print('\nExport Part_A Done!  \n\nExport Part_B Start...')

    model_B = InternVL_PartB(PROMPT_HEAD_LENGTH, MAX_SEQ_LENGTH, num_image_token)
    torch.onnx.export(
        model_B,
        (hidden_states, vision_embed, ids_len),
        onnx_model_B,
        input_names=[
            'hidden_states',
            'vision_embed',
            'ids_len'
        ],
        output_names=['hidden_states_with_vision'],
        do_constant_folding=True,
        opset_version=17
    )
    del model_B
    del vision_embed
    print('\nExport Part_B Done! \n\nExport Part_C Start...')

    model_C = InternVL_PartC(embed_data, scale, zero_point, hidden_size, MAX_SEQ_LENGTH)
    torch.onnx.export(
        model_C,
        (input_ids, ids_len),
        onnx_model_C,
        input_names=[
            'input_ids',
            'ids_len'
        ],
        output_names=['hidden_states'],
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
    torch.onnx.export(
        model,
        (hidden_states, attention_mask, past_key_states, past_value_states, history_len, ids_len, split_factor),
        onnx_model_D,
        input_names=[
            'hidden_states',
            'attention_mask',
            'past_key_states',
            'past_value_states',
            'history_len',
            'ids_len',
            'split_factor'
        ],
        output_names=['max_logit_ids', 'past_key_states', 'past_value_states'],
        do_constant_folding=True,
        opset_version=17
    )
    del model
    del hidden_states
    del attention_mask
    del past_key_states
    del past_value_states
    del history_len
    del ids_len
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
in_name_B2 = in_name_B[2].name
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
in_name_D2 = in_name_D[2].name
in_name_D3 = in_name_D[3].name
in_name_D4 = in_name_D[4].name
in_name_D5 = in_name_D[5].name
in_name_D6 = in_name_D[6].name
out_name_D0 = out_name_D[0].name
out_name_D1 = out_name_D[1].name
out_name_D2 = out_name_D[2].name


# Pre-process inputs
if is_valid_image_path(image_path):
    image = Image.open(image_path)
    image = image.resize((INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0]))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    pixel_values = np.transpose(np.array(image).astype(np.uint8), (2, 0, 1))
    pixel_values = np.expand_dims(pixel_values, axis=0)
    use_vision = True
else:
    use_vision = False

prompt = f"<|im_start|>user\n<img></img>\n{query}<|im_end|><|im_start|>assistant\n"
token = tokenizer(prompt, return_tensors='pt')['input_ids']
ids_len = np.array([token.shape[1]], dtype=np.int64)
input_ids = np.zeros((1, MAX_SEQ_LENGTH), dtype=np.int32)
input_ids[:, :ids_len[0]] = token
attention_mask = np.array([-65504.0], dtype=np.float32)
history_len = np.array([0], dtype=np.int64)
split_factor = np.array([PROMPT_HEAD_LENGTH], dtype=np.int64)
past_key_states = np.zeros((num_layers, num_key_value_heads, MAX_SEQ_LENGTH, head_dim), dtype=np.float16)
past_values_states = past_key_states
num_image_token_plus = num_image_token + PROMPT_HEAD_LENGTH
num_decode = 0


# Start to run LLM
hidden_states = ort_session_C.run(
    [out_name_C0],
    {
        in_name_C0: input_ids,
        in_name_C1: ids_len
    })[0]

if use_vision:
    print('\nStart to Process the Image...')
    start_time = time.time()
    vision_embed = ort_session_A.run(
        [out_name_A0],
        {in_name_A0: pixel_values})[0]
    hidden_states = ort_session_B.run(
        [out_name_B0],
        {
            in_name_B0: hidden_states,
            in_name_B1: vision_embed,
            in_name_B2: ids_len
        })[0]
    ids_len += num_image_token
    split_factor = np.array([num_image_token_plus], dtype=np.int64)
    print(f'\nImage Process Complete. Time Cost: {(time.time() - start_time):.3f} seconds')

print('\nTest Question: ' + query + "\n\nInternVL Answering:\n")
start_time = time.time()
while (num_decode < max_single_chat_length) & (history_len < MAX_SEQ_LENGTH):
    token_id, past_key_states, past_values_states = ort_session_D.run(
        [out_name_D0, out_name_D1, out_name_D2],
        {
            in_name_D0: hidden_states,
            in_name_D1: attention_mask,
            in_name_D2: past_key_states,
            in_name_D3: past_values_states,
            in_name_D4: history_len,
            in_name_D5: ids_len,
            in_name_D6: split_factor
        })
    if (token_id == 2) | (token_id == 92542):  # the stop_id in Intern is "2" & "92542"
        break
    else:
        num_decode += 1
        if num_decode < 2:
            history_len += ids_len
            ids_len = np.array([1], dtype=np.int64)
            attention_mask = np.array([0.0], dtype=np.float32)
            split_factor = np.array([PROMPT_HEAD_LENGTH], dtype=np.int64)
        else:
            history_len += 1
        input_ids[:, 0] = token_id
        hidden_states = ort_session_C.run(
            [out_name_C0],
            {
                in_name_C0: input_ids,
                in_name_C1: ids_len
            })[0]
        print(tokenizer.decode(token_id), end="", flush=True)

print(f"\n\nText Generate Speed: {num_decode / (time.time() - start_time):.3f} token/s")
