import shutil
import gc
import torch

try:
    from export_config import INPUT_IMAGE_SIZE, IMAGE_RESIZE, MAX_SEQ_LENGTH, WIDTH_FACTOR, HEIGHT_FACTOR
except:
    # Default Values if import failed
    INPUT_IMAGE_SIZE = [960, 960]                                                                  # Input image shape. Should be a multiple of GPU group (e.g., 16) for optimal efficiency.
    HEIGHT_FACTOR = 10                                                                             # Adjust this value to determine the resize shape and vision resolution.
    WIDTH_FACTOR = 10                                                                              # Adjust this value to determine the resize shape and vision resolution.
    IMAGE_RESIZE = [HEIGHT_FACTOR * 28, WIDTH_FACTOR * 28]                                         # 28 = self.patch_size * self.merge_size
    MAX_SEQ_LENGTH = 1024                                                                          # The max token length. Note, this value include the 10 tokens for system prompt and (HEIGHT_FACTOR * WIDTH_FACTOR) tokens for image prompt. Hence, only (MAX_SEQ_LENGTH - (HEIGHT_FACTOR * WIDTH_FACTOR) - 10) tokens for query + response.


path = r'/home/dake/Downloads/Qwen2-VL-2B-Instruct/'                                               # Set the folder path where the Qwen2-VL whole project downloaded.
# Replace the original "modeling_qwen2_vl.py" with the modified "modeling_qwen2_vl.py", which stored at the folder "modeling_modified".
modified_path_A = r'/home/dake/Downloads/QwenVL/modeling_modified/part_ABCD/modeling_qwen2_vl.py'  # The path where the modified modeling_qwen2_vl.py stored.
onnx_model_A = r'/home/dake/Downloads/Qwen/QwenVL_A.onnx'                                          # Assign a path where the exported QwenVL model stored.
onnx_model_B = r'/home/dake/Downloads/Qwen/QwenVL_B.onnx'
onnx_model_C = r'/home/dake/Downloads/Qwen/QwenVL_C.onnx'
onnx_model_D = r'/home/dake/Downloads/Qwen/QwenVL_D.onnx'
transformers_qwen2_path = r'/home/dake/anaconda3/envs/python_311b/lib/python3.11/site-packages/transformers/models/qwen2_vl/modeling_qwen2_vl.py'  # The original modeling_qwen2_vl.py path which was stored in the transformers python package.

shutil.copyfile(modified_path_A, transformers_qwen2_path)
shutil.copyfile("export_config.py", transformers_qwen2_path.replace("modeling_qwen2_vl", "export_config"))

from transformers import Qwen2VLForConditionalGeneration


def quantize_to_uint8(tensor, scale, zero_point):
    return ((tensor - zero_point) * scale).round().clamp(0, 255).to(torch.uint8)


class QwenVL_PartB(torch.nn.Module):
    def __init__(self, embed_data, scale, zero_point, hidden_size, image_token_id):
        super(QwenVL_PartB, self).__init__()
        self.embed_data = embed_data
        self.scale = scale
        self.zero_point = zero_point
        self.hidden_size = hidden_size
        self.image_token_id = image_token_id

    def forward(self, input_ids, kv_seq_len):
        return self.embed_data[input_ids] * self.scale[input_ids] + self.zero_point[input_ids], torch.zeros((1, 1, kv_seq_len), dtype=torch.float32)


class QwenVL_PartC(torch.nn.Module):
    def __init__(self, max_seq_len):
        super(QwenVL_PartC, self).__init__()
        self.position_ids = torch.arange(0, max_seq_len, dtype=torch.int16).repeat(3, 1, 1)
        self.position_ids_2 = torch.ones((3, 1, 1), dtype=torch.float32)

    def forward(self, ids_len, kv_seq_len):
        return self.position_ids[:, :, :ids_len].float(), self.position_ids_2 + ids_len,  (1 - torch.tril(torch.ones([1, ids_len, kv_seq_len], dtype=torch.float32))) * -65504.0


class QwenVL_PartD(torch.nn.Module):
    def __init__(self, width_factor, height_factor, prompt_head_len, max_seq_len):
        super(QwenVL_PartD, self).__init__()
        self.image_factor = width_factor * height_factor
        self.prompt_head_len = prompt_head_len
        self.max_seq_len = max_seq_len
        self.image_factor_plus = self.image_factor + self.prompt_head_len
        self.position_ids = torch.arange(0, self.max_seq_len, dtype=torch.int16).repeat(3, 1, 1)
        self.position_ids_2 = torch.ones((3, 1, 1), dtype=torch.float32) - self.image_factor + width_factor
        self.position_ids[0, :, self.prompt_head_len: self.image_factor_plus] = self.prompt_head_len
        j = self.prompt_head_len
        for i in range(self.prompt_head_len, self.image_factor_plus, width_factor):
            self.position_ids[1, :, i: i + width_factor] = j
            j += 1
        self.start_id = self.prompt_head_len + width_factor
        fill_id = torch.arange(self.prompt_head_len, self.start_id, dtype=torch.int16)
        for i in range(self.start_id, self.image_factor_plus, width_factor):
            self.position_ids[2, :, i: i + width_factor] = fill_id
        self.fill_tail_position = torch.arange(self.start_id, self.max_seq_len, dtype=torch.int16).repeat(3, 1, 1)

    def forward(self, hidden_states, image_embed, ids_len):
        part_1, part_2 = torch.split(hidden_states, [self.prompt_head_len, hidden_states.shape[0] - self.prompt_head_len], dim=0)
        position_ids = self.position_ids[:, :, :ids_len]
        position_ids[:, :, self.image_factor_plus:] = self.fill_tail_position[:, :, :ids_len - self.image_factor_plus]
        return torch.cat((part_1, image_embed, part_2), dim=0), position_ids.float(), self.position_ids_2 + ids_len


# Load the model
with torch.inference_mode():
    model = Qwen2VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True)
    max_seq_len = MAX_SEQ_LENGTH  # Please modify the same variable, which declared in the modified modeling_qwen2.py on line 1001, at the same time.
    num_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // num_heads
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    image_token_id = model.config.image_token_id

    prompt_head_len = 5  # \n<|im_start|>user\n<|vision_start|>
    ids_len = torch.tensor([10], dtype=torch.long)  # "10" is just a dummy value.
    history_len = torch.tensor([10], dtype=torch.long)  # "10" is just a dummy value.
    image_embed_size = WIDTH_FACTOR * HEIGHT_FACTOR
    image_embed = torch.ones((image_embed_size, hidden_size), dtype=torch.float32)
    pixel_values = torch.ones([1, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]]).to(torch.float32)
    image_pad_len = torch.tensor([image_embed_size], dtype=torch.long)
    ids_len = ids_len + image_pad_len
    kv_seq_len = ids_len + history_len
    input_ids = torch.ones(ids_len, dtype=torch.int32)
    hidden_states = torch.ones((ids_len, hidden_size), dtype=torch.float32)

    model.model.embed_tokens.weight.requires_grad = False
    data = model.model.embed_tokens.weight.data
    zero_point = (torch.min(data, dim=1)[0]).unsqueeze(1)
    scale = ((torch.max(data, dim=1)[0] - zero_point[:, 0]) / 255.0).unsqueeze(1)
    embed_data = quantize_to_uint8(data, 1.0 / scale, zero_point)

    print('\nExport Part_A Start...')
    torch.onnx.export(
        model,
        (pixel_values,),
        onnx_model_A,
        input_names=[
            'pixel_values'
        ],
        output_names=['image_embed'],
        do_constant_folding=True,
        opset_version=17
    )
    del model
    del pixel_values
    gc.collect()
    print('\nExport Part_A Done!  \n\nExport Part_B Start...')

    model = QwenVL_PartB(embed_data, scale, zero_point, hidden_size, image_token_id)
    torch.onnx.export(
        model,
        (input_ids, kv_seq_len),
        onnx_model_B,
        input_names=[
            'input_ids',
            'kv_seq_len'
        ],
        output_names=['hidden_states', 'attention_mask'],
        dynamic_axes={
            'input_ids': {0: 'ids_len'},
            'hidden_states': {0: 'ids_len'},
            'attention_mask': {2: 'kv_seq_len'}
        },
        do_constant_folding=True,
        opset_version=17
    )
    del model
    del embed_data
    del data
    del scale
    del zero_point
    del input_ids
    gc.collect()
    print('\nExport Part_B Done! \n\nExport Part_C Start...')

    model = QwenVL_PartC(max_seq_len)
    torch.onnx.export(
        model,
        (ids_len, kv_seq_len),
        onnx_model_C,
        input_names=[
            'ids_len',
            'kv_seq_len'
        ],
        output_names=['position_ids', 'position_ids_2', 'attention_mask'],
        dynamic_axes={
            'position_ids': {2: 'ids_len'},
            'attention_mask': {1: 'ids_len', 2: 'kv_seq_len'}
        },
        do_constant_folding=True,
        opset_version=17
    )
    del model
    del kv_seq_len
    print('\nExport Part_C Done! \n\nExport Part_C Start...')

    model = QwenVL_PartD(WIDTH_FACTOR, HEIGHT_FACTOR, prompt_head_len, max_seq_len)
    torch.onnx.export(
        model,
        (hidden_states, image_embed, ids_len),
        onnx_model_D,
        input_names=[
            'hidden_states',
            'image_embed',
            'ids_len'
        ],
        output_names=['hidden_states_out', 'position_ids', 'position_ids_2'],
        dynamic_axes={
            'hidden_states': {0: 'ids_len'},
            'hidden_states_out': {0: 'ids_len'},
            'position_ids': {2: 'ids_len'}
        },
        do_constant_folding=True,
        opset_version=17
    )
    del model
    del hidden_states
    del image_embed
    gc.collect()
    print('\nExport Part_D Done! \n\nNext, please execute the QwenVL_Export_E.py to export the last part and run the test.')

