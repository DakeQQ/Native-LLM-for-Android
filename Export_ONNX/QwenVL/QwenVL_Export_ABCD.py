import shutil
import gc
import torch
import site

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
onnx_model_A = r'/home/dake/Downloads/Qwen/QwenVL_A.onnx'                                          # Assign a path where the exported QwenVL model stored.
onnx_model_B = r'/home/dake/Downloads/Qwen/QwenVL_B.onnx'
onnx_model_C = r'/home/dake/Downloads/Qwen/QwenVL_C.onnx'
onnx_model_D = r'/home/dake/Downloads/Qwen/QwenVL_D.onnx'

python_package_path = site.getsitepackages()[-1]
shutil.copyfile("./modeling_modified/part_ABCD/modeling_qwen2_vl.py", python_package_path + "/transformers/models/qwen2_vl/modeling_qwen2_vl.py")
shutil.copyfile("export_config.py", python_package_path + "/transformers/models/qwen2_vl/export_config.py")

from transformers import Qwen2VLForConditionalGeneration


def quantize_to_uint8(tensor, scale, zero_point):
    return ((tensor - zero_point) * scale).round().clamp(0, 255).to(torch.uint8)


class QwenVL_PartB(torch.nn.Module):
    def __init__(self, embed_data, scale, zero_point, hidden_size, max_seq_len):
        super(QwenVL_PartB, self).__init__()
        self.embed_data = embed_data
        self.scale = scale.half()
        self.zero_point = zero_point.half()
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

    def forward(self, input_ids, ids_len):
        ids = input_ids[:ids_len]
        hidden_states = self.embed_data[ids] * self.scale[ids] + self.zero_point[ids]
        return torch.cat((hidden_states, torch.zeros(self.max_seq_len - ids_len, self.hidden_size, dtype=torch.float16)), dim=0)


class QwenVL_PartC(torch.nn.Module):
    def __init__(self, max_seq_len):
        super(QwenVL_PartC, self).__init__()
        self.position_ids = torch.arange(0, max_seq_len, dtype=torch.float16).repeat(1, 3, 1, 1)

    def forward(self, dummy):
        return self.position_ids[dummy]


class QwenVL_PartD(torch.nn.Module):
    def __init__(self, width_factor, height_factor, prompt_head_len, max_seq_len):
        super(QwenVL_PartD, self).__init__()
        self.image_factor = width_factor * height_factor
        self.prompt_head_len = prompt_head_len
        self.max_seq_len = max_seq_len
        self.image_factor_plus = self.image_factor + self.prompt_head_len
        self.position_ids = torch.arange(0, self.max_seq_len, dtype=torch.float16).repeat(3, 1, 1)
        self.position_ids[0, :, self.prompt_head_len: self.image_factor_plus] = self.prompt_head_len
        j = self.prompt_head_len
        for i in range(self.prompt_head_len, self.image_factor_plus, width_factor):
            self.position_ids[1, :, i: i + width_factor] = j
            j += 1
        self.start_id = self.prompt_head_len + width_factor
        fill_id = torch.arange(self.prompt_head_len, self.start_id, dtype=torch.float16)
        for i in range(self.start_id, self.image_factor_plus, width_factor):
            self.position_ids[2, :, i: i + width_factor] = fill_id
        self.fill_tail_position = torch.arange(self.start_id, self.max_seq_len, dtype=torch.float16).repeat(3, 1, 1)

    def forward(self, hidden_states, image_embed, ids_len, ids_len_minus, split_factor):
        part_1, part_2, _, part_3 = torch.split(hidden_states, [self.prompt_head_len, ids_len_minus, self.image_factor, split_factor], dim=0)
        self.position_ids[:, :, self.image_factor_plus:ids_len] = self.fill_tail_position[:, :, :ids_len - self.image_factor_plus]
        return torch.cat((part_1, image_embed, part_2, part_3), dim=0), self.position_ids


# Load the model
with torch.inference_mode():
    model = Qwen2VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True).eval()
    max_seq_len = MAX_SEQ_LENGTH
    num_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // num_heads
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    prompt_head_len = 5  # \n<|im_start|>user\n<|vision_start|>
    ids_len = torch.tensor([10], dtype=torch.long)  # "10" is just a dummy value.
    history_len = torch.tensor([10], dtype=torch.long)  # "10" is just a dummy value.
    image_embed_size = WIDTH_FACTOR * HEIGHT_FACTOR
    image_embed = torch.ones((image_embed_size, hidden_size), dtype=torch.float16)
    pixel_values = torch.ones([1, 3, INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]]).to(torch.float32)
    image_pad_len = torch.tensor([image_embed_size], dtype=torch.long)
    input_ids = torch.ones(max_seq_len, dtype=torch.int32)
    hidden_states = torch.ones((max_seq_len, hidden_size), dtype=torch.float16)
    ids_len_minus = torch.tensor(ids_len[0] - prompt_head_len, dtype=torch.int32)
    ids_len = ids_len + image_pad_len
    kv_seq_len = ids_len + history_len
    split_factor = torch.tensor(max_seq_len - ids_len[0], dtype=torch.int32)
    dummy = torch.tensor(0, dtype=torch.int32)

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

    model = QwenVL_PartB(embed_data, scale, zero_point, hidden_size, max_seq_len)
    torch.onnx.export(
        model,
        (input_ids, ids_len),
        onnx_model_B,
        input_names=[
            'input_ids',
            'ids_len'
        ],
        output_names=['hidden_states'],
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
        (dummy,),
        onnx_model_C,
        input_names=[
            'dummy'
        ],
        output_names=['position_ids'],
        do_constant_folding=True,
        opset_version=17
    )
    del model
    del dummy
    print('\nExport Part_C Done! \n\nExport Part_C Start...')

    model = QwenVL_PartD(WIDTH_FACTOR, HEIGHT_FACTOR, prompt_head_len, max_seq_len)
    torch.onnx.export(
        model,
        (hidden_states, image_embed, ids_len, ids_len_minus, split_factor),
        onnx_model_D,
        input_names=[
            'hidden_states',
            'image_embed',
            'ids_len',
            'ids_len_minus',
            'split_factor'
        ],
        output_names=['hidden_states', 'position_ids'],
        do_constant_folding=True,
        opset_version=17
    )
    print('\nExport Part_D Done! \n\nNext, please execute the QwenVL_Export_E.py to export the last part and run the test.')

