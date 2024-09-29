import torch
from transformers import AutoModelForCausalLM
import shutil
import gc

model_folder_path = 'C:/Users/Downloads/Gemma2-2B-it'  # Set the folder path where the Gemma whole project downloaded.
# Replace the original "modeling_gemma2.py" with the modified "modeling_gemma2.py", which stored at the folder "modeling_modified".
modified_path_A = './modeling_modified_A/modeling_gemma2.py'  # The path where the modified modeling_gemma2.py stored.
onnx_model_A = 'C:/Users/Downloads/Gemma_ONNX/part_A/Gemma_A.onnx'  # Assign a path where the exported Gemma model stored.
transformers_gemma2_path = 'C:/Users/dake/.conda/envs/python_311/Lib/site-packages/transformers/models/gemma2/modeling_gemma2.py'  # The original modeling_gemma2.py path which was stored in the transformers python package.

# Load the model
shutil.copyfile(modified_path_A, transformers_gemma2_path)
model = AutoModelForCausalLM.from_pretrained(model_folder_path, torch_dtype=torch.float32, device_map='cpu', trust_remote_code=True).eval()
max_seq_len = 1024  # Please modify the same variable, which declared in the modified modeling_gemma2.py on line 868, at the same time.
num_heads = model.config.num_attention_heads
num_key_value_heads = model.config.num_key_value_heads
head_dim = model.config.head_dim
num_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
normalizer = torch.tensor(hidden_size ** 0.5, dtype=torch.float32)

# Generate dummies for torch.onnx.export()
input_ids = torch.ones(max_seq_len, dtype=torch.int32)
ids_len = torch.tensor([10], dtype=torch.long)  # "10" is just a dummy value.


def quantize_to_uint8(tensor, scale, zero_point):
    return ((tensor - zero_point) * scale).round().clamp(0, 255).to(torch.uint8)


model.model.embed_tokens.weight.requires_grad = False
data = (model.model.embed_tokens.weight.data) * normalizer
zero_point = (torch.min(data, dim=1)[0]).unsqueeze(1)
scale = ((torch.max(data, dim=1)[0] - zero_point[:, 0]) / 255.0).unsqueeze(1)
embed_data = quantize_to_uint8(data, 1.0 / scale, zero_point)
model.register_buffer("scale", scale)
model.register_buffer("zero_point", zero_point)
model.register_buffer("embed_data", embed_data)
del embed_data
del data
del scale
del zero_point
gc.collect()

print('Export Part_A start ...')
torch.onnx.export(
    model, (
        input_ids, ids_len),
    onnx_model_A,
    input_names=[
        'input_ids',
        'ids_len'
    ],
    output_names=['hidden_state'],
    do_constant_folding=True,
    opset_version=17)
print('Part_A Done!')

