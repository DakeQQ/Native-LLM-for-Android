import time
import torch
import numpy as np
import onnxruntime
from transformers import AutoModelForCausalLM, AutoTokenizer
import shutil

model_folder_path = 'C:/Users/Downloads/Gemma1.1-2B-it'  # set the folder path where the Gemma whole project downloaded.
modified_path_A = 'C:./modeling_modified_A/modeling_gemma.py'  # The path where store the modified part_A modeling_gemma.py
transformers_gemma_path = 'C:/Users/dake/.conda/envs/python_311/Lib/site-packages/transformers/models/gemma/modeling_gemma.py'  # The original modeling_gemma.py path which was stored in the transformers python package.
onnx_model_A = 'C:/Users/Downloads/Gemma_ONNX/Gemma_part_A.onnx'  # Assign a path where the exported Gemma_part_A model stored.

# Load the model
shutil.copyfile(modified_path_A, transformers_gemma_path)
model = AutoModelForCausalLM.from_pretrained(model_folder_path, torch_dtype=torch.float32, device_map='cpu', trust_remote_code=True).float().eval()
max_seq_len = 1024  # Please modify the same variable, which declared in the modified modeling_gemma.py on line 937, at the same time.
num_heads = model.config.num_attention_heads
num_key_value_heads = model.config.num_key_value_heads
head_dim = model.config.hidden_size // num_heads
num_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size

# Generate dummies for torch.onnx.export()
input_ids = torch.ones((1, max_seq_len), dtype=torch.int32)
ids_len = torch.zeros(1, dtype=torch.long) + 10  # "10" is just a dummy value.

print('Export Part_A start ...')
torch.onnx.export(
    model, (input_ids, ids_len),
    onnx_model_A,
    input_names=[
        'input_ids',
        'ids_len'
    ],
    output_names=['hidden_states_B'],
    do_constant_folding=True,
    opset_version=17)
print('Export Part_A done!')

