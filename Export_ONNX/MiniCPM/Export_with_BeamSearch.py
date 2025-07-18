import gc
import time
import torch
import numpy as np
import onnxruntime
from transformers import AutoModelForCausalLM, AutoTokenizer


path = '/home/DakeQQ/Downloads/MiniCPM4-0.5B'                                     # Set the folder path where the MiniCPM4 or BitCPM4 whole project downloaded.
onnx_model_A = '/home/DakeQQ/Downloads/MiniCPM_ONNX/MiniCPM.onnx'                 # Assign a path where the exported MiniCPM4 model stored.
onnx_model_B = '/home/DakeQQ/Downloads/MiniCPM_ONNX/MiniCPM_Embed.onnx'           # Assign a path where the exported MiniCPM4 model stored.
onnx_model_C = '/home/DakeQQ/Downloads/MiniCPM_ONNX/Greedy_Search.onnx'           # Assign a path where the exported MiniCPM4 model stored.
onnx_model_D = '/home/DakeQQ/Downloads/MiniCPM_ONNX/First_Beam_Search.onnx'       # Assign a path where the exported MiniCPM4 model stored.
onnx_model_E = '/home/DakeQQ/Downloads/MiniCPM_ONNX/Second_Beam_Search.onnx'      # Assign a path where the exported MiniCPM4 model stored.
onnx_model_F = '/home/DakeQQ/Downloads/MiniCPM_ONNX/Reset_Penality.onnx'          # Assign a path where the exported MiniCPM4 model stored.


STOP_TOKEN = [2, 73440]                                                         # The stop_id in MiniCPM4 is "2" & "73440"
MAX_SEQ_LEN = 4096                                                              # The max context length.
REPEAT_PENALITY = 0.95                                                          # Range from 0.0 to 1.0; "1.0" means no penality.
PENALITY_RANGE = 30                                                             # Penalizes the most recent output. "30" means the last 30 tokens.
USE_BEAM_SEARCH = True                                                          # Use beam search or greedy search.
TOP_K = 5                                                                       # The top k candidate in decoding.
BEAM_SIZE = 5                                                                   # Number of beams in searching.
test_query = "地球最高的山是哪座山？"                                               # The test query after the export process.


def quantize_to_uint8(tensor, scale, zero_point):
    return ((tensor - zero_point) * scale).round().clamp(0, 255).to(torch.uint8)


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
        pass

    def forward(self, repeat_penality, logits, penality_value):
        max_logits_idx = torch.argmax(logits * repeat_penality, dim=-1, keepdim=True)
        repeat_penality[:, max_logits_idx.squeeze(-1)] = penality_value
        return repeat_penality, max_logits_idx.int()


class FIRST_BEAM_SEARCH(torch.nn.Module):
    def __init__(self, num_layers):
        super(FIRST_BEAM_SEARCH, self).__init__()
        self.num_keys_values = num_layers + num_layers
        self.save_keys_values = [None] * self.num_keys_values
        self.batch_indices = torch.arange(255, dtype=torch.uint8)

    def forward(self, *all_inputs):
        save_id = all_inputs[-5]
        repeat_penality = all_inputs[-4]
        logits = all_inputs[-3]
        penality_value = all_inputs[-2]
        beam_size = all_inputs[-1]
        logits = torch.log_softmax(logits, dim=-1)
        top_beam_prob, top_beam_indices = torch.topk(logits, dim=-1, k=beam_size, sorted=False, largest=True)
        for i in range(self.num_keys_values):
            self.save_keys_values[i] = torch.cat([all_inputs[i] for _ in range(beam_size)], dim=0)
        top_beam_indices = top_beam_indices.transpose(0, 1)
        batch_indices = self.batch_indices[:beam_size].long()
        repeat_penality[batch_indices, top_beam_indices] *= penality_value
        top_beam_indices = top_beam_indices.int()
        save_id = torch.cat([save_id, top_beam_indices], dim=-1)
        max_logits_idx = top_beam_indices[0]
        return *self.save_keys_values, save_id, repeat_penality, top_beam_prob.transpose(0, 1), batch_indices, top_beam_indices, max_logits_idx


class SECOND_BEAM_SEARCH(torch.nn.Module):
    def __init__(self, num_layers):
        super(SECOND_BEAM_SEARCH, self).__init__()
        self.num_keys_values = num_layers + num_layers
        self.save_keys_values = [None] * self.num_keys_values
        self.batch_indices = torch.arange(255, dtype=torch.uint8)

    def forward(self, *all_inputs):
        save_id = all_inputs[-8]
        repeat_penality = all_inputs[-7]
        previous_prob = all_inputs[-6]
        batch_indices = all_inputs[-5]
        logits = all_inputs[-4]
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
        top_beam_indices = top_beam_indices.int().unsqueeze(-1)
        save_id = torch.cat([save_id[beam_index], top_beam_indices], dim=-1)
        max_logits_idx = top_beam_indices[0]
        return *self.save_keys_values, save_id, repeat_penality, top_beam_prob.unsqueeze(-1), top_beam_indices, max_logits_idx


class RESET_PENALITY(torch.nn.Module):
    def __init__(self):
        super(RESET_PENALITY, self).__init__()
        pass

    def forward(self, save_id, repeat_penality, penality_reset_count, batch_indices):
        repeat_penality[batch_indices, save_id[batch_indices, penality_reset_count[batch_indices]]] = 1.0
        penality_reset_count += 1
        return save_id, repeat_penality, penality_reset_count


class MINICPM_EMBED(torch.nn.Module):
    def __init__(self, minicpm):
        super(MINICPM_EMBED, self).__init__()
        self.minicpm = minicpm
        self.minicpm.model.embed_tokens.weight.data *= self.minicpm.model.config.scale_emb

    def forward(self, input_ids):
        return self.minicpm.model.embed_tokens(input_ids)


class MINICPM(torch.nn.Module):
    def __init__(self, minicpm, max_seq_len, num_heads, num_key_value_heads, head_dim, num_layers):
        super(MINICPM, self).__init__()
        self.minicpm = minicpm
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim_half = head_dim // 2
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.variance_epsilon = float(1e-6)

        scale_factor = float(head_dim ** -0.25)
        scale_depth_factor = float(self.minicpm.model.layers._modules['0'].scale_depth * (num_layers ** -0.5))

        for i in range(num_layers):
            self.minicpm.model.layers._modules[f'{i}'].self_attn.q_proj.weight.data *= scale_factor
            self.minicpm.model.layers._modules[f'{i}'].self_attn.k_proj.weight.data *= scale_factor
            self.minicpm.model.layers._modules[f'{i}'].self_attn.o_proj.weight.data *= scale_depth_factor
            self.minicpm.model.layers._modules[f'{i}'].mlp.down_proj.weight.data *= scale_depth_factor
        self.minicpm.model.norm.weight.data *= float(self.minicpm.config.dim_model_base / self.minicpm.config.hidden_size)

        self.cos_rotary_pos_emb = self.minicpm.model.layers._modules['0'].self_attn.rotary_emb.cos_cached[:max_seq_len].unsqueeze(0).unsqueeze(0).half()
        self.sin_rotary_pos_emb = self.minicpm.model.layers._modules['0'].self_attn.rotary_emb.sin_cached[:max_seq_len].unsqueeze(0).unsqueeze(0).half()

        self.save_key = [None] * num_layers
        self.save_value = [None] * num_layers
        self.attention_mask = (1 - torch.tril(torch.ones([1, 1, max_seq_len, max_seq_len], dtype=torch.int8))) * -128

    def forward(self, *all_inputs):
        history_len = all_inputs[-4]
        hidden_states = all_inputs[-3]
        ids_len = all_inputs[-2]
        kv_seq_len = history_len + ids_len
        rotary_pos_emb_cos_q = self.cos_rotary_pos_emb[..., history_len:kv_seq_len, :].float()
        rotary_pos_emb_sin_q = self.sin_rotary_pos_emb[..., history_len:kv_seq_len, :].float()
        rotary_pos_emb_cos_k = rotary_pos_emb_cos_q.transpose(-1, -2).unsqueeze(0)
        rotary_pos_emb_sin_k = rotary_pos_emb_sin_q.transpose(-1, -2).unsqueeze(0)
        batch_size = hidden_states.shape[0].unsqueeze(0)
        attention_mask = (self.attention_mask[..., :ids_len, :kv_seq_len] * all_inputs[-1]).float()
        for i, layer in enumerate(self.minicpm.model.layers):
            hidden_states_norm = layer.input_layernorm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
            q = layer.self_attn.q_proj(hidden_states_norm).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            k = layer.self_attn.k_proj(hidden_states_norm).view(batch_size, -1, 1, self.num_key_value_heads, self.head_dim).permute(0, 3, 2, 4, 1)
            v = layer.self_attn.v_proj(hidden_states_norm).view(batch_size, -1, 1, self.num_key_value_heads, self.head_dim).transpose(1, 3)
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
            hidden_states = layer.post_attention_layernorm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
            hidden_states = layer.mlp.down_proj(layer.mlp.act_fn(layer.mlp.gate_proj(hidden_states)) * layer.mlp.up_proj(hidden_states))
            hidden_states += residual
        hidden_states = hidden_states[:, -1]
        hidden_states = self.minicpm.model.norm.weight * (hidden_states / torch.sqrt(hidden_states.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
        logits = self.minicpm.lm_head(hidden_states)
        return *self.save_key, *self.save_value, kv_seq_len, logits


print('Export start ...')
with torch.inference_mode():
    # Load the original model
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, device_map='cpu', trust_remote_code=True, low_cpu_mem_usage=True).eval()
    head_dim = model.model.layers._modules['0'].self_attn.head_dim
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    num_key_value_heads = model.config.num_key_value_heads
    hidden_size = model.model.embed_tokens.embedding_dim
    vocab_size = model.vocab_size

    # Build an optimized model
    minicpm = MINICPM(model, MAX_SEQ_LEN, num_heads, num_key_value_heads, head_dim, num_layers)

    # Generate dummies for torch.onnx.export()
    attention_mask = torch.tensor([0], dtype=torch.int8)
    ids_len = torch.tensor([10], dtype=torch.int64)   # "10" is just a dummy value.
    hidden_states = torch.ones((3, ids_len, hidden_size), dtype=torch.float32)
    history_len = torch.zeros(1, dtype=torch.int64)
    past_keys = torch.zeros((hidden_states.shape[0], num_key_value_heads, 1, head_dim, 0), dtype=torch.float32)
    past_values = torch.zeros((hidden_states.shape[0], num_key_value_heads, 1, 0, head_dim), dtype=torch.float32)

    # Prepare input and output names
    all_inputs = []
    input_names = []
    output_names = []
    dynamic_axes = {'hidden_states': {0: 'batch', 1: 'ids_len'}}
    for i in range(num_layers):
        name = f'in_key_{i}'
        input_names.append(name)
        all_inputs.append(past_keys)
        dynamic_axes[name] = {0: 'batch', 4: 'history_len'}
        name = f'out_key_{i}'
        output_names.append(name)
        dynamic_axes[name] = {0: 'batch', 4: 'history_len_plus_ids_len'}
    for i in range(num_layers):
        name = f'in_value_{i}'
        input_names.append(name)
        all_inputs.append(past_values)
        dynamic_axes[name] = {0: 'batch', 3: 'history_len'}
        name = f'out_value_{i}'
        output_names.append(name)
        dynamic_axes[name] = {0: 'batch', 3: 'history_len_plus_ids_len'}
    input_names.append('history_len')
    all_inputs.append(history_len)
    input_names.append('hidden_states')
    all_inputs.append(hidden_states)
    input_names.append('ids_len')
    all_inputs.append(ids_len)
    input_names.append('attention_mask')
    all_inputs.append(attention_mask)
    output_names.append('kv_seq_len')
    output_names.append('logits')
    dynamic_axes['logits'] = {0: 'batch'}

    torch.onnx.export(
        minicpm,
        tuple(all_inputs),
        onnx_model_A,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=17
    )
    del hidden_states
    del ids_len
    del history_len
    del attention_mask
    del input_names
    del output_names
    del dynamic_axes
    del all_inputs
    del minicpm

    embed = MINICPM_EMBED(model)
    input_ids = torch.zeros((3, 10), dtype=torch.int32)
    torch.onnx.export(
        embed,
        (input_ids,),
        onnx_model_B,
        input_names=['input_ids'],
        output_names=['hidden_state'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'ids_len'},
            'hidden_state': {0: 'batch', 1: 'ids_len'},
        },
        do_constant_folding=True,
        opset_version=17
    )
    del embed
    del input_ids
    del model

    greedy = GREEDY_SEARCH()
    beam_size = torch.tensor([BEAM_SIZE], dtype=torch.int64)
    repeat_penality = torch.ones((beam_size, vocab_size), dtype=torch.float32)
    penality_reset_count = torch.zeros(beam_size, dtype=torch.int32)
    logits = torch.randn((beam_size, vocab_size), dtype=torch.float32)
    penality_value = torch.tensor(REPEAT_PENALITY, dtype=torch.float32)
    batch_indices = torch.arange(BEAM_SIZE, dtype=torch.int64)

    torch.onnx.export(
        greedy,
        (repeat_penality, logits, penality_value),
        onnx_model_C,
        input_names=['repeat_penality_in', 'logits', 'penality_value'],
        output_names=['repeat_penality_out', 'max_logits_idx'],
        dynamic_axes={
            'logits': {0: 'batch'},
            'repeat_penality_in': {0: 'batch'},
            'repeat_penality_out': {0: 'batch'},
            'max_logits_idx': {0: 'batch'}
        },
        do_constant_folding=True,
        opset_version=17
    )
    del greedy

    first_beam_search = FIRST_BEAM_SEARCH(num_layers)
    topK = torch.tensor([TOP_K], dtype=torch.int64)
    save_id = torch.zeros((beam_size, 10), dtype=torch.int32)
    previous_prob = torch.zeros((beam_size, 1), dtype=torch.float32)
    past_keys_greedy = past_keys[[0]]
    past_values_greedy = past_values[[0]]
    logits = torch.ones((beam_size, vocab_size), dtype=torch.float32)

    all_inputs = []
    input_names = []
    output_names = []
    dynamic_axes = {'save_id_in': {0: 'batch', 1: 'history_len'}}
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
    input_names.append('save_id_in')
    all_inputs.append(save_id)
    input_names.append('repeat_penality_in')
    all_inputs.append(repeat_penality)
    input_names.append('logits')
    all_inputs.append(logits[[0]])
    input_names.append('penality_value')
    all_inputs.append(penality_value)
    input_names.append('beam_size')
    all_inputs.append(beam_size)
    output_names.append('save_id_out')
    output_names.append('repeat_penality_out')
    output_names.append('batch_indices')
    output_names.append('top_beam_prob')
    output_names.append('top_beam_indices')
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
        onnx_model_D,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=17
    )
    del first_beam_search
    
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
    input_names.append('save_id_in')
    all_inputs.append(save_id)
    input_names.append('repeat_penality_in')
    all_inputs.append(repeat_penality)
    input_names.append('previous_prob')
    all_inputs.append(previous_prob)
    input_names.append('batch_indices')
    all_inputs.append(batch_indices)
    input_names.append('logits')
    all_inputs.append(logits)
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
        onnx_model_E,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        opset_version=17
    )

    reset_penality = RESET_PENALITY()
    torch.onnx.export(
        reset_penality,
        (save_id, repeat_penality, penality_reset_count, batch_indices),
        onnx_model_F,
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
        opset_version=17
    )

    del batch_indices
    del reset_penality
    del second_beam_search
    del past_keys_greedy
    del past_values_greedy
    del logits
    del previous_prob
    del save_id
    del repeat_penality
    del penality_reset_count
    del topK
    del input_names
    del output_names
    del dynamic_axes
    del all_inputs
    gc.collect()

print('\nExport done!\n\nStart running the MiniCPM by ONNXRuntime.\nNow loading . . . it could cost minutes.')

# Run the exported model by ONNX Runtime
max_single_chat_length = MAX_SEQ_LEN                  # It an adjustable value, but must less than max_seq_len.
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
stop_words = []
for i in STOP_TOKEN:
    stop_words.append(tokenizer.decode(i))

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

ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
amount_of_outputs_A = len(out_name_A)
num_layers = (amount_of_outputs_A - 2) // 2
in_name_A = [in_name_A[i].name for i in range(len(in_name_A))]
out_name_A = [out_name_A[i].name for i in range(amount_of_outputs_A)]
amount_of_outputs_A -= 1

ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_B = ort_session_B.get_inputs()[0].name
out_name_B = ort_session_B.get_outputs()[0].name

ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_C = ort_session_C.get_inputs()
out_name_C = ort_session_C.get_outputs()
in_name_C = [in_name_C[i].name for i in range(len(in_name_C))]
out_name_C = [out_name_C[i].name for i in range(len(out_name_C))]

ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_D = ort_session_D.get_inputs()
out_name_D = ort_session_D.get_outputs()
in_name_D = [in_name_D[i].name for i in range(len(in_name_D))]
out_name_D = [out_name_D[i].name for i in range(len(out_name_D))]

ort_session_E = onnxruntime.InferenceSession(onnx_model_E, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_E = ort_session_E.get_inputs()
out_name_E = ort_session_E.get_outputs()
in_name_E = [in_name_E[i].name for i in range(len(in_name_E))]
out_name_E = [out_name_E[i].name for i in range(len(out_name_E))]

ort_session_F = onnxruntime.InferenceSession(onnx_model_F, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_F = ort_session_F.get_inputs()
out_name_F = ort_session_F.get_outputs()
in_name_F = [in_name_F[i].name for i in range(len(in_name_F))]
out_name_F = [out_name_F[i].name for i in range(len(out_name_F))]


# Pre-process inputs
if USE_BEAM_SEARCH and (TOP_K < BEAM_SIZE):
    TOP_K = BEAM_SIZE

if (TOP_K < 2) or (BEAM_SIZE < 2):
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")

if not USE_BEAM_SEARCH:
    TOP_K = 1
    BEAM_SIZE = 1

if REPEAT_PENALITY != 1.0:
    do_repeat_penality = True
else:
    do_repeat_penality = False
    

prompt = f'<|im_start|>user\n{test_query}<|im_end|>\n<|im_start|>assistant\n'
tokens = tokenizer(prompt, return_tensors='np')['input_ids'].astype(np.int32)
batch_size = tokens.shape[0]
topK = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([TOP_K], dtype=np.int64), 'cpu', 0)
beam_size = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([BEAM_SIZE], dtype=np.int64), 'cpu', 0)
input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, 'cpu', 0)
ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([tokens.shape[-1]], dtype=np.int64), 'cpu', 0)
history_len = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int64), 'cpu', 0)
attention_mask = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), 'cpu', 0)
past_keys_A = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((batch_size, num_key_value_heads, 1, head_dim, 0), dtype=np.float32), 'cpu', 0)
past_values_A = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((batch_size, num_key_value_heads, 1, 0, head_dim), dtype=np.float32), 'cpu', 0)
save_id = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((BEAM_SIZE, 0), dtype=np.int32), 'cpu', 0)
repeat_penality = onnxruntime.OrtValue.ortvalue_from_numpy(np.ones((BEAM_SIZE, vocab_size), dtype=np.float32), 'cpu', 0)
penality_value = onnxruntime.OrtValue.ortvalue_from_numpy(np.array(REPEAT_PENALITY, dtype=np.float32), 'cpu', 0)

if do_repeat_penality:
    penality_reset_count = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(BEAM_SIZE, dtype=np.int32), 'cpu', 0)
    if not USE_BEAM_SEARCH:
        save_id_greedy = np.zeros(MAX_SEQ_LEN, dtype=np.int32)
        penality_reset_count_greedy = 0

num_keys_values = num_layers + num_layers
num_keys_values_plus_3 = num_keys_values + 3
num_decode = 0
print(f'\n\nTest Question: {test_query}\nMiniCPM Answering:\n')

if USE_BEAM_SEARCH:
    print("(Beam Search does not display the immediate decoding results; the best result is shown only after the entire decoding process is complete.)\n")

hidden_states = ort_session_B.run_with_ort_values([out_name_B], {in_name_B: input_ids})[0]

input_feed_A = {
    in_name_A[-4]: history_len,
    in_name_A[-3]: hidden_states,
    in_name_A[-2]: ids_len,
    in_name_A[-1]: attention_mask
}

for i in range(num_layers):
    input_feed_A[in_name_A[i]] = past_keys_A
for i in range(num_layers, num_keys_values):
    input_feed_A[in_name_A[i]] = past_values_A

input_feed_C = {
    in_name_C[0]: repeat_penality,
    in_name_C[2]: penality_value
}

input_feed_D = {
    in_name_D[-5]: save_id,
    in_name_D[-4]: repeat_penality,
    in_name_D[-2]: penality_value,
    in_name_D[-1]: beam_size
}

input_feed_E = {
    in_name_E[-3]: penality_value,
    in_name_E[-2]: beam_size,
    in_name_E[-1]: topK
}

# Start to run LLM
start_time = time.time()
while num_decode < max_single_chat_length:
    all_outputs_A = ort_session_A.run_with_ort_values(out_name_A, input_feed_A)
    if USE_BEAM_SEARCH:
        if num_decode < 1:
            for i in range(num_keys_values):
                input_feed_D[in_name_D[i]] = all_outputs_A[i]
            input_feed_D[in_name_D[-3]] = all_outputs_A[-1]
            all_outputs_D = ort_session_D.run_with_ort_values(out_name_D, input_feed_D)
            max_logits_idx = onnxruntime.OrtValue.numpy(all_outputs_D[-1])
            input_feed_F = {in_name_F[3]: all_outputs_D[-3]}
            input_feed_E[in_name_E[num_keys_values_plus_3]] = all_outputs_D[-3]
        else:
            input_feed_E[in_name_E[-4]] = all_outputs_A[-1]
            for i in range(num_keys_values):
                input_feed_E[in_name_E[i]] = all_outputs_A[i]
            all_outputs_E = ort_session_E.run_with_ort_values(out_name_E, input_feed_E)
            max_logits_idx = onnxruntime.OrtValue.numpy(all_outputs_E[-1])
        if max_logits_idx in STOP_TOKEN:
            save_id = onnxruntime.OrtValue.numpy(all_outputs_E[num_keys_values])[0]
            for i in range(num_decode, -1, -1):
                if save_id[i] not in STOP_TOKEN:
                    sentence = tokenizer.decode(save_id[:i+1])
                    break
            print(f"\nBest: {sentence}")
            break
        if do_repeat_penality and (num_decode >= PENALITY_RANGE):
            input_feed_F[in_name_F[0]] = all_outputs_E[num_keys_values]
            input_feed_F[in_name_F[1]] = all_outputs_E[-4]
            input_feed_F[in_name_F[2]] = penality_reset_count
            all_outputs_F = ort_session_F.run_with_ort_values(out_name_F, input_feed_F)
            input_feed_E[in_name_E[num_keys_values]] = all_outputs_F[0]
            input_feed_E[in_name_E[-4]] = all_outputs_F[1]
            penality_reset_count = all_outputs_F[2]
        if num_decode < 1:
            for i in range(num_keys_values):
                input_feed_A[in_name_A[i]] = all_outputs_D[i]
            for i in range(num_keys_values, num_keys_values_plus_3):
                input_feed_E[in_name_E[i]] = all_outputs_D[i]
            input_feed_A[in_name_A[-3]] = ort_session_B.run_with_ort_values([out_name_B], {in_name_B: all_outputs_D[-2]})[0]
        else:
            for i in range(num_keys_values):
                input_feed_A[in_name_A[i]] = all_outputs_E[i]
            for i in range(num_keys_values, num_keys_values_plus_3):
                input_feed_E[in_name_E[i]] = all_outputs_E[i]
            input_feed_A[in_name_A[-3]] = ort_session_B.run_with_ort_values([out_name_B], {in_name_B: all_outputs_E[-2]})[0]
        for i in range(num_keys_values, amount_of_outputs_A):
            input_feed_A[in_name_A[i]] = all_outputs_A[i]
    else:
        input_feed_C[in_name_C[1]] = all_outputs_A[-1]
        all_outputs_C = ort_session_C.run_with_ort_values(out_name_C, input_feed_C)
        max_logits_idx = onnxruntime.OrtValue.numpy(all_outputs_C[-1])[0][0]
        if max_logits_idx in STOP_TOKEN:
            break
        input_feed_A[in_name_A[-3]] = ort_session_B.run_with_ort_values([out_name_B], {in_name_B: all_outputs_C[-1]})[0]
        if do_repeat_penality and (num_decode >= PENALITY_RANGE) and (save_id_greedy[penality_reset_count_greedy] != max_logits_idx):
            repeat_penality = onnxruntime.OrtValue.numpy(all_outputs_C[0])
            repeat_penality[:, penality_reset_count_greedy] = 1.0
            penality_reset_count_greedy += 1
        input_feed_C[in_name_C[0]] = all_outputs_C[0]
        save_id_greedy[num_decode] = max_logits_idx
        for i in range(amount_of_outputs_A):
            input_feed_A[in_name_A[i]] = all_outputs_A[i]
        print(tokenizer.decode(max_logits_idx), end="", flush=True)
    if num_decode < 1:
        input_feed_A[in_name_A[-1]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), 'cpu', 0)
        input_feed_A[in_name_A[-2]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), 'cpu', 0)
    num_decode += 1

print(f"\n\nDecode: {((num_decode + 1) / (time.time() - start_time)):.3f} token/s")
