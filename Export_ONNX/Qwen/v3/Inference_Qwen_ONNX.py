import argparse
import time
from pathlib import Path
import numpy as np
import onnxruntime
from onnxruntime.capi import _pybind_state as C
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Qwen split-ONNX inference demo.")
    parser.add_argument(
        "--model-folder",
        type=Path,
        default=Path(__file__).resolve().parent / "Qwen_Optimized",
        help="Folder containing the split ONNX graphs. Defaults to Qwen_Optimized; use Qwen_ONNX for freshly exported graphs.",
    )
    parser.add_argument(
        "--tokenizer-folder",
        type=Path,
        default=Path(r"/home/iamj/Downloads/Qwen3-0.6B"),
        help="HF checkpoint/tokenizer folder used to tokenize the demo prompt.",
    )
    return parser.parse_args()


args = parse_args()


download_path                  = str(args.tokenizer_folder.expanduser().resolve())
onnx_folder                    = args.model_folder.expanduser().resolve()
onnx_model_Metadata            = str(onnx_folder / "LLM_Metadata.onnx")
onnx_model_Embed               = str(onnx_folder / "LLM_Embed.onnx")
onnx_model_Main                = str(onnx_folder / "LLM_Main.onnx")
onnx_model_Rotary_Text_Prefill = str(onnx_folder / "LLM_RotaryPrefill.onnx")
onnx_model_Rotary_Text_Decode  = str(onnx_folder / "LLM_RotaryDecode.onnx")
onnx_model_Greedy              = str(onnx_folder / "LLM_Greedy.onnx")
onnx_model_First_Beam          = str(onnx_folder / "LLM_FirstBeam.onnx")
onnx_model_Second_Beam         = str(onnx_folder / "LLM_SecondBeam.onnx")
onnx_model_Penalty             = str(onnx_folder / "LLM_Penalty.onnx")
onnx_model_Argmax              = str(onnx_folder / "LLM_Argmax.onnx")

TEST_THINK_MODE          = False
TEST_QUERY               = "地球最高的山峰是什么？"

USE_BEAM_SEARCH          = False
REPEAT_PENALTY           = 1.0                     # 0.0 ~ 1.0; No penalty = 1.0
PENALTY_RANGE            = 20
TOP_K                    = 3
BEAM_SIZE                = 3                       # Must be <= export MAX_BEAM_SIZE.

ORT_LOG                  = False
ORT_FP16                 = False                   # CPU FP16 requires ARM64-v8.2a+.
ORT_Accelerate_Providers = []                      # ORT execution providers; ['CUDAExecutionProvider', 'DmlExecutionProvider', 'OpenVINOExecutionProvider']
MAX_THREADS              = 0                       # auto
DEVICE_ID                = 0


def bind_ort_in_buf(binding, names, values):
    for name, val in zip(names, values):
        binding.bind_ortvalue_input(name, val)


def bind_ort_out_buf(binding, names, values):
    for name, val in zip(names, values):
        binding.bind_ortvalue_output(name, val)


def bind_ort_out(binding, names, device):
    for name in names:
        binding._iobinding.bind_output(name, device)


def create_ort_with_data(data, dtype, device, device_id):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.array(data, dtype=dtype), device, device_id)


def create_ort_with_shape(shape, dtype, device, device_id):
    return onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros(shape, dtype=dtype), device, device_id)


def create_session(model_path, _session_opts, _providers, _provider_options, _disabled_optimizers):
    return onnxruntime.InferenceSession(
        model_path,
        sess_options=_session_opts,
        providers=_providers,
        provider_options=_provider_options,
        disabled_optimizers=_disabled_optimizers)


def get_in_names(session):
    return [x.name for x in session.get_inputs()]


def get_out_names(session):
    return [x.name for x in session.get_outputs()]


def run(session, binding):
    session.run_with_iobinding(binding, run_options=run_options)


def load_metadata_carrier(model_path):
    meta_opts = onnxruntime.SessionOptions()
    meta_opts.log_severity_level = 0 if ORT_LOG else 4
    meta_opts.log_verbosity_level = 4
    meta_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    try:
        meta_session = onnxruntime.InferenceSession(
            model_path,
            sess_options=meta_opts,
            providers=['CPUExecutionProvider'],
        )
    except Exception as exc:
        raise RuntimeError(
            "LLM_Metadata.onnx is required to preload Native-LLM metadata. "
            "Re-export with Export_Qwen.py or copy the metadata carrier into the model folder."
        ) from exc

    metadata = meta_session.get_modelmeta().custom_metadata_map
    if not metadata.get('native_llm_metadata_version'):
        raise RuntimeError(
            "LLM_Metadata.onnx carries no Native-LLM metadata_props. Re-export the model with Export_Qwen.py."
        )
    return metadata


# Read metadata before creating real sessions so ORT_FP16/session settings follow the exported graphs.
model_meta = load_metadata_carrier(onnx_model_Metadata)
ORT_FP16 = model_meta.get('activations_fp16') == '1'


session_opts = onnxruntime.SessionOptions()
run_options  = onnxruntime.RunOptions()

for opt in (session_opts, run_options):
    opt.log_severity_level  = 0 if ORT_LOG else 4
    opt.log_verbosity_level = 4

session_opts.inter_op_num_threads     = MAX_THREADS
session_opts.intra_op_num_threads     = MAX_THREADS
session_opts.execution_mode           = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

_session_configs = {
    'session.set_denormal_as_zero':                  '1',
    'session.intra_op.allow_spinning':               '1',
    'session.inter_op.allow_spinning':               '1',
    'session.enable_quant_qdq_cleanup':              '1',
    'session.qdq_matmulnbits_accuracy_level':        '2' if ORT_FP16 else '4',
    'session.use_device_allocator_for_initializers': '1',
    'session.graph_optimizations_loop_level':        '2',
    'optimization.enable_gelu_approximation':        '1',
    'optimization.minimal_build_optimizations':      '',
    'optimization.enable_cast_chain_elimination':    '1',
    'optimization.disable_specified_optimizers':
        'CastFloat16Transformer;FuseFp16InitializerToFp32NodeTransformer' if ORT_FP16 else ''
}
for k, v in _session_configs.items():
    session_opts.add_session_config_entry(k, v)

run_options.add_run_config_entry('disable_synchronize_execution_providers', '0')

disabled_optimizers = ['CastFloat16Transformer', 'FuseFp16InitializerToFp32NodeTransformer'] if ORT_FP16 else None


if "OpenVINOExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_type':              'CPU',
        'precision':                'ACCURACY',
        'num_of_threads':           MAX_THREADS if MAX_THREADS != 0 else 8,
        'num_streams':              1,
        'enable_opencl_throttling': False,
        'enable_qdq_optimizer':     False,
        'disable_dynamic_shapes':   False
    }]
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

elif "CUDAExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                          DEVICE_ID,
        'gpu_mem_limit':                      24 * (1024 **3),
        'arena_extend_strategy':              'kNextPowerOfTwo',
        'cudnn_conv_algo_search':             'EXHAUSTIVE',
        'sdpa_kernel':                        '2',
        'use_tf32':                           '1',
        'fuse_conv_bias':                     '0',
        'cudnn_conv_use_max_workspace':       '1',
        'cudnn_conv1d_pad_to_nc1d':           '0',
        'tunable_op_enable':                  '0',
        'tunable_op_tuning_enable':           '0',
        'tunable_op_max_tuning_duration_ms':  10,
        'do_copy_in_default_stream':          '0',
        'enable_cuda_graph':                  '0',
        'prefer_nhwc':                        '0',
        'enable_skip_layer_norm_strict_mode': '0',
        'use_ep_level_unified_stream':        '0'
    }]
    device_type      = 'cuda'
    _ort_device_type = C.OrtDevice.cuda()

elif "DmlExecutionProvider" in ORT_Accelerate_Providers:
    provider_options = [{
        'device_id':                  DEVICE_ID,
        'performance_preference':     'high_performance',
        'device_filter':              'gpu',
        'disable_metacommands':       'false',
        'enable_graph_capture':       'false',
        'enable_graph_serialization': 'false'
    }]
    device_type      = 'dml'
    _ort_device_type = C.OrtDevice.dml()

else:
    provider_options = None
    device_type      = 'cpu'
    _ort_device_type = C.OrtDevice.cpu()

packed_settings = {
    "_session_opts":        session_opts,
    "_providers":           ORT_Accelerate_Providers,
    "_provider_options":    provider_options,
    "_disabled_optimizers": disabled_optimizers
}

_ort_device_type = C.OrtDevice(_ort_device_type, C.OrtDevice.default_memory(), DEVICE_ID)
kv_device = 'cpu' if 'dml' in device_type else device_type

print(
    '\nStart running the LLM by ONNXRuntime.\n'
    'Now loading . . . it could cost minutes.'
)


ort_session_Embed = create_session(onnx_model_Embed, **packed_settings)
binding_Embed     = ort_session_Embed.io_binding()
in_name_Embed     = get_in_names(ort_session_Embed)[0]
out_name_Embed    = get_out_names(ort_session_Embed)[0]

ort_session_Rotary_Text_Prefill = create_session(onnx_model_Rotary_Text_Prefill, **packed_settings)
binding_Rotary_Text_Prefill     = ort_session_Rotary_Text_Prefill.io_binding()
in_name_Rotary_Text_Prefill     = get_in_names(ort_session_Rotary_Text_Prefill)
out_name_Rotary_Text_Prefill    = get_out_names(ort_session_Rotary_Text_Prefill)
out_meta_Rotary_Text_Prefill    = ort_session_Rotary_Text_Prefill._outputs_meta

ort_session_Rotary_Text_Decode = create_session(onnx_model_Rotary_Text_Decode, **packed_settings)
binding_Rotary_Text_Decode     = ort_session_Rotary_Text_Decode.io_binding()
in_name_Rotary_Text_Decode     = get_in_names(ort_session_Rotary_Text_Decode)[0]
out_name_Rotary_Text_Decode    = get_out_names(ort_session_Rotary_Text_Decode)
out_meta_Rotary_Text_Decode    = ort_session_Rotary_Text_Decode._outputs_meta

ort_session_Main = create_session(onnx_model_Main, **packed_settings)
binding_Main     = ort_session_Main.io_binding()
print(f"\nUsable Providers: {ort_session_Main.get_providers()}")


in_name_Main  = get_in_names(ort_session_Main)
out_name_Main = get_out_names(ort_session_Main)
in_meta_Main  = ort_session_Main._inputs_meta

_NP_DTYPES = {
    'float32': np.float32, 'float16': np.float16,
    'int8':    np.int8,    'uint8':   np.uint8,
    'int32':   np.int32,   'int64':   np.int64,
}

num_keys_values_Main = int(model_meta['kv_num_tensors'])
num_layers_Main      = int(model_meta['num_layers'])
_kv_blocks_per_layer = int(model_meta['kv_blocks_per_layer'])   # 2 plain, 4 symmetric quant, 6 asymmetric quant.
_kv_sym_rt           = model_meta['kv_symmetric'] == '1'

if len(in_name_Main) != num_keys_values_Main + 4 or len(out_name_Main) != num_keys_values_Main + 1:
    raise RuntimeError(
        "LLM_Metadata.onnx disagrees with LLM_Main.onnx I/O: "
        f"metadata KV={num_keys_values_Main}, Main inputs={len(in_name_Main)}, outputs={len(out_name_Main)}."
    )

num_keys_values_Main_plus_1 = num_keys_values_Main + 1
num_keys_values_Main_plus_2 = num_keys_values_Main + 2
num_keys_values_Main_plus_3 = num_keys_values_Main + 3

in_name_Main_kv      = in_name_Main[:num_keys_values_Main]
out_name_Main_kv     = out_name_Main[:num_keys_values_Main]
in_name_Main_others  = in_name_Main[num_keys_values_Main:]
out_name_Main_logits = out_name_Main[num_keys_values_Main]

kv_dtype_Main = _NP_DTYPES[model_meta['kv_cache_elem_type']]
vocab_size    = int(model_meta['vocab_size'])

# F16 attention surgery can retype the mask without changing hidden_states.
hidden_dtype_Main = np.float16 if 'float16' in in_meta_Main[num_keys_values_Main].type else np.float32
mask_dtype_Main   = np.float16 if 'float16' in out_meta_Rotary_Text_Prefill[out_name_Rotary_Text_Prefill.index('attention_mask')].type else np.float32

MAX_SEQ_LEN = int(model_meta['max_seq_len'])
STOP_TOKEN = [int(t) for t in model_meta.get('eos_token_ids', '').split(',') if t]
if not STOP_TOKEN:
    STOP_TOKEN = [int(model_meta['chat_endoftext_id']), int(model_meta['chat_im_end_id'])]


# Scale/bias shapes come from graph I/O because grouped KV quantization packs dims there.
if _kv_blocks_per_layer >= 4:
    scale_dtype = _NP_DTYPES[model_meta['kv_scale_bias_elem_type']]

    if _kv_sym_rt:
        k_scale_shape   = list(in_meta_Main[num_layers_Main * 2].shape)
        k_scale_shape[0] = 1
        k_scale_shape[-1] = 0
        v_scale_shape   = list(in_meta_Main[num_layers_Main * 3].shape)
        v_scale_shape[0] = 1
        v_scale_shape[3] = 0
        k_scales        = create_ort_with_shape(tuple(k_scale_shape), scale_dtype, kv_device, DEVICE_ID)
        k_biases        = None
        v_scales        = create_ort_with_shape(tuple(v_scale_shape), scale_dtype, kv_device, DEVICE_ID)
        v_biases        = None
    else:
        k_scale_shape   = list(in_meta_Main[num_layers_Main * 2].shape)
        k_scale_shape[0] = 1
        k_scale_shape[-1] = 0
        v_scale_idx     = num_layers_Main * 4
        v_scale_shape   = list(in_meta_Main[v_scale_idx].shape)
        v_scale_shape[0] = 1
        v_scale_shape[3] = 0
        k_scales        = create_ort_with_shape(tuple(k_scale_shape), scale_dtype, kv_device, DEVICE_ID)
        k_biases        = create_ort_with_shape(tuple(k_scale_shape), scale_dtype, kv_device, DEVICE_ID)
        v_scales        = create_ort_with_shape(tuple(v_scale_shape), scale_dtype, kv_device, DEVICE_ID)
        v_biases        = create_ort_with_shape(tuple(v_scale_shape), scale_dtype, kv_device, DEVICE_ID)
else:
    k_scales = None

past_keys_Main   = create_ort_with_shape((1, in_meta_Main[0].shape[1],               1, in_meta_Main[0].shape[3],               0), kv_dtype_Main, kv_device, DEVICE_ID)
past_values_Main = create_ort_with_shape((1, in_meta_Main[num_layers_Main].shape[1], 1, 0, in_meta_Main[num_layers_Main].shape[4]), kv_dtype_Main, kv_device, DEVICE_ID)


if USE_BEAM_SEARCH and TOP_K < BEAM_SIZE:
    TOP_K = BEAM_SIZE

if TOP_K < 2 or BEAM_SIZE < 2:
    USE_BEAM_SEARCH = False
    print("\nInappropriate Beam Search setting detected. Falling back to Greedy Search.")

if not USE_BEAM_SEARCH:
    BEAM_SIZE = 1

USE_PENALTY = (REPEAT_PENALTY != 1.0)


tokenizer = AutoTokenizer.from_pretrained(download_path, trust_remote_code=True)

STOP_TOKEN_SET = set(STOP_TOKEN)

prompt = (
    f'<|im_start|>user\n{TEST_QUERY}<|im_end|>\n<|im_start|>assistant\n'
    if TEST_THINK_MODE else
    f'<|im_start|>user\n{TEST_QUERY}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
)

tokens      = tokenizer(prompt, return_tensors='np')['input_ids'].astype(np.int32)
num_prefill = tokens.shape[-1]


input_ids        = onnxruntime.OrtValue.ortvalue_from_numpy(tokens,   device_type, DEVICE_ID)
ids_len          = create_ort_with_data([num_prefill], np.int64, device_type, DEVICE_ID)
init_history_len = create_ort_with_data([0],           np.int64, device_type, DEVICE_ID)
init_cache_len   = create_ort_with_data([0],           np.int64, device_type, DEVICE_ID)
topK             = create_ort_with_data([TOP_K],       np.int64, device_type, DEVICE_ID)
beam_size        = create_ort_with_data([BEAM_SIZE],   np.int64, device_type, DEVICE_ID)

attention_mask_buf = create_ort_with_shape((1, 1, 1, 1, 1),                                            mask_dtype_Main, device_type, DEVICE_ID)
rotary_cos_buf     = create_ort_with_shape(out_meta_Rotary_Text_Decode[0].shape,                              hidden_dtype_Main, device_type, DEVICE_ID)
rotary_sin_buf     = create_ort_with_shape(out_meta_Rotary_Text_Decode[1].shape,                              hidden_dtype_Main, device_type, DEVICE_ID)
hidden_states_buf  = create_ort_with_shape((BEAM_SIZE, 1, in_meta_Main[num_keys_values_Main].shape[2]), hidden_dtype_Main, device_type, DEVICE_ID)
save_id_buf        = create_ort_with_shape((BEAM_SIZE, 0),                                              np.int32,          device_type, DEVICE_ID)

prefill_logits_buf = create_ort_with_shape((1, vocab_size),         hidden_dtype_Main, device_type, DEVICE_ID)
decode_logits_buf  = create_ort_with_shape((BEAM_SIZE, vocab_size), hidden_dtype_Main, device_type, DEVICE_ID)
max_idx_buf        = create_ort_with_shape((1, 1),                  np.int32,          device_type, DEVICE_ID)


if USE_BEAM_SEARCH:
    print("\nBeam Search does not display immediate decoding results...")

    ort_session_First_Beam     = create_session(onnx_model_First_Beam, **packed_settings)
    binding_First_Beam         = ort_session_First_Beam.io_binding()
    in_name_First_Beam         = get_in_names(ort_session_First_Beam)
    out_name_First_Beam        = get_out_names(ort_session_First_Beam)
    in_name_First_Beam_parts   = in_name_First_Beam[:num_keys_values_Main_plus_1]
    out_name_First_Beam_parts  = out_name_First_Beam[:num_keys_values_Main_plus_1]
    out_name_First_Beam_others = out_name_First_Beam[num_keys_values_Main_plus_1:]

    ort_session_Second_Beam     = create_session(onnx_model_Second_Beam, **packed_settings)
    binding_Second_Beam         = ort_session_Second_Beam.io_binding()
    in_name_Second_Beam         = get_in_names(ort_session_Second_Beam)
    out_name_Second_Beam        = get_out_names(ort_session_Second_Beam)
    in_name_Second_Beam_parts   = in_name_Second_Beam[:num_keys_values_Main_plus_1]
    out_name_Second_Beam_parts  = out_name_Second_Beam[:num_keys_values_Main_plus_1]
    out_name_Second_Beam_others = out_name_Second_Beam[num_keys_values_Main_plus_1:]

    beam_ids_buf   = create_ort_with_shape((BEAM_SIZE, 1), np.int32,          device_type, DEVICE_ID)
    beam_score_buf = create_ort_with_shape((BEAM_SIZE, 1), hidden_dtype_Main, device_type, DEVICE_ID)

    bind_ort_in_buf(binding_First_Beam, in_name_First_Beam[num_keys_values_Main_plus_1: num_keys_values_Main_plus_3], [save_id_buf, beam_size])
    bind_ort_in_buf(binding_Second_Beam, in_name_Second_Beam[num_keys_values_Main_plus_3:], [beam_size, topK])
else:
    ort_session_Greedy = create_session(onnx_model_Greedy, **packed_settings)
    binding_Greedy     = ort_session_Greedy.io_binding()
    in_name_Greedy     = get_in_names(ort_session_Greedy)
    out_name_Greedy    = get_out_names(ort_session_Greedy)
    binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id_buf)

    ort_session_Argmax = create_session(onnx_model_Argmax, **packed_settings)
    binding_Argmax     = ort_session_Argmax.io_binding()
    in_name_Argmax     = get_in_names(ort_session_Argmax)[0]
    out_name_Argmax    = get_out_names(ort_session_Argmax)[0]
    save_id_numpy      = np.zeros(MAX_SEQ_LEN, dtype=np.int32)


if USE_PENALTY:
    ort_session_Penalty = create_session(onnx_model_Penalty, **packed_settings)
    binding_Penalty     = ort_session_Penalty.io_binding()
    in_name_Penalty     = get_in_names(ort_session_Penalty)
    out_name_Penalty    = get_out_names(ort_session_Penalty)[0]

    penalty_dtype = np.float16 if 'float16' in ort_session_Penalty._inputs_meta[2].type else np.float32
    penalty_value = create_ort_with_data([REPEAT_PENALTY], penalty_dtype, device_type, DEVICE_ID)
    penalty_range = create_ort_with_data([PENALTY_RANGE],  np.int64,      device_type, DEVICE_ID)

    bind_ort_in_buf(binding_Penalty, in_name_Penalty[2:], [penalty_value, penalty_range])


is_prefill_step = True
prefill_start_time = time.time()

binding_Embed.bind_ortvalue_input(in_name_Embed, input_ids)
bind_ort_out(binding_Embed, [out_name_Embed], _ort_device_type)
run(ort_session_Embed, binding_Embed)
hidden_states = binding_Embed.get_outputs()[0]

binding_Embed.bind_ortvalue_input(in_name_Embed, max_idx_buf)

bind_ort_in_buf(binding_Rotary_Text_Prefill, in_name_Rotary_Text_Prefill, [ids_len, init_history_len, init_cache_len])
bind_ort_out(binding_Rotary_Text_Prefill, out_name_Rotary_Text_Prefill, _ort_device_type)
run(ort_session_Rotary_Text_Prefill, binding_Rotary_Text_Prefill)
rotary_cos, rotary_sin, attention_mask, kv_seq_len = binding_Rotary_Text_Prefill.get_outputs()

binding_Rotary_Text_Decode.bind_ortvalue_input(in_name_Rotary_Text_Decode, kv_seq_len)
bind_ort_out_buf(binding_Rotary_Text_Decode, out_name_Rotary_Text_Decode, [rotary_cos_buf, rotary_sin_buf, kv_seq_len])

bind_ort_in_buf(binding_Main, in_name_Main_others, [hidden_states, rotary_cos, rotary_sin, attention_mask])

i = 0
for _ in range(num_layers_Main):
    binding_Main.bind_ortvalue_input(in_name_Main[i], past_keys_Main)
    i += 1
for _ in range(num_layers_Main):
    binding_Main.bind_ortvalue_input(in_name_Main[i], past_values_Main)
    i += 1
if k_scales is not None:
    if k_biases is not None:
        for j in (k_scales, k_biases, v_scales, v_biases):
            for _ in range(num_layers_Main):
                binding_Main.bind_ortvalue_input(in_name_Main[i], j)
                i += 1
    else:
        for j in (k_scales, v_scales):
            for _ in range(num_layers_Main):
                binding_Main.bind_ortvalue_input(in_name_Main[i], j)
                i += 1

bind_ort_out(binding_Main, out_name_Main_kv, _ort_device_type)
binding_Main.bind_ortvalue_output(out_name_Main_logits, prefill_logits_buf)

if USE_PENALTY:
    binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], prefill_logits_buf)
    binding_Penalty.bind_ortvalue_output(out_name_Penalty,  prefill_logits_buf)

if USE_BEAM_SEARCH:
    binding_First_Beam.bind_ortvalue_input(in_name_First_Beam[num_keys_values_Main], prefill_logits_buf)
elif USE_PENALTY:
    binding_Greedy.bind_ortvalue_input(in_name_Greedy[0],   prefill_logits_buf)
    binding_Greedy.bind_ortvalue_output(out_name_Greedy[0], max_idx_buf)
else:
    binding_Argmax.bind_ortvalue_input(in_name_Argmax,   prefill_logits_buf)
    binding_Argmax.bind_ortvalue_output(out_name_Argmax, max_idx_buf)


print(f'\nTest Question: {TEST_QUERY}\nLLM Answering:')

num_decode     = 0
generate_limit = MAX_SEQ_LEN - num_prefill

while num_decode < generate_limit:

    run(ort_session_Main, binding_Main)
    outputs_Main = binding_Main.get_outputs()

    if USE_PENALTY and num_decode >= PENALTY_RANGE:
        binding_Penalty.bind_ortvalue_input(in_name_Penalty[1], save_id)
        run(ort_session_Penalty, binding_Penalty)

    if USE_BEAM_SEARCH:
        if is_prefill_step:
            bind_ort_in_buf(binding_First_Beam, in_name_First_Beam_parts, outputs_Main)
            bind_ort_out(binding_First_Beam, out_name_First_Beam_parts, _ort_device_type)
            bind_ort_out_buf(binding_First_Beam, out_name_First_Beam_others, [beam_score_buf, beam_ids_buf, max_idx_buf])
            run(ort_session_First_Beam, binding_First_Beam)
            outputs_Beam = binding_First_Beam.get_outputs()
        else:
            bind_ort_in_buf(binding_Second_Beam, in_name_Second_Beam_parts, outputs_Main)
            bind_ort_out(binding_Second_Beam, out_name_Second_Beam_parts, _ort_device_type)
            if num_decode < 2:
                binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_Main_plus_2], beam_score_buf)
                bind_ort_out_buf(binding_Second_Beam, out_name_Second_Beam_others, [beam_score_buf, beam_ids_buf, max_idx_buf])
            run(ort_session_Second_Beam, binding_Second_Beam)
            outputs_Beam = binding_Second_Beam.get_outputs()

        max_logits_idx = max_idx_buf.numpy().flat[0]
        if max_logits_idx in STOP_TOKEN_SET:
            break

        save_id = outputs_Beam[num_keys_values_Main]
        bind_ort_in_buf(binding_Main, in_name_Main_kv, outputs_Beam)
        binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_Main_plus_1], save_id)

    else:
        if USE_PENALTY:
            binding_Greedy._iobinding.bind_output(out_name_Greedy[1], _ort_device_type)
            run(ort_session_Greedy, binding_Greedy)
            save_id = binding_Greedy.get_outputs()[1]
        else:
            run(ort_session_Argmax, binding_Argmax)

        max_logits_idx = max_idx_buf.numpy().flat[0]
        if max_logits_idx in STOP_TOKEN_SET:
            break

        if USE_PENALTY:
            binding_Greedy.bind_ortvalue_input(in_name_Greedy[1], save_id)
        else:
            save_id_numpy[num_decode] = max_logits_idx

        bind_ort_in_buf(binding_Main, in_name_Main_kv, outputs_Main)

        print(tokenizer.decode(max_logits_idx), end="", flush=True)

    bind_ort_out(binding_Main, out_name_Main_kv, _ort_device_type)

    if is_prefill_step:

        bind_ort_in_buf(binding_Main, in_name_Main_others, [hidden_states_buf, rotary_cos_buf, rotary_sin_buf, attention_mask_buf])
        binding_Main.bind_ortvalue_output(out_name_Main_logits, decode_logits_buf)

        binding_Embed.bind_ortvalue_output(out_name_Embed, hidden_states_buf)

        if USE_PENALTY:
            binding_Penalty.bind_ortvalue_input(in_name_Penalty[0], decode_logits_buf)
            binding_Penalty.bind_ortvalue_output(out_name_Penalty, decode_logits_buf)

        if USE_BEAM_SEARCH:
            binding_Second_Beam.bind_ortvalue_input(in_name_Second_Beam[num_keys_values_Main], decode_logits_buf)
            binding_Embed.bind_ortvalue_input(in_name_Embed, beam_ids_buf)
        elif USE_PENALTY:
            binding_Greedy.bind_ortvalue_input(in_name_Greedy[0], decode_logits_buf)
        else:
            binding_Argmax.bind_ortvalue_input(in_name_Argmax, decode_logits_buf)

        is_prefill_step = False

        decode_start_time = time.time()
        prefill_elapsed = decode_start_time - prefill_start_time

    run(ort_session_Embed, binding_Embed)
    run(ort_session_Rotary_Text_Decode, binding_Rotary_Text_Decode)
    num_decode += 1


decode_end_time = time.time()

if num_decode < 2:
    prefill_elapsed = 0.0
    decode_elapsed = 0.0
else:
    decode_elapsed = decode_end_time - decode_start_time

total_elapsed = decode_end_time - prefill_start_time

prefill_tokens_per_second = num_prefill / prefill_elapsed if prefill_elapsed > 0 else 0.0
decode_tokens_per_second = num_decode / decode_elapsed if decode_elapsed > 0 else 0.0
overall_tokens_per_second = (num_decode + 1) / total_elapsed if total_elapsed > 0 else 0.0

if USE_PENALTY or USE_BEAM_SEARCH:
    result = tokenizer.decode(save_id.numpy().flat[:num_decode], skip_special_tokens=True)
else:
    result = tokenizer.decode(save_id_numpy[:num_decode], skip_special_tokens=True)

print(
    f"\n\n{'─' * 56}\n"
    f"  📝 Generated Output\n"
    f"{'─' * 56}\n"
    f"{result}\n"
    f"{'─' * 56}\n\n"
    f"  ⚡ Performance Summary\n"
    f"{'─' * 56}\n"
    f"  {'Phase':<12} {'Speed':>14} {'Tokens':>8} {'Time':>10}\n"
    f"  {'─' * 48}\n"
    f"  {'Prefill':<12} {prefill_tokens_per_second:>10.2f} t/s {num_prefill:>8d} {prefill_elapsed:>8.3f}s\n"
    f"  {'Decode':<12} {decode_tokens_per_second:>10.2f} t/s {num_decode:>8d} {decode_elapsed:>8.3f}s\n"
    f"  {'─' * 48}\n"
    f"  {'Overall':<12} {overall_tokens_per_second:>10.2f} t/s {num_decode:>8d} {total_elapsed:>8.3f}s\n"
    f"{'─' * 56}\n"
)
