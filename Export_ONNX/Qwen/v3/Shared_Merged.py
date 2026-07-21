"""Build Qwen v3 merged ONNX graphs backed by one shared initializer bundle.

The production layout keeps one prefill and one decode graph per strategy
(greedy, penalty-greedy, sampling), while KV management stays in small standalone
graphs. Large Main initializers are written once to
LLM_SharedInitializers.onnx(.data); small shape constants stay embedded.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

MIN_SHARED_INITIALIZER_ELEMENTS = 1024
_UNSHAREABLE_INIT_TYPES = frozenset(
    getattr(TensorProto, name)
    for name in ("UINT4", "INT4", "FLOAT4E2M1")
    if hasattr(TensorProto, name)
)

# Rotary/decode-head nodes are shell; everything else in a merged graph is Main.
SHELL_PREFIXES = (
    "prefill_",
    "decode_",
    "greedy_",
    "penalty_greedy_",
    "penalty_",
    "sampling_",
)

_DEFAULT_MODEL_FILE_NAMES = {
    "main": "LLM_Main.onnx",
    "rotary_prefill": "LLM_RotaryPrefill.onnx",
    "rotary_decode": "LLM_RotaryDecode.onnx",
    "greedy": "LLM_Greedy.onnx",
    "sampling": "LLM_TopKTopPSampling.onnx",
    "penalty_greedy": "LLM_PenaltyGreedy.onnx",
    "penalty": "LLM_Penalty.onnx",
    "argmax": "LLM_Argmax.onnx",
    "prefill_greedy": "LLM_TextPrefillGreedy.onnx",
    "prefill_penalty_greedy": "LLM_TextPrefillPenaltyGreedy.onnx",
    "prefill_sampling": "LLM_TextPrefillSampling.onnx",
    "decode_greedy": "LLM_DecodeGreedy.onnx",
    "decode_penalty_greedy": "LLM_DecodePenaltyGreedy.onnx",
    "decode_sampling": "LLM_DecodeSampling.onnx",
    "shared_initializers": "LLM_SharedInitializers.onnx",
}

PREFILL_GREEDY_MODEL_NAME         = _DEFAULT_MODEL_FILE_NAMES["prefill_greedy"]
PREFILL_PENALTY_GREEDY_MODEL_NAME = _DEFAULT_MODEL_FILE_NAMES["prefill_penalty_greedy"]
PREFILL_SAMPLING_MODEL_NAME       = _DEFAULT_MODEL_FILE_NAMES["prefill_sampling"]
DECODE_GREEDY_MODEL_NAME          = _DEFAULT_MODEL_FILE_NAMES["decode_greedy"]
DECODE_PENALTY_GREEDY_MODEL_NAME  = _DEFAULT_MODEL_FILE_NAMES["decode_penalty_greedy"]
DECODE_SAMPLING_MODEL_NAME        = _DEFAULT_MODEL_FILE_NAMES["decode_sampling"]
SHARED_MODEL_NAME                 = _DEFAULT_MODEL_FILE_NAMES["shared_initializers"]
SHARED_DATA_NAME                  = SHARED_MODEL_NAME + ".data"

_MERGED_CONSTITUENT_ROLES = (
    "main",
    "rotary_prefill",
    "rotary_decode",
    "greedy",
    "sampling",
    "penalty_greedy",
    "penalty",
    "argmax",
)
MERGED_CONSTITUENT_GRAPHS = tuple(_DEFAULT_MODEL_FILE_NAMES[role] for role in _MERGED_CONSTITUENT_ROLES)
_LEGACY_SHARED_ARTIFACTS = (
    "shared_initializers.npz",
    "shared_initializers.manifest.json",
    "shared_initializers.onnx",
    "shared_initializers.onnx.data",
)


def load_model(path: Path) -> onnx.ModelProto:
    return onnx.load(str(path), load_external_data=True)


def save_model(model: onnx.ModelProto, path: Path) -> None:
    """Save a merged graph without producing any per-graph .data file.

    Shared initializers were already redirected to LLM_SharedInitializers.onnx.data,
    so the remaining embedded (small) initializers are tiny and stay in the .onnx.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_name(path.name + ".data").unlink(missing_ok=True)
    onnx.save(model, str(path))


def _external_data_map(init: TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in init.external_data}


def save_shared_initializers_from_tensors(shared: dict[str, TensorProto], path: Path) -> None:
    """Write LLM_SharedInitializers.onnx + LLM_SharedInitializers.onnx.data from raw TensorProtos.

    Each tensor's raw bytes are streamed straight into one external-data file and only
    lightweight external references are kept in the shared .onnx. Writing bytes directly
    (instead of onnx.save_model's copy-then-reserialize with save_as_external_data) keeps
    peak memory near the source size rather than several times it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data_name = path.name + ".data"
    data_path = path.with_name(data_name)
    path.unlink(missing_ok=True)
    data_path.unlink(missing_ok=True)

    ref_inits: list[TensorProto] = []
    offset = 0
    with open(data_path, "wb") as data_file:
        for name, tensor in sorted(shared.items()):
            raw = tensor.raw_data
            if not raw:
                raw = numpy_helper.to_array(tensor).tobytes()
            data_file.write(raw)
            length = len(raw)
            ref = TensorProto()
            ref.name = name
            ref.data_type = tensor.data_type
            ref.dims.extend(tensor.dims)
            ref.data_location = TensorProto.EXTERNAL
            for key, value in (("location", data_name), ("offset", str(offset)), ("length", str(length))):
                entry = ref.external_data.add()
                entry.key = key
                entry.value = value
            ref_inits.append(ref)
            offset += length

    graph = onnx.helper.make_graph(nodes=[], name="shared_initializers", inputs=[], outputs=[], initializer=ref_inits)
    model = onnx.helper.make_model(graph, producer_name="Shared_Merged.py")
    model.ir_version = 10
    model.metadata_props.add(key="native_llm_shared_initializers", value="1")
    model.metadata_props.add(key="initializer_count", value=str(len(ref_inits)))
    onnx.save_model(model, str(path))


def shared_external_data_map(shared_model_path: Path) -> dict[str, dict[str, str]]:
    model = onnx.load(str(shared_model_path), load_external_data=False)
    return {init.name: _external_data_map(init) for init in model.graph.initializer}


def _write_shared_initializer_bundle(
    shared: dict[str, TensorProto], shared_model_path: Path
) -> dict[str, dict[str, str]]:
    save_shared_initializers_from_tensors(shared, shared_model_path)
    return shared_external_data_map(shared_model_path)


def make_external_initializer_ref(init: TensorProto, external_data: dict[str, str]) -> TensorProto:
    ref = TensorProto()
    ref.name = init.name
    ref.data_type = init.data_type
    ref.dims.extend(init.dims)
    ref.data_location = TensorProto.EXTERNAL
    for key in ("location", "offset", "length", "checksum", "basepath"):
        value = external_data.get(key)
        if value is not None:
            entry = ref.external_data.add()
            entry.key = key
            entry.value = value
    if "location" not in external_data:
        raise RuntimeError(f"Shared initializer {init.name!r} is missing external data location.")
    return ref


def redirect_shared_initializers_to_external(
    model: onnx.ModelProto, external_by_name: dict[str, dict[str, str]]
) -> int:
    """Rewrite each initializer that appears in external_by_name to an external ref."""
    rewritten = []
    count = 0
    for init in model.graph.initializer:
        external = external_by_name.get(init.name)
        if external is not None:
            rewritten.append(make_external_initializer_ref(init, external))
            count += 1
        else:
            rewritten.append(init)
    del model.graph.initializer[:]
    model.graph.initializer.extend(rewritten)
    return count


def prefixed(model: onnx.ModelProto, prefix: str) -> onnx.ModelProto:
    import onnx.compose

    return onnx.compose.add_prefix(
        model,
        prefix,
        rename_nodes=True,
        rename_edges=True,
        rename_inputs=True,
        rename_outputs=True,
        rename_initializers=True,
        rename_value_infos=True,
    )


def value_info_by_name(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    items = list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    return {item.name: item for item in items}


def set_graph_outputs(model: onnx.ModelProto, output_names: list[str]) -> None:
    by_name = value_info_by_name(model)
    missing = [name for name in output_names if name not in by_name]
    if missing:
        raise RuntimeError(f"Merged graph is missing output value_info for: {missing}")
    del model.graph.output[:]
    model.graph.output.extend([by_name[name] for name in output_names])


def copy_metadata(dst: onnx.ModelProto, *sources: onnx.ModelProto) -> None:
    existing = {prop.key: prop for prop in dst.metadata_props}
    for source in sources:
        for prop in source.metadata_props:
            if prop.key in existing:
                existing[prop.key].value = prop.value
            else:
                existing[prop.key] = dst.metadata_props.add(key=prop.key, value=prop.value)


def merge_models_no_check(
    first: onnx.ModelProto,
    second: onnx.ModelProto,
    io_map: list[tuple[str, str]],
) -> onnx.ModelProto:
    """Merge two models without running onnx.checker on a >2GiB in-memory proto.

    ``first`` and ``second`` are only read: every field is copied into a fresh ``merged``
    graph via protobuf append/extend, and the ``second``-side input remap is applied to
    merged's own node copies. The sources stay unmutated (safe to reuse across shells), so
    no defensive deepcopy of the potentially multi-GB inputs is needed.
    """
    source_by_target = {target: source for source, target in io_map}
    mapped_sources = set(source_by_target.values())
    mapped_targets = set(source_by_target)

    merged = onnx.ModelProto()
    merged.ir_version = max(first.ir_version, second.ir_version)
    merged.producer_name = "Shared_Merged.py"
    merged.graph.name = f"{first.graph.name}_{second.graph.name}_merged"

    opsets: dict[str, int] = {}
    for model in (first, second):
        for opset in model.opset_import:
            opsets[opset.domain] = max(opsets.get(opset.domain, 0), opset.version)
    for domain, version in sorted(opsets.items()):
        merged.opset_import.add(domain=domain, version=version)

    seen_inputs = set()
    for value in list(first.graph.input) + [v for v in second.graph.input if v.name not in mapped_targets]:
        if value.name not in seen_inputs:
            merged.graph.input.append(value)
            seen_inputs.add(value.name)

    init_by_name: dict[str, onnx.TensorProto] = {}
    for init in list(first.graph.initializer) + list(second.graph.initializer):
        existing = init_by_name.get(init.name)
        if existing is None:
            init_by_name[init.name] = init
        elif existing is not init and existing.SerializeToString() != init.SerializeToString():
            raise RuntimeError(f"Initializer name collision with different data: {init.name}")
    merged.graph.initializer.extend(init_by_name.values())

    merged.graph.node.extend(first.graph.node)
    second_start = len(merged.graph.node)
    merged.graph.node.extend(second.graph.node)
    if source_by_target:
        for node in merged.graph.node[second_start:]:
            for index, name in enumerate(node.input):
                replacement = source_by_target.get(name)
                if replacement is not None:
                    node.input[index] = replacement

    seen_vi = {item.name for item in merged.graph.input}
    seen_vi.update(init_by_name)
    for value in list(first.graph.value_info) + list(second.graph.value_info):
        if value.name not in seen_vi:
            merged.graph.value_info.append(value)
            seen_vi.add(value.name)

    seen_outputs = set()
    for value in [v for v in first.graph.output if v.name not in mapped_sources] + list(second.graph.output):
        if value.name not in seen_outputs:
            merged.graph.output.append(value)
            seen_outputs.add(value.name)

    copy_metadata(merged, first, second)
    return merged


def load_main_with_shared_initializers(
    source_folder: Path,
    min_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
    model_file_names: dict[str, str] | None = None,
) -> tuple[onnx.ModelProto, dict[str, TensorProto]]:
    """Load LLM_Main.onnx and split its initializers into embedded vs. shared.

    Only weight-like numeric initializers with >= min_elements elements are
    marked shared; small shape constants stay embedded in the returned model.
    The shared dict holds *references* into ``main`` (no copy), so it must be
    consumed (streamed to disk) before ``main`` is redirected to external refs.
    """
    main = load_model(source_folder / _model_file_name(model_file_names, "main"))
    shared: dict[str, TensorProto] = {}
    for init in main.graph.initializer:
        if _is_shareable_initializer(init, min_elements):
            shared[init.name] = init
    if not shared:
        raise RuntimeError("LLM_Main.onnx has no initializer to share.")
    return main, shared


def _initializer_num_elements(init: TensorProto) -> int:
    total = 1
    for dim in init.dims:
        total *= int(dim)
    return total


def _is_shareable_initializer(init: TensorProto, min_elements: int) -> bool:
    if init.data_type in (TensorProto.UNDEFINED, TensorProto.STRING):
        return False
    return _initializer_num_elements(init) >= min_elements


def _main_kv_output_names(main: onnx.ModelProto) -> list[str]:
    return [out.name for out in main.graph.output if out.name.startswith("out_")]


def _merge_rotary_into_main(
    source_folder: Path,
    main: onnx.ModelProto,
    kind: str,
    model_file_names: dict[str, str] | None = None,
):
    """Fuse RotaryPrefill/RotaryDecode into Main. Returns (merged, kv_seq_len_output, rotary_model)."""
    if kind == "prefill":
        rotary = prefixed(
            load_model(source_folder / _model_file_name(model_file_names, "rotary_prefill")),
            "prefill_",
        )
        merged = merge_models_no_check(
            rotary, main,
            io_map=[
                ("prefill_rotary_cos", "rotary_cos"),
                ("prefill_rotary_sin", "rotary_sin"),
                ("prefill_attention_mask", "attention_mask"),
            ],
        )
        return merged, "prefill_kv_seq_len", rotary

    rotary = prefixed(
        load_model(source_folder / _model_file_name(model_file_names, "rotary_decode")),
        "decode_",
    )
    mask_info = next(item for item in main.graph.input if item.name == "attention_mask")
    mask_dtype = onnx.helper.tensor_dtype_to_np_dtype(mask_info.type.tensor_type.elem_type)
    rotary.graph.initializer.append(
        numpy_helper.from_array(np.zeros((1, 1, 1, 1, 1), dtype=mask_dtype), name="decode_zero_attention_mask")
    )
    merged = merge_models_no_check(
        rotary, main,
        io_map=[
            ("decode_rotary_cos", "rotary_cos"),
            ("decode_rotary_sin", "rotary_sin"),
            ("decode_zero_attention_mask", "attention_mask"),
        ],
    )
    return merged, "decode_kv_seq_len", rotary


def _finalize(merged, main, rotary, output_names):
    set_graph_outputs(merged, output_names)
    _order_kv_inputs_first(merged)
    copy_metadata(merged, main, rotary)
    merged.producer_name = "Shared_Merged.py"
    return merged


def _order_kv_inputs_first(model: onnx.ModelProto) -> None:
    kv_inputs = [value for value in model.graph.input if value.name.startswith("in_")]
    other_inputs = [value for value in model.graph.input if not value.name.startswith("in_")]
    del model.graph.input[:]
    model.graph.input.extend(kv_inputs + other_inputs)


def _merge_greedy(source_folder, main, kind, model_file_names=None):
    merged, kv_seq_len, rotary = _merge_rotary_into_main(source_folder, main, kind, model_file_names)
    greedy = prefixed(
        load_model(source_folder / _model_file_name(model_file_names, "greedy")),
        "greedy_",
    )
    merged = merge_models_no_check(merged, greedy, io_map=[("logits", "greedy_logits")])
    return _finalize(merged, main, rotary,
                     _main_kv_output_names(main) + ["greedy_max_logits_idx", kv_seq_len])


def _merge_sampling(source_folder, main, kind, model_file_names=None):
    merged, kv_seq_len, rotary = _merge_rotary_into_main(source_folder, main, kind, model_file_names)
    sampling = prefixed(
        load_model(source_folder / _model_file_name(model_file_names, "sampling")),
        "sampling_",
    )
    merged = merge_models_no_check(merged, sampling, io_map=[("logits", "sampling_logits")])
    return _finalize(merged, main, rotary,
                     _main_kv_output_names(main) + ["sampling_sampled_id", "sampling_save_id_out", kv_seq_len])


def _merge_penalty_greedy_prefill(source_folder, main, kind, model_file_names=None):
    merged, kv_seq_len, rotary = _merge_rotary_into_main(source_folder, main, kind, model_file_names)
    penalty_greedy = prefixed(
        load_model(source_folder / _model_file_name(model_file_names, "penalty_greedy")),
        "penalty_greedy_",
    )
    merged = merge_models_no_check(merged, penalty_greedy, io_map=[("logits", "penalty_greedy_logits")])
    return _finalize(merged, main, rotary,
                     _main_kv_output_names(main) + [
                         "penalty_greedy_max_logits_idx", "penalty_greedy_save_id_out", kv_seq_len
                     ])


def merge_prefill_greedy(
    source_folder: Path, main: onnx.ModelProto, model_file_names: dict[str, str] | None = None
) -> onnx.ModelProto:
    return _merge_greedy(source_folder, main, "prefill", model_file_names)


def merge_decode_greedy(
    source_folder: Path, main: onnx.ModelProto, model_file_names: dict[str, str] | None = None
) -> onnx.ModelProto:
    return _merge_greedy(source_folder, main, "decode", model_file_names)


def merge_prefill_penalty_greedy(
    source_folder: Path, main: onnx.ModelProto, model_file_names: dict[str, str] | None = None
) -> onnx.ModelProto:
    return _merge_penalty_greedy_prefill(source_folder, main, "prefill", model_file_names)


def merge_decode_penalty_greedy(
    source_folder: Path, main: onnx.ModelProto, model_file_names: dict[str, str] | None = None
) -> onnx.ModelProto:
    merged, kv_seq_len, rotary = _merge_rotary_into_main(source_folder, main, "decode", model_file_names)
    penalty = prefixed(
        load_model(source_folder / _model_file_name(model_file_names, "penalty")),
        "penalty_",
    )
    penalty_greedy = prefixed(
        load_model(source_folder / _model_file_name(model_file_names, "penalty_greedy")),
        "penalty_greedy_",
    )
    merged = merge_models_no_check(merged, penalty, io_map=[("logits", "penalty_logits_in")])
    merged = merge_models_no_check(merged, penalty_greedy, io_map=[("penalty_logits_out", "penalty_greedy_logits")])
    return _finalize(merged, main, rotary,
                     _main_kv_output_names(main) + [
                         "penalty_greedy_max_logits_idx", "penalty_greedy_save_id_out", kv_seq_len
                     ])


def merge_prefill_sampling(
    source_folder: Path, main: onnx.ModelProto, model_file_names: dict[str, str] | None = None
) -> onnx.ModelProto:
    return _merge_sampling(source_folder, main, "prefill", model_file_names)


def merge_decode_sampling(
    source_folder: Path, main: onnx.ModelProto, model_file_names: dict[str, str] | None = None
) -> onnx.ModelProto:
    return _merge_sampling(source_folder, main, "decode", model_file_names)


# --------------------------------------------------------------------------- #
# Top-level production entry points
# --------------------------------------------------------------------------- #
# Each entry: (output file name, recipe, [split graphs the recipe needs besides Main]).
def _model_file_name(model_file_names: dict[str, str] | None, role: str) -> str:
    if model_file_names is None:
        return _DEFAULT_MODEL_FILE_NAMES[role]
    return model_file_names.get(role, _DEFAULT_MODEL_FILE_NAMES[role])


def _recipe_with_names(recipe, model_file_names: dict[str, str] | None):
    if model_file_names is None:
        return recipe

    def wrapped(source_folder, main):
        return recipe(source_folder, main, model_file_names)

    wrapped.__name__ = recipe.__name__
    return wrapped


_MERGED_BUILD_SPECS = (
    ("prefill_greedy", merge_prefill_greedy, ("rotary_prefill", "greedy")),
    ("prefill_penalty_greedy", merge_prefill_penalty_greedy, ("rotary_prefill", "penalty_greedy")),
    ("prefill_sampling", merge_prefill_sampling, ("rotary_prefill", "sampling")),
    ("decode_greedy", merge_decode_greedy, ("rotary_decode", "greedy")),
    ("decode_penalty_greedy", merge_decode_penalty_greedy,
     ("rotary_decode", "penalty", "penalty_greedy")),
    ("decode_sampling", merge_decode_sampling, ("rotary_decode", "sampling")),
)


def make_merged_build_plan(model_file_names: dict[str, str] | None = None):
    return [
        (
            _model_file_name(model_file_names, output_role),
            _recipe_with_names(recipe, model_file_names),
            [_model_file_name(model_file_names, dependency_role) for dependency_role in dependency_roles],
        )
        for output_role, recipe, dependency_roles in _MERGED_BUILD_SPECS
    ]


MERGED_BUILD_PLAN = make_merged_build_plan()


def _node_is_shell(node: onnx.NodeProto) -> bool:
    return any(output.startswith(SHELL_PREFIXES) for output in node.output)


def _used_inputs(nodes: list[onnx.NodeProto]) -> set[str]:
    return {name for node in nodes for name in node.input if name}


def _copy_node_with_input_remap(node: onnx.NodeProto, remap: dict[str, str]) -> onnx.NodeProto:
    copied = copy.deepcopy(node)
    for index, name in enumerate(copied.input):
        copied.input[index] = remap.get(name, name)
    return copied


def _copy_value_info_with_name(value_info: onnx.ValueInfoProto, name: str) -> onnx.ValueInfoProto:
    copied = copy.deepcopy(value_info)
    copied.name = name
    return copied


def _merge_opsets(dst: onnx.ModelProto, *sources: onnx.ModelProto) -> None:
    opsets: dict[str, int] = {}
    for model in (dst, *sources):
        for opset in model.opset_import:
            opsets[opset.domain] = max(opsets.get(opset.domain, 0), opset.version)
    del dst.opset_import[:]
    for domain, version in sorted(opsets.items()):
        dst.opset_import.add(domain=domain, version=version)


def _target_rotary_remap(target: onnx.ModelProto) -> dict[str, str]:
    names = {value.name for value in target.graph.input}
    names.update(init.name for init in target.graph.initializer)
    for node in target.graph.node:
        names.update(node.output)
    if "decode_rotary_cos" not in names:
        return {}
    return {
        "prefill_rotary_cos": "decode_rotary_cos",
        "prefill_rotary_sin": "decode_rotary_sin",
        "prefill_attention_mask": "decode_zero_attention_mask",
    }


def transplant_quantized_main(target: onnx.ModelProto, quantized_primary: onnx.ModelProto) -> onnx.ModelProto:
    """Replace target's unquantized Main block with the quantized Main block from primary.

    Merged graph layout is [rotary shell] + [Main] + [decode head shell]. Shell nodes are identified
    by prefixed output names; Main nodes keep the original unprefixed tensor names across strategies.
    """
    remap = _target_rotary_remap(target)
    primary_main_nodes = [
        _copy_node_with_input_remap(node, remap)
        for node in quantized_primary.graph.node
        if not _node_is_shell(node)
    ]
    if not primary_main_nodes:
        raise RuntimeError("Quantized primary graph has no Main node block to transplant.")

    merged = copy.deepcopy(target)
    new_nodes: list[onnx.NodeProto] = []
    inserted = False
    for node in target.graph.node:
        if _node_is_shell(node):
            new_nodes.append(node)
        elif not inserted:
            new_nodes.extend(primary_main_nodes)
            inserted = True
    if not inserted:
        new_nodes.extend(primary_main_nodes)

    primary_inits = {init.name: init for init in quantized_primary.graph.initializer}
    target_inits = {init.name: init for init in target.graph.initializer}
    main_init_names = _used_inputs(primary_main_nodes) & set(primary_inits)
    used = _used_inputs(new_nodes)

    new_initializers: list[TensorProto] = []
    seen_inits: set[str] = set()

    def add_init(init: TensorProto) -> None:
        if init.name in seen_inits:
            return
        new_initializers.append(init)
        seen_inits.add(init.name)

    for init in target.graph.initializer:
        if init.name in used and init.name not in main_init_names:
            add_init(init)
    for init in quantized_primary.graph.initializer:
        if init.name in used and init.name in main_init_names:
            add_init(init)
    for name in sorted(used):
        if name not in seen_inits and name in target_inits:
            add_init(target_inits[name])
        if name not in seen_inits and name in primary_inits:
            add_init(primary_inits[name])

    del merged.graph.node[:]
    merged.graph.node.extend(new_nodes)
    del merged.graph.initializer[:]
    merged.graph.initializer.extend(new_initializers)

    existing = {value.name for value in merged.graph.input}
    existing.update(value.name for value in merged.graph.output)
    existing.update(init.name for init in merged.graph.initializer)
    value_infos: list[onnx.ValueInfoProto] = []

    def add_value_info(value_info: onnx.ValueInfoProto, name: str | None = None) -> None:
        vi_name = name or value_info.name
        if vi_name in existing:
            return
        value_infos.append(_copy_value_info_with_name(value_info, vi_name))
        existing.add(vi_name)

    for value_info in quantized_primary.graph.value_info:
        add_value_info(value_info, remap.get(value_info.name, value_info.name))
    for value_info in target.graph.value_info:
        if value_info.name.startswith(SHELL_PREFIXES):
            add_value_info(value_info)

    del merged.graph.value_info[:]
    merged.graph.value_info.extend(value_infos)
    _merge_opsets(merged, quantized_primary)
    _order_kv_inputs_first(merged)
    return merged


def extract_and_write_shared(
    models: dict[str, onnx.ModelProto] | list[onnx.ModelProto],
    shared_model_path: Path,
    primary_model: onnx.ModelProto | None = None,
    min_shared_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
) -> dict[str, dict[str, str]]:
    """Extract primary Main initializers to the shared ONNX external-data bundle and redirect models."""
    model_values = list(models.values()) if isinstance(models, dict) else list(models)
    if not model_values:
        raise RuntimeError("No merged models were provided for shared initializer extraction.")
    source = primary_model or model_values[0]
    main_nodes = [node for node in source.graph.node if not _node_is_shell(node)]
    main_inputs = _used_inputs(main_nodes)

    shared: dict[str, TensorProto] = {}
    for init in source.graph.initializer:
        if init.name in main_inputs and _is_shareable_initializer(init, min_shared_elements):
            shared[init.name] = init
    if not shared:
        raise RuntimeError("Quantized primary Main has no initializer to share.")

    external_by_name = _write_shared_initializer_bundle(shared, shared_model_path)
    del shared
    for model in model_values:
        redirect_shared_initializers_to_external(model, external_by_name)
    return external_by_name


def _build_available_merged_graphs(
    source_folder: Path,
    out_folder: Path,
    build_plan,
    main: onnx.ModelProto,
    external_by_name: dict[str, dict[str, str]],
) -> dict[str, Path]:
    graphs: dict[str, Path] = {}
    for name, recipe, dependencies in build_plan:
        if not all((source_folder / dependency).exists() for dependency in dependencies):
            continue
        merged = recipe(source_folder, main)
        redirect_shared_initializers_to_external(merged, external_by_name)
        out_path = out_folder / name
        save_model(merged, out_path)
        graphs[name] = out_path
        del merged
    return graphs


def build_shared_merged_bundle(
    source_folder: Path,
    out_folder: Path | None = None,
    min_shared_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS,
    model_file_names: dict[str, str] | None = None,
) -> dict:
    """Build every available merged decode-strategy graph + the shared weight bundle.

    Produces, inside out_folder (default = source_folder):
        LLM_TextPrefillGreedy.onnx / LLM_TextPrefillPenaltyGreedy.onnx /
        LLM_TextPrefillSampling.onnx
        LLM_DecodeGreedy.onnx / LLM_DecodePenaltyGreedy.onnx /
        LLM_DecodeSampling.onnx
        LLM_SharedInitializers.onnx + LLM_SharedInitializers.onnx.data
    A strategy graph is skipped only if its split decode head is absent. Every merged graph references
    the single LLM_SharedInitializers.onnx.data blob; no LLM_*.onnx.data / .npz / manifest are produced.

    Merged-only: when out_folder == source_folder the absorbed split constituents
    (Main / RotaryPrefill / RotaryDecode / Greedy / TopKTopPSampling / PenaltyGreedy / Penalty) and
    their weight files are deleted, so the large weights exist once and only the merged modules remain.

    """
    source_folder = Path(source_folder)
    out_folder = Path(out_folder) if out_folder is not None else source_folder
    out_folder.mkdir(parents=True, exist_ok=True)

    build_plan = make_merged_build_plan(model_file_names)
    main_name = _model_file_name(model_file_names, "main")
    shared_model_name = _model_file_name(model_file_names, "shared_initializers")
    shared_data_name = (
        model_file_names.get("shared_initializers_data", shared_model_name + ".data")
        if model_file_names is not None else shared_model_name + ".data"
    )
    if shared_data_name != shared_model_name + ".data":
        raise RuntimeError(
            "Shared initializer data file must be named after the shared ONNX model: "
            f"got {shared_data_name!r}, expected {shared_model_name + '.data'!r}."
        )

    if not (source_folder / main_name).exists():
        raise FileNotFoundError(source_folder / main_name)

    shared_model_path = out_folder / shared_model_name
    shared_data_path = out_folder / shared_data_name

    # Remove any legacy forbidden artifacts from older prototypes.
    for artifact in _LEGACY_SHARED_ARTIFACTS:
        (out_folder / artifact).unlink(missing_ok=True)

    # Stream Main's shareable weights once into the shared blob, then strip Main to external refs so
    # every merged graph is built from a light-weight (data-less) Main. `shared` holds references into
    # Main, so it is released before the redirect frees Main's in-memory weights.
    main_for_merge, shared = load_main_with_shared_initializers(source_folder, min_shared_elements, model_file_names)
    external_by_name = _write_shared_initializer_bundle(shared, shared_model_path)
    del shared
    redirect_shared_initializers_to_external(main_for_merge, external_by_name)

    graphs = _build_available_merged_graphs(
        source_folder, out_folder, build_plan, main_for_merge, external_by_name
    )

    result: dict = {
        "graphs": graphs,
        "shared_model": shared_model_path,
        "shared_data": shared_data_path,
    }
    # Merged-only layout: delete the absorbed split constituents in-place so only the merged
    # modules + the single LLM_SharedInitializers.onnx.data weight blob remain.
    if out_folder.resolve() == source_folder.resolve():
        result["removed_constituents"] = delete_merged_constituents(
            source_folder, protected_names=(shared_model_name, shared_data_name)
        )
    return result


def _external_locations(onnx_path: Path) -> set[str]:
    model = onnx.load(str(onnx_path), load_external_data=False)
    locations: set[str] = set()
    for init in model.graph.initializer:
        if init.data_location == TensorProto.EXTERNAL:
            loc = _external_data_map(init).get("location")
            if loc:
                locations.add(loc)
    return locations


def delete_merged_constituents(folder: Path, protected_names: tuple[str, ...] | set[str] | None = None) -> list[str]:
    """Delete the split graphs absorbed into the merged graphs, plus their weight files.

    The merged graphs are self-contained (small embedded constants) and reference the single
    LLM_SharedInitializers.onnx.data blob, so LLM_Main / LLM_RotaryPrefill / LLM_RotaryDecode /
    LLM_Greedy / LLM_PenaltyGreedy and their external weight files are redundant once merged. LLM_SharedInitializers.onnx(.data)
    is never removed. Returns the list of deleted file names.
    """
    folder = Path(folder)
    protected = set(protected_names or (SHARED_MODEL_NAME, SHARED_DATA_NAME))
    removed: list[str] = []
    for name in MERGED_CONSTITUENT_GRAPHS:
        onnx_path = folder / name
        if not onnx_path.exists():
            continue
        for loc in _external_locations(onnx_path):
            if loc in protected:
                continue
            ext = folder / loc
            if ext.exists():
                ext.unlink()
                removed.append(ext.name)
        onnx_path.unlink()
        removed.append(onnx_path.name)
        data_path = onnx_path.with_name(onnx_path.name + ".data")
        if data_path.exists() and data_path.name not in protected:
            data_path.unlink()
            removed.append(data_path.name)
    return removed


# --------------------------------------------------------------------------- #
# Inference-side shared-initializer attachment (mmap + OrtValue + add_initializer)
# --------------------------------------------------------------------------- #
def attach_shared_initializers(session_options, shared_model_path: Path):
    """Attach shared initializers to a SessionOptions via add_initializer.

    Mirrors the Android runtime path: each shared initializer is mmap'd over
    LLM_SharedInitializers.onnx.data (zero-copy) and injected as an OrtValue. The
    returned (arrays, ort_values) MUST be kept alive for the whole session
    lifetime, otherwise the mmaps/OrtValues are garbage-collected and the
    session reads freed memory.
    """
    import onnxruntime as ort  # lazy: export path must not require onnxruntime

    shared_model_path = Path(shared_model_path)
    shared_model = onnx.load(str(shared_model_path), load_external_data=False)
    arrays: dict[str, np.ndarray] = {}
    ort_values: list = []
    for init in shared_model.graph.initializer:
        if init.data_type in _UNSHAREABLE_INIT_TYPES:
            continue
        ext = _external_data_map(init)
        location = ext.get("location")
        if not location:
            raise RuntimeError(f"Shared initializer {init.name!r} is not stored as external data.")
        data_path = shared_model_path.parent / location
        offset = int(ext.get("offset", "0"))
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(init.data_type)
        shape = tuple(int(dim) for dim in init.dims)
        array = np.memmap(data_path, dtype=np_dtype, mode="r", offset=offset, shape=shape)
        arrays[init.name] = array
        ort_value = ort.OrtValue.ortvalue_from_numpy(array)
        ort_values.append(ort_value)
        session_options.add_initializer(init.name, ort_value)
    return arrays, ort_values
