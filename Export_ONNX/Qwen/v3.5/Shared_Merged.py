"""Build Qwen v3.5 multimodal merged ONNX graphs with shared initializers.

The v3.5 language path differs from text-only Qwen v3 because vision features are
prepared by separate image/video preprocess + vision graphs, then inserted before
LLM_Main. This builder keeps those expensive vision front-end graphs standalone
and merges the language-side path for every runtime scenario:

* text prefill:  Embed -> RotaryTextPrefill -> Main -> decode head
* image prefill: Embed -> ConcatImage -> RotaryImagePrefill -> Main -> decode head
* video prefill: Embed -> ConcatVideo -> RotaryVideoPrefill -> Main -> decode head
* text/image/video decode: Embed -> modality RotaryDecode -> Main -> decode head

Each scenario is emitted for greedy, penalty-greedy, and top-k/top-p sampling.
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

# Rotary cos/sin tables share one phase-only name across text/image/video but differ in
# data AND shape (position spans differ), so shared-initializer selection always skips this
# suffix; folding them would give the image/video graphs a wrong-shaped external reference.
ROTARY_TABLE_SUFFIX = "_rotary_pos_emb"

SHELL_PREFIXES = (
    "embed_",
    "concat_image_",
    "concat_video_",
    "prefill_",
    "decode_",
    "greedy_",
    "penalty_greedy_",
    "penalty_",
    "sampling_",
)

SHARED_MODEL_NAME = "LLM_SharedInitializers.onnx"
SHARED_DATA_NAME = SHARED_MODEL_NAME + ".data"

MERGED_CONSTITUENT_GRAPHS = (
    "LLM_Embed.onnx",
    "LLM_Concat_Image.onnx",
    "LLM_Concat_Video.onnx",
    "LLM_Rotary_Image_Prefill.onnx",
    "LLM_Rotary_Image_Decode.onnx",
    "LLM_Rotary_Video_Prefill.onnx",
    "LLM_Rotary_Video_Decode.onnx",
    "LLM_RotaryPrefill.onnx",
    "LLM_RotaryDecode.onnx",
    "LLM_Main.onnx",
    "LLM_Greedy.onnx",
    "LLM_TopKTopPSampling.onnx",
    "LLM_PenaltyGreedy.onnx",
    "LLM_Penalty.onnx",
    "LLM_Argmax.onnx",
)

MODALITIES = ("text", "image", "video")
STRATEGIES = ("greedy", "penalty_greedy", "sampling")
_LEGACY_SHARED_ARTIFACTS = (
    "shared_initializers.npz",
    "shared_initializers.manifest.json",
    "shared_initializers.onnx",
    "shared_initializers.onnx.data",
)


def _build_default_model_file_names() -> dict[str, str]:
    names = {
        "metadata": "LLM_Metadata.onnx",
        "embed": "LLM_Embed.onnx",
        "vision": "LLM_Vision.onnx",
        "image_preprocess": "LLM_Image_Preprocess.onnx",
        "video_preprocess": "LLM_Video_Preprocess.onnx",
        "concat_image": "LLM_Concat_Image.onnx",
        "concat_video": "LLM_Concat_Video.onnx",
        "rotary_image_prefill": "LLM_Rotary_Image_Prefill.onnx",
        "rotary_image_decode": "LLM_Rotary_Image_Decode.onnx",
        "rotary_video_prefill": "LLM_Rotary_Video_Prefill.onnx",
        "rotary_video_decode": "LLM_Rotary_Video_Decode.onnx",
        "rotary_text_prefill": "LLM_RotaryPrefill.onnx",
        "rotary_text_decode": "LLM_RotaryDecode.onnx",
        "main": "LLM_Main.onnx",
        "greedy": "LLM_Greedy.onnx",
        "sampling": "LLM_TopKTopPSampling.onnx",
        "penalty_greedy": "LLM_PenaltyGreedy.onnx",
        "penalty": "LLM_Penalty.onnx",
        "argmax": "LLM_Argmax.onnx",
        "shared_initializers": SHARED_MODEL_NAME,
    }
    names["shared_initializers_data"] = names["shared_initializers"] + ".data"
    for modality in MODALITIES:
        title = modality.capitalize()
        names[f"{modality}_prefill_greedy"] = f"LLM_{title}PrefillGreedy.onnx"
        names[f"{modality}_prefill_penalty_greedy"] = f"LLM_{title}PrefillPenaltyGreedy.onnx"
        names[f"{modality}_prefill_sampling"] = f"LLM_{title}PrefillSampling.onnx"
        names[f"{modality}_decode_greedy"] = f"LLM_{title}DecodeGreedy.onnx"
        names[f"{modality}_decode_penalty_greedy"] = f"LLM_{title}DecodePenaltyGreedy.onnx"
        names[f"{modality}_decode_sampling"] = f"LLM_{title}DecodeSampling.onnx"
    return names


_DEFAULT_MODEL_FILE_NAMES = _build_default_model_file_names()


def default_model_file_names() -> dict[str, str]:
    return dict(_DEFAULT_MODEL_FILE_NAMES)


def _model_file_name(model_file_names: dict[str, str] | None, key: str) -> str:
    names = _DEFAULT_MODEL_FILE_NAMES if model_file_names is None else model_file_names
    return names[key]


def load_model(path: Path) -> onnx.ModelProto:
    return onnx.load(str(path), load_external_data=True)


def save_model(model: onnx.ModelProto, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_name(path.name + ".data").unlink(missing_ok=True)
    onnx.save(model, str(path))


def _external_data_map(init: TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in init.external_data}


def save_shared_initializers_from_tensors(shared: dict[str, TensorProto], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data_name = path.name + ".data"
    data_path = path.with_name(data_name)
    path.unlink(missing_ok=True)
    data_path.unlink(missing_ok=True)

    # Stream each tensor's raw bytes straight into one external-data file and keep only
    # lightweight external references in the shared .onnx. Writing bytes directly (instead
    # of onnx.save_model's copy-then-reserialize) keeps peak memory near the source size.
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


def redirect_shared_initializers_to_external(model: onnx.ModelProto, external_by_name: dict[str, dict[str, str]]) -> int:
    rewritten = []
    count = 0
    for init in model.graph.initializer:
        external = external_by_name.get(init.name)
        if external is None:
            rewritten.append(init)
        else:
            rewritten.append(make_external_initializer_ref(init, external))
            count += 1
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
                dst.metadata_props.add(key=prop.key, value=prop.value)


def merge_models_no_check(first: onnx.ModelProto, second: onnx.ModelProto, io_map: list[tuple[str, str]]) -> onnx.ModelProto:
    # `first` and `second` are only read: every field is copied into a fresh `merged` graph
    # via protobuf append/extend, and the `second`-side input remap is applied to merged's
    # node copies. Sources stay unmutated (reusable across shells) with no defensive deepcopy.
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

    init_by_name: dict[str, TensorProto] = {}
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


def _initializer_num_elements(init: TensorProto) -> int:
    total = 1
    for dim in init.dims:
        total *= int(dim)
    return total


def _is_shareable_initializer(init: TensorProto, min_elements: int) -> bool:
    if init.data_type in (TensorProto.UNDEFINED, TensorProto.STRING):
        return False
    return _initializer_num_elements(init) >= min_elements


def _state_output_names(main: onnx.ModelProto) -> list[str]:
    return [out.name for out in main.graph.output if out.name.startswith("out_")]


def _order_state_inputs_first(model: onnx.ModelProto) -> None:
    state_inputs = [value for value in model.graph.input if value.name.startswith("in_")]
    other_inputs = [value for value in model.graph.input if not value.name.startswith("in_")]
    del model.graph.input[:]
    model.graph.input.extend(state_inputs + other_inputs)


def _with_embed(source_folder: Path, model_file_names: dict[str, str] | None, embed: onnx.ModelProto | None = None) -> onnx.ModelProto:
    if embed is not None:
        return embed
    return prefixed(load_model(source_folder / _model_file_name(model_file_names, "embed")), "embed_")


def _with_rotary(source_folder: Path, modality: str, phase: str, model_file_names: dict[str, str] | None) -> onnx.ModelProto:
    return prefixed(load_model(source_folder / _model_file_name(model_file_names, f"rotary_{modality}_{phase}")), f"{phase}_")


def _hidden_source_for_prefill(source_folder: Path, modality: str, embed: onnx.ModelProto, model_file_names: dict[str, str] | None):
    if modality == "text":
        return embed, "embed_text_hidden_states"

    concat_key = "concat_image" if modality == "image" else "concat_video"
    concat_prefix = "concat_image_" if modality == "image" else "concat_video_"
    concat = prefixed(load_model(source_folder / _model_file_name(model_file_names, concat_key)), concat_prefix)
    merged = merge_models_no_check(embed, concat, [("embed_text_hidden_states", f"{concat_prefix}text_hidden_states")])
    return merged, f"{concat_prefix}concat_hidden_states"


def _deepstack_input_names(main: onnx.ModelProto) -> list[str]:
    names = [value.name for value in main.graph.input if value.name.startswith("deepstack_features_")]
    return sorted(names, key=lambda name: int(name.rsplit("_", 1)[1]))


def _zero_like_model(source_info: onnx.ValueInfoProto, output_name: str, opset: int) -> onnx.ModelProto:
    input_info = copy.deepcopy(source_info)
    input_info.name = f"{output_name}_input"
    output_info = copy.deepcopy(source_info)
    output_info.name = output_name
    element_type = source_info.type.tensor_type.elem_type
    numpy_dtype = onnx.helper.tensor_dtype_to_np_dtype(element_type)
    zero_name = f"{output_name}_scalar"
    zero = numpy_helper.from_array(np.array(0, dtype=numpy_dtype), name=zero_name)
    node = onnx.helper.make_node(
        "Mul", [input_info.name, zero_name], [output_name], name=f"{output_name}_node"
    )
    graph = onnx.helper.make_graph(
        [node], f"{output_name}_graph", [input_info], [output_info], [zero]
    )
    return onnx.helper.make_model(
        graph,
        producer_name="Shared_Merged.py",
        opset_imports=[onnx.helper.make_opsetid("", opset)],
    )


def _with_zero_deepstack(
    hidden_model: onnx.ModelProto,
    hidden_name: str,
    deepstack_inputs: list[str],
    phase: str,
) -> tuple[onnx.ModelProto, list[tuple[str, str]]]:
    if not deepstack_inputs:
        return hidden_model, []
    hidden_info = value_info_by_name(hidden_model).get(hidden_name)
    if hidden_info is None:
        raise RuntimeError(f"Cannot build deepstack zeros: missing value_info for {hidden_name!r}.")
    default_opset = max(
        (item.version for item in hidden_model.opset_import if item.domain == ""),
        default=13,
    )
    zero_name = f"{phase}_zero_deepstack"
    zero_model = _zero_like_model(hidden_info, zero_name, default_opset)
    hidden_model = merge_models_no_check(
        hidden_model,
        zero_model,
        [(hidden_name, f"{zero_name}_input")],
    )
    return hidden_model, [(zero_name, input_name) for input_name in deepstack_inputs]


def _prefill_deepstack_map(
    hidden_model: onnx.ModelProto,
    hidden_name: str,
    main: onnx.ModelProto,
    modality: str,
) -> tuple[onnx.ModelProto, list[tuple[str, str]]]:
    deepstack_inputs = _deepstack_input_names(main)
    if not deepstack_inputs:
        return hidden_model, []
    if modality == "text":
        return _with_zero_deepstack(hidden_model, hidden_name, deepstack_inputs, "prefill")

    concat_prefix = "concat_image_" if modality == "image" else "concat_video_"
    available = set(value_info_by_name(hidden_model))
    io_map = []
    for index, input_name in enumerate(deepstack_inputs):
        output_name = f"{concat_prefix}out_deepstack_feature_{index}"
        if output_name not in available:
            raise RuntimeError(
                f"{modality} concat graph is missing deepstack output {output_name!r}."
            )
        io_map.append((output_name, input_name))
    return hidden_model, io_map


def _base_prefill(source_folder: Path, main: onnx.ModelProto, modality: str, model_file_names: dict[str, str] | None, embed: onnx.ModelProto | None = None):
    embed = _with_embed(source_folder, model_file_names, embed)
    hidden_model, hidden_name = _hidden_source_for_prefill(source_folder, modality, embed, model_file_names)
    hidden_model, deepstack_map = _prefill_deepstack_map(
        hidden_model, hidden_name, main, modality
    )
    rotary = _with_rotary(source_folder, modality, "prefill", model_file_names)
    merged = merge_models_no_check(hidden_model, rotary, [])
    merged = merge_models_no_check(
        merged,
        main,
        [
            (hidden_name, "hidden_states"),
            ("prefill_rotary_cos", "rotary_cos"),
            ("prefill_rotary_sin", "rotary_sin"),
            ("prefill_attention_mask", "attention_mask"),
        ] + deepstack_map,
    )
    return merged, "prefill_kv_seq_len", [hidden_model, rotary]


def _base_decode(source_folder: Path, main: onnx.ModelProto, modality: str, model_file_names: dict[str, str] | None, embed: onnx.ModelProto | None = None):
    embed = _with_embed(source_folder, model_file_names, embed)
    embed, deepstack_map = _with_zero_deepstack(
        embed,
        "embed_text_hidden_states",
        _deepstack_input_names(main),
        "decode",
    )
    rotary = _with_rotary(source_folder, modality, "decode", model_file_names)
    mask_info = next(item for item in main.graph.input if item.name == "attention_mask")
    mask_dtype = onnx.helper.tensor_dtype_to_np_dtype(mask_info.type.tensor_type.elem_type)
    rotary.graph.initializer.append(
        numpy_helper.from_array(np.zeros((1, 1, 1, 1, 1), dtype=mask_dtype), name="decode_zero_attention_mask")
    )
    merged = merge_models_no_check(embed, rotary, [])
    merged = merge_models_no_check(
        merged,
        main,
        [
            ("embed_text_hidden_states", "hidden_states"),
            ("decode_rotary_cos", "rotary_cos"),
            ("decode_rotary_sin", "rotary_sin"),
            ("decode_zero_attention_mask", "attention_mask"),
        ] + deepstack_map,
    )
    return merged, "decode_kv_seq_len_next", [embed, rotary]


def _finalize(merged: onnx.ModelProto, output_names: list[str], *metadata_sources: onnx.ModelProto) -> onnx.ModelProto:
    set_graph_outputs(merged, output_names)
    _order_state_inputs_first(merged)
    copy_metadata(merged, *metadata_sources)
    merged.producer_name = "Shared_Merged.py"
    return merged


def _merge_greedy(source_folder: Path, main: onnx.ModelProto, modality: str, phase: str, model_file_names: dict[str, str] | None, embed: onnx.ModelProto | None = None):
    base, kv_seq, meta = (_base_prefill if phase == "prefill" else _base_decode)(source_folder, main, modality, model_file_names, embed)
    greedy = prefixed(load_model(source_folder / _model_file_name(model_file_names, "greedy")), "greedy_")
    merged = merge_models_no_check(base, greedy, [("logits", "greedy_logits")])
    return _finalize(merged, _state_output_names(main) + ["greedy_max_logits_idx", kv_seq], main, *meta, greedy)


def _merge_sampling(source_folder: Path, main: onnx.ModelProto, modality: str, phase: str, model_file_names: dict[str, str] | None, embed: onnx.ModelProto | None = None):
    base, kv_seq, meta = (_base_prefill if phase == "prefill" else _base_decode)(source_folder, main, modality, model_file_names, embed)
    sampling = prefixed(load_model(source_folder / _model_file_name(model_file_names, "sampling")), "sampling_")
    merged = merge_models_no_check(base, sampling, [("logits", "sampling_logits")])
    return _finalize(merged, _state_output_names(main) + ["sampling_sampled_id", "sampling_save_id_out", kv_seq], main, *meta, sampling)


def _merge_penalty_greedy(source_folder: Path, main: onnx.ModelProto, modality: str, phase: str, model_file_names: dict[str, str] | None, embed: onnx.ModelProto | None = None):
    base, kv_seq, meta = (_base_prefill if phase == "prefill" else _base_decode)(source_folder, main, modality, model_file_names, embed)
    penalty_greedy = prefixed(load_model(source_folder / _model_file_name(model_file_names, "penalty_greedy")), "penalty_greedy_")
    if phase == "prefill":
        merged = merge_models_no_check(base, penalty_greedy, [("logits", "penalty_greedy_logits")])
    else:
        penalty = prefixed(load_model(source_folder / _model_file_name(model_file_names, "penalty")), "penalty_")
        merged = merge_models_no_check(base, penalty, [("logits", "penalty_logits_in")])
        merged = merge_models_no_check(merged, penalty_greedy, [("penalty_logits_out", "penalty_greedy_logits")])
        meta.append(penalty)
    return _finalize(
        merged,
        _state_output_names(main) + ["penalty_greedy_max_logits_idx", "penalty_greedy_save_id_out", kv_seq],
        main,
        *meta,
        penalty_greedy,
    )


_STRATEGY_MERGERS = {
    "greedy": _merge_greedy,
    "penalty_greedy": _merge_penalty_greedy,
    "sampling": _merge_sampling,
}


def _recipe(modality: str, phase: str, strategy: str):
    merge = _STRATEGY_MERGERS.get(strategy)

    def build(source_folder: Path, main: onnx.ModelProto, model_file_names: dict[str, str] | None = None, embed: onnx.ModelProto | None = None):
        if merge is None:
            raise ValueError(f"Unknown strategy {strategy!r}")
        return merge(source_folder, main, modality, phase, model_file_names, embed)

    build.__name__ = f"merge_{modality}_{phase}_{strategy}"
    return build


def _deps_for(modality: str, phase: str, strategy: str, model_file_names: dict[str, str] | None) -> list[str]:
    deps = [_model_file_name(model_file_names, "embed"), _model_file_name(model_file_names, "main")]
    if phase == "prefill" and modality != "text":
        deps.append(_model_file_name(model_file_names, "concat_image" if modality == "image" else "concat_video"))
    deps.append(_model_file_name(model_file_names, f"rotary_{modality}_{phase}"))
    if strategy == "greedy":
        deps.append(_model_file_name(model_file_names, "greedy"))
    elif strategy == "penalty_greedy":
        deps.append(_model_file_name(model_file_names, "penalty_greedy"))
        if phase == "decode":
            deps.append(_model_file_name(model_file_names, "penalty"))
    elif strategy == "sampling":
        deps.append(_model_file_name(model_file_names, "sampling"))
    return deps


_MERGED_BUILD_SPECS = tuple(
    (modality, phase, strategy, _recipe(modality, phase, strategy))
    for modality in MODALITIES
    for phase in ("prefill", "decode")
    for strategy in STRATEGIES
)


def make_merged_build_plan(model_file_names: dict[str, str] | None = None):
    return [
        (
            _model_file_name(model_file_names, f"{modality}_{phase}_{strategy}"),
            recipe,
            _deps_for(modality, phase, strategy, model_file_names),
        )
        for modality, phase, strategy, recipe in _MERGED_BUILD_SPECS
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


def _target_main_remap(target: onnx.ModelProto) -> dict[str, str]:
    names = {value.name for value in target.graph.input}
    names.update(init.name for init in target.graph.initializer)
    for node in target.graph.node:
        names.update(node.output)
    remap = {}
    if "decode_rotary_cos" not in names:
        if "concat_image_concat_hidden_states" in names:
            remap["embed_text_hidden_states"] = "concat_image_concat_hidden_states"
        elif "concat_video_concat_hidden_states" in names:
            remap["embed_text_hidden_states"] = "concat_video_concat_hidden_states"
        return remap
    remap.update({
        "prefill_rotary_cos": "decode_rotary_cos",
        "prefill_rotary_sin": "decode_rotary_sin",
        "prefill_attention_mask": "decode_zero_attention_mask",
    })
    return remap


def _target_deepstack_sources(target: onnx.ModelProto) -> list[str]:
    names = {value.name for value in target.graph.input}
    names.update(init.name for init in target.graph.initializer)
    for node in target.graph.node:
        names.update(node.output)

    for prefix in (
        "concat_image_out_deepstack_feature_",
        "concat_video_out_deepstack_feature_",
    ):
        sources = [name for name in names if name.startswith(prefix)]
        if sources:
            return sorted(sources, key=lambda name: int(name.rsplit("_", 1)[1]))
    for zero_name in ("decode_zero_deepstack", "prefill_zero_deepstack"):
        if zero_name in names:
            return [zero_name]
    return []


def remap_transplanted_deepstack_inputs(
    nodes: list[onnx.NodeProto], target: onnx.ModelProto
) -> int:
    """Map text-prefill donor zeros to the target shell's deepstack tensors."""
    donor_prefix = "prefill_zero_deepstack_"

    def is_donor_feature(name: str) -> bool:
        if name == "prefill_zero_deepstack":
            return True
        return name.startswith(donor_prefix) and name[len(donor_prefix):].isdigit()

    locations = []
    for node in nodes:
        for input_index, name in enumerate(node.input):
            if is_donor_feature(name):
                locations.append((node, input_index))
    if not locations:
        return 0

    sources = _target_deepstack_sources(target)
    if not sources:
        raise RuntimeError(
            "Quantized Main uses deepstack inputs, but the target shell provides none."
        )
    if len(sources) == 1:
        for node, input_index in locations:
            node.input[input_index] = sources[0]
        return len(locations)
    if len(locations) != len(sources):
        raise RuntimeError(
            "Deepstack transplant mismatch: "
            f"donor uses={len(locations)}, target sources={len(sources)}."
        )
    for (node, input_index), source in zip(locations, sources):
        node.input[input_index] = source
    return len(locations)


def transplant_quantized_main(target: onnx.ModelProto, quantized_primary: onnx.ModelProto) -> onnx.ModelProto:
    remap = _target_main_remap(target)
    primary_main_nodes = [
        _copy_node_with_input_remap(node, remap)
        for node in quantized_primary.graph.node
        if not _node_is_shell(node)
    ]
    if not primary_main_nodes:
        raise RuntimeError("Quantized primary graph has no Main node block to transplant.")
    remap_transplanted_deepstack_inputs(primary_main_nodes, target)

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
        if init.name not in seen_inits:
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
        if vi_name not in existing:
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
    _order_state_inputs_first(merged)
    return merged


def _collect_shareable_initializers(sources: list[onnx.ModelProto], min_shared_elements: int) -> dict[str, TensorProto]:
    shared: dict[str, TensorProto] = {}
    for source in sources:
        for init in source.graph.initializer:
            name = init.name
            if name in shared or name.endswith(ROTARY_TABLE_SUFFIX):
                continue
            if _is_shareable_initializer(init, min_shared_elements):
                shared[name] = init
    return shared


def write_shared_initializers(sources: onnx.ModelProto | list[onnx.ModelProto], shared_model_path: Path, min_shared_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS) -> dict[str, dict[str, str]]:
    """Serialize the shared-initializer blob from one or more source graphs.

    Only initializers with at least ``min_shared_elements`` elements are eligible,
    de-duplicated by name (first source wins). Modality/phase-specific rotary tables
    are always excluded (see ``ROTARY_TABLE_SUFFIX``). The referenced tensors are not
    copied here; ``save_shared_initializers_from_tensors`` streams their raw bytes
    directly to disk, keeping peak memory low for multi-GB weights.
    """
    if isinstance(sources, onnx.ModelProto):
        sources = [sources]
    shared = _collect_shareable_initializers(sources, min_shared_elements)
    if not shared:
        raise RuntimeError("No shareable initializers were found for the shared blob.")
    save_shared_initializers_from_tensors(shared, shared_model_path)
    return shared_external_data_map(shared_model_path)


def _available_build_plan(source_folder: Path, model_file_names: dict[str, str] | None):
    return [
        (file_name, recipe, deps)
        for file_name, recipe, deps in make_merged_build_plan(model_file_names)
        if all((source_folder / dep).exists() for dep in deps)
    ]


def _prepare_shared_language_sources(
    source_folder: Path,
    shared_model_path: Path,
    model_file_names: dict[str, str] | None,
    min_shared_elements: int,
) -> tuple[onnx.ModelProto, onnx.ModelProto, dict[str, dict[str, str]]]:
    main = load_model(source_folder / _model_file_name(model_file_names, "main"))
    embed = _with_embed(source_folder, model_file_names)
    external_by_name = write_shared_initializers(
        [main, embed], shared_model_path, min_shared_elements
    )
    redirect_shared_initializers_to_external(main, external_by_name)
    redirect_shared_initializers_to_external(embed, external_by_name)
    return main, embed, external_by_name


def _build_available_merged_graphs(
    source_folder: Path,
    out_folder: Path,
    available_plan,
    main: onnx.ModelProto,
    embed: onnx.ModelProto,
    model_file_names: dict[str, str] | None,
    external_by_name: dict[str, dict[str, str]],
) -> dict[str, Path]:
    graphs: dict[str, Path] = {}
    for file_name, recipe, _ in available_plan:
        model = recipe(source_folder, main, model_file_names, embed=embed)
        redirect_shared_initializers_to_external(model, external_by_name)
        out_path = out_folder / file_name
        save_model(model, out_path)
        graphs[file_name] = out_path
        del model
    return graphs


def build_shared_merged_bundle(source_folder: Path, out_folder: Path | None = None, min_shared_elements: int = MIN_SHARED_INITIALIZER_ELEMENTS, model_file_names: dict[str, str] | None = None, delete_constituents: bool = False) -> dict:
    source_folder = Path(source_folder)
    out_folder = Path(out_folder) if out_folder is not None else source_folder
    out_folder.mkdir(parents=True, exist_ok=True)

    main_name = _model_file_name(model_file_names, "main")
    if not (source_folder / main_name).exists():
        raise FileNotFoundError(source_folder / main_name)

    shared_model_name = _model_file_name(model_file_names, "shared_initializers")
    shared_data_name = _model_file_name(model_file_names, "shared_initializers_data")
    expected_data_name = shared_model_name + ".data"
    if shared_data_name != expected_data_name:
        raise RuntimeError(f"Shared initializer data must be {expected_data_name!r}, got {shared_data_name!r}.")

    for legacy in _LEGACY_SHARED_ARTIFACTS:
        (out_folder / legacy).unlink(missing_ok=True)

    available = _available_build_plan(source_folder, model_file_names)
    if not available:
        raise RuntimeError("No complete multimodal merged graph recipes are available from the exported split graphs.")

    # Only Main weights and the embedding table are reused verbatim by every strategy graph:
    # write those two into the shared blob and redirect to external refs, then build and save
    # graphs one at a time (low peak memory). Rotary tables stay inline (per-graph data/shape).
    shared_model_path = out_folder / shared_model_name
    main, embed, external_by_name = _prepare_shared_language_sources(
        source_folder, shared_model_path, model_file_names, min_shared_elements
    )
    graphs = _build_available_merged_graphs(
        source_folder,
        out_folder,
        available,
        main,
        embed,
        model_file_names,
        external_by_name,
    )

    result: dict = {
        "graphs": graphs,
        "shared_model": shared_model_path,
        "shared_data": out_folder / shared_data_name,
    }
    if delete_constituents and out_folder.resolve() == source_folder.resolve():
        result["removed_constituents"] = delete_merged_constituents(
            source_folder,
            protected_names=(shared_model_name, shared_data_name),
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
    folder = Path(folder)
    protected = set(protected_names or (SHARED_MODEL_NAME, SHARED_DATA_NAME))
    removed: list[str] = []
    for name in MERGED_CONSTITUENT_GRAPHS:
        onnx_path = folder / name
        if not onnx_path.exists():
            continue
        for loc in _external_locations(onnx_path):
            if loc not in protected:
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


def attach_shared_initializers(session_options, shared_model_path: Path):
    import onnxruntime as ort

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