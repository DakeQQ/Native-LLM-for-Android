"""
Standalone `tokenizer.txt` exporter.

Extracted from `llmexport.py`'s tokenizer export step so you can produce ONLY the
plain-text `tokenizer.txt` without running the full model export pipeline.

Usage (no command line needed):
    1. Set MODEL_PATH below to your downloaded model folder (the directory that
       contains the tokenizer files, e.g. tokenizer.model / tokenizer.json /
       vocab / merges.txt, and usually config.json).
    2. Optionally set OUTPUT_DIR (defaults to the current folder) and MODEL_TYPE.
    3. Click "Run" in your IDE (or run `python export_tokenizer.py`).

The result is written to `<OUTPUT_DIR>/vocab_<model_name>.txt` (model name taken
from the MODEL_PATH folder).

This script is fully self-contained: the tokenizer export logic is inlined from
MNN's `utils/tokenizer.py`, so it does NOT depend on the MNN repo. Only the
`transformers` package (plus `sentencepiece` for SentencePiece models) is needed.
It always writes the plain-text `tokenizer.txt` (never the binary `tokenizer.mtok`).

NVIDIA NeMo ASR models are also supported: point MODEL_PATH at a `.nemo` archive
(e.g. Nemotron ASR) or a folder containing one, and the embedded SentencePiece
tokenizer is extracted and exported directly (no HuggingFace tokenizer required).
"""

import os
import glob
import json
import base64
import tarfile

from transformers import AutoTokenizer

# ── Configuration ─────────────────────────────────────────────────────────────
# Path to the downloaded model folder (contains the tokenizer files).
MODEL_PATH = "/home/DakeQQ/Downloads/Qwen3.5-0.8B"

# Directory to write tokenizer.txt into. Set to None to write into the current folder.
OUTPUT_DIR = None

# Optional model type override. Set to None to auto-detect from the model's
# config.json "model_type" (falls back gracefully if missing). MODEL_TYPE only
# adjusts a few tokenizer special-cases (e.g. gemma3 / gemma3-text vocab count,
# glm_ocr stop token); any type not listed still exports via the default map.
# NeMo (.nemo) SentencePiece ASR tokenizers (e.g. Nemotron) are auto-detected by
# file format and ignore MODEL_TYPE entirely.
#
# Supported model types (mirrors MNN llmexport `ModelMapper`, model_mapper.py):
# ┌────────────────┬──────────────────────────────────────────────────────────┐
# │ Family         │ model_type value(s) as found in config.json              │
# ├────────────────┼──────────────────────────────────────────────────────────┤
# │ LLaMA / dense  │ llama, internlm, mobilellm, baichuan, llama4_text, mimo, │
# │                │ poi_qwen2_mtp, openelm, hunyuan_v1_dense, gpt_oss        │
# │ Qwen (text)    │ qwen, qwen2, qwen3, qwen3_moe, qwen3_5, qwen3_5_moe      │
# │ Phi            │ phi, phi-msft                                            │
# │ Gemma (text)   │ gemma2, gemma3_text, gemma4                              │
# │ ChatGLM        │ chatglm, chatglm2                                        │
# │ MiniCPM        │ minicpm                                                  │
# │ LFM2 (text)    │ lfm2, lfm2_moe                                           │
# │ Vision-Lang    │ deepseek-vl, internvl_chat, gemma3, idefics3, smolvlm,   │
# │                │ llava_qwen2, qwen2_vl, qwen2_5_vl, qwen3_vl,             │
# │                │ qwen3_vl_moe, minicpmv, glm_ocr, lfm2_vl                 │
# │ Audio / Omni   │ qwen2_5_omni, qwen2_audio, funaudiochat, lfm2_audio      │
# └────────────────┴──────────────────────────────────────────────────────────┘
MODEL_TYPE = None
# ──────────────────────────────────────────────────────────────────────────────


# ── NVIDIA NeMo (.nemo) tokenizer support ─────────────────────────────────
# NeMo ASR checkpoints (e.g. Nemotron) ship the tokenizer as a SentencePiece model
# packed inside the `.nemo` tar archive, with no HuggingFace tokenizer config. These
# helpers locate and load that SentencePiece model so it can be exported directly.
_NEMO_HF_MARKERS = ("tokenizer.json", "tokenizer_config.json", "vocab.json")


def detect_nemo_tokenizer(model_path):
    """Detect an NVIDIA NeMo SentencePiece tokenizer source.

    Returns one of:
      ("nemo", "/path/to/model.nemo")          -- a `.nemo` archive (a file, or one found in a folder)
      ("spm",  "/path/to/xxx_tokenizer.model") -- an already-extracted NeMo SentencePiece model
    or ``None`` when ``model_path`` looks like a regular HuggingFace model folder.
    """
    path = os.path.abspath(os.path.expanduser(model_path))
    if os.path.isfile(path) and path.endswith(".nemo"):
        return ("nemo", path)
    if os.path.isdir(path):
        nemo_files = sorted(glob.glob(os.path.join(path, "*.nemo")))
        if nemo_files:
            return ("nemo", nemo_files[0])
        # A folder may hold an extracted NeMo SentencePiece model but no HuggingFace config.
        if not any(os.path.isfile(os.path.join(path, marker)) for marker in _NEMO_HF_MARKERS):
            sp_candidates = sorted(glob.glob(os.path.join(path, "*_tokenizer.model")))
            plain = os.path.join(path, "tokenizer.model")
            if os.path.isfile(plain):
                sp_candidates.insert(0, plain)
            if sp_candidates:
                return ("spm", sp_candidates[0])
    return None


def _read_nemo_tokenizer_bytes(source):
    """Return the raw SentencePiece model bytes for a detected NeMo ``source``."""
    kind, location = source
    if kind == "spm":
        with open(location, "rb") as handle:
            return handle.read()
    # kind == "nemo": pull the SentencePiece model out of the tar archive in-memory.
    with tarfile.open(location, "r:*") as tar:
        tok_member = next(
            (m for m in tar.getmembers()
             if m.isfile() and (os.path.basename(m.name).endswith("_tokenizer.model")
                                or os.path.basename(m.name) == "tokenizer.model")),
            None,
        )
        if tok_member is None:
            raise SystemExit(
                f"No SentencePiece tokenizer (*_tokenizer.model / tokenizer.model) found inside {location}."
            )
        with tar.extractfile(tok_member) as src:
            return src.read()


def _load_nemo_sp_model(source):
    """Load the NeMo SentencePiece model from a detected ``source`` into a processor."""
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise SystemExit(
            "The `sentencepiece` package is required to export NeMo (.nemo) tokenizers. "
            "Install it with: pip install sentencepiece"
        ) from exc
    return spm.SentencePieceProcessor(model_proto=_read_nemo_tokenizer_bytes(source))


class TokenizerExporter:
    """Self-contained tokenizer -> plain-text `tokenizer.txt` exporter.

    All logic is inlined from MNN's `utils/tokenizer.py` so this script has no
    dependency on the MNN repo. Only `transformers` (and `sentencepiece` for
    SentencePiece models) are required at runtime. It always writes the
    plain-text `tokenizer.txt` format (never the binary `tokenizer.mtok`).
    """

    def __init__(self, tokenizer_path, model_type):
        self.tokenizer_path = tokenizer_path
        self.model_type = model_type
        self.is_nemo = False
        self.nemo_source = None
        self.nemo_sp = None

        nemo_source = detect_nemo_tokenizer(tokenizer_path)
        if nemo_source is not None:
            # NVIDIA NeMo (.nemo) SentencePiece ASR tokenizer: no HuggingFace tokenizer is present.
            self.is_nemo = True
            self.tokenizer = None
            self.nemo_source = nemo_source
            self.nemo_sp = _load_nemo_sp_model(nemo_source)
            self.stop_ids = self._collect_nemo_stop_ids(self.nemo_sp)
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=True)
        self.stop_ids = self._collect_stop_ids(tokenizer_path, model_type)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, model_type):
        return cls(pretrained_model_name_or_path, model_type)

    def _collect_stop_ids(self, tokenizer_path, model_type):
        from collections.abc import Iterable
        stop_ids = []
        stop_ids.append(self.tokenizer.eos_token_id)
        if hasattr(self.tokenizer, 'im_end_id'):
            stop_ids.append(self.tokenizer.im_end_id)
        try:
            eot_id = self.tokenizer.encode('<|eot_id|>')
            if len(eot_id) == 1:
                stop_ids.append(eot_id[0])
            eot_id = self.tokenizer.encode('<end_of_turn>')
            if len(eot_id) == 2 and eot_id[0] == 2:
                stop_ids.append(eot_id[1])
        except Exception:
            pass
        if hasattr(self.tokenizer, 'generation_config') and self.tokenizer.generation_config is not None:
            eos_token_id = self.tokenizer.generation_config.eos_token_id
            if isinstance(eos_token_id, int):
                stop_ids.append(eos_token_id)
            elif isinstance(eos_token_id, Iterable):
                for id in eos_token_id:
                    stop_ids.append(id)
        gen_cfg_path = os.path.join(tokenizer_path, 'generation_config.json')
        if os.path.isfile(gen_cfg_path):
            try:
                with open(gen_cfg_path, 'r') as f:
                    gen_cfg = json.load(f)
                eos_token_id = gen_cfg.get('eos_token_id')
                if isinstance(eos_token_id, int):
                    stop_ids.append(eos_token_id)
                elif isinstance(eos_token_id, Iterable):
                    for id in eos_token_id:
                        stop_ids.append(id)
            except Exception:
                pass
        # gemma4: <turn|> is end-of-turn
        try:
            turn_ids = self.tokenizer.encode('<turn|>', add_special_tokens=False)
            if len(turn_ids) == 1 and turn_ids[0] not in stop_ids:
                stop_ids.append(turn_ids[0])
        except Exception:
            pass
        if model_type == 'glm_ocr':
            user_ids = self.tokenizer.encode('<|user|>', add_special_tokens=False)
            if len(user_ids) == 1:
                stop_ids.append(user_ids[0])
        stop_ids = [stop_id for stop_id in stop_ids if stop_id is not None]
        stop_ids = list(set(stop_ids))
        return stop_ids

    def _collect_nemo_stop_ids(self, sp_model):
        """Stop ids for a NeMo SentencePiece tokenizer (EOS only, when the model defines one)."""
        stop_ids = []
        eos_id = sp_model.eos_id()
        if isinstance(eos_id, int) and eos_id >= 0:
            stop_ids.append(eos_id)
        return list(set(stop_ids))

    def _export_nemo(self, save_directory, model_path):
        """Export a NeMo (.nemo) SentencePiece tokenizer to the plain-text SENTENCEPIECE format."""
        os.makedirs(save_directory, exist_ok=True)
        sp_model = self.nemo_sp

        # TOKENIZER MAGIC NUMBER / TYPE (mirrors the SentencePiece branch of export()).
        MAGIC_NUMBER = 430
        SENTENCEPIECE = 0
        # TOKENIZER TOKEN TYPES
        NORMAL = 1; UNKNOWN = 2; CONTROL = 3; UNUSED = 5; BYTE = 6

        # Special tokens = SentencePiece control + unknown pieces (e.g. <unk>).
        special_list = [i for i in range(sp_model.GetPieceSize())
                        if sp_model.IsControl(i) or sp_model.IsUnknown(i)]
        prefix_list = [sp_model.bos_id()] if sp_model.bos_id() >= 0 else []
        stop_ids = self.stop_ids

        vocab_list = []
        for i in range(sp_model.GetPieceSize()):
            token = sp_model.IdToPiece(i)
            score = sp_model.GetScore(i)
            if sp_model.IsUnknown(i):
                token_type = UNKNOWN
            elif sp_model.IsControl(i):
                token_type = CONTROL
            elif sp_model.IsUnused(i):
                token_type = UNUSED
            elif sp_model.IsByte(i):
                token_type = BYTE
            else:
                token_type = NORMAL
            if '▁' in token:
                token = token.replace('▁', ' ')
            token_encode = base64.b64encode(token.encode("utf-8")).decode("utf8")
            vocab_list.append(f'{token_encode} {score} {token_type}\n')

        model_name = os.path.basename(os.path.normpath(model_path))
        if model_name.endswith('.nemo'):
            model_name = model_name[:-len('.nemo')]
        file_path = os.path.join(save_directory, f"vocab_{model_name}.txt")

        with open(file_path, "w", encoding="utf8") as fp:
            fp.write(f'{MAGIC_NUMBER} {SENTENCEPIECE}\n')
            fp.write(f'{len(special_list)} {len(stop_ids)} {len(prefix_list)}\n')
            for group in (special_list, stop_ids, prefix_list):
                for token in group:
                    fp.write(str(token) + ' ')
            fp.write('\n')
            fp.write(f'{len(vocab_list)}\n')
            for vocab in vocab_list:
                fp.write(vocab)

        print(f"NeMo SentencePiece tokenizer: {sp_model.GetPieceSize()} pieces, "
              f"{len(special_list)} special, {len(stop_ids)} stop, {len(prefix_list)} prefix.")
        return file_path

    def export(self, save_directory, model_path=None, model_type=None):
        # Use provided values or fall back to instance values
        if model_path is None:
            model_path = self.tokenizer_path
        if model_type is None:
            model_type = self.model_type

        # NeMo (.nemo) SentencePiece ASR tokenizers use a dedicated, HuggingFace-free export path.
        if self.is_nemo:
            return self._export_nemo(save_directory, model_path)

        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # TOKENIZER MAGIC NUMBER
        MAGIC_NUMBER = 430
        # TOKENIZER TYPE
        SENTENCEPIECE = 0; TIKTOIKEN = 1; BERT = 2; HUGGINGFACE = 3

        def write_line(fp, *args):
            for arg in args:
                for token in arg:
                    fp.write(str(token) + ' ')
            fp.write('\n')

        def write_header(fp, type, speicals, prefix=[]):
            fp.write(f'{MAGIC_NUMBER} {type}\n')
            fp.write(f'{len(speicals)} {len(self.stop_ids)} {len(prefix)}\n')
            write_line(fp, speicals, self.stop_ids, prefix)

        model_name = os.path.basename(os.path.normpath(model_path))
        file_path = os.path.join(save_directory, f"vocab_{model_name}.txt")

        # Collect special tokens from various sources
        special_list = list(self.tokenizer.added_tokens_decoder.keys())
        if hasattr(self.tokenizer, 'special_tokens'):
            for k, v in self.tokenizer.special_tokens.items():
                special_list.append(v)
        if hasattr(self.tokenizer, 'all_special_ids'):
            special_list.extend(self.tokenizer.all_special_ids)
        if hasattr(self.tokenizer, 'gmask_token_id'):
            special_list.append(self.tokenizer.gmask_token_id)

        # Handle generation_config special tokens
        if hasattr(self.tokenizer, 'generation_config') and self.tokenizer.generation_config is not None:
            generation_config = self.tokenizer.generation_config
            if hasattr(generation_config, 'user_token_id'):
                special_list.append(generation_config.user_token_id)
            if hasattr(generation_config, 'assistant_token_id'):
                special_list.append(generation_config.assistant_token_id)

        vocab_list = []
        prefix_list = []

        # Get prefix tokens
        if hasattr(self.tokenizer, 'get_prefix_tokens'):
            prefix_list = self.tokenizer.get_prefix_tokens()

        # Simple prefix token detection
        if len(prefix_list) == 0:
            try:
                test_txt = 'A'
                ids = self.tokenizer.encode(test_txt)
                get_txt = self.tokenizer.decode(ids[-1])
                if len(ids) > 1 and get_txt == test_txt:
                    prefix_list += ids[:-1]
            except Exception:
                pass

        # Load SentencePiece model if available
        sp_model = None
        tokenizer_model = os.path.join(model_path, 'tokenizer.model')
        ice_text_model = os.path.join(model_path, 'ice_text.model')

        try:
            import sentencepiece as spm
            if os.path.exists(tokenizer_model):
                sp_model = spm.SentencePieceProcessor(tokenizer_model)
            elif os.path.exists(ice_text_model):
                sp_model = spm.SentencePieceProcessor(ice_text_model)
        except Exception:
            sp_model = None

        # Check for merge file (BERT/HuggingFace tokenizers)
        merge_file = os.path.join(model_path, 'merges.txt')
        merge_txt = merge_file if os.path.exists(merge_file) else None

        if sp_model is not None:
            # SentencePiece tokenizer export
            NORMAL = 1; UNKNOWN = 2; CONTROL = 3
            USER_DEFINED = 4; UNUSED = 5; BYTE = 6

            for i in range(sp_model.GetPieceSize()):
                token = sp_model.IdToPiece(i)
                score = sp_model.GetScore(i)
                token_type = NORMAL
                if sp_model.IsUnknown(i):
                    token_type = UNKNOWN
                elif sp_model.IsControl(i):
                    token_type = CONTROL
                elif sp_model.IsUnused(i):
                    token_type = UNUSED
                elif sp_model.IsByte(i):
                    token_type = BYTE

                # Handle special cases for specific models
                if model_path == 'Chatglm_6b':
                    if '<n>' in token: token = '\n'
                    if '<|tab|>' in token: token = '\t'
                    if '<|blank_' in token: token = ' ' * int(token[8:token.find('|>')])
                if '▁' in token: token = token.replace('▁', ' ')

                token_encode = base64.b64encode(token.encode("utf-8")).decode("utf8")
                vocab_list.append(f'{token_encode} {score} {token_type}\n')

            # Add special tokens to vocab_list
            for index in special_list:
                if index >= len(vocab_list):
                    try:
                        token = self.tokenizer.decode(index)
                        token_encode = base64.b64encode(token.encode("utf-8")).decode("utf8")
                        vocab_list.append(f'{token_encode} {0} {NORMAL}\n')
                    except Exception:
                        pass

            # Write SentencePiece format
            with open(file_path, "w", encoding="utf8") as fp:
                write_header(fp, SENTENCEPIECE, special_list, prefix_list)
                if model_type == "gemma3" or model_type == "gemma3-text":
                    fp.write(f'{len(vocab_list) + 1}\n')  # +1 for image_soft_token
                else:
                    fp.write(f'{len(vocab_list)}\n')
                for vocab in vocab_list:
                    fp.write(vocab)

        elif hasattr(self.tokenizer, 'mergeable_ranks'):
            # TikToken tokenizer export
            vocab_list = []
            for k, v in self.tokenizer.mergeable_ranks.items():
                line = base64.b64encode(k).decode("utf8") + "\n"
                vocab_list.append(line)
            if hasattr(self.tokenizer, 'special_tokens'):
                for k, v in self.tokenizer.special_tokens.items():
                    line = base64.b64encode(k.encode("utf-8")).decode("utf8") + "\n"
                    vocab_list.append(line)
            if hasattr(self.tokenizer, 'added_tokens_decoder'):
                for k, v in self.tokenizer.added_tokens_decoder.items():
                    line = base64.b64encode(v.__str__().encode("utf-8")).decode("utf8") + "\n"
                    vocab_list.append(line)

            # Write TikToken format
            with open(file_path, "w", encoding="utf8") as fp:
                write_header(fp, TIKTOIKEN, special_list, prefix_list)
                fp.write(f'{len(vocab_list)}\n')
                for vocab in vocab_list:
                    fp.write(vocab)

        elif merge_txt is not None:
            # HuggingFace/BERT tokenizer export
            merge_list = []
            vocab = self.tokenizer.get_vocab()
            special_list = list(self.tokenizer.added_tokens_decoder.keys())
            vocab_list = ['<unk>' for i in range(len(vocab))]

            # Load vocab
            for k, v in vocab.items():
                vocab_list[int(v)] = k

            # Load merge
            with open(merge_txt, 'rt') as merge:
                for line in merge.readlines():
                    merge_list.append(line)

            # Write HuggingFace format
            with open(file_path, "w", encoding="utf8") as fp:
                write_header(fp, HUGGINGFACE, special_list)
                fp.write(f'{len(vocab_list)} {len(merge_list)}\n')
                for v in vocab_list:
                    fp.write(v + '\n')
                for m in merge_list:
                    fp.write(m)
        else:
            # Auto-detect tokenizer type and export
            tokenizer_class_name = type(self.tokenizer).__name__.lower()
            vocab = self.tokenizer.get_vocab()

            # Check for SentencePiece-based tokenizers
            if ('xlmroberta' in tokenizer_class_name or
                'roberta' in tokenizer_class_name or
                'sentencepiece' in tokenizer_class_name or
                hasattr(self.tokenizer, 'sp_model') or
                (hasattr(self.tokenizer, 'vocab_file') and
                 self.tokenizer.vocab_file and 'sentencepiece' in self.tokenizer.vocab_file.lower()) or
                # Check for SentencePiece patterns (▁ prefix)
                (len(vocab) > 0 and any('▁' in token for token in list(vocab.keys())[:100]))):
                tokenizer_type = SENTENCEPIECE
                print(f"Detected SentencePiece-based tokenizer: {tokenizer_class_name}")
            elif 'bert' in tokenizer_class_name:
                tokenizer_type = BERT
                print(f"Detected BERT tokenizer: {tokenizer_class_name}")
            else:
                tokenizer_type = TIKTOIKEN
                print(f"Detected TikToken tokenizer: {tokenizer_class_name}")

            vocab = self.tokenizer.get_vocab()

            if tokenizer_type == SENTENCEPIECE:
                # Handle SentencePiece tokenizer
                vocab_list = []
                NORMAL = 1

                for token, token_id in sorted(vocab.items(), key=lambda x: x[1]):
                    try:
                        token_bytes = token.encode('utf-8')
                        token_b64 = base64.b64encode(token_bytes).decode('utf-8')
                        vocab_list.append(f'{token_b64} 0.0 {NORMAL}\n')
                    except Exception as e:
                        print(f"Warning: Failed to encode SentencePiece token '{token}': {e}")
                        token_b64 = base64.b64encode('▁'.encode('utf-8')).decode('utf-8')
                        vocab_list.append(f'{token_b64} 0.0 {NORMAL}\n')

                with open(file_path, "w", encoding="utf8") as fp:
                    write_header(fp, SENTENCEPIECE, special_list, prefix_list)
                    fp.write(f'{len(vocab_list)}\n')
                    for vocab_line in vocab_list:
                        fp.write(vocab_line)
            else:
                # Handle BERT or TikToken tokenizer
                def unicode_to_byte(u: int):
                    # Handle special unicode mappings for BERT tokenizers
                    if u >= 256 and u <= 288:
                        return u - 256
                    if u >= 289 and u <= 322:
                        return u - 162
                    if u == 323:
                        return 173
                    return u

                vocab_list = ['<unk>' for i in range(len(vocab))]

                for k, v in vocab.items():
                    if tokenizer_type == BERT:
                        try:
                            vocab_list[int(v)] = k.encode('utf-8')
                        except Exception as e:
                            try:
                                vocab_list[int(v)] = bytes([unicode_to_byte(ord(c)) for c in k])
                            except Exception as e2:
                                print(f"Warning: Failed to encode token '{k}' with id {v}: {e2}")
                                vocab_list[int(v)] = k.encode('utf-8', errors='replace')
                    else:
                        try:
                            vocab_list[int(v)] = bytes([unicode_to_byte(ord(c)) for c in k])
                        except Exception as e2:
                            print(f"Warning: Failed to encode token '{k}' with id {v}: {e2}")
                            vocab_list[int(v)] = k.encode('utf-8', errors='replace')

                with open(file_path, "w", encoding="utf8") as fp:
                    write_header(fp, tokenizer_type, special_list)
                    fp.write(f'{len(vocab_list)}\n')
                    for v in vocab_list:
                        line = base64.b64encode(v).decode("utf8") + "\n"
                        fp.write(line)

        return file_path


def read_model_type(model_path):
    """Best-effort read of `model_type` from the model folder's config files."""
    for name in ("config.json", "llm_config.json"):
        config_path = os.path.join(model_path, name)
        if os.path.isfile(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception:
                continue
            model_type = config.get("model_type")
            if model_type:
                return model_type
    return None


def main():
    model_path = os.path.abspath(os.path.expanduser(MODEL_PATH))
    is_nemo_file = os.path.isfile(model_path) and model_path.endswith(".nemo")
    if not (os.path.isdir(model_path) or is_nemo_file):
        raise SystemExit(f"MODEL_PATH must be a model directory or a .nemo file: {model_path}")

    output_dir = OUTPUT_DIR if OUTPUT_DIR else os.getcwd()
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    model_type = MODEL_TYPE if MODEL_TYPE else read_model_type(model_path)

    print(f"Loading tokenizer from: {model_path}")
    print(f"Model type            : {model_type}")
    if detect_nemo_tokenizer(model_path) is not None:
        print("Detected NVIDIA NeMo (.nemo) SentencePiece ASR tokenizer.")

    tokenizer = TokenizerExporter.from_pretrained(model_path, model_type)
    out_path = tokenizer.export(output_dir, model_path, model_type)

    if not os.path.isfile(out_path):
        raise SystemExit(f"Export failed: {out_path} was not created.")
    print(f"Exported {os.path.basename(out_path)} -> {out_path}")


if __name__ == "__main__":
    main()
