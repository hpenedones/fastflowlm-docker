#!/usr/bin/env python3
"""Scaffold a Qwen3.5 MoE converter spike in a local FLM_Q4NX_Converter checkout.

This does not claim that the public converter can already convert
`Qwen3.5-35B-A3B`. Instead, it creates the minimum file/config skeleton needed
to start a text-only `qwen3_5_moe` spike using the public GPT-OSS MoE packing
path as the closest available baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


CONSTANTS_PATH = Path("q4nx/constants.py")
MODELS_INIT_PATH = Path("q4nx/models/__init__.py")
GPT_OSS_MODEL_PATH = Path("q4nx/models/gpt_oss.py")
QWEN3_CONFIG_PATH = Path("configs/qwen3.json")
NEW_MODEL_PATH = Path("q4nx/models/qwen3_5_moe.py")
NEW_CONFIG_PATH = Path("configs/qwen3_5_moe.json")


QWEN35_MODEL_SOURCE = """from .gpt_oss import GPTOSS
from ..constants import ModelArch


class Qwen35MoE(GPTOSS, model_arch=ModelArch.QWEN3_5_MOE):
    \"\"\"Scaffold converter for text-only `qwen3_5_moe` experiments.

    This intentionally reuses the public GPT-OSS MoE packing path as the
    nearest public baseline. Before attempting real Qwen3.5 conversion, review:
    - GGUF tensor names for router and expert weights
    - hybrid linear/full-attention tensor metadata
    - any Qwen3.5-specific norms or projection layouts
    \"\"\"

    pass
"""


QWEN35_CONFIG = {
    "q4nx_config": {
        "row_block_size": 32,
        "col_block_size": 128,
        "parallel_size": 16,
        "keep_block_in_2D": True,
    },
    "default_tensor_type": "Q4_1",
    "notes": [
        "Scaffold config for qwen3_5_moe text-only experiments.",
        "This starts from the public GPT-OSS MoE packing path, not from a verified Qwen3.5 converter.",
        "Review the GGUF tensor names and hybrid-attention metadata before attempting a real conversion.",
        "Vision weights are intentionally out of scope for this scaffold.",
    ],
    "name_map": {
        "embedding": {
            "gguf_name": "token_embd.weight",
            "q4nx_name": "model.embed_tokens.weight",
        },
        "q_proj": {
            "gguf_name": "blk.{bid}.attn_q.weight",
            "q4nx_name": "model.layers.{bid}.self_attn.q_proj.weight",
        },
        "q_norm": {
            "gguf_name": "blk.{bid}.attn_q_norm.weight",
            "q4nx_name": "model.layers.{bid}.self_attn.q_norm.weight",
        },
        "k_proj": {
            "gguf_name": "blk.{bid}.attn_k.weight",
            "q4nx_name": "model.layers.{bid}.self_attn.k_proj.weight",
        },
        "k_norm": {
            "gguf_name": "blk.{bid}.attn_k_norm.weight",
            "q4nx_name": "model.layers.{bid}.self_attn.k_norm.weight",
        },
        "v_proj": {
            "gguf_name": "blk.{bid}.attn_v.weight",
            "q4nx_name": "model.layers.{bid}.self_attn.v_proj.weight",
        },
        "o_proj": {
            "gguf_name": "blk.{bid}.attn_output.weight",
            "q4nx_name": "model.layers.{bid}.self_attn.o_proj.weight",
        },
        "up_proj": {
            "gguf_name": "blk.{bid}.ffn_up_exps.weight",
            "q4nx_name": "model.layers.{bid}.ffn_up_exps.weight",
        },
        "up_proj_bias": {
            "gguf_name": "blk.{bid}.ffn_up_exps.bias",
            "q4nx_name": "model.layers.{bid}.mlp.experts.up_proj_bias",
        },
        "gate_proj": {
            "gguf_name": "blk.{bid}.ffn_gate_exps.weight",
            "q4nx_name": "model.layers.{bid}.ffn_gate_exps.weight",
        },
        "gate_proj_bias": {
            "gguf_name": "blk.{bid}.ffn_gate_exps.bias",
            "q4nx_name": "model.layers.{bid}.mlp.experts.gate_proj_bias",
        },
        "down_exp_proj": {
            "gguf_name": "blk.{bid}.ffn_down_exps.weight",
            "q4nx_name": "model.layers.{bid}.ffn_down_exps.weight",
        },
        "down_exp_proj_bias": {
            "gguf_name": "blk.{bid}.ffn_down_exps.bias",
            "q4nx_name": "model.layers.{bid}.mlp.experts.down_proj_bias",
        },
        "pre_attn_norm": {
            "gguf_name": "blk.{bid}.attn_norm.weight",
            "q4nx_name": "model.layers.{bid}.input_layernorm.weight",
        },
        "post_attn_norm": {
            "gguf_name": "blk.{bid}.ffn_norm.weight",
            "q4nx_name": "model.layers.{bid}.post_attention_layernorm.weight",
        },
        "router_weight": {
            "gguf_name": "blk.{bid}.ffn_gate_inp.weight",
            "q4nx_name": "model.layers.{bid}.mlp.router.weight",
        },
        "router_bias": {
            "gguf_name": "blk.{bid}.ffn_gate_inp.bias",
            "q4nx_name": "model.layers.{bid}.mlp.router.bias",
        },
        "model_final_norm": {
            "gguf_name": "output_norm.weight",
            "q4nx_name": "model.norm.weight",
        },
        "lm_head": {
            "gguf_name": "output.weight",
            "q4nx_name": "lm_head.weight",
        },
    },
}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SystemExit(f"Required file not found: {path}") from exc


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def ensure_anchor_patch(text: str, needle: str, addition: str) -> str:
    if addition in text:
        return text
    if needle not in text:
        raise SystemExit(f"Could not find expected anchor in target file: {needle!r}")
    return text.replace(needle, needle + addition, 1)


def patch_constants(text: str) -> str:
    text = ensure_anchor_patch(
        text,
        "    QWEN3   = auto()\n",
        "    QWEN3_5_MOE = auto()\n",
    )
    text = ensure_anchor_patch(
        text,
        '    ModelArch.QWEN3:   ["qwen3"],\n',
        '    ModelArch.QWEN3_5_MOE: ["qwen3_5_moe", "qwen3.5", "qwen3.5-moe"],\n',
    )
    text = ensure_anchor_patch(
        text,
        '    ModelArch.QWEN3:   "qwen3.json",\n',
        '    ModelArch.QWEN3_5_MOE: "qwen3_5_moe.json",\n',
    )
    return text


def patch_models_init(text: str) -> str:
    text = ensure_anchor_patch(
        text,
        "from .qwen3 import Qwen3\n",
        "from .qwen3_5_moe import Qwen35MoE\n",
    )
    if "'Qwen35MoE'" not in text:
        old = "__all__ = ['Qwen3VL', 'Llama', 'LFM2', 'Qwen3', 'Qwen2', 'Qwen2VL', 'Gemma3', 'Phi4', 'GPTOSS', 'Nanbeige']"
        new = "__all__ = ['Qwen3VL', 'Llama', 'LFM2', 'Qwen3', 'Qwen35MoE', 'Qwen2', 'Qwen2VL', 'Gemma3', 'Phi4', 'GPTOSS', 'Nanbeige']"
        if old not in text:
            raise SystemExit("Could not find expected __all__ declaration in q4nx/models/__init__.py")
        text = text.replace(old, new, 1)
    return text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scaffold a qwen3_5_moe converter spike into a local "
            "FLM_Q4NX_Converter checkout."
        )
    )
    parser.add_argument(
        "converter_root",
        type=Path,
        help="Path to a local FLM_Q4NX_Converter checkout",
    )
    parser.add_argument(
        "--replace-existing-files",
        action="store_true",
        help="Overwrite existing scaffold files if they already exist",
    )
    return parser


def verify_checkout(root: Path) -> None:
    required = [
        CONSTANTS_PATH,
        MODELS_INIT_PATH,
        GPT_OSS_MODEL_PATH,
        QWEN3_CONFIG_PATH,
    ]
    for relative_path in required:
        path = root / relative_path
        if not path.exists():
            raise SystemExit(
                f"{root} does not look like an FLM_Q4NX_Converter checkout; "
                f"missing {relative_path}"
            )


def create_file(path: Path, content: str, replace: bool) -> None:
    if path.exists() and not replace:
        raise SystemExit(
            f"Refusing to overwrite existing scaffold file: {path}. "
            "Pass --replace-existing-files to overwrite it."
        )
    write_text(path, content)


def main() -> int:
    args = build_parser().parse_args()
    root = args.converter_root.resolve()

    verify_checkout(root)

    constants_path = root / CONSTANTS_PATH
    models_init_path = root / MODELS_INIT_PATH
    new_model_path = root / NEW_MODEL_PATH
    new_config_path = root / NEW_CONFIG_PATH

    patched_constants = patch_constants(read_text(constants_path))
    patched_models_init = patch_models_init(read_text(models_init_path))

    write_text(constants_path, patched_constants)
    write_text(models_init_path, patched_models_init)
    create_file(new_model_path, QWEN35_MODEL_SOURCE, args.replace_existing_files)
    create_file(
        new_config_path,
        json.dumps(QWEN35_CONFIG, indent=4) + "\n",
        args.replace_existing_files,
    )

    print(f"Patched: {constants_path}")
    print(f"Patched: {models_init_path}")
    print(f"Created: {new_model_path}")
    print(f"Created: {new_config_path}")
    print("Next manual steps:")
    print("  1. Inspect a real Qwen3.5 GGUF and update configs/qwen3_5_moe.json tensor names.")
    print("  2. Review whether GPTOSS post-processing matches Qwen3.5 expert/router layouts.")
    print("  3. Test converter detection with `python convert.py -f qwen3_5_moe ...`.")
    print("  4. Treat hybrid-attention and vision support as follow-up work, not part of this scaffold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
