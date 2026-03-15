#!/usr/bin/env python3
"""Inspect a Hugging Face model repo for FastFlowLM/Q4NX bring-up work.

This helper focuses on the first two Qwen3.5 bring-up tasks:
1. Specify the architecture gap against the current public FastFlowLM converter.
2. Prove whether an official GGUF export path exists in the target repo.

It intentionally uses only the Python standard library so it can run without
additional dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_REPO = "Qwen/Qwen3.5-35B-A3B"
DEFAULT_REVISION = "main"
SUPPORTED_CONVERTER_ARCHES = (
    "gemma3",
    "gpt-oss",
    "lfm2",
    "llama",
    "nanbeige",
    "phi4",
    "qwen2",
    "qwen2vl",
    "qwen3",
    "qwen3vl",
)


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "fastflowlm-docker/inspect"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def load_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc


def hf_raw_config_url(repo_id: str, revision: str) -> str:
    quoted_repo = urllib.parse.quote(repo_id, safe="/")
    quoted_rev = urllib.parse.quote(revision, safe="")
    return f"https://huggingface.co/{quoted_repo}/raw/{quoted_rev}/config.json"


def hf_tree_url(repo_id: str, revision: str) -> str:
    quoted_repo = urllib.parse.quote(repo_id, safe="/")
    quoted_rev = urllib.parse.quote(revision, safe="")
    return f"https://huggingface.co/api/models/{quoted_repo}/tree/{quoted_rev}?recursive=1"


def normalize_tree_items(tree: Any) -> list[dict[str, Any]]:
    if isinstance(tree, list):
        return [item for item in tree if isinstance(item, dict)]
    return []


def get_nested(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def summarize_architecture(config: dict[str, Any]) -> dict[str, Any]:
    text_config = get_nested(config, "text_config") or {}
    vision_config = get_nested(config, "vision_config") or {}

    layer_types = text_config.get("layer_types") or []
    layer_type_set = set(layer_types)

    num_experts = text_config.get("num_experts")
    num_experts_per_tok = text_config.get("num_experts_per_tok")
    shared_expert_intermediate_size = text_config.get("shared_expert_intermediate_size")

    return {
        "architectures": config.get("architectures", []),
        "model_type": config.get("model_type"),
        "text_model_type": text_config.get("model_type"),
        "vision_model_type": vision_config.get("model_type") if vision_config else None,
        "num_hidden_layers": text_config.get("num_hidden_layers"),
        "hidden_size": text_config.get("hidden_size"),
        "max_position_embeddings": text_config.get("max_position_embeddings"),
        "has_vision": bool(vision_config),
        "uses_moe": bool(num_experts or num_experts_per_tok or shared_expert_intermediate_size),
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "has_shared_expert": shared_expert_intermediate_size is not None,
        "uses_hybrid_attention": bool(layer_types) and len(layer_type_set) > 1,
        "layer_types": layer_types,
        "full_attention_interval": text_config.get("full_attention_interval"),
        "uses_linear_attention": "linear_attention" in layer_type_set,
        "uses_full_attention": "full_attention" in layer_type_set,
    }


def detect_gap(summary: dict[str, Any]) -> list[str]:
    reasons: list[str] = []

    model_type = summary.get("model_type")
    text_model_type = summary.get("text_model_type")
    if model_type not in SUPPORTED_CONVERTER_ARCHES and text_model_type not in SUPPORTED_CONVERTER_ARCHES:
        reasons.append(
            f"unsupported public converter model_type: {model_type!r} / {text_model_type!r}"
        )

    if summary.get("uses_hybrid_attention"):
        reasons.append("hybrid attention pattern detected (linear + full attention)")

    if summary.get("uses_linear_attention"):
        reasons.append("linear attention / DeltaNet-style layers detected")

    if summary.get("uses_moe"):
        reasons.append(
            "Mixture-of-Experts fields detected "
            f"(experts={summary.get('num_experts')}, active={summary.get('num_experts_per_tok')})"
        )

    if summary.get("has_vision"):
        reasons.append("vision_config present; model is multimodal")

    return reasons


def find_gguf_files(tree_items: list[dict[str, Any]]) -> list[str]:
    gguf_files: list[str] = []
    for item in tree_items:
        path = item.get("path")
        if isinstance(path, str) and path.lower().endswith(".gguf"):
            gguf_files.append(path)
    return sorted(gguf_files)


def print_section(title: str) -> None:
    print(f"\n## {title}")


def print_key_value(label: str, value: Any) -> None:
    print(f"- {label}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a Hugging Face model repo for FastFlowLM/Q4NX bring-up work."
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help=f"Hugging Face repo id (default: {DEFAULT_REPO})",
    )
    parser.add_argument(
        "--revision",
        default=DEFAULT_REVISION,
        help=f"Repo revision (default: {DEFAULT_REVISION})",
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        help="Use a local config.json instead of downloading from Hugging Face",
    )
    parser.add_argument(
        "--tree-file",
        type=Path,
        help="Use a local Hugging Face tree JSON dump instead of querying the API",
    )
    parser.add_argument(
        "--skip-config",
        action="store_true",
        help="Skip config lookup and inspect only the repo tree (useful for GGUF-only repos)",
    )
    parser.add_argument(
        "--skip-tree",
        action="store_true",
        help="Skip tree lookup and inspect only the config",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of human-readable text",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    config: dict[str, Any] | None = None
    config_found = False
    config_source = "skipped"
    tree_items: list[dict[str, Any]] = []
    tree_found = False
    tree_source = "skipped"

    try:
        if args.skip_config:
            config_source = "skipped by --skip-config"
        elif args.config_file:
            config = load_json_file(args.config_file)
            config_source = str(args.config_file)
            config_found = True
        else:
            config_source = hf_raw_config_url(args.repo, args.revision)
            try:
                loaded_config = fetch_json(config_source)
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
            else:
                if isinstance(loaded_config, dict):
                    config = loaded_config
                    config_found = True

        if args.skip_tree:
            tree_source = "skipped by --skip-tree"
        elif args.tree_file:
            tree_items = normalize_tree_items(load_json_file(args.tree_file))
            tree_source = str(args.tree_file)
            tree_found = True
        else:
            tree_source = hf_tree_url(args.repo, args.revision)
            try:
                tree_items = normalize_tree_items(fetch_json(tree_source))
                tree_found = True
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    tree_items = []
                else:
                    raise
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"HTTP error while querying Hugging Face: {exc}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Network error while querying Hugging Face: {exc}") from exc

    summary = summarize_architecture(config) if config is not None else None
    gap_reasons = detect_gap(summary) if summary is not None else []
    gguf_files = find_gguf_files(tree_items)

    report = {
        "repo": args.repo,
        "revision": args.revision,
        "config_source": config_source,
        "config_found": config_found,
        "tree_source": tree_source,
        "tree_found": tree_found,
        "supported_public_converter_arches": list(SUPPORTED_CONVERTER_ARCHES),
        "architecture": summary,
        "gap_reasons": gap_reasons,
        "gguf_files": gguf_files,
        "gguf_found": bool(gguf_files),
        "needs_new_converter_arch": bool(gap_reasons) if summary is not None else None,
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"Repository: {args.repo}@{args.revision}")
    print(f"Config source: {config_source}")
    print(f"Tree source: {tree_source}")

    print_section("Architecture summary")
    if summary is None:
        print("- config_found: no")
        print("- note: no config.json was available; architecture checks were skipped")
    else:
        print_key_value("architectures", summary["architectures"])
        print_key_value("model_type", summary["model_type"])
        print_key_value("text_model_type", summary["text_model_type"])
        print_key_value("vision_model_type", summary["vision_model_type"])
        print_key_value("num_hidden_layers", summary["num_hidden_layers"])
        print_key_value("hidden_size", summary["hidden_size"])
        print_key_value("max_position_embeddings", summary["max_position_embeddings"])
        print_key_value("has_vision", summary["has_vision"])
        print_key_value("uses_moe", summary["uses_moe"])
        if summary["uses_moe"]:
            print_key_value("num_experts", summary["num_experts"])
            print_key_value("num_experts_per_tok", summary["num_experts_per_tok"])
            print_key_value("has_shared_expert", summary["has_shared_expert"])
        print_key_value("uses_hybrid_attention", summary["uses_hybrid_attention"])
        if summary["layer_types"]:
            print_key_value("layer_types", summary["layer_types"])
            print_key_value("full_attention_interval", summary["full_attention_interval"])

    print_section("Public FastFlowLM converter check")
    print_key_value("supported_converter_arches", ", ".join(SUPPORTED_CONVERTER_ARCHES))
    if summary is None:
        print("- verdict: unknown (no config available to compare against converter support)")
    elif gap_reasons:
        print("- verdict: new converter/runtime work is likely required")
        print("- reasons:")
        for reason in gap_reasons:
            print(f"  - {reason}")
    else:
        print("- verdict: architecture looks close to existing public converter support")

    print_section("GGUF path check")
    if gguf_files:
        print("- gguf_found: yes")
        print("- gguf_files:")
        for path in gguf_files:
            print(f"  - {path}")
    else:
        print("- gguf_found: no")
        if tree_found:
            print("- note: the target repo does not expose `.gguf` files in the queried tree")
        else:
            print("- note: tree lookup was skipped or unavailable")

    print_section("Likely next steps")
    if summary is None:
        print("- re-run without `--skip-config` against a source repo that exposes config.json")
    elif gap_reasons:
        print("- compare this config against the pinned public `qwen3` converter path")
        print("- add a dedicated `qwen3.5` converter model class instead of tweaking `qwen3` in place")
        print("- audit FastFlowLM runtime support for hybrid attention, DeltaNet, and MoE decode/prefill")
    if not gguf_files:
        print("- identify a trusted GGUF export path or community GGUF repo before attempting Q4NX conversion")
    else:
        print("- validate the GGUF metadata against the converter's expected architecture naming")

    return 0


if __name__ == "__main__":
    sys.exit(main())
