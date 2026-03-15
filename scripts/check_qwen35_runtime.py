#!/usr/bin/env python3
"""Audit a local FastFlowLM checkout for a Qwen3.5 runtime spike.

This does not patch the upstream runtime. It tells you which reusable pieces
already exist and which Qwen3.5 blockers are still missing from public code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ALL_MODELS_PATH = Path("src/include/AutoModel/all_models.hpp")
QWEN3_HEADER_PATH = Path("src/include/AutoModel/modeling_qwen3.hpp")
QWEN3_CPP_PATH = Path("src/common/AutoModel/modeling_qwen3.cpp")
QWEN3_NPU_PATH = Path("src/include/models/qwen3/qwen3_npu.hpp")
QWEN3_SEQUENCE_PATH = Path("src/include/models/qwen3/qwen3_npu_sequence.hpp")
GPT_OSS_NPU_PATH = Path("src/include/models/gpt_oss/gpt_oss_npu.hpp")
MODEL_LIST_PATH = Path("src/model_list.json")
XCLBINS_PATH = Path("src/xclbins")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SystemExit(f"Required file not found: {path}") from exc


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Required JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse JSON from {path}: {exc}") from exc


def search_terms(root: Path, terms: list[str]) -> dict[str, list[str]]:
    matches: dict[str, list[str]] = {term: [] for term in terms}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".cpp", ".hpp", ".h", ".json"}:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for term in terms:
            if term in content:
                matches[term].append(str(path.relative_to(root)))
    return matches


def collect_report(root: Path, expected_xclbin_dir: str) -> dict[str, Any]:
    all_models = read_text(root / ALL_MODELS_PATH)
    qwen3_header = read_text(root / QWEN3_HEADER_PATH)
    qwen3_cpp = read_text(root / QWEN3_CPP_PATH)
    qwen3_npu = read_text(root / QWEN3_NPU_PATH)
    qwen3_sequence = read_text(root / QWEN3_SEQUENCE_PATH)
    gpt_oss_npu = read_text(root / GPT_OSS_NPU_PATH)
    model_list = read_json(root / MODEL_LIST_PATH)
    xclbins_dir = root / XCLBINS_PATH

    model_families = sorted(model_list.get("models", {}).keys())
    xclbin_dirs = sorted(
        child.name for child in xclbins_dir.iterdir() if child.is_dir()
    )
    keyword_hits = search_terms(
        root,
        ["linear_attention", "full_attention", "router", "expert", "qwen3_5_moe"],
    )

    return {
        "runtime_root": str(root),
        "expected_xclbin_dir": expected_xclbin_dir,
        "reusable_public_bits": {
            "qwen3_family_registered": '"qwen3"' in all_models,
            "gpt_oss_family_registered": '"gpt-oss"' in all_models,
            "qwen3_automodel_class_present": "class Qwen3 : public AutoModel" in qwen3_header,
            "qwen3_loader_uses_qwen3_npu": "std::make_unique<qwen3_npu>" in qwen3_cpp,
            "qwen3_loader_calls_shared_load": "_shared_load_model" in qwen3_cpp,
            "qwen3_sequence_present": "class qwen3_npu_sequence" in qwen3_sequence,
            "gpt_oss_npu_present": "class gpt_oss_npu" in gpt_oss_npu,
        },
        "missing_qwen35_bits": {
            "family_in_all_models_hpp": any(
                token in all_models for token in ("qwen3.5", "qwen3_5", "qwen3_5_moe")
            ),
            "family_in_model_list": any(
                family in model_list.get("models", {})
                for family in ("qwen3.5", "qwen3_5_moe", "qwen35")
            ),
            "expected_xclbin_dir_exists": (xclbins_dir / expected_xclbin_dir).is_dir(),
        },
        "keyword_hits": keyword_hits,
        "sample_existing_xclbins": [
            name for name in xclbin_dirs if name.startswith(("Qwen3", "GPT-OSS"))
        ],
        "recommended_patch_targets": [
            str(ALL_MODELS_PATH),
            str(QWEN3_HEADER_PATH),
            str(QWEN3_CPP_PATH),
            str(MODEL_LIST_PATH),
            str(XCLBINS_PATH / expected_xclbin_dir),
        ],
        "blocker_summary": [
            "Public FastFlowLM already has qwen3 and gpt-oss runtime pieces to reuse.",
            "Public FastFlowLM does not register a qwen3.5 / qwen3_5_moe family yet.",
            f"Expected xclbin directory `{expected_xclbin_dir}` is missing.",
            "If `linear_attention` / `full_attention` do not appear in runtime files, hybrid-attention support is probably still missing.",
            "If `expert` / `router` hits are sparse or unrelated to runtime kernels, MoE routing support likely still needs new work.",
        ],
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"Runtime root: {report['runtime_root']}")
    print(f"Expected xclbin dir: {report['expected_xclbin_dir']}")

    print("\n## Reusable public bits")
    for key, value in report["reusable_public_bits"].items():
        print(f"- {key}: {value}")

    print("\n## Missing qwen3.5 bits")
    for key, value in report["missing_qwen35_bits"].items():
        print(f"- {key}: {value}")

    print("\n## Keyword hits")
    for term, files in report["keyword_hits"].items():
        if files:
            preview = ", ".join(files[:5])
            suffix = " ..." if len(files) > 5 else ""
            print(f"- {term}: {preview}{suffix}")
        else:
            print(f"- {term}: no hits")

    print("\n## Existing Qwen3 / GPT-OSS xclbins")
    for name in report["sample_existing_xclbins"]:
        print(f"- {name}")

    print("\n## Recommended patch targets")
    for path in report["recommended_patch_targets"]:
        print(f"- {path}")

    print("\n## Blocker summary")
    for line in report["blocker_summary"]:
        print(f"- {line}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a local FastFlowLM checkout for a Qwen3.5 runtime spike."
    )
    parser.add_argument(
        "runtime_root",
        type=Path,
        help="Path to a local FastFlowLM checkout",
    )
    parser.add_argument(
        "--expected-xclbin-dir",
        default="Qwen3.5-35B-A3B-NPU2",
        help="Expected xclbin directory name for the target runtime artifact",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of human-readable text",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.runtime_root.resolve()

    for required in (
        ALL_MODELS_PATH,
        QWEN3_HEADER_PATH,
        QWEN3_CPP_PATH,
        QWEN3_NPU_PATH,
        QWEN3_SEQUENCE_PATH,
        GPT_OSS_NPU_PATH,
        MODEL_LIST_PATH,
        XCLBINS_PATH,
    ):
        if not (root / required).exists():
            raise SystemExit(
                f"{root} does not look like a FastFlowLM checkout; missing {required}"
            )

    report = collect_report(root, args.expected_xclbin_dir)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
