#!/usr/bin/env python3
"""Generate an upstream-ready Qwen3.5 bring-up summary from local reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"JSON report not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse JSON from {path}: {exc}") from exc


def bool_word(value: Any) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


def get_architecture_line(report: dict[str, Any], key: str, fallback: str = "unknown") -> str:
    architecture = report.get("architecture")
    if not isinstance(architecture, dict):
        return fallback
    value = architecture.get(key, fallback)
    return str(value)


def render_markdown(
    hf_report: dict[str, Any],
    runtime_report: dict[str, Any],
    community_report: dict[str, Any] | None,
) -> str:
    lines: list[str] = []
    lines.append("# Qwen3.5-35B-A3B FastFlowLM bring-up summary")
    lines.append("")
    lines.append("## Target")
    lines.append(
        f"- Hugging Face repo: `{hf_report.get('repo', 'unknown')}@{hf_report.get('revision', 'unknown')}`"
    )
    lines.append(f"- `model_type`: `{get_architecture_line(hf_report, 'model_type')}`")
    lines.append(f"- `text_model_type`: `{get_architecture_line(hf_report, 'text_model_type')}`")
    lines.append(f"- MoE detected: **{bool_word(hf_report.get('architecture', {}).get('uses_moe') if isinstance(hf_report.get('architecture'), dict) else None)}**")
    lines.append(
        f"- Hybrid attention detected: **{bool_word(hf_report.get('architecture', {}).get('uses_hybrid_attention') if isinstance(hf_report.get('architecture'), dict) else None)}**"
    )
    lines.append(
        f"- Vision config present: **{bool_word(hf_report.get('architecture', {}).get('has_vision') if isinstance(hf_report.get('architecture'), dict) else None)}**"
    )
    lines.append("")
    lines.append("## Artifact path")
    lines.append(
        f"- Official source repo exposes GGUF: **{bool_word(hf_report.get('gguf_found'))}**"
    )
    if hf_report.get("gguf_files"):
        for path in hf_report["gguf_files"]:
            lines.append(f"  - `{path}`")
    if community_report is not None:
        lines.append(
            f"- Community GGUF repo: `{community_report.get('repo', 'unknown')}@{community_report.get('revision', 'unknown')}`"
        )
        lines.append(
            f"- Community repo exposes GGUF: **{bool_word(community_report.get('gguf_found'))}**"
        )
        for path in community_report.get("gguf_files", []):
            lines.append(f"  - `{path}`")
    lines.append("")
    lines.append("## Public converter/runtime gap")
    for reason in hf_report.get("gap_reasons", []):
        lines.append(f"- {reason}")
    lines.append(
        f"- Public runtime qwen3 family present: **{bool_word(runtime_report.get('reusable_public_bits', {}).get('qwen3_family_registered'))}**"
    )
    lines.append(
        f"- Public runtime GPT-OSS family present: **{bool_word(runtime_report.get('reusable_public_bits', {}).get('gpt_oss_family_registered'))}**"
    )
    lines.append(
        f"- `qwen3.5` family wired in `all_models.hpp`: **{bool_word(runtime_report.get('missing_qwen35_bits', {}).get('family_in_all_models_hpp'))}**"
    )
    lines.append(
        f"- `qwen3.5` family present in `model_list.json`: **{bool_word(runtime_report.get('missing_qwen35_bits', {}).get('family_in_model_list'))}**"
    )
    lines.append(
        f"- Expected xclbin dir `{runtime_report.get('expected_xclbin_dir', 'unknown')}` exists: **{bool_word(runtime_report.get('missing_qwen35_bits', {}).get('expected_xclbin_dir_exists'))}**"
    )
    lines.append("")
    lines.append("## Requested upstream work")
    lines.append("- Converter:")
    lines.append("  - add `qwen3_5_moe` architecture handling in `q4nx/constants.py`")
    lines.append("  - add a dedicated `q4nx/models/qwen3_5_moe.py`")
    lines.append("  - add `configs/qwen3_5_moe.json` with verified Qwen3.5 tensor names")
    lines.append("- Runtime:")
    for path in runtime_report.get("recommended_patch_targets", []):
        lines.append(f"  - inspect or patch `{path}`")
    lines.append("- Kernels / packaging:")
    lines.append(
        f"  - provide a public xclbin set for `{runtime_report.get('expected_xclbin_dir', 'unknown')}` or document the supported equivalent"
    )
    lines.append("")
    lines.append("## Reproduction")
    lines.append(
        f"- `python3 scripts/inspect_hf_model.py --repo {hf_report.get('repo', 'Qwen/Qwen3.5-35B-A3B')} --json`"
    )
    lines.append(
        f"- `python3 scripts/check_qwen35_runtime.py /path/to/FastFlowLM --expected-xclbin-dir {runtime_report.get('expected_xclbin_dir', 'Qwen3.5-35B-A3B-NPU2')} --json`"
    )
    if community_report is not None:
        lines.append(
            f"- `python3 scripts/inspect_hf_model.py --repo {community_report.get('repo', 'lmstudio-community/Qwen3.5-35B-A3B-GGUF')} --json`"
        )
    return "\n".join(lines) + "\n"


def render_plain(
    hf_report: dict[str, Any],
    runtime_report: dict[str, Any],
    community_report: dict[str, Any] | None,
) -> str:
    lines: list[str] = []
    lines.append("Qwen3.5-35B-A3B FastFlowLM bring-up summary")
    lines.append("")
    lines.append(f"Target repo: {hf_report.get('repo', 'unknown')}@{hf_report.get('revision', 'unknown')}")
    lines.append(f"model_type: {get_architecture_line(hf_report, 'model_type')}")
    lines.append(f"text_model_type: {get_architecture_line(hf_report, 'text_model_type')}")
    lines.append(
        "Architecture flags: "
        f"MoE={bool_word(hf_report.get('architecture', {}).get('uses_moe') if isinstance(hf_report.get('architecture'), dict) else None)}, "
        f"hybrid_attention={bool_word(hf_report.get('architecture', {}).get('uses_hybrid_attention') if isinstance(hf_report.get('architecture'), dict) else None)}, "
        f"vision={bool_word(hf_report.get('architecture', {}).get('has_vision') if isinstance(hf_report.get('architecture'), dict) else None)}"
    )
    lines.append(f"Official GGUF present: {bool_word(hf_report.get('gguf_found'))}")
    if community_report is not None:
        lines.append(
            f"Community GGUF repo: {community_report.get('repo', 'unknown')} (gguf_found={bool_word(community_report.get('gguf_found'))})"
        )
    lines.append("Gap reasons:")
    for reason in hf_report.get("gap_reasons", []):
        lines.append(f"  - {reason}")
    lines.append("Runtime blockers:")
    for line in runtime_report.get("blocker_summary", []):
        lines.append(f"  - {line}")
    lines.append("Recommended upstream patch targets:")
    for path in runtime_report.get("recommended_patch_targets", []):
        lines.append(f"  - {path}")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an upstream-ready Qwen3.5 summary from "
            "`inspect_hf_model.py` and `check_qwen35_runtime.py` JSON reports."
        )
    )
    parser.add_argument("--hf-report", required=True, type=Path, help="JSON output from inspect_hf_model.py for the official repo")
    parser.add_argument("--runtime-report", required=True, type=Path, help="JSON output from check_qwen35_runtime.py")
    parser.add_argument(
        "--community-gguf-report",
        type=Path,
        help="Optional JSON output from inspect_hf_model.py for a community GGUF repo",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Emit Markdown instead of plain text",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    hf_report = load_json(args.hf_report)
    runtime_report = load_json(args.runtime_report)
    community_report = load_json(args.community_gguf_report) if args.community_gguf_report else None

    if args.markdown:
        sys.stdout.write(render_markdown(hf_report, runtime_report, community_report))
    else:
        sys.stdout.write(render_plain(hf_report, runtime_report, community_report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
