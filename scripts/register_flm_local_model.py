#!/usr/bin/env python3
"""Create a local FastFlowLM model_list.json overlay entry from a supported tag.

This is useful for experimenting with manually copied local Q4NX artifacts for
families and sizes that FastFlowLM already supports. It does not bypass runtime
limits for brand-new model sizes or unsupported architectures.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path


LOCAL_ONLY_URL = "local://manual-copy-required"


def parse_tag(tag: str) -> tuple[str, str]:
    if ":" not in tag:
        raise argparse.ArgumentTypeError(
            f"Expected a tag of the form family:size, got {tag!r}"
        )

    family, size = tag.split(":", 1)
    if not family or not size:
        raise argparse.ArgumentTypeError(
            f"Expected a tag of the form family:size, got {tag!r}"
        )
    return family, size


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def load_model_list(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Base model list not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse JSON from {path}: {exc}") from exc


def get_entry(data: dict, tag: str) -> dict:
    family, size = parse_tag(tag)
    try:
        return deepcopy(data["models"][family][size])
    except KeyError as exc:
        raise SystemExit(f"Template tag not found in model list: {tag}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a local FastFlowLM model_list.json overlay entry from an "
            "existing supported template tag."
        ),
        epilog=(
            "Example:\n"
            "  python3 scripts/register_flm_local_model.py \\\n"
            "    --base ~/.config/flm/model_list.base.json \\\n"
            "    --output ~/.config/flm/model_list.local.json \\\n"
            "    --template-tag qwen3:8b \\\n"
            "    --new-tag qwen3-local:8b \\\n"
            "    --name Qwen3-8B-Local-NPU2"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base", required=True, type=Path, help="Base model_list.json")
    parser.add_argument("--output", required=True, type=Path, help="Output overlay path")
    parser.add_argument(
        "--template-tag",
        required=True,
        help="Existing FastFlowLM tag to clone, such as qwen3:8b",
    )
    parser.add_argument(
        "--new-tag",
        required=True,
        help="New overlay tag, such as qwen3-local:8b",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Directory name under the model root, such as Qwen3-8B-Local-NPU2",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing entry if the new tag already exists",
    )
    parser.add_argument("--size-bytes", type=int, help="Override the size field in bytes")
    parser.add_argument(
        "--default-context-length",
        type=int,
        help="Override default_context_length",
    )
    parser.add_argument("--flm-min-version", help="Override flm_min_version")
    parser.add_argument("--family", help="Override details.family")
    parser.add_argument("--parameter-size", help="Override details.parameter_size")
    parser.add_argument("--quantization-level", help="Override details.quantization_level")
    parser.add_argument("--think", type=parse_bool, help="Override details.think")
    parser.add_argument(
        "--think-toggleable",
        type=parse_bool,
        help="Override details.think_toggleable",
    )
    parser.add_argument("--vlm", type=parse_bool, help="Override the top-level vlm flag")
    parser.add_argument("--footprint", type=float, help="Override footprint")
    parser.add_argument(
        "--file",
        dest="files",
        action="append",
        help="Replace the required file list; repeat for each file",
    )
    parser.add_argument(
        "--label",
        dest="labels",
        action="append",
        help="Replace labels; repeat for each label",
    )
    parser.add_argument(
        "--url",
        default=LOCAL_ONLY_URL,
        help=(
            "Remote base URL for flm pull. Defaults to a local-only placeholder "
            "so accidental pull attempts fail clearly."
        ),
    )
    parser.add_argument(
        "--file-url",
        default=LOCAL_ONLY_URL,
        help="Remote Hugging Face tree URL for flm pull metadata lookup",
    )
    parser.add_argument(
        "--mkdir",
        action="store_true",
        help="Create the expected model directory next to the output config",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    base_data = load_model_list(args.base)
    entry = get_entry(base_data, args.template_tag)
    new_family, new_size = parse_tag(args.new_tag)

    if "models" not in base_data or not isinstance(base_data["models"], dict):
        raise SystemExit("Base model list is missing a top-level 'models' object")

    family_bucket = base_data["models"].setdefault(new_family, {})
    if new_size in family_bucket and not args.replace:
        raise SystemExit(
            f"Entry {args.new_tag} already exists in {args.output}; "
            "pass --replace to overwrite it"
        )

    entry["name"] = args.name
    entry["url"] = args.url
    entry["file_url"] = args.file_url

    if args.size_bytes is not None:
        entry["size"] = args.size_bytes
    if args.default_context_length is not None:
        entry["default_context_length"] = args.default_context_length
    if args.flm_min_version is not None:
        entry["flm_min_version"] = args.flm_min_version
    if args.vlm is not None:
        entry["vlm"] = args.vlm
    if args.footprint is not None:
        entry["footprint"] = args.footprint
    if args.files:
        entry["files"] = args.files
    if args.labels is not None:
        entry["label"] = args.labels

    details = entry.setdefault("details", {})
    if args.family is not None:
        details["family"] = args.family
    if args.parameter_size is not None:
        details["parameter_size"] = args.parameter_size
    if args.quantization_level is not None:
        details["quantization_level"] = args.quantization_level
    if args.think is not None:
        details["think"] = args.think
    if args.think_toggleable is not None:
        details["think_toggleable"] = args.think_toggleable

    family_bucket[new_size] = entry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(base_data, indent=4) + "\n", encoding="utf-8")

    model_root = args.output.parent / base_data.get("model_path", "models")
    expected_dir = model_root / args.name
    if args.mkdir:
        expected_dir.mkdir(parents=True, exist_ok=True)

    print(f"Wrote overlay config: {args.output}")
    print(f"Registered tag: {args.new_tag}")
    print(f"Expected model directory: {expected_dir}")
    print("Required files:")
    for filename in entry.get("files", []):
        print(f"  - {expected_dir / filename}")
    print(
        "Note: this only registers a local overlay entry. "
        "You must copy the required files yourself, and unsupported FastFlowLM "
        "families/sizes still will not run."
    )
    if args.url == LOCAL_ONLY_URL or args.file_url == LOCAL_ONLY_URL:
        print(
            "Note: flm pull is intentionally disabled for this overlay entry "
            "unless you provide real --url/--file-url values."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
