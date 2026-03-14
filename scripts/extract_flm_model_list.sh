#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <docker-image> <output-path>" >&2
    echo "Example: $0 fastflowlm ~/.config/flm/model_list.base.json" >&2
}

if [[ $# -ne 2 ]]; then
    usage
    exit 1
fi

image="$1"
output="$2"

container_id="$(docker create "$image")"
cleanup() {
    docker rm -f "$container_id" >/dev/null 2>&1 || true
}
trap cleanup EXIT

mkdir -p "$(dirname "$output")"
docker cp "$container_id:/opt/fastflowlm/share/flm/model_list.json" "$output"

echo "Wrote $output"
