# FastFlowLM Docker — Run LLMs on AMD Ryzen AI NPU (Linux)

Run large language models on your AMD Ryzen AI NPU under Linux, using
[FastFlowLM](https://github.com/FastFlowLM/FastFlowLM) inside Docker.

## What this is

As of March 2026, running LLMs on AMD's XDNA2 NPU under Linux is not
straightforward. AMD's official Ryzen AI 1.7 stack is missing a critical
shared library (`onnxruntime_providers_ryzenai.so`) on Linux
([amd/RyzenAI-SW#333](https://github.com/amd/RyzenAI-SW/issues/333)),
and FastFlowLM doesn't yet ship Linux binaries in releases
([FastFlowLM#381](https://github.com/FastFlowLM/FastFlowLM/issues/381)).

This Dockerfile builds FastFlowLM from source and packages everything needed
into a minimal container that talks directly to the NPU.

## Supported hardware

Any AMD processor with an XDNA2 NPU:
- Ryzen AI 9 HX 370/375 (Strix Point)
- Ryzen AI 9 HX 395 (Strix Halo)
- Ryzen AI Max / Max+ (Kraken Point)
- And other XDNA2-based APUs

## Host prerequisites

| Requirement | How to check |
|---|---|
| Linux kernel ≥ 6.11 with `amdxdna` driver | `lsmod \| grep amdxdna` |
| NPU device node | `ls -la /dev/accel/accel0` |
| NPU firmware ≥ 1.1.0.0 | `ls /lib/firmware/amdnpu/` |
| Docker installed | `docker --version` |
| memlock unlimited (recommended) | `ulimit -l` |

### Quick host setup (Ubuntu 24.04)

```bash
# Install XRT and NPU userspace (if not already)
sudo add-apt-repository ppa:amd-team/xrt
sudo apt update && sudo apt install libxrt-npu2

# Set memlock to unlimited (needs reboot)
echo -e "* soft memlock unlimited\n* hard memlock unlimited" | sudo tee -a /etc/security/limits.conf
sudo reboot
```

## Build the Docker image

```bash
git clone https://github.com/hpenedones/fastflowlm-docker.git
cd fastflowlm-docker
docker build -t fastflowlm .
```

The build takes ~5-10 minutes (Rust compilation + FLM C++ build).
The resulting image is ~484MB (multi-stage build, runtime-only).

## Usage

```bash
# List available NPU models
docker run --rm fastflowlm list

# Download a model (mount cache so it persists)
docker run --rm -v ~/.config/flm:/root/.config/flm fastflowlm pull llama3.2:1b

# Chat with the model on your NPU
docker run -it --rm \
  --device=/dev/accel/accel0 \
  --ulimit memlock=-1:-1 \
  -v ~/.config/flm:/root/.config/flm \
  fastflowlm run llama3.2:1b

# Validate NPU setup
docker run --rm \
  --device=/dev/accel/accel0 \
  --ulimit memlock=-1:-1 \
  fastflowlm validate

# Run OpenAI-compatible API server on port 8000
docker run -d --rm \
  --device=/dev/accel/accel0 \
  --ulimit memlock=-1:-1 \
  -v ~/.config/flm:/root/.config/flm \
  -p 8000:8000 \
  fastflowlm serve
```

## Available models

As of FLM v0.9.35, these models run on the NPU:

| Model | Size | Command |
|---|---|---|
| Llama 3.2 1B | ~1.2 GB | `run llama3.2:1b` |
| Llama 3.2 3B | ~1.8 GB | `run llama3.2:3b` |
| Llama 3.1 8B | ~4.5 GB | `run llama3.1:8b` |
| Qwen3 0.6B | ~0.4 GB | `run qwen3:0.6b` |
| Qwen3 1.7B | ~1.0 GB | `run qwen3:1.7b` |
| Qwen3 4B | ~2.3 GB | `run qwen3:4b` |
| Gemma3 1B | ~0.7 GB | `run gemma3:1b` |
| DeepSeek-R1 8B | ~4.5 GB | `run deepseek-r1:8b` |
| Phi-4 Mini 4B | ~2.3 GB | `run phi4-mini-it:4b` |

Run `flm list` for the complete list.

## How it works

The Dockerfile:
1. **Build stage**: Installs all build dependencies (cmake, ninja, Rust, Boost,
   FFmpeg, FFTW3, XRT headers), clones FastFlowLM, and compiles from source
2. **Runtime stage**: Copies only the built binary, NPU kernel libraries (`.so`),
   and xclbin files into a minimal Ubuntu image with runtime dependencies

The container accesses the NPU via `--device=/dev/accel/accel0`. The host
kernel's `amdxdna` driver handles the actual hardware communication.

## Troubleshooting

**`flm validate` shows no NPU**: Make sure you passed `--device=/dev/accel/accel0`
and the host has the `amdxdna` driver loaded.

**Permission denied on /dev/accel/accel0**: Check device permissions on the host
(`ls -la /dev/accel/accel0`). You may need to add your user to the `render` group
or run the container with `--group-add render`.

**Low memlock limit**: The NPU needs a high memlock limit. Pass `--ulimit memlock=-1:-1`
to Docker, or set unlimited memlock in `/etc/security/limits.conf` on the host and reboot.

## Credits

- [FastFlowLM](https://github.com/FastFlowLM/FastFlowLM) — the NPU LLM runtime
- [AMD XDNA Driver](https://github.com/amd/xdna-driver) — Linux NPU driver
- [AMD XRT](https://launchpad.net/~amd-team/+archive/ubuntu/xrt) — Xilinx Runtime

## License

The Dockerfile itself is MIT licensed. FastFlowLM and AMD's libraries have their
own licenses — see their respective repositories.
