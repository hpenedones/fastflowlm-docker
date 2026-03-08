# =============================================================================
# FastFlowLM on AMD Ryzen AI NPU — Ubuntu 24.04
# =============================================================================
# Runs LLMs on the AMD XDNA2 NPU (Strix Point, Kraken Point, etc.) on Linux.
#
# Prerequisites (on the HOST, not in the container):
#   - AMD Ryzen AI processor with NPU (Strix Point / Kraken Point / etc.)
#   - Linux kernel 6.11+ with amdxdna driver (in-tree from 7.0, or via amdxdna-dkms)
#   - NPU device visible at /dev/accel/accel0
#   - NPU firmware ≥ 1.1.0.0 (in /lib/firmware/amdnpu/)
#   - Docker with --device passthrough support
#
# Build:
#   docker build -t fastflowlm .
#
# Run (interactive chat):
#   docker run -it --rm \
#     --device=/dev/accel/accel0 \
#     --ulimit memlock=-1:-1 \
#     -v ~/.config/flm:/root/.config/flm \
#     fastflowlm run llama3.2:1b
#
# The model cache is in /root/.config/flm inside the container.
# Mounting it as a volume avoids re-downloading models on every run.
#
# Other examples:
#   docker run ... fastflowlm list              # list available models
#   docker run ... fastflowlm pull qwen3:1.7b   # download a model
#   docker run ... fastflowlm validate          # check NPU setup
#   docker run ... fastflowlm serve             # OpenAI-compatible API server
#
# (where "..." = --device=/dev/accel/accel0 --ulimit memlock=-1:-1 -v ~/.config/flm:/root/.config/flm)
# =============================================================================

# ---------------------
# Stage 1: Build
# ---------------------
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    ca-certificates \
    curl \
    pkg-config \
    libboost-program-options-dev \
    libcurl4-openssl-dev \
    libfftw3-dev \
    libavformat-dev \
    libavcodec-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libreadline-dev \
    uuid-dev \
    libdrm-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (needed for tokenizers-cpp FFI bindings)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# XRT headers + libs (needed at build time for xrt_bo.h, xrt_kernel.h etc.)
# Install from AMD's PPA (provides XRT 2.21+)
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:amd-team/xrt \
    && apt-get update \
    && apt-get install -y --no-install-recommends libxrt-dev libxrt-npu2 \
    && rm -rf /var/lib/apt/lists/*

# Clone FastFlowLM
WORKDIR /build
RUN git clone --recurse-submodules https://github.com/FastFlowLM/FastFlowLM.git

# Build (override XRT paths — PPA installs to /usr, not /opt/xilinx/xrt)
WORKDIR /build/FastFlowLM/src
RUN cmake --preset linux-default \
      -DXRT_INCLUDE_DIR=/usr/include \
      -DXRT_LIB_DIR=/usr/lib/x86_64-linux-gnu \
    && cmake --build build -j8

# Install to /opt/fastflowlm
RUN cmake --install build

# ---------------------
# Stage 2: Runtime
# ---------------------
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Runtime dependencies only (no -dev packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    software-properties-common \
    libboost-program-options1.83.0 \
    libcurl4 \
    libfftw3-single3 \
    libfftw3-double3 \
    libfftw3-long3 \
    libavformat60 \
    libavcodec60 \
    libavutil58 \
    libswscale7 \
    libswresample4 \
    libreadline8t64 \
    && rm -rf /var/lib/apt/lists/*

# XRT runtime (provides libxrt_coreutil.so.2 for NPU access)
RUN add-apt-repository -y ppa:amd-team/xrt \
    && apt-get update \
    && apt-get install -y --no-install-recommends libxrt-npu2 \
    && rm -rf /var/lib/apt/lists/*

# Copy FastFlowLM installation from builder
COPY --from=builder /opt/fastflowlm /opt/fastflowlm

# Symlink so `flm` is in PATH
RUN ln -sf /opt/fastflowlm/bin/flm /usr/local/bin/flm

# Model cache directory
RUN mkdir -p /root/.config/flm

# FLM needs the NPU xclbin files at a known path
ENV FLM_XCLBIN_PATH=/opt/fastflowlm/share/flm/xclbins

ENTRYPOINT ["flm"]
CMD ["--help"]
