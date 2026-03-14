# =============================================================================
# FastFlowLM on AMD Ryzen AI NPU — Ubuntu 24.04
# =============================================================================
# Runs LLMs on the AMD XDNA/XDNA2 NPU (Strix Point, Kraken Point, etc.) on Linux.
#
# Prerequisites (on the HOST, not in the container):
#   - AMD Ryzen AI processor with NPU (Strix Point / Kraken Point / etc.)
#   - Linux kernel 6.11+ with amdxdna driver (in-tree from 6.14+, or via amdxdna-dkms)
#   - NPU device visible at /dev/accel/accel0
#   - NPU firmware in /lib/firmware/amdnpu/ (or /usr/lib/firmware/amdnpu/)
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
# Stage 1: Build XRT from source
# ---------------------
FROM ubuntu:24.04 AS xrt-builder

ENV DEBIAN_FRONTEND=noninteractive

# XRT build dependencies (from xdna-driver's xrtdeps.sh, trimmed for Docker)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    file \
    git \
    ca-certificates \
    pkg-config \
    jq \
    wget \
    libboost-dev \
    libboost-filesystem-dev \
    libboost-program-options-dev \
    libcurl4-openssl-dev \
    libdrm-dev \
    libdw-dev \
    libelf-dev \
    libffi-dev \
    libgtest-dev \
    libjson-glib-dev \
    libncurses5-dev \
    libprotoc-dev \
    libssl-dev \
    libsystemd-dev \
    libudev-dev \
    libyaml-dev \
    lsb-release \
    ocl-icd-dev \
    ocl-icd-opencl-dev \
    opencl-headers \
    pciutils \
    protobuf-compiler \
    python3 \
    libpython3-dev \
    python3-pybind11 \
    pybind11-dev \
    rapidjson-dev \
    systemtap-sdt-dev \
    uuid-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone xdna-driver (includes XRT as submodule)
WORKDIR /build
RUN git clone --recurse-submodules https://github.com/amd/xdna-driver.git

# Build XRT base (headers + libs)
WORKDIR /build/xdna-driver/xrt/build
RUN ./build.sh -npu -opt

# Install XRT base .deb
RUN apt-get update && apt install -y ./Release/xrt_*.deb && rm -rf /var/lib/apt/lists/*

# Build XRT NPU plugin
WORKDIR /build/xdna-driver/build
RUN ./build.sh -release -nokmod

# Install XRT plugin .deb
RUN apt install -y ./Release/xrt_plugin*.deb

# ---------------------
# Stage 2: Build FastFlowLM
# ---------------------
FROM ubuntu:24.04 AS flm-builder

ENV DEBIAN_FRONTEND=noninteractive
ARG FASTFLOWLM_REPO=https://github.com/FastFlowLM/FastFlowLM.git
ARG FASTFLOWLM_COMMIT=5b33fd97afc4d07a5673fff0efdec4e15fe61a1e

# FastFlowLM build dependencies
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

# Copy XRT installation from xrt-builder
COPY --from=xrt-builder /opt/xilinx/xrt /opt/xilinx/xrt

# Clone FastFlowLM at a pinned commit so model/runtime support stays reproducible.
WORKDIR /build
RUN git clone --recurse-submodules "${FASTFLOWLM_REPO}" FastFlowLM \
    && cd FastFlowLM \
    && git checkout "${FASTFLOWLM_COMMIT}" \
    && git submodule update --init --recursive

# Build (point at XRT from source build)
WORKDIR /build/FastFlowLM/src
RUN cmake --preset linux-default \
      -DXRT_INCLUDE_DIR=/opt/xilinx/xrt/include \
      -DXRT_LIB_DIR=/opt/xilinx/xrt/lib \
    && cmake --build build -j8

# Install to /opt/fastflowlm
RUN cmake --install build

# ---------------------
# Stage 3: Runtime
# ---------------------
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Runtime dependencies only (no -dev packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
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
    libdrm2 \
    && rm -rf /var/lib/apt/lists/*

# Copy XRT runtime libraries (from source build)
COPY --from=xrt-builder /opt/xilinx/xrt/lib /opt/xilinx/xrt/lib
COPY --from=xrt-builder /opt/xilinx/xrt/setup.sh /opt/xilinx/xrt/setup.sh

# Add XRT libs to linker path
ENV LD_LIBRARY_PATH="/opt/xilinx/xrt/lib:${LD_LIBRARY_PATH}"

# Copy FastFlowLM installation from builder
COPY --from=flm-builder /opt/fastflowlm /opt/fastflowlm

# Symlink so `flm` is in PATH
RUN ln -sf /opt/fastflowlm/bin/flm /usr/local/bin/flm

# Model cache directory
RUN mkdir -p /root/.config/flm

# FLM needs the NPU xclbin files at a known path
ENV FLM_XCLBIN_PATH=/opt/fastflowlm/share/flm/xclbins

ENTRYPOINT ["flm"]
CMD ["--help"]
