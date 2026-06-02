FROM ubuntu:24.04

# RK3588 (Cortex-A76/A55) is ARMv8.2-A — it does NOT support Pointer
# Authentication (PAC, ARMv8.3-A).  Debian Trixie ARM64 packages
# (python:3.12-slim) are compiled with PAC guards; the kernel returns
# ENOEXEC for those binaries on RK3588, causing "exec format error" at
# container startup.  Ubuntu 24.04 ARM64 packages do NOT use PAC and
# run correctly on ARMv8.2-A hardware.
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3.12 python3.12-dev python3-pip python3.12-venv \
       libgomp1 wget curl sudo git build-essential \
       ffmpeg libsm6 libxext6 \
    && rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

# Use a virtual environment to avoid Ubuntu 24.04's PEP 668
# "externally-managed-environment" restriction.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir "setuptools<82.0.0"

WORKDIR /opt/rkllama


################################################## llama.cpp #######################################################

ARG TARGETARCH

RUN apt-get update && \
    apt-get install -y gcc-14 g++-14 build-essential git cmake libssl-dev wget lsb-release software-properties-common gnupg apt-utils

RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 21

RUN apt-get install -y libomp5 clang-21 libclang-cpp21-dev libomp-21-dev

RUN git clone https://github.com/invisiofficial/rk-llama.cpp.git

WORKDIR /opt/rkllama/rk-llama.cpp

RUN ARCH="${TARGETARCH:-}" && \
    if [ -z "$ARCH" ]; then ARCH="$(uname -m)"; fi && \
    if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then \
        rm -rf build && \
        cmake -S . -B build \
            #-DCMAKE_BUILD_TYPE=Release \
            #-DGGML_NATIVE=OFF \
            #-DLLAMA_BUILD_TESTS=OFF \
            #-DGGML_BACKEND_DL=ON \
            #-DGGML_CPU_ALL_VARIANTS=ON \
            -DLLAMA_RKNPU2=ON \
	        -DCMAKE_C_COMPILER=clang-21 \
            -DCMAKE_CXX_COMPILER=clang++-21 && \
        cmake --build build -j "$(nproc)"; \
    else \
        echo "rknpu2 image: unsupported architecture (need arm64/aarch64), got TARGETARCH=${TARGETARCH} uname=${ARCH}"; \
        exit 1; \
    fi

################################################## llama.cpp #######################################################

WORKDIR /opt/rkllama

# Copy RKLLM runtime library explicitly
COPY ./src/rkllama/lib/librkllmrt.so /usr/lib/
RUN chmod 755 /usr/lib/librkllmrt.so && ldconfig

# Copy RKNN runtime library explicitly
COPY ./src/rkllama/lib/librknnrt.so /usr/lib/
RUN chmod 755 /usr/lib/librknnrt.so && ldconfig

# Copy the source and other resources of the RKllama project
COPY ./src /opt/rkllama/src
RUN mkdir /opt/rkllama/models
COPY README.md LICENSE pyproject.toml /opt/rkllama/

# Install RKllama project
RUN pip install --no-cache-dir .

EXPOSE 8080

# If you want to change the port see the
# documentation/configuration.md for the INI file settings.
CMD ["rkllama_server", "--models", "/opt/rkllama/models", "--llamacpp" , "/opt/rkllama/rk-llama.cpp/build/bin"]
