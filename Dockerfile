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
CMD ["rkllama_server", "--models", "/opt/rkllama/models"]
