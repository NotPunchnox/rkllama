# Dockerfile for RKLLama
# Improved by Yoann Vanitou <yvanitou@gmail.com>

FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 wget curl sudo git build-essential \
    && rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

WORKDIR /opt/rkllama

# Copy RKLLM runtime library explicitly
COPY ./src/rkllama/lib/librkllmrt.so /usr/lib/
RUN chmod 755 /usr/lib/librkllmrt.so && ldconfig

# Copy RKNN runtime library explicitly
COPY ./src/rkllama/lib/librknnrt.so /usr/lib/
RUN chmod 755 /usr/lib/librknnrt.so && ldconfig

COPY ./src /opt/rkllama/src
COPY ./models /opt/rkllama/models
COPY README.md LICENSE pyproject.toml /opt/rkllama/

# Install RKNNLite toolkit
RUN python -m pip install .

EXPOSE 8080

CMD ["rkllama_server", "--models", "/opt/rkllama/models"]
# If you want to change the port see
# documentation/configuration.md for the INI file settings.



