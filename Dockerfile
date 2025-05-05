# Dockerfile for RKLLama
# Improved by Yoann Vanitou <yvanitou@gmail.com>

FROM python:slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 wget curl sudo git build-essential \
    && rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

# Install RKNPU driver
RUN cd /tmp \
    && git clone https://github.com/rockchip-linux/rknpu.git \
    && cd rknpu \
    && mkdir -p /usr/lib \
    && cp -r drivers/linux-aarch64/usr/lib/* /usr/lib/ \
    && cp -r rknn/rknn_api/librknn_api/lib/* /usr/lib/ \
    && cp -r rknn/rknn_utils/librknn_utils/lib/* /usr/lib/ \
    && ldconfig \
    && cd .. \
    && rm -rf rknpu

WORKDIR /opt/rkllama

# Copy RKLLM runtime library explicitly
COPY ./lib/librkllmrt.so /usr/lib/
RUN chmod 755 /usr/lib/librkllmrt.so && ldconfig

COPY ./lib /opt/rkllama/lib
COPY ./src /opt/rkllama/src
COPY ./models /opt/rkllama/models
COPY requirements.txt README.md LICENSE *.sh *.py /opt/rkllama/
RUN chmod +x setup.sh && ./setup.sh --no-conda

EXPOSE 8080

CMD ["/usr/local/bin/rkllama", "serve"]
# If you want to change the port see
# documentation/configuration.md for the INI file settings.
