FROM ubuntu:24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip wget curl \
    && rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

WORKDIR /opt/rkllama
RUN chmod +x setup.sh && ./setup.sh

COPY . /opt/rkllama/

EXPOSE 5000

CMD ["rkllama", "serve"]
