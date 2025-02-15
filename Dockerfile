FROM --platform=linux/arm64 python:slim as build

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget curl sudo \
    && rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

WORKDIR /opt/rkllama

COPY . /opt/rkllama/
RUN chmod +x setup.sh && ./setup.sh

EXPOSE 5000

CMD ["rkllama", "serve"]
