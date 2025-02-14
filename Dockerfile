FROM ubuntu:24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends git python3 python3-pip wget curl build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip \
    && python3 -m pip install flask \

RUN git clone -b "Without-minconda" https://github.com/notpunchnox/rkllama /opt/rkllama

WORKDIR /opt/rkllama
RUN chmod +x setup.sh && ./setup.sh

EXPOSE 5000

CMD ["rkllama", "serve"]
