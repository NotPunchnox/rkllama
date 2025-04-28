FROM python:slim

WORKDIR /opt/rkllama

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    less \
    libgomp1 \
    procps \
    sudo \
    wget \
    nano \
    && rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

COPY ./lib ./lib
COPY ./src ./src
COPY ./config ./config
COPY requirements.txt README.md LICENSE *.sh *.py *.ini ./

RUN bash ./setup.sh --no-conda

EXPOSE 8080

CMD ["/usr/local/bin/rkllama", "serve"]
# If you want to change the port see
# documentation/configuration.md for the INI file settings.
