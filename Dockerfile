FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu20.04

# set timezone
ENV TZ=Europe/Berlin
ENV BASE_DIR=/home/dwk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir -p /usr/share/man/man1 \
 && apt-get update \
 && apt-get install -y --no-install-recommends openjdk-8-jdk wget gradle \
    build-essential libgraphviz-dev tar zlib1g-dev libffi-dev libreadline-gplv2-dev \
    libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev \
    p7zip-full wget git
RUN wget https://www.python.org/ftp/python/3.8.2/Python-3.8.2.tar.xz \
 && tar -xf Python-3.8.2.tar.xz \
 && cd Python-3.8.2 \
 && ./configure --enable-optimizations \
 && make -j 4 \
 && make install

RUN python3 -m pip install --upgrade pip \
 && pip install setuptools wheel \
 && pip install --upgrade setuptools \
 && ln -s /usr/local/bin/python3 /usr/local/bin/python

COPY . ${BASE_DIR}
WORKDIR ${BASE_DIR}

RUN ./scripts/env_cuda.sh
RUN ./scripts/setup_joern.sh
RUN ./scripts/download.sh
