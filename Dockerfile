FROM debian AS builder

FROM builder as dev-envs

RUN apt update
RUN apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget
RUN wget https://www.python.org/ftp/python/3.10.8/Python-3.10.8.tgz
RUN tar xzf Python-3.10.8.tgz
WORKDIR Python-3.10.8
RUN ./configure --enable-optimizations
RUN make install
WORKDIR /
RUN python3 -m ensurepip --upgrade
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install install numpy pandas tensorflow jupyter notebook nbconvert ipykernel tqdm