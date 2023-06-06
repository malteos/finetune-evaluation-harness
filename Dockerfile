# This image is for development/model-training (e.g. sweeps) and will not be deployed on production machines.

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

WORKDIR /app
SHELL ["/bin/bash", "-c"]

# install git, python & torch
RUN apt update && \
    apt install -y vim git python3.10 python3.10-venv && \
    python3.10 -m venv venv

RUN source venv/bin/activate && \
    pip install torch==1.13.1

# Install git-lfs for huggingface hub cli
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs
RUN git lfs install

# app files and install dependencies
RUN apt install -y python3.10-dev

# copy req file first
COPY requirements.txt /app/requirements.txt

# A hacky but working way to exclude deepspeed, apex and torch from the requirements (we install them manually or they are part of base image)
RUN sed -i 's/^torch/# &/' /app/requirements.txt
RUN source venv/bin/activate && pip install -r /app/requirements.txt
RUN sed -i 's/^# \(torch\)/\1/' /app/requirements.txt

# Copy all other project files
WORKDIR /app
COPY . /app

# Create non-root user and give permissions to /app
RUN useradd -ms /bin/bash docker
RUN chown -R docker:docker /app

USER docker

# expose the Jupyter port 8888
EXPOSE 8888

ENV PATH=/app/venv/bin:$PATH

WORKDIR /app

CMD ["/bin/bash"]
