ARG BASE_TAG=21.12-py3
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch

FROM $BASE_IMAGE:$BASE_TAG AS main

RUN mkdir /app
WORKDIR /app

# Install git-lfs for huggingface hub cli
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get install git-lfs
RUN git lfs install

# copy req file first
COPY requirements.txt /app/requirements.txt

# A hacky but working way to exclude deepspeed, apex and torch from the requirements (we install them manually or they are part of base image)
RUN sed -i 's/^torch/# &/' /app/requirements.txt
RUN pip install -r /app/requirements.txt
RUN sed -i 's/^# \(torch\)/\1/' /app/requirements.txt

# install scispacy model
RUN pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz

# Copy all other project files
WORKDIR /app
COPY . /app

# Create non-root user and give permissions to /app
RUN useradd -ms /bin/bash docker
RUN chown -R docker:docker /app

USER docker

# expose the Jupyter port 8888
EXPOSE 8888

# print versions
RUN nvcc --version
RUN python --version
RUN python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
# RUN python -c "import transformers; print(transformers.__version__);"
# RUN ds_report

CMD ["jupyter", "notebook"]
