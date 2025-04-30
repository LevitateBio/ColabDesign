FROM nvidia/cuda:12.6.1-runtime-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    aria2 \
    build-essential \
    curl \
    git \
    tar \
    wget \
    unzip \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN apt-get update && apt-get install -y \
    cuda-compat-12.6

# Add NVIDIA repository and install CUDNN 9.8
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y \
    libcudnn9-cuda-12=9.8.0.87-1 \
    libcudnn9-dev-cuda-12=9.8.0.87-1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip --no-cache-dir install torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121 \
    && pip --no-cache-dir install nvidia-cudnn-cu12==9.8.0.* --extra-index-url https://download.pytorch.org/whl/cu121 \
    && pip --no-cache-dir install dgl \
    && pip --no-cache-dir install dglgo -f https://data.dgl.ai/wheels-test/repo.html \
    && pip --no-cache-dir install hydra-core pyrsistent jedi omegaconf icecream scipy opt_einsum opt_einsum_fx wandb e3nn decorator pynvml fire biopython ipython \
    && pip --no-cache-dir install git+https://github.com/NVIDIA/dllogger#egg=dllogger

# Copy local ColabDesign files instead of cloning
COPY . /opt/ColabDesign
RUN pip --no-cache-dir install -e /opt/ColabDesign \
    && git clone https://github.com/LevitateBio/RFdiffusion.git \
    && pip --no-cache-dir install -e ./RFdiffusion/env/SE3Transformer \
    && pip --no-cache-dir install -e ./RFdiffusion --no-deps

# Download schedules
RUN cd /opt \
    && aria2c -s 16  -x 16 https://files.ipd.uw.edu/krypton/schedules.zip \
    && unzip schedules.zip \
    && rm schedules.zip

# Must be hardcoded for RFDiffusion
WORKDIR /opt/RFdiffusion

# Add python symlink
RUN ln -sf /usr/bin/python3 /usr/bin/python

#Install the proper version of jax
RUN pip install -U "jax[cuda12]"
