ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
ARG DEBIAN_FRONTEND=noninteractive

##############################################
# You should modify this to match your GPU compute capability
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5+PTX"
##############################################

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Install dependencies
RUN echo
RUN apt-get update
RUN apt-get install -y git openssh-client ffmpeg libsm6 libxext6 ninja-build
RUN apt-get install -y tmux
RUN apt-get clean
RUN apt-get -y install build-essential \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \ 
        libnss3-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev \
        libsqlite3-dev \
        libbz2-dev \
        wget \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get purge -y imagemagick imagemagick-6-common 
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /packages

RUN pip install spconv-cu116 iopath 
RUN pip install fvcore>=0.1.5
RUN FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.4"
RUN pip install tensorboardX

RUN apt-get update
RUN apt-get install -y libeigen3-dev

ARG USER_ID
ARG GROUP_ID

# Switch to same user as host system
RUN addgroup --gid $GROUP_ID pdd && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

WORKDIR /packages/3dlabgen
RUN pip install open3d
RUN pip install lightning==2.2 pytorch-lightning==2.2
RUN pip install torch==2.0.1
RUN pip install ipdb 
RUN pip install pycpd
RUN pip install optuna
RUN pip install trimesh
RUN pip install pygeodesic==0.1.7
RUN apt-get install unzip

USER root
RUN apt install -y gfortran
RUN pip install pc-skeletor
USER user

ENTRYPOINT ["bash", "-c"]

