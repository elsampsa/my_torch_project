## FROM ufoym/deepo
## based on this excellent Docker image: https://github.com/ufoym/deepo#GPU
#
## WARNING: the very first step!  check with "pip3 install tensorflow-gpu==2.2.0" to see what tf versions you have available!
## after that, proceed in picking up the correct nvidia image with compatible cuda and cudnn versions
## tensorflow compatible versions here: https://www.tensorflow.org/install/source#gpu
## tags are here: https://hub.docker.com/r/nvidia/cuda/tags
##
# FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04
## ..there's something wrong with that image... too "low-level".  Let's use this instead:
# FROM tensorflow/tensorflow:2.3.0-gpu
## ..that's based on ubuntu 18 LTS (old)
## this is more recent (ubuntu 20.04 focal) & has CUDA 11.2:
FROM tensorflow/tensorflow:2.9.1-gpu

USER root
ENV PATH="/usr/local/cuda/bin:$PATH"

## opencv requires:
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
libsm6 libxext6 libxrender-dev

RUN pip3 install --upgrade pip

RUN pip3 install \
    numpy \
    scipy \
    pandas \
    cloudpickle \
    scikit-image>=0.14.2 \
    scikit-learn \
    matplotlib \
    Cython \
    jupyter \
    ipython \
    torch==1.9.1 \
    torchvision==0.10.1 \
    pytorch-msssim
## see torch & torchvision compat here:
## https://pypi.org/project/torchvision/

## detectron2 & torch & cuda compat here:
## https://detectron2.readthedocs.io/en/latest/tutorials/install.html
RUN python3 -m pip install detectron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html

RUN pip3 install opencv-python==4.2.0.34 \
    progressbar \
    tqdm \
    PyYAML \
    tensorboard \
    Pillow \
    scikit-image
## .. keep that manually up-to-date with "requirements.txt" ..?
## installing directly from github: https://stackoverflow.com/questions/20101834/pip-install-from-git-repo-branch
## torch might be built with cuda 10.2 but it is compatible with cuda 11.2 (cuda 11.2 supports 10.2 instructions):
RUN python3 -c "import torch; print('\ntorch was compiled with cuda',torch.version.cuda, 'torch version:', torch.__version__,'\n')"
##
## additiona notes:
## tried this image: pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel
## ..but it's outdated & too stripped (can't install those opencv deps. etc.)
