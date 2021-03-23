## FROM ufoym/deepo
## based on this excellent Docker image: https://github.com/ufoym/deepo#GPU
#
## WARNING: the very first step!  check with "pip3 install tensorflow-gpu==2.2.0" to see what tf versions you have available!
## after that, proceed in picking up the correct nvidia image with compatible cuda and cudnn versions
## tensorflow compatible versions here: https://www.tensorflow.org/install/source#gpu
## tags are here: https://hub.docker.com/r/nvidia/cuda/tags

# FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04
## there's something wrong with that image... too "low-level".  Let's use this instead:
FROM tensorflow/tensorflow:2.3.0-gpu

# opencv requires:
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
libsm6 libxext6 libxrender-dev
# python3-pip

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
    torch \
    torchvision

USER root
#
RUN pip3 install opencv-python==4.2.0.34
RUN pip3 install progressbar
RUN pip3 install tqdm
RUN pip3 install PyYAML
RUN pip3 install tensorboard
RUN pip3 install Pillow scikit-image
##
## KEEP THIS MANUALLY UP-TO-DATE with "requirements.txt"
##
