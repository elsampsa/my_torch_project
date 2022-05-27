#!/bin/bash

# on your host:
# notebook in 8889
# tensorboard in 8890

notebook_dir="../notebook"
python_module_dir=".." # hot-reload code from this python package
data_dir="../../data/my_torch_project" 
train_dir="../train"

docker run --cap-add all --runtime=nvidia --gpus 1 -it -p 8889:8888 -p 8890:8890 \
--mount type=bind,source="$(pwd)"/$notebook_dir,target=/mnt/notebook \
--mount type=bind,source="$(pwd)"/$python_module_dir,target=/mnt/python-module \
--mount type=bind,source="$(pwd)"/$data_dir,target=/mnt/data \
--mount type=bind,source="$(pwd)"/$train_dir,target=/mnt/train \
--mount source=torch_vol,target=/root/.cache/torch \
--ipc=host \
my-pytorch-env \
/bin/bash -c \
"\
pip3 install --no-deps -e /mnt/python-module && \
rm -rf ~/.nv/ && \
nvidia-smi && \
tensorboard --bind_all --port 8890 --logdir /mnt/train/runs & \
jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir='/mnt/notebook' \
"
## here's a variant if you'd like to hot-reload install all subdirectories from /mnt/python-module/*
#/bin/bash -c '\
#cd /mnt/python-module && \
#echo $PWD && \
#for d in $(ls -1); do echo "installing "$d; cd /mnt/python-module/$d; pip3 install --no-deps -e .; cd ..; done && \
#rm -rf ~/.nv/ && \
#nvidia-smi && \
#tensorboard --bind_all --port 8890 --logdir /mnt/train/runs & \
#jupyter notebook --no-browser --ip=0.0.0.0 --allow-root --NotebookApp.token= --notebook-dir="/mnt/notebook" \
#'
