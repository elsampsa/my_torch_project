#!/bin/bash

# on your host:
# notebook in 8889
# tensorboard in 8890

notebook_dir="../notebook"
python_module_dir=".."
data_dir="../../data/my_torch_project"
train_dir="../train"

docker run --cap-add all --runtime=nvidia --gpus 1 -it -p 8889:8888 -p 8890:8890 \
--mount type=bind,source="$(pwd)"/$notebook_dir,target=/mnt/notebook \
--mount type=bind,source="$(pwd)"/$python_module_dir,target=/mnt/python-module \
--mount type=bind,source="$(pwd)"/$data_dir,target=/mnt/data \
--mount type=bind,source="$(pwd)"/$train_dir,target=/mnt/train \
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
