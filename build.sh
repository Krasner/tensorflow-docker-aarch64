#!/bin/bash
export TF_VER="${TF_VER:-2.20}"
export CUDA_VER="${CUDA_VER:-12.5.1}"
export UBUNTU_VER="${UBUNTU_VER:-22.04}"
export PYTHON_VER="${PYTHON_VER:-3.11}"

echo $TF_VER

sudo docker build \
    --build-arg TF_VER=$TF_VER \
    --build-arg CUDA_VER=$CUDA_VER \
    --build-arg UBUNTU_VER=$UBUNTU_VER \
    --build-arg PYTHON_VER=$PYTHON_VER \
    -t "krasnera/tensorflow-${TF_VER}-cuda-${CUDA_VER}-python-${PYTHON_VER}-ubuntu-${UBUNTU_VER}-aarch64:latest" .