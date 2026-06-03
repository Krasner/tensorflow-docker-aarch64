# Tensorflow docker image for aarch64 (with GPU enabled)

## Build
Use `build.sh` to build the docker container. There are the following arguments:

`UBUNTU_VER` defaults to 22.04 \
`CUDA_VER` defaults to 12.5 \
`PYTHON_VER` defaults to 3.11 \
`TF_VER` defaults to 2.20 

### Notes:
- To build tensorflow 2.19 see branch r2.19 since it requires slightly different bazel build configs

## Run
```
sudo docker run --rm --gpus=all -it tensorflow-${TF_VER}-cuda-${CUDA_VER}-python-${PYTHON_VER}-ubuntu-${UBUNTU_VER}-aarch64:latest /bin/bash
```

## Deployed
| Tensorflow |  CUDA  | Python | Ubuntu | docker image |
|:----------:|:------:|:------:|:------:|:------------:|
|2.21|12.5.1|3.11 |22.04|krasnera/tensorflow-2.21-cuda-12.5.1-python-3.11-ubuntu-22.04-aarch64|
|2.20|12.5.1|3.11 |22.04|krasnera/tensorflow-2.20-cuda-12.5.1-python-3.11-ubuntu-22.04-aarch64|
|2.19|12.5.1|3.11 |22.04|krasnera/tensorflow-2.19-cuda-12.5.1-python-3.11-ubuntu-22.04-aarch64|

## Notes
### GPU access in docker
If you see the following error when running the container with `docker run --gpus=all`
```
docker: Error response from daemon: failed to discover GPU vendor from CDI: no known GPU vendor found
```

You may need to install `nvidia-container-toolkit`:

See https://stackoverflow.com/questions/75118992/docker-error-response-from-daemon-could-not-select-device-driver-with-capab

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
&& sudo apt-get update

sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker

sudo systemctl restart docker
```

## Building for different tensorflow versions
Reference the following table https://www.tensorflow.org/install/source#gpu for the correct CUDA, python, clang, bazel versions needed to build tensorflow from source.

Currently clang 18.1.8 and bazel 7.4.1 is hardcoded into the Dockerfile.
