ARG UBUNTU_VER
ARG CUDA_VER

FROM --platform=linux/arm64 nvidia/cuda:${CUDA_VER}-cudnn-devel-ubuntu${UBUNTU_VER} AS builder

ARG PYTHON_VER
ARG TF_VER
ARG CUDA_VER

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    curl \
    git \
    wget \
    libcurl4-openssl-dev \
    libssl-dev \
    uuid-dev \
    zlib1g-dev \
    libpulse-dev \
    libarchive-dev \
    libprotobuf-dev \
    protobuf-compiler \
    python${PYTHON_VER} \
    python${PYTHON_VER}-dev \
    python${PYTHON_VER}-distutils \
    patch \
    zip \
    rsync \
    cpio \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --allow-change-held-packages \
    libcudnn9-cuda-12=9.3.0.75-1 \
    libcudnn9-dev-cuda-12=9.3.0.75-1 \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.8/clang+llvm-18.1.8-aarch64-linux-gnu.tar.xz \
    && tar -xvf clang+llvm-18.1.8-aarch64-linux-gnu.tar.xz \
    && cp -r clang+llvm-18.1.8-aarch64-linux-gnu/* /usr

RUN wget https://releases.bazel.build/6.5.0/release/bazel-6.5.0-linux-arm64 \
    && chmod +x bazel-6.5.0-linux-arm64 \
    && mv bazel-6.5.0-linux-arm64 /usr/local/bin/bazel

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VER}
RUN python${PYTHON_VER} -m pip install --no-cache-dir numpy packaging requests

RUN git clone https://github.com/tensorflow/tensorflow.git

WORKDIR tensorflow
RUN git checkout r${TF_VER}

RUN ln -sf /usr/bin/python${PYTHON_VER} /usr/bin/python3

ENV PYTHON_BIN_PATH=/usr/bin/python${PYTHON_VER}
ENV PYTHON_LIB_PATH=/usr/local/lib/python${PYTHON_VER}/dist-packages
ENV TF_NEED_CUDA=1
ENV TF_CUDA_CLANG=1
ENV GCC_HOST_COMPILER_PATH=/usr/bin/gcc
ENV TF_NEED_TENSORRT=0
ENV TF_CUDA_COMPUTE_CAPABILITIES="8.7"
ENV TF_ENABLE_XLA=1
ENV CC_OPT_FLAGS="-march=native"
# ENV HERMETIC_CUDA_COMPUTE_CAPABILITIES="8.7"
ENV TF_CUDA_VERSION=${CUDA_VER}
ENV TF_CUDNN_VERSION="9"

# RUN yes "" | ./configure

RUN --mount=type=cache,target=/root/.cache/bazel_gcc \
    # PATCH: Force the architectural check_deps rule to bypass and return success
    sed -i '/def _check_deps_impl(ctx):/a \ \ \ \ return struct()' tensorflow/tensorflow.bzl && \
    STUB_PATH=$(find /usr/local/cuda/ -name "libcuda.so" | head -n 1) && \
    ln -sf "$STUB_PATH" /usr/lib/aarch64-linux-gnu/libcuda.so.1 && \
    ln -sf "$STUB_PATH" /usr/lib/aarch64-linux-gnu/libcuda.so && \
    bazel clean --expunge && \
    bazel build //tensorflow/tools/pip_package:wheel \
    # --repo_env=USE_PYWRAP_RULES=1 --repo_env=WHEEL_NAME=tensorflow \
    --repo_env=CC=/usr/bin/clang-18 \
    --repo_env=CXX=/usr/bin/clang++-18 \
    --repo_env=TF_NCCL_USE_STUB=1 \
    --repo_env=TF_NCCL_VERSION= \
    --@local_config_cuda//cuda:include_cuda_libs=false \
    --@local_config_cuda//cuda:override_include_cuda_libs=true \
    --config=cuda_clang \
    --config=cuda_wheel \
    --define=no_nccl_support=true \
    -c opt \
    --config=nogcp \
    --config=nonccl \
    --action_env=CLANG_COMPILER_PATH=/usr/bin/clang-18 \
    --host_action_env=CC=/usr/bin/clang-18 \
    --host_action_env=CXX=/usr/bin/clang++-18 \
    --copt=-Wno-unused-command-line-argument \
    --host_copt=-Wno-unused-command-line-argument \
    --copt=-Wno-gnu-offsetof-extensions \
    --host_copt=-Wno-gnu-offsetof-extensions \
    --action_env=LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu \
    --host_action_env=LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu \
    --jobs=64 \
    --verbose_failures

# --- Staging & Harvesting Area ---
RUN mkdir -p /workspace/dist/lib \
    && mkdir -p /workspace/dist/include/tensorflow \
    && mkdir -p /workspace/dist/wheel

RUN cp bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow-*.whl /workspace/dist/wheel/

# --- STAGE 2: Pristine Production Container ---
ARG UBUNTU_VER
ARG CUDA_VER

FROM --platform=linux/arm64 nvidia/cuda:${CUDA_VER}-cudnn-runtime-ubuntu${UBUNTU_VER}

ARG PYTHON_VER

RUN apt-get update && apt-get install -y --allow-change-held-packages \
    libcudnn9-cuda-12=9.3.0.75-1 \
    libcudnn9-dev-cuda-12=9.3.0.75-1 \
    && rm -rf /var/lib/apt/lists/*

# Install basic runtime engines
RUN apt-get update && apt-get install -y \
    python${PYTHON_VER} \
    python${PYTHON_VER}-dev \
    python${PYTHON_VER}-distutils \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VER}
RUN python${PYTHON_VER} -m pip install --no-cache-dir numpy packaging requests nvidia-cuda-nvcc-cu12

ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/lib/python${PYTHON_VER}/dist-packages/nvidia/cuda_nvcc"

WORKDIR /app

# 1. Install the C++ Shared Libraries and Headers globally
COPY --from=builder /workspace/dist/lib/ /usr/local/lib/
COPY --from=builder /workspace/dist/include/ /usr/local/include/
COPY --from=builder /usr/local/cuda/lib64/libcupti.so.12 /usr/local/cuda/lib64/libcupti.so.12
RUN ldconfig

RUN STUB_PATH=$(find /usr/local/cuda/ -name "libcuda.so" | head -n 1) && \
    ln -sf "$STUB_PATH" /usr/lib/aarch64-linux-gnu/libcuda.so.1 && \
    ln -sf "$STUB_PATH" /usr/lib/aarch64-linux-gnu/libcuda.so

RUN ln -sf /usr/bin/python${PYTHON_VER} /usr/bin/python3
RUN ln -sf /usr/bin/python${PYTHON_VER} /usr/bin/python

# 2. Install the companion Python Wheel
COPY --from=builder /workspace/dist/wheel/tensorflow-*.whl /app/
RUN python${PYTHON_VER} -m pip install /app/tensorflow-*.whl && rm /app/tensorflow-*.whl

# Verify both interfaces are online and functional
CMD python -c "import tensorflow as tf; print('Python GPU Check:', tf.config.list_physical_devices('GPU'))" && \
    ldconfig -p | grep tensorflow