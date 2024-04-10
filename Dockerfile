# 1. Test setup:
# docker run -it --rm --gpus all nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04 nvidia-smi
#
# If the above does not work, try adding the --privileged flag
# and changing the command to `sh -c 'ldconfig -v && nvidia-smi'`.
#
# 2. Start training:
# docker build -f  dreamerv3/Dockerfile -t img . && \
# docker run -it --rm --gpus all -v ~/logdir:/logdir img \
#   sh scripts/xvfb_run.sh python3 dreamerv3/train.py \
#   --logdir "/logdir/$(date +%Y%m%d-%H%M%S)" \
#   --configs dmc_vision --task dmc_walker_walk
#
# 3. See results:
# tensorboard --logdir ~/logdir

# System
# FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
FROM nvidia/cudagl:11.4.1-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/San_Francisco

# Upgrade python to 3.9
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.9 python3.9-dev python3.9-distutils && \
    apt-get install -y python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

ENV PYTHONUNBUFFERED 1
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PIP_NO_CACHE_DIR 1
RUN apt-get update && apt-get install -y \
  ffmpeg git python3-pip vim libglew-dev \
  x11-xserver-utils xvfb \
  && apt-get clean
RUN pip3 install --upgrade pip

# Envs
ENV MUJOCO_GL egl
ENV DMLAB_DATASET_PATH /dmlab_data
COPY /dreamerv3/embodied/scripts /scripts
RUN sh scripts/install-dmlab.sh
# RUN sh scripts/install-atari.sh
# RUN sh scripts/install-minecraft.sh
ENV NUMBA_CACHE_DIR=/tmp
RUN pip3 install crafter
RUN pip3 install dm_control
RUN pip3 install robodesk
RUN pip3 install bsuite
RUN pip3 install robosuite

# Agent
RUN pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip3 install jaxlib
RUN pip3 install tensorflow_probability
RUN pip3 install optax
RUN pip3 install tensorflow-cpu
ENV XLA_PYTHON_CLIENT_MEM_FRACTION 0.8

# Google Cloud DNS cache (optional)
ENV GCS_RESOLVE_REFRESH_SECS=60
ENV GCS_REQUEST_CONNECTION_TIMEOUT_SECS=300
ENV GCS_METADATA_REQUEST_TIMEOUT_SECS=300
ENV GCS_READ_REQUEST_TIMEOUT_SECS=300
ENV GCS_WRITE_REQUEST_TIMEOUT_SECS=600

# Mount and install racecar_gym
COPY /racecar_gym/ /home/racecar_gym/
RUN cd /home/racecar_gym/ && pip3 install -e .

# Embodied
RUN pip3 install numpy cloudpickle ruamel.yaml rich zmq msgpack
RUN pip3 install gymnasium
COPY /dreamerv3/ /home/dreamerv3/
RUN chown -R 1000:root /home/dreamerv3/embodied && chmod -R 775 /home/dreamerv3/embodied

WORKDIR /home/dreamerv3/
