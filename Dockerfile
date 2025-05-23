# 1. Test setup:
# docker run -it --rm --gpus all pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime nvidia-smi
#
# If the above does not work, try adding the --privileged flag
# and changing the command to `sh -c 'ldconfig -v && nvidia-smi'`.
#
# 2. Start training:
# docker build -f  Dockerfile -t img . && \
# docker run -it --rm --gpus all -v $PWD:/workspace -u $(id -u):$(id -g) img \
#   sh xvfb_run.sh python3 dreamer.py \
#   --configs dmc_vision --task dmc_walker_walk \
#   --logdir "./logdir/dmc_walker_walk"
#
# 3. See results:
# tensorboard --logdir ~/logdir

# System
# FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
FROM nvidia/cudagl:11.4.1-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/San_Francisco
ENV PYTHONUNBUFFERED 1
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PIP_NO_CACHE_DIR 1
RUN apt-get update && apt-get install -y \
    vim libglew2.1 libgl1-mesa-glx libosmesa6 \
    wget unrar cmake g++ libgl1-mesa-dev \
    libx11-6 openjdk-8-jdk x11-xserver-utils xvfb \
    && apt-get clean

# Install python3.10 and pip
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3.10-distutils && \
    apt-get install -y python3-pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN apt update && apt install -y curl
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Envs
ENV NUMBA_CACHE_DIR=/tmp

# dmc setup
RUN pip3 install tensorboard
RUN pip3 install gym==0.19.0
RUN pip3 install mujoco==2.3.5
RUN pip3 install dm_control==1.0.9
RUN pip3 install moviepy

# crafter setup
RUN pip3 install crafter

RUN pip3 install numpy==1.23.0

# Install pytorch 
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install pybullet scipy gymnasium yamldataclassconfig nptyping==1.4.4 pettingzoo==1.22.3
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev
RUN pip3 install opencv-python
COPY /racecar_gym/ /home/racecar_gym/
RUN cd /home/racecar_gym/ && pip3 install -e . --no-deps

# Install ros noetic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt install curl -y
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-get update && apt-get install -y ros-noetic-desktop

# Install rosdep
RUN apt install python3-rosdep -y  && rosdep init && rosdep update

# Allow sourcing
RUN rm /bin/sh && ln -s /bin/bash /bin/sh 

# Install rospkg in conda environment
RUN pip3 install rospkg

# Install ros2 foxy
RUN apt-get update && apt-get install -y curl gnupg2 lsb-release
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN sh -c 'echo "deb [arch=amd64,arm64] http://packages.ros.org/ros2/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros2-latest.list'
RUN apt-get update && apt-get install -y ros-foxy-desktop

#  Alias for sourcing
# sr1 -> source ros 1
# sr2 -> source ros 2
RUN echo "alias sr1='source /opt/ros/noetic/setup.bash'" >> ~/.bashrc
RUN echo "alias sr2='source /opt/ros/foxy/setup.bash'" >> ~/.bashrc

# Install ackermann_msgs in ros1 and ros2
RUN apt-get install -y ros-noetic-ackermann-msgs ros-foxy-ackermann-msgs

# Install ros1 bridge
RUN pip install -U colcon-common-extensions
COPY /ros/ros2_ws /home/ros2_ws
RUN cd /home/ros2_ws && source /opt/ros/noetic/setup.bash && source /opt/ros/foxy/setup.bash && \
        colcon build --symlink-install --packages-select ros1_bridge --cmake-force-configure

# Install tmux
RUN apt-get install -y tmux

# # Set python3.8 as default (for roscore to work)
RUN rm -r /usr/bin/python3 && ln -s /usr/bin/python3.8 /usr/bin/python3

# workdir /home/dreamerv3
WORKDIR /home/dreamerv3