# Use Python 3.10 base image
FROM mcr.microsoft.com/devcontainers/python:0-3.10

# Install required system packages
RUN apt-get update && apt-get install -y \
    git cmake libgl1-mesa-glx libglfw3 libglew-dev \
    libosmesa6-dev patchelf ffmpeg xvfb swig wget unzip && \
    apt-get clean

# Install Python packages
RUN pip install --upgrade pip

# Pin Cython to <3 and ensure mujoco-py compatibility
RUN pip install Cython==0.29.36

# Install all Python dependencies
RUN pip install \
    gymnasium[mujoco,other] mujoco mujoco-py==2.1.2.14 \
    numpy matplotlib torch torchvision \
    stable-baselines3 stable-baselines3[extra] sb3-contrib \
    huggingface_hub huggingface_sb3 pyyaml tqdm pyvirtualdisplay "shimmy>=0.2.1"


# Download and extract MuJoCo 2.1.0 to ~/.mujoco
RUN mkdir -p /root/.mujoco && \
    wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco.tar.gz && \
    tar -xzf /tmp/mujoco.tar.gz -C /root/.mujoco && \
    rm /tmp/mujoco.tar.gz

# Set MuJoCo environment variables
ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
ENV LD_LIBRARY_PATH="/root/.mujoco/mujoco210/bin"
ENV LIBGL_ALWAYS_SOFTWARE=1

# Clone and setup RL Zoo
WORKDIR /workspaces/go2_sim_rl
RUN git clone https://github.com/DLR-RM/rl-baselines3-zoo.git && \
    cd rl-baselines3-zoo && \
    pip install -r requirements.txt