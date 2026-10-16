ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base

ARG ROS_DISTRO
ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-c"]

# Use fast mirror for apt updates
RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|http://de.archive.ubuntu.com/ubuntu/|g' /etc/apt/sources.list

# Install system dependencies & ROS 2 packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-std-msgs \
    ros-${ROS_DISTRO}-sensor-msgs \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch wheels first with CUDA 12.8 support
RUN pip3 install --no-cache-dir \
    torch==2.9.1+cu128 \
    torchvision==0.24.1+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Copy package source files
COPY . /app/ros2_ws/src/bob_q21img

# Build ROS 2 workspace
WORKDIR /app/ros2_ws
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && colcon build --packages-select bob_q21img

WORKDIR /app

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "run", "bob_q21img", "tti", "--ros-args", "-r", "__ns:=/q21img", "--params-file", "/app/ros2_ws/src/bob_q21img/config/tti.yaml"]
