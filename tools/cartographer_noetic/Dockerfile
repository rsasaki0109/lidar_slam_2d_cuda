FROM ros:melodic-ros-base

SHELL ["/bin/bash", "-lc"]

# ROS Noetic (focal) apt no longer ships cartographer binaries; Melodic (bionic) still does.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-melodic-cartographer \
    ros-melodic-cartographer-ros \
    ros-melodic-tf2-ros \
    ros-melodic-tf2-tools \
    ros-melodic-robot-state-publisher \
    ros-melodic-xacro \
    ros-melodic-rosbag \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

CMD ["bash", "-lc", "echo 'Cartographer Melodic image ready'; bash"]
