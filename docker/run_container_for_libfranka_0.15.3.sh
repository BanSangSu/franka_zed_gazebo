#!/bin/bash
#
# Group rheinrobot: Run Franka ROS container

## For Athena side robot, libfranka_0.18.2
# CONTAINER_NAME="rheinrobot_franka_project_athena"
# IMAGE_NAME="bandi0605/rheinrobot:ban_experiments_test_0.3.9_libfranka_0.18.2"

## For Poseidon side robot, libfranka_0.15.3
# CONTAINER_NAME="rheinrobot_franka_project_poseidon"
# IMAGE_NAME="bandi0605/rheinrobot:ban_experiments_test_0.3.9_libfranka_0.15.3"
#

set -e

CONTAINER_NAME="rheinrobot_franka_project_poseidon"
IMAGE_NAME="bandi0605/rheinrobot:ban_experiments_test_0.3.9_libfranka_0.15.3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_SRC="${SCRIPT_DIR}/ros_ws/src"

# Allow GUI (run once per session)
xhost +local:root 2>/dev/null || true

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container ${CONTAINER_NAME} is already running. Attaching..."
    exec docker exec -it "${CONTAINER_NAME}" bash
  else
    echo "Starting existing container ${CONTAINER_NAME}..."
    docker start "${CONTAINER_NAME}"
    exec docker exec -it "${CONTAINER_NAME}" bash
  fi
else
  echo "Creating and starting container ${CONTAINER_NAME}..."
  
  # Automatically remove the container on exit
  # exec docker run -it --rm \
  
  # Keep the container after it stops
  exec docker run -it \
    --name "${CONTAINER_NAME}" \
    --network host \
    --privileged \
    -e DISPLAY="${DISPLAY}" \
    -e QT_X11_NO_MITSHM=1 \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e TERM=xterm-256color \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v /dev:/dev \
    -v "${WORKSPACE_SRC}:/opt/ros_ws/src/local_src" \
    --gpus all \
    "${IMAGE_NAME}" \
    bash
fi

# If you need specific features, add command below on the docker run -it  
    # --ipc=host \    # Enable shared memory to prevent bus errors in ROS/GUI
    # -e ROS_NAMESPACE=/group_rheinrobot \ # Optional: Groups all ROS nodes under this namespace
    # --runtime=nvidia \ # Use NVIDIA runtime for GPU support (if --gpus all is not working)
