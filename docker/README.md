# **Franka ROS Noetic Docker Laptop Lab (FR3 + Vision)**

This repository provides a complete **Docker-based development environment for controlling the Franka Emika FR3 robot using ROS Noetic**, specifically designed for a **laptop or workstation setup with GPU support**.

This setup is intended for the following scenario:

- ✅ **Control PC is a GPU-equipped laptop/workstation**
- ✅ **No dedicated Franka NUC control PC**
- ✅ Vision, perception, and planning run directly on the laptop
- ✅ Robot is accessed via Ethernet from the laptop

The environment supports:

- **MoveIt** – motion planning
- **Intel RealSense**
- **ZED Cameras**
- **Pinocchio** – robot dynamics
- **Full GPU + GUI** support via **NVIDIA Container Toolkit**

---

## ⚡ Quick Start

### 1. Start the Container
To run everything at once, use the shell script (🌟 Recommended):
```bash
./run_container_for_libfranka_0.15.3
```

Alternatively, follow these step-by-step instructions:
```bash
# Allow GUI access for the container (run once per reboot)
xhost +local:root

# Build and start the container
docker compose up -d  # NOTE: Don't add --build

# Enter the container
docker exec -it franka_ros_lab bash

# Source ROS environment (inside container)
source /opt/ros/noetic/setup.bash
source /opt/ros_ws/devel/setup.bash
```

> 📁 **About `local_src`:**  
> The container mounts your local workspace (`/home/irobman-students/ros_ws/src`) to `/opt/ros_ws/src/local_src` inside the container.  
> This allows you to:
> - Develop ROS packages on your host machine
> - Build and test them inside the container
> - Keep your code synchronized between host and container
> 
> To use packages from `local_src`, build them and source the workspace:
> ```bash
> cd /opt/ros_ws
> catkin_make  # or catkin build
> source devel/setup.bash
> ```

### 2. Run MoveIt

Inside the container:

```bash
roslaunch panda_moveit_config franka_control.launch robot_ip:=10.10.10.10
```

### 3. Run ZED Camera

In a new terminal (or tmux session), enter the container and launch:

```bash
# Enter the container
docker exec -it franka_ros_lab bash

# Source ROS environment
source /opt/ros/noetic/setup.bash
source /opt/ros_ws/devel/setup.bash

# Launch ZED camera (choose your model)
roslaunch zed_wrapper zed2.launch      # For ZED 2
# roslaunch zed_wrapper zed2i.launch  # For ZED 2i
# roslaunch zed_wrapper zedm.launch   # For ZED Mini
```

> ⚠️ **Note:** The ZED camera must be connected to a USB 3.0 (blue) port.

### 4. Commit Your Changes from Container

If you've made changes inside the container (installed packages, modified configurations, etc.) and want to save them as a new image:

```bash
# Stop the container first (if running)
docker compose down

# Commit the container to a new image with a custom tag
# IMPORTANT: Use a custom tag to avoid overwriting the base image
docker commit franka_ros_lab franka-ros:custom

# Or with a more descriptive tag
docker commit franka_ros_lab franka-ros:my-changes-v1
```

> ⚠️ **Important:**  
> - **Never commit to the base image tag** (`franka-ros:base`) to preserve the original image
> - Always use a **custom tag** (e.g., `franka-ros:custom`, `franka-ros:my-setup`)
> - To use your custom image, update `docker-compose.yaml`:
>   ```yaml
>   image: franka-ros:custom  # Change from franka-ros:base
>   ```

---

## 📋 **Prerequisites (Host Machine & Robot)**

### ✅ Host Machine Requirements

Before running the container, ensure your **laptop / workstation** has:

- **NVIDIA Driver** installed (CUDA recommended)
- **Docker** + **Docker Compose** installed
- **NVIDIA Container Toolkit** installed (required for GPU + OpenGL)
- **Ubuntu 20.04** (recommended for ROS Noetic)

---

### ✅ Franka Robot & Library Compatibility (Critical)

This ROS Noetic laptop setup is **strictly matched to the following versions**:

| Component            | Required Version |
|----------------------|------------------|
| **Robot Firmware**   | ✅ **5.7.2** |
| **libfranka (Docker)** | ✅ **0.14.2** |
| **ROS Distribution** | ✅ **ROS Noetic** |

> 🔎 **Why these exact versions are required**
>
> - `libfranka 0.14.2` is the **last stable release officially compatible with firmware 5.7.x**  
> - Newer `libfranka (≥0.15)` **requires newer robot firmware** and may fail with:
>   - Torque control
>   - Rate-limiting APIs
>   - Impedance controllers
> - ROS Noetic `franka_ros` stacks were developed and tested against **libfranka 0.14.x**

> 📚 References:
>
> - Franka Emika libfranka release notes (v0.14.2)
> - Franka System Image 5.7.x documentation
> - franka_ros ROS Noetic compatibility matrix

---

## 🧰 Install NVIDIA Container Toolkit

If NVIDIA Container Toolkit is not installed, run:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
---

## 🌐 **Network Configuration (Required for Real Robot)**

Use a wired connection to the robot with:

* **Host IP:** `10.10.10.50`
* **Netmask:** `255.255.255.0`
* **Robot IP:** `10.10.10.10`

Verify connection:

```bash
ping 10.10.10.10
```

---

## 🛠️ **1. Setup & Installation**

### **Folder Structure**

Your working directory should contain:

```
.
├── Dockerfile
├── docker-compose.yaml
├── ros_entrypoint.sh
└── README.md
```

---

### **Build the Docker Image**

```bash
# Allow GUI access for the container (run once per reboot)
xhost +local:root

# Build and start the container
docker compose up -d --build
```

---

## 🚀 **2. Running & Entering the Docker Container**

### **Enter the container:**

```bash
docker exec -it franka_ros_lab bash
```

### **(Optional) Use tmux inside the container**

```bash
tmux
```

### **Stop the environment**

```bash
docker compose down
```

---

## 🤖 **3. Launching Franka FR3 + MoveIt (Quick Demo)**

Inside the container, launch the FR3 with MoveIt:

```bash
roslaunch panda_moveit_config franka_control.launch robot_ip:=10.10.10.10
```

---

## 📷 **4. Launching Intel RealSense**

The container includes the `realsense2_camera` package.

**Troubleshooting:**
If you see repeated "Success" reset messages, add `initial_reset:=true`.

### **Basic RGB + Depth stream**

```bash
roslaunch realsense2_camera rs_camera.launch initial_reset:=true
```

### **With Point Cloud**

```bash
roslaunch realsense2_camera rs_camera.launch \
    initial_reset:=true \
    filters:=pointcloud
```

### **RViz Topics**

* `/camera/color/image_raw`
* `/camera/depth/color/points`

---

## 👁️ **5. Launching ZED Camera**

The repository includes the **zed-ros-wrapper** (SDK 4.1).

⚠️ **Important:** The ZED camera **must** be connected to a **USB 3.0 (blue) port**.

### **Launch commands**

```bash
# ZED 2
roslaunch zed_wrapper zed2.launch

# ZED 2i
roslaunch zed_wrapper zed2i.launch

# ZED Mini
roslaunch zed_wrapper zedm.launch
```

### **RViz Topic**

* `/zed2/zed_node/rgb/image_rect_color`

