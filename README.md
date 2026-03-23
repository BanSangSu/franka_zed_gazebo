# 🐼 Franka ZED Noetic

An integrated ROS workspace for controlling the **Franka Emika Panda Robot** with **ZED2 Camera (Eye-to-Hand configuration)** in both **simulation (Gazebo)** and **real robot** environments. This project includes support for multiple tasks (manipulation, grasping, flipping, stacking) with dual robot configurations (**Poseidon** and **Athena**).

<p align="center">
  <img src="images/gazebo.png" width="600"/>
</p>

---

## 📋 Table of Contents

1. [🐳 Docker Setup](#docker-setup)
2. [🤖 Robot Configuration (Poseidon & Athena)](#robot-configuration)
3. [🎮 Simulation (Gazebo)](#simulation)
4. [🦾 Real Robot Setup](#real-robot-setup)
5. [🚀 Robot System Launchers](#robot-system-launchers)
6. [🔧 Quick Reference](#quick-reference)
7. [📚 Additional Resources](#additional-resources)
8. [⚙️ System Specifications](#system-specifications)
9. [📝 Notes](#notes)

---

<h2 id="docker-setup">🐳 Docker Setup</h2>

This project provides **Docker containers** with pre-installed dependencies for seamless development and deployment.

### Prerequisites

- **Docker** and **Docker Compose** installed
- **NVIDIA GPU** (optional but recommended)
- **X11 forwarding** enabled for GUI support

### Available Docker Configurations

Two container scripts are provided for different **libfranka versions**:

#### **1. For Poseidon Robot (libfranka 0.15.3)**

```bash
cd docker/
./run_container_for_libfranka_0.15.3.sh
```

**Configuration Details:**
- **Container Name:** `rheinrobot_franka_project_poseidon`
- **Image:** `bandi0605/rheinrobot:ban_experiments_test_0.3.9_libfranka_0.15.3`
- **libfranka Version:** 0.15.3
- **Robot IP:** 192.168.2.55

#### **2. For Athena Robot (libfranka 0.18.2)**

```bash
cd docker/
./run_container_for_libfranka_0.18.2.sh
```

**Configuration Details:**
- **Container Name:** `rheinrobot_franka_project_athena`
- **Image:** `bandi0605/rheinrobot:ban_experiments_test_0.3.9_libfranka_0.18.2`
- **libfranka Version:** 0.18.2
- **Robot IP:** 10.10.10.10

### Docker Setup Instructions

#### Step 1: Allow GUI Access (One-time setup per session)

```bash
xhost +local:root
```

#### Step 2: Run the Container

Choose the appropriate script for your robot:

```bash
# For Poseidon
./docker/run_container_for_libfranka_0.15.3.sh

# OR for Athena
./docker/run_container_for_libfranka_0.18.2.sh
```

The script will:
- ✅ Check if the container is already running
- ✅ Attach to existing container if running
- ✅ Create and start a new container if needed
- ✅ Mount local workspace for development
- ✅ Enable GPU support
- ✅ Enable GUI forwarding

#### Step 3: Inside the Container

Once inside the container, source the ROS environment:

```bash
source /opt/ros/noetic/setup.bash
source /opt/ros_ws/devel/setup.bash
```

Activate the Python virtual environment using `uv`:

```bash
source /opt/ros_ws/.venv/bin/activate
```

This activates the Python virtual environment for managing Python dependencies and scripts within the workspace.

#### Step 4: Access from Another Terminal

To open another terminal in the same container:

```bash
# For Poseidon
docker exec -it rheinrobot_franka_project_poseidon bash

# For Athena
docker exec -it rheinrobot_franka_project_athena bash
```

#### Step 5: Update Repository

Keep your local fork up-to-date:

```bash
cd /opt/ros_ws/src/franka_zed_gazebo
git pull --recurse-submodules

# Or if you already cloned the repo:
git submodule update --init --recursive

cd ../..
```

### Docker Shell Script Details

Both `run_container_for_libfranka_0.15.3.sh` and `run_container_for_libfranka_0.18.2.sh` handle:

- **Container Management:** Checks if container exists, is running, or needs to be created
- **GPU Support:** `--gpus all` flag for NVIDIA GPU acceleration
- **Network:** `--network host` for ROS communication
- **Display:** X11 socket forwarding for Gazebo/RViZ GUI
- **Device Access:** `/dev` mounting for cameras and hardware
- **Workspace Mounting:** Local source code synced to `/opt/ros_ws/src/local_src`

---

<h2 id="robot-configuration">🤖 Robot Configuration</h2>

This workspace supports two Franka Panda robots with different configurations:

### Poseidon Robot

| Parameter | Value |
|-----------|-------|
| **Robot IP** | `192.168.2.55` |
| **libfranka Version** | 0.15.3 |
| **Container Name** | `rheinrobot_franka_project_poseidon` |
| **Docker Image** | `bandi0605/rheinrobot:ban_experiments_test_0.3.9_libfranka_0.15.3` |
| **ZED Camera** | Eye-to-Hand (mounted with gripper in FoV) |

### Athena Robot

| Parameter | Value |
|-----------|-------|
| **Robot IP** | `10.10.10.10` |
| **libfranka Version** | 0.18.2 |
| **Container Name** | `rheinrobot_franka_project_athena` |
| **Docker Image** | `bandi0605/rheinrobot:ban_experiments_test_0.3.9_libfranka_0.18.2` |
| **ZED Camera** | Eye-to-Hand (mounted with gripper in FoV) |

---
<h2 id="simulation">🎮 Simulation (Gazebo)</h2>

### Quick Start - Simulation Only

Launch the full simulation environment with Gazebo, RViZ, and MoveIt:

```bash
roslaunch franka_zed_gazebo moveit_gazebo_panda.launch
```

This launches:
- **Gazebo Simulator** with the Panda robot and Pearl lab environment
- **RViZ** visualization
- **MoveIt!** Motion Planning interface
- **ZED2 camera** simulation (image topics available)

You can then:
1. Use MoveIt to plan and execute robot motions
2. Visualize camera feeds by adding topics in RViZ
3. Spawn cubes on the table for manipulation tasks

> **Note:** Edit `scripts/spawn_cubes.py` to adjust the number and placement of cubes.

> **Important:** Make sure to update **topic names** in the perception and control nodes according to your Gazebo configuration. The default topic names for camera feeds and robot state may vary depending on your launch parameters. Check topics with `rostopic list` and set the correct topics in the Python scripts like `perception_service_node.py` to ensure they match your actual ROS topics.

### Running Robot System Tasks in Simulation

After launching the simulation, start the robot system orchestration:

```bash
roslaunch franka_zed_gazebo robot_system_moveit.launch
```

---
<h2 id="real-robot-setup">🦾 Real Robot Setup</h2>

### Prerequisites

1. **Enable Franka Control Interface (FCI)**
   - Access the robot controller via browser at the robot's IP address
   - Set the robot to **Execution mode**
   - Unlock the robot

2. **ZED2 Camera Mount**
   - Check the `3d_prints/` folder for the appropriate mount (with/without gripper in FoV)
   - Ensure camera is properly secured and calibrated

3. **Network Connection**
   - Robot must be reachable on its configured IP address
   - Host PC should be on the same network

### Real Robot Launch Instructions

**IMPORTANT:** Before launching, you **MUST configure the correct robot IP** in the launch file.

#### Configuring Robot IP Address

Edit the appropriate real robot launch file and set the `robot_ip` parameter:

**For MoveIt Control:**
- File: `launch/real_robot_zed2_moveit.launch`
- Default: `10.10.10.10` (Athena)
- For Poseidon: Change to `192.168.2.55`

**For Impedance Control:**
- File: `launch/real_robot_zed2_impedance.launch`
- Default: `10.10.10.10` (Athena)
- For Poseidon: Change to `192.168.2.55`

Example configuration in the launch file:
```xml
<arg name="robot_ip" default="10.10.10.10" />
<!-- 10.10.10.10 Athena -->
<!-- 192.168.2.55 Poseidon -->
```

#### Launch Real Robot with MoveIt

```bash
roslaunch franka_zed_gazebo real_robot_zed2_moveit.launch
```

This launches:
- MoveIt! motion planning interface
- ZED2 camera with eye-to-hand calibration
- Robot control via FCI

**For Flipping Task:** Load the cartesian impedance controller:

```bash
rosservice call /controller_manager/load_controller "name: 'cartesian_impedance_example_controller'"
```

This command is commented in the [real_robot_zed2_moveit.launch](./launch/real_robot_zed2_moveit.launch) file for reference.

#### Launch Real Robot with Impedance Control

```bash
roslaunch franka_zed_gazebo real_robot_zed2_impedance.launch
```

This launches:
- Impedance controller for compliant manipulation
- Joint impedance gains loaded from `config/joint_impedance_command_controller.yaml`
- ZED2 camera integration

> **Manual Robot Operation:**
> To manually move the robot in Programming mode, press the two buttons on the left side of the end-effector simultaneously. See the [Franka Handbook](https://www.generationrobots.com/media/franka-emika-robot-handbook.pdf) for exact button locations.

---

<h2 id="robot-system-launchers">🚀 Robot System Launchers</h2>

The robot system uses **task-specific orchestration** with perception, planning, and control modules. Each launcher corresponds to a different task.

### Available Robot System Launchers

#### 1. **MoveIt-based Manipulation**

```bash
roslaunch franka_zed_gazebo robot_system_moveit.launch
```

**Includes:**
- Perception service (object detection & grasping)
- MoveIt! motion planning
- MoveIt-based control (Position control)
- Orchestrator node (coordinates execution)
- Behavior Tree Visualization (rqt_py_trees)

**Use for:** General manipulation tasks, pick-and-place, reaching

---

#### 2. **Impedance Control**

```bash
roslaunch franka_zed_gazebo robot_system_impedance.launch
```

**Includes:**
- Perception service
- Cartesian motion planning
- Impedance controller (compliant control)
- Orchestrator node
- Behavior Tree Visualization

**Use for:** Tasks requiring force feedback, gentle interactions, contact-rich manipulation

**Configuration:**
- Impedance gains: `config/joint_impedance_command_controller.yaml`
- Cartesian stiffness and damping parameters can be tuned here

---

#### 3. **Flipping Task**

```bash
roslaunch franka_zed_gazebo robot_system_flipping.launch
```

**Includes:**
- Perception service
- Cartesian motion planning
- MoveIt-based control
- **Flipping-specific orchestrator** (`orchestrate_node_flipping_task.py`)
- Behavior Tree Visualization

**Use for:** Object flipping and reorientation tasks

**Features:**
- Specialized planning for flipping motions
- Grasp stability assessment
- Multi-phase motion execution

---

#### 4. **Stacking Task**

```bash
roslaunch franka_zed_gazebo robot_system_stacking.launch
```

**Includes:**
- Perception service
- Motion planning
- Control system
- **Stacking-specific orchestrator** (`orchestrate_node_stacking_cube_task.py`)
- Behavior Tree Visualization

**Use for:** Cube stacking and tower-building tasks

**Features:**
- Stability-aware placement
- Multi-cube coordination
- Height-aware planning


---

<h2 id="quick-reference">🔧 Quick Reference</h2>

### Starting a Simulation Task

```bash
# 1. Terminal 1: Launch simulation with MoveIt
roslaunch franka_zed_gazebo moveit_gazebo_panda.launch

# 2. Terminal 2: Launch task system (choose one)
roslaunch franka_zed_gazebo robot_system_moveit.launch
# OR for impedance
roslaunch franka_zed_gazebo robot_system_impedance.launch
# OR for flipping
roslaunch franka_zed_gazebo robot_system_flipping.launch
```

### Starting Real Robot Task

```bash
# IMPORTANT: Update robot_ip and camera calibration (zed_eye2hand.launch) in the launch file first!

# 1. Terminal 1: Launch real robot MoveIt
roslaunch franka_zed_gazebo real_robot_zed2_moveit.launch

# 2. Terminal 2: Launch task system
roslaunch franka_zed_gazebo robot_system_moveit.launch
# OR for impedance
roslaunch franka_zed_gazebo robot_system_impedance.launch
# OR for flipping
roslaunch franka_zed_gazebo robot_system_flipping.launch
```

---

<h2 id="additional-resources">📚 Additional Resources</h2>

- **Franka Handbook:** [PDF Link](https://www.generationrobots.com/media/franka-emika-robot-handbook.pdf)
- **Original Repository:** [pearl-robot-lab/franka_zed_gazebo](https://github.com/pearl-robot-lab/franka_zed_gazebo)
- **libfranka Documentation:** [frankaemika.github.io](https://frankaemika.github.io/docs/installation_linux.html)
- **MoveIt Documentation:** [ros-planning.github.io](https://ros-planning.github.io/moveit_tutorials/)
- **ZED Camera Documentation:** [stereolabs.com](https://github.com/stereolabs/zed-ros-wrapper)

---

<h2 id="system-specifications">⚙️ System Specifications</h2>

This project is built and tested on the following software stack:

| Component | Version |
|-----------|---------|
| **ROS** | Noetic |
| **Ubuntu** | 20.04 LTS |
| **Python** | 3.8+ |
| **libfranka (Poseidon)** | 0.15.3 |
| **libfranka (Athena)** | 0.18.2 |

### Python Dependencies

- **Contact-GraspNet (Pytorch version)** - Grasp generation and planning
- **PyTorch** - Neural network framework
- **NumPy, SciPy** - Scientific computing
- **OpenCV** - Computer vision
- **py_trees** - Behavior tree framework
- **ROS Python packages** - rospy, geometry_msgs, std_msgs, sensor_msgs, etc.

---

<h2 id="notes">📝 Notes</h2>


- **Camera Mount:** The ZED2 can be mounted with or without the gripper in the field of view. Check `urdf/panda_camera.urdf.xacro` to select the appropriate configuration.
- **Network Configuration:** Ensure the host PC can reach the robot IP address over Ethernet.
- **GPU Optimization:** The Docker containers are configured for NVIDIA GPU support. Make sure the NVIDIA Container Runtime is installed.
- **Virtual Environment:** The workspace uses a Python virtual environment at `/opt/ros_ws/.venv` inside the container.

