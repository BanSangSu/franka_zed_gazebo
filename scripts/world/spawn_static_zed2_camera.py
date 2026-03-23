#!/usr/bin/env python3
import rospy
import os
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose, Point, Quaternion
from tf.transformations import quaternion_from_euler
import subprocess

def spawn_zed2_camera():
    rospy.init_node('spawn_zed2_node', anonymous=True)
    
    # 1. Configuration
    xacro_file = rospy.get_param(
        '~xacro_file', 
        '/opt/ros_ws/src/franka_zed_gazebo/urdf/static_zed2_camera.xacro' 
    )
    model_name = "static_zed2_camera"
    
    # Position: 1m on left side, 0.5m high, rotated 180 deg (3.14 rad) to face toward origin
    spawn_pose = Pose(
        Point(1.0, 0.0, 0.5), 
        Quaternion(*quaternion_from_euler(0, 3.14, 0)) 
    )

    # 2. Process Xacro -> URDF
    urdf_string = ""

    # Get from Parameter Server
    # This ensures Gazebo gets the EXACT same model as Rviz
    if rospy.has_param("static_camera_description"):
        rospy.loginfo("Found 'static_camera_description' on parameter server. Using it.")
        urdf_string = rospy.get_param("static_camera_description")
    
    # Manual Load (Fallback)
    else:
        rospy.logwarn("Param 'static_camera_description' not found! Falling back to manual Xacro generation.")
        xacro_file = rospy.get_param(
            '~xacro_file', 
            '/opt/ros_ws/src/franka_zed_gazebo/urdf/static_zed2_camera.xacro' 
        )

    try:
        process = subprocess.Popen(['xacro', xacro_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        urdf_bytes, stderr = process.communicate()
        
        if process.returncode != 0:
            rospy.logerr("Xacro failed: " + str(stderr))
            return

        urdf_string = urdf_bytes.decode('utf-8') 
        rospy.loginfo(f"Successfully converted Xacro to URDF")
        
    except Exception as e:
        rospy.logerr(f"Failed to run xacro command: {e}")
        return

    # 3. Wait for Gazebo Service
    rospy.loginfo("Waiting for gazebo/spawn_urdf_model service...")
    try:
        rospy.wait_for_service("gazebo/spawn_urdf_model", timeout=10.0)
    except rospy.ROSException:
        rospy.logerr("Gazebo service unavailable!")
        return
    
    # 4. Call the Service
    try:
        spawn_model_prox = rospy.ServiceProxy("gazebo/spawn_urdf_model", SpawnModel)
        
        spawn_model_prox(
            model_name,       # Name in Gazebo
            urdf_string,      # The XML we generated
            model_name,       # Robot namespace
            spawn_pose,       # Initial pose
            "world"           # Reference frame
        )
        rospy.loginfo(f"Spawned model '{model_name}' successfully!")
        
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

if __name__ == '__main__':
    try:
        # Fixed function call name here
        spawn_zed2_camera()
    except rospy.ROSInterruptException:
        pass
