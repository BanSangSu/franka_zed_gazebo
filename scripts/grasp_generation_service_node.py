#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
import os
import sys
import argparse
import torch

# ROS Messages
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseArray
from cv_bridge import CvBridge
from sensor_msgs.point_cloud2 import read_points
from contact_graspnet_pytorch.visualization_utils_o3d import visualize_grasps, show_image

# --- Contact Graspnet Imports ---
# Ensure this path points to your contact_graspnet_pytorch folder
sys.path.append(os.path.join(os.path.dirname(__file__), 'contact_graspnet_pytorch'))

from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from contact_graspnet_pytorch import config_utils
from contact_graspnet_pytorch.contact_graspnet_pytorch.checkpoints import CheckpointIO 

# We will use a standard service or a custom one. 
# For this example, I will assume a wrapper class that can be imported 
# or a service definition. Let's build a class that processes the data.

class GraspServiceNode:
    def __init__(self, init_node=False):
        if init_node:
            # rospy.init_node('contact_graspnet_service')
            rospy.init_node('contact_graspnet_service')
        
        self.bridge = CvBridge()
        
        # --- Parameters ---
        self.ckpt_dir = rospy.get_param('~ckpt_dir', 'checkpoints/contact_graspnet')
        
        current_dir= os.path.dirname(os.path.abspath(__file__))
        package_path = os.path.dirname(current_dir)
        rospy.loginfo(f"Package exists at: {package_path}")
        
        ckpt_base = os.path.join(package_path, self.ckpt_dir)
        rospy.loginfo(f"Looking for checkpoints in: {ckpt_base}")


        self.forward_passes = rospy.get_param('~forward_passes', 1)
        self.z_range = rospy.get_param('~z_range', [0.2, 1.8])
        self.pc_segments_flag = rospy.get_param('~pc_segments_flag', True)
        self.local_regions = rospy.get_param('~local_regions', True)
        self.filter_grasps = rospy.get_param('~filter_grasps', True)
        self.use_cam_boxes = rospy.get_param('~use_cam_boxes', True)

        # --- Load Model ---
        rospy.loginfo("Loading Contact Graspnet Model...")
        global_config = config_utils.load_config(ckpt_base, batch_size=self.forward_passes)
        self.grasp_estimator = GraspEstimator(global_config)
        
        # Load Checkpoint
        model_checkpoint_dir = os.path.join(ckpt_base, 'checkpoints')
        rospy.loginfo(f"Check checkpoints dir: {model_checkpoint_dir}")
        checkpoint_io = CheckpointIO(checkpoint_dir=model_checkpoint_dir, model=self.grasp_estimator.model)
        try:
            checkpoint_io.load('model.pt')
            rospy.loginfo("Model loaded successfully.")
        except FileExistsError:
            rospy.logerr("No model checkpoint found.")

        # --- Service Definition ---
        # Note: In a real deployment, define a .srv file like DetectGrasps.srv
        # For this code to run immediately, we will use a Topic-based Request/Reply 
        # or a Service if you have one. 
        # Here I will simulate the processing logic which you can wrap in your specific Service callback.
        
        # Placeholder for service server
        # self.service = rospy.Service('detect_grasps', DetectGrasps, self.handle_grasp_request)
        rospy.loginfo("Grasp Service Ready (Logic Implemented).")

    def predict(self, rgb, depth, segmap, pc_full, K):
        """
        Main inference logic to be called by the Service Callback.
        
        Args:
            rgb: HxWx3 uint8
            depth: HxW float32 (meters)
            segmap: HxW float (object IDs)
            K: 3x3 Camera Matrix
        Returns:
            Dictionary mapping object_id -> list of grasps (4x4 matrices)
        """
        
        # 1. Convert Depth to Point Cloud
        # The library's extract_point_clouds expects depth in meters)
        pc_segments = {}

        if segmap is None and (self.local_regions or self.filter_grasps):
            raise ValueError('Need segmentation map to extract local regions or filter grasps')

        if pc_full is None or self.pc_segments_flag:
            print('Converting depth to point cloud(s)...')
            pc_full, pc_segments, pc_colors = self.grasp_estimator.extract_point_clouds(
                depth, K, segmap=segmap, rgb=rgb,
                skip_border_objects=False, 
                z_range=self.z_range
            )
        else:
            self.local_regions = False
            self.filter_grasps = False
            self.use_cam_boxes = False
        
        print(f"Full point cloud: {pc_full.shape}")

        print('Generating Grasps...')

        print(f"pc_segments keys: {list(pc_segments.keys())}")
        # 2. Predict Grasps
        # pred_grasps_cam is a dict: {obj_id: (N, 4, 4)}
        pred_grasps_cam, scores, contact_pts, _ = self.grasp_estimator.predict_scene_grasps(
            pc_full, 
            pc_segments=pc_segments, 
            local_regions=self.local_regions, 
            filter_grasps=self.filter_grasps, 
            forward_passes=self.forward_passes,
            use_cam_boxes=self.use_cam_boxes
        )

        if len(pc_segments) == 0:
            return {}

        # show_image(rgb, segmap)
        # visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)        
        return pred_grasps_cam, scores

    def process_ros_data(self, rgb_msg, depth_msg, segmap, pc_msg, k_msg):
        # Convert ROS to Numpy
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            # segmap = self.bridge.imgmsg_to_cv2(segmap_msg, "mono8") # Assuming IDs are 0-255


            if k_msg.K is not None:
                if isinstance(k_msg.K,str):
                    cam_K = eval(k_msg.K)
                cam_K = np.array(k_msg.K).reshape(3,3)
            
            points = read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True)
            pc_full = np.array(list(points))   # or convert to NumPy if you prefer
            # Run Inference
            grasps_dict, scores_dict = self.predict(rgb, depth, segmap, pc_full, cam_K)

            # Format results for ROS (This would go into your Service Response)
            # Returning a simplified structure for demonstration
            results = []
            for obj_id, grasp_matrices in grasps_dict.items():
                 # grasp_matrices is (N, 4, 4)
                 # Get the best grasp (highest score) or all of them
                 obj_scores = scores_dict[obj_id]
                 best_idx = np.argmax(obj_scores)
                 best_grasp = grasp_matrices[best_idx]
                 results.append({
                     "id": int(obj_id),
                     "pose": best_grasp,
                     "score": obj_scores[best_idx]
                 })
                 
            return results

        except Exception as e:
            rospy.logerr(f"Grasp Prediction Error: {e}")
            return []

if __name__ == '__main__':
    # Add argument parsing if needed to override defaults
    node = GraspServiceNode(init_node=True)
    rospy.spin()