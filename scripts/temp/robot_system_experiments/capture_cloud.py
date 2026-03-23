#!/usr/bin/env python3
import rospy
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import struct
import sys

# CONFIG
TOPIC_NAME = "/zedr/zed_node/point_cloud/cloud_registered"
TARGET_FRAMES = 10  # How many frames to average

class BurstCapture:
    def __init__(self):
        rospy.init_node('burst_capture_node', anonymous=True)
        
        self.points_buffer = []
        self.colors_buffer = []
        self.frame_count = 0
        
        rospy.loginfo(f"--- STARTING BURST CAPTURE ({TARGET_FRAMES} Frames) ---")
        rospy.loginfo(f"Listening on {TOPIC_NAME}...")
        
        self.sub = rospy.Subscriber(TOPIC_NAME, PointCloud2, self.callback)
        
    def callback(self, ros_cloud):
        # Stop if we have enough data
        if self.frame_count >= TARGET_FRAMES:
            return

        self.frame_count += 1
        rospy.loginfo(f"Capturing Frame {self.frame_count}/{TARGET_FRAMES}...")
        
        # 1. Parse Points (Using your reference logic)
        gen = pc2.read_points(ros_cloud, field_names=("x", "y", "z", "rgb"), skip_nans=True)
        
        # Temporary lists for this frame
        frame_xyz = []
        frame_rgb = []
        
        for p in gen:
            x, y, z, float_rgb = p
            frame_xyz.append([x, y, z])
            
            # Pack/Unpack RGB (Exact copy of your working logic)
            packed = struct.pack('f', float_rgb)
            b, g, r, a = struct.unpack('BBBB', packed)
            frame_rgb.append([r, g, b])

        # 2. Add to Main Buffer
        self.points_buffer.extend(frame_xyz)
        self.colors_buffer.extend(frame_rgb)
        
        # 3. Check if done
        if self.frame_count >= TARGET_FRAMES:
            self.save_and_exit()

    def save_and_exit(self):
        rospy.loginfo("--- CAPTURE COMPLETE. PROCESSING... ---")
        self.sub.unregister() # Stop listening
        
        if not self.points_buffer:
            rospy.logerr("Error: Buffer is empty. No points captured.")
            rospy.signal_shutdown("Failure")
            return

        # Convert to Open3D
        xyz = np.array(self.points_buffer)
        colors = np.array(self.colors_buffer) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Save merged file
        filename = "burst_capture.ply"
        o3d.io.write_point_cloud(filename, pcd)
        
        rospy.loginfo(f"SUCCESS: Saved '{filename}'")
        rospy.loginfo(f"Total Points: {len(xyz)} (Merged from {TARGET_FRAMES} frames)")
        rospy.loginfo("Run your detection script on this file now.")
        
        rospy.signal_shutdown("Done")

if __name__ == '__main__':
    try:
        BurstCapture()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass