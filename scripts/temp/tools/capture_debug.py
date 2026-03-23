#!/usr/bin/env python3
import rospy
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2
import struct
import sys

# Ensure this matches your active topic!
TOPIC_NAME = "/static_zed2_camera/static_zed2/zed_node/point_cloud/cloud_registered" 

def callback(ros_cloud):
    rospy.loginfo("--- PointCloud Received ---")
    rospy.loginfo(f"Cloud Dimensions: {ros_cloud.width} x {ros_cloud.height}")
    
    # Check 1: Is the message empty?
    if ros_cloud.width * ros_cloud.height == 0:
        rospy.logwarn("CRITICAL: Received an empty PointCloud (0 points)!")
        rospy.signal_shutdown("Empty Cloud")
        return

    rospy.loginfo("Parsing points... (This might take a second)")

    # NOTE: We set skip_nans=True. If 100% of points are NaN, the loop below will never run.
    gen = pc2.read_points(ros_cloud, field_names=("x", "y", "z", "rgb"), skip_nans=True)
    
    xyz = []
    rgb_standard = [] 
    
    count = 0
    debug_printed = False

    for p in gen:
        x, y, z, float_rgb = p
        xyz.append([x, y, z])
        
        # Packing/Unpacking RGB
        packed = struct.pack('f', float_rgb)
        b, g, r, a = struct.unpack('BBBB', packed)
        
        rgb_standard.append([r, g, b])
        
        # DEBUG: Always print the FIRST valid point we see, no matter the color
        if not debug_printed:
            rospy.loginfo(f"SUCCESS: Found valid points!")
            rospy.loginfo(f"Sample Point -> X:{x:.2f}, Y:{y:.2f}, Z:{z:.2f}")
            rospy.loginfo(f"Sample Color -> R:{r}, G:{g}, B:{b} (Int: {struct.unpack('I', packed)[0]})")
            debug_printed = True
        
        count += 1

    rospy.loginfo(f"Total Valid Points Processed: {count}")

    if count == 0:
        rospy.logerr("ERROR: The cloud has dimensions, but ALL points were NaN (Invalid Depth).")
        rospy.logerr("Troubleshooting: 1. Is the camera blocked? 2. Is the object too close (<30cm)?")
        rospy.signal_shutdown("No Valid Data")
        return

    # Create Open3D objects
    xyz = np.array(xyz)
    colors = np.array(rgb_standard) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save
    filename = "debug_standard.ply"
    o3d.io.write_point_cloud(filename, pcd)
    rospy.loginfo(f"Saved {filename} with {len(pcd.points)} points.")
    
    rospy.signal_shutdown("Done")

def capture():
    rospy.init_node('capture_debug_v2', anonymous=True)
    rospy.loginfo(f"Listening on {TOPIC_NAME}...")
    rospy.Subscriber(TOPIC_NAME, PointCloud2, callback)
    rospy.spin()

if __name__ == '__main__':
    capture()