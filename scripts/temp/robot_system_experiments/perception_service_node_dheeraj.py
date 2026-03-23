#!/usr/bin/env python3
import rospy
import tf2_ros
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseArray, Pose, Vector3, PoseStamped
import numpy as np
import open3d as o3d
import cv2
import struct
import sys
import os
import moveit_commander 
from scipy.spatial.transform import Rotation as R_scipy

# --- IMPORT CUSTOM SERVICE ---
try:
    from franka_zed_gazebo.srv import PerceptionService, PerceptionServiceResponse
except ImportError:
    print("ERROR: Could not import PerceptionService.")
    sys.exit(1)

# --- SETUP PATHS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

try:
    import config
    from cloud_to_image import Cloud_to_Image
    from processor_backup2 import CubeSegmenter
    from detector3 import CubeDetector
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

# --- CONFIGURATION ---
TOPIC_CLOUD = "/zedr/zed_node/point_cloud/cloud_registered"
TOPIC_MARKERS = "/perception/detected_cubes"

# FRAMES
CAMERA_FRAME_ID = "zedr_left_camera_frame" 
ROBOT_FRAME_ID = "world"

TARGET_FRAMES = 25 

class PerceptionServiceNode:
    def __init__(self):
        rospy.init_node('perception_service_node', anonymous=False)
        
        # --- MoveIt Setup ---
        moveit_commander.roscpp_initialize(sys.argv)
        self.scene = moveit_commander.PlanningSceneInterface()
        rospy.sleep(1.0) 
        self.clear_all_objects()
        rospy.loginfo("Startup: Cleared collision scene.")
        
        # --- TF ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # --- Publishers ---
        self.marker_pub = rospy.Publisher(TOPIC_MARKERS, MarkerArray, queue_size=10, latch=True)
        self.srv = rospy.Service('/perception_service', PerceptionService, self.trigger_callback)
        
        # --- Workers ---
        self.points_buffer = []
        self.colors_buffer = []
        self.frame_count = 0
        self.capture_done = False
        
        self.preprocessor = Cloud_to_Image()
        self.cleaner = CubeSegmenter()
        self.detector = CubeDetector()
        
        rospy.loginfo(f"READY: Service '/perception_service' waiting. Frame: {ROBOT_FRAME_ID}")

    def trigger_callback(self, req):
        rospy.loginfo("--- SERVICE REQUEST RECEIVED ---")
        
        success_capture = self.capture_burst()
        if not success_capture:
            return PerceptionServiceResponse(success=False, message="Capture Failed")

        pcd = self.build_open3d_cloud()
        results = self.run_pipeline(pcd) 
        
        if not results:
            self.clear_all_objects() # Safety clear
            return PerceptionServiceResponse(
                success=True, 
                message="Scan complete. No objects found.",
                num_cubes=0,
                cube_poses=PoseArray()
            )

        # Response
        response = PerceptionServiceResponse()
        response.success = True
        response.message = f"Detected {len(results)} objects."
        response.num_cubes = len(results)
        response.cube_poses = PoseArray()
        response.cube_poses.header.frame_id = ROBOT_FRAME_ID
        response.cube_poses.header.stamp = rospy.Time.now()
        
        for i, (pos, quat, angle) in enumerate(results):
            p = Pose()
            p.position.x, p.position.y, p.position.z = pos
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat
            response.cube_poses.poses.append(p)
            response.confidences.append(1.0)
            response.dimensions.append(Vector3(0.045, 0.045, 0.045))
            response.labels.append("cube")

        self.publish_markers(results)
        self.update_cube_collision(results)
        
        rospy.loginfo(response.message)
        return response

    def clear_all_objects(self):
        """Removes both cubes and the table"""
        self.scene.remove_world_object() 
        rospy.sleep(0.1)

    def add_table_to_scene(self, math_data, T_global):
        """Adds the detected table plane to MoveIt"""
        table_z = math_data["table_z_flat"]
        R_cam_to_flat = math_data["R"]
        R_flat_to_cam = R_cam_to_flat.T
        p_flat = np.array([0, 0, table_z]) 
        
        # Rotate to Camera Frame
        p_cam = R_flat_to_cam @ p_flat
        
        # Transform to Robot Frame
        p_cam_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])
        p_robot = (T_global @ p_cam_h)[:3]
        
        # Orientation
        R_robot_table = T_global[:3, :3] @ R_flat_to_cam
        quat_robot = R_scipy.from_matrix(R_robot_table).as_quat()
        
        # Add to Scene
        p = PoseStamped()
        p.header.frame_id = ROBOT_FRAME_ID
        p.header.stamp = rospy.Time.now()
        p.pose.position.x, p.pose.position.y, p.pose.position.z = p_robot
        p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w = quat_robot
        p.pose.position.z -= 0.03
        
        self.scene.add_box("detected_table", p, size=(1.5, 1.5, 0.02))

    def update_cube_collision(self, results):
        """Updates just the cubes"""
        # Remove old cubes
        old = self.scene.get_known_object_names()
        to_remove = [n for n in old if n.startswith("cube_")]
        if to_remove:
            for name in to_remove:
                self.scene.remove_world_object(name)

        # Add new cubes
        for i, (pos, quat, angle) in enumerate(results):
            p = PoseStamped()
            p.header.frame_id = ROBOT_FRAME_ID
            p.header.stamp = rospy.Time.now()
            p.pose.position.x, p.pose.position.y, p.pose.position.z = pos
            p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w = quat
            self.scene.add_box(f"cube_{i}", p, size=(0.045, 0.045, 0.045))

    def capture_burst(self):
        self.points_buffer = []
        self.colors_buffer = []
        self.frame_count = 0
        self.capture_done = False
        self.subscriber = rospy.Subscriber(TOPIC_CLOUD, PointCloud2, self.capture_callback)
        rate = rospy.Rate(10)
        start_time = rospy.Time.now()
        while not self.capture_done and not rospy.is_shutdown():
            if (rospy.Time.now() - start_time).to_sec() > 5.0:
                self.subscriber.unregister()
                return False
            rate.sleep()
        return len(self.points_buffer) > 0

    def capture_callback(self, ros_cloud):
        if self.frame_count >= TARGET_FRAMES: return
        self.frame_count += 1
        gen = pc2.read_points(ros_cloud, field_names=("x", "y", "z", "rgb"), skip_nans=True)
        for p in gen:
            x, y, z, float_rgb = p
            self.points_buffer.append([x, y, z])
            packed = struct.pack('f', float_rgb)
            b, g, r, a = struct.unpack('BBBB', packed)
            self.colors_buffer.append([r, g, b])
        if self.frame_count >= TARGET_FRAMES:
            self.subscriber.unregister()
            self.capture_done = True

    def build_open3d_cloud(self):
        xyz = np.array(self.points_buffer)
        colors = np.array(self.colors_buffer) / 255.0
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd.remove_non_finite_points()

    def run_pipeline(self, pcd):
        try:
            # 1. PREPROCESS
            self.preprocessor.process_live(pcd)
            math_data = self.preprocessor.transform_data
            raw_main, raw_table = self.preprocessor.get_result_images()
            if raw_main is None: return []

            # 2. TF LOOKUP (Needed for Table & Cubes)
            try:
                trans = self.tf_buffer.lookup_transform(ROBOT_FRAME_ID, CAMERA_FRAME_ID, rospy.Time(0), rospy.Duration(1.0))
                T_global = self.msg_to_numpy(trans.transform)
                
                # --- ADD TABLE TO COLLISION SCENE ---
                self.add_table_to_scene(math_data, T_global)
                
            except Exception as ex:
                rospy.logerr(f"TF Failed: {ex}")
                return []

            # 3. CLEAN
            self.cleaner.set_images(raw_main, raw_table)
            self.cleaner.process_segmentation()
            clean_img = self.cleaner.get_clean_image()
            if clean_img is None: return []

            # 4. DETECT
            self.detector.set_image(clean_img)
            pixel_poses = self.detector.detect_pose()
            if not pixel_poses: return []
            
            # 5. CAMERA FRAME
            camera_results = self.detector.calculate_camera_coordinates(pixel_poses, math_data)

            # 6. TRANSFORM CUBES
            final_results = []
            for (pos_cam, quat_cam, angle) in camera_results:
                p_robot = (T_global @ np.array([pos_cam[0], pos_cam[1], pos_cam[2], 1.0]))[:3]
                R_cube_cam = R_scipy.from_quat(quat_cam).as_matrix()
                R_robot_cam = T_global[:3, :3]
                quat_robot = R_scipy.from_matrix(R_robot_cam @ R_cube_cam).as_quat()
                final_results.append((p_robot, quat_robot, angle))
                
            return final_results
            
        except Exception as e:
            rospy.logerr(f"Pipeline Error: {e}")
            return []

    def msg_to_numpy(self, transform):
        t = [transform.translation.x, transform.translation.y, transform.translation.z]
        q = [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]
        T = np.eye(4)
        T[:3, :3] = R_scipy.from_quat(q).as_matrix()
        T[:3, 3] = t
        return T

    def publish_markers(self, results):
        marker_array = MarkerArray()
        del_m = Marker(); del_m.action = Marker.DELETEALL; marker_array.markers.append(del_m)
        for i, (pos, quat, angle) in enumerate(results):
            m = Marker()
            m.header.frame_id = ROBOT_FRAME_ID
            m.header.stamp = rospy.Time.now()
            m.ns = "cubes"; m.id = i; m.type = Marker.CUBE; m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = pos
            m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = quat
            m.scale.x = m.scale.y = m.scale.z = 0.045
            m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.0; m.color.a = 0.8
            marker_array.markers.append(m)
        self.marker_pub.publish(marker_array)

if __name__ == '__main__':
    try:
        PerceptionServiceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass