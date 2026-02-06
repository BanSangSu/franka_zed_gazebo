#!/usr/bin/env python3

import rospy
import message_filters
import sys
import os
import numpy as np
import cv2
import tf2_ros
import tf2_geometry_msgs
import moveit_commander
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, Pose, PoseStamped, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image as RosImage, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from sklearn.decomposition import PCA
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# --- Runtime Path Adjustment ---
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from detect_n_segment import MultiDetectorSAM 

class Cube:
    def __init__(self, id, position, orientation, confidence, dimensions):
        self.id = id
        self.position = position
        self.orientation = orientation
        self.confidence = confidence
        self.dimensions = dimensions

class SamCubeDetector:
    def __init__(self):
        rospy.init_node('sam_cube_detector', anonymous=True)
        
        # 1. Initialize MoveIt Planning Scene (From Old Code)
        moveit_commander.roscpp_initialize(sys.argv)
        self.scene = moveit_commander.PlanningSceneInterface()
        
        # 2. Parameters
        self.detector_type = rospy.get_param('~detector_type', 'florence2')
        self.sam_checkpoint = rospy.get_param('~sam_checkpoint', 'sam_vit_b_01ec64.pth')
        self.sam_model_type = rospy.get_param('~sam_model_type', 'vit_b')
        
        self.prompt = rospy.get_param('~prompt', 'small cube')
        self.world_frame = rospy.get_param('~world_frame', 'world')
        
        self.known_cube_size = 0.045 # Standard size from old code
        
        # Table ROI Bounds (From Old Code)
        self.table_x_min, self.table_x_max = 0.3, 1.0
        self.table_y_min, self.table_y_max = -0.4, 0.4

        # # Target cube point cloud for visualization and matching (a simple cube centered at origin)        
        # box_mesh = o3d.geometry.TriangleMesh.create_box(
        #     width=self.known_cube_size, 
        #     height=self.known_cube_size, 
        #     depth=self.known_cube_size
        # )
        # box_mesh.translate([-self.known_cube_size/2, -self.known_cube_size/2, -self.known_cube_size/2])
        # self.synthetic_source_pcd = box_mesh.sample_points_uniformly(number_of_points=1000)
        
        # Camera Intrinsics and Bridge
        self.bridge = CvBridge()
        self.camera_K = None
        self.camera_frame_id = None
        
        # 3. Initialize AI Pipeline
        detector_config = {"model_name": "microsoft/Florence-2-base"}
        self.pipeline = MultiDetectorSAM(
            detector_type=self.detector_type,
            detector_config=detector_config,
            sam_checkpoint=self.sam_checkpoint,
            sam_model_type=self.sam_model_type
        )
        
        # # ROS Communication
        # image_topic = rospy.get_param('~image_topic', "/zed2/zed_node/rgb/image_rect_color")
        # camera_info_topic = rospy.get_param('~camera_info_topic', "/zed2/zed_node/rgb/camera_info")
        # depth_topic = rospy.get_param('~depth_topic', "/zed2/zed_node/depth/depth_registered")
        # ZED2 topics
        image_topic = rospy.get_param('~image_topic', "static_zed2_camera/static_zed2/zed_node/left/image_rect_color")
        camera_info_topic = rospy.get_param('~camera_info_topic', "static_zed2_camera/static_zed2/zed_node/left/camera_info")
        depth_topic = rospy.get_param('~depth_topic', "static_zed2_camera/static_zed2/zed_node/depth/depth_registered")
        point_cloud_topic = rospy.get_param('~point_cloud_topic', "static_zed2_camera/static_zed2/zed_node/point_cloud/cloud_registered")
        
        # Subscribers
        image_sub = message_filters.Subscriber(image_topic, RosImage)
        camera_info_sub = message_filters.Subscriber(camera_info_topic, CameraInfo)
        depth_sub = message_filters.Subscriber(depth_topic, RosImage)
        point_cloud_sub = message_filters.Subscriber(point_cloud_topic, PointCloud2)

        ts = message_filters.ApproximateTimeSynchronizer([image_sub, camera_info_sub, depth_sub, point_cloud_sub], 10, 0.1)
        ts.registerCallback(self.detection_callback)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Publishers
        self.centroid_point_pub = rospy.Publisher('/sam_cube_detection/cubes_centroid_point', PointCloud2, queue_size=10)
        self.mapped_point_cloud = rospy.Publisher('/sam_cube_detection/cubes_mapped_point_cloud', PointCloud2, queue_size=10)
        self.seg_image_pub = rospy.Publisher('/sam_cube_detection/segmentation_image', RosImage, queue_size=10)
        
        rospy.loginfo(f"SAM+Open3D Hybrid Node Started. Frame: {self.world_frame}")

    def detection_callback(self, image_msg, camera_info_msg, depth_msg, point_cloud_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            
            if self.camera_K is None:
                self.camera_K = np.array(camera_info_msg.K).reshape(3, 3)
                self.camera_frame_id = camera_info_msg.header.frame_id
            
            # AI Inference
            result = self.pipeline.detect_and_segment(cv_image, prompt=self.prompt, conf=0.25)

            if result is None or 'detection' not in result or len(result['detection']['scores']) == 0:
                rospy.loginfo("No cubes detected.")
                return
            
            # rospy.loginfo(f"Detections: {len(result['detection']['scores'])} cubes found.")
            self.visualize_and_publish(result, cv_image, image_msg.header)
            self.process_detections(result, depth_image, point_cloud_msg, image_msg.header)
            
        except Exception as e:
            rospy.logerr(f"Callback error: {e}")

    def process_detections(self, result, depth_image, point_cloud_msg, header):
        masks = result['segmentation']['masks']
        scores = result['detection']['scores']
        
        # Clean up scene
        for obj_name in self.scene.get_known_object_names():
            if obj_name.startswith("cube_"):
                self.scene.remove_world_object(obj_name)
        
        # Get the transform from camera to world once per callback
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame, 
                self.camera_frame_id, 
                header.stamp, 
                rospy.Duration(0.1)
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logerr("TF Lookup failed")
            return
        
        detected_cubes = []
        all_points_world = []
        for idx, mask in enumerate(masks):
            # Get pixel indices where mask is True
            # np.where returns (rows, cols), but read_points needs (u, v) which is (cols, rows)
            rows, cols = np.where(mask > 0)
            uvs = np.stack([cols, rows], axis=1).tolist()
            
            # DIRECT MAPPING: Pull XYZ from the point cloud for these pixels
            # Field names 'x', 'y', 'z' are standard for ZED cloud_registered
            point_gen = pc2.read_points(point_cloud_msg, field_names=("x", "y", "z"), skip_nans=True, uvs=uvs)
            points_cam = np.array(list(point_gen))

            # Check if we have enough valid 3D points
            if len(points_cam) < 50:
                continue
                
            # Transform points to World Frame
            points_world = self.transform_to_world(points_cam, transform)
            w_centroid = np.mean(points_world, axis=0)

            # ROI Filter
            if not (self.table_x_min < w_centroid[0] < self.table_x_max and 
                    self.table_y_min < w_centroid[1] < self.table_y_max):
                continue
                
            # ROUGH ESTIMATE (PCA)
            pca_points = points_world - w_centroid
            pca = PCA(n_components=2).fit(pca_points[:, :2])
            angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])

            # Build Initial Guess Matrix for ICP
            initial_guess = np.eye(4)
            rot_mat = R.from_euler('z', angle).as_matrix()
            initial_guess[0:3, 0:3] = rot_mat
            initial_guess[0:3, 3] = [w_centroid[0], w_centroid[1], self.known_cube_size / 2.0]

            # FINE REFINEMENT (Open3D ICP)
            # Convert raw points to Open3D format and clean noise
            # 1. Physically re-allocate memory to a new buffer
            # Using .astype() forces a new allocation even if the input is already float64

            # observed_target_pcd = o3d.geometry.PointCloud()
            # observed_target_pcd.points = o3d.utility.Vector3dVector(points_world)
            # observed_target_pcd, _ = observed_target_pcd.remove_statistical_outlier(20, 2.0)

            # # Run Point-to-Point ICP
            # reg_p2p = o3d.pipelines.registration.registration_icp(
            #     self.synthetic_source_pcd, observed_target_pcd, 0.02, initial_guess,
            #     o3d.pipelines.registration.TransformationEstimationPointToPoint()
            # )

            # Extract Refined Pose
            # final_transform = reg_p2p.transformation

            
            final_transform = initial_guess
            refined_pos = final_transform[0:3, 3]
            refined_q = R.from_matrix(final_transform[0:3, 0:3]).as_quat()

            # Fix Z height to table surface
            refined_pos[2] = self.known_cube_size / 2.0

            # Update Scene
            self.update_planning_scene(idx, refined_pos, refined_q)
            detected_cubes.append(Cube(idx, refined_pos, refined_q, scores[idx], [self.known_cube_size]*3))

            all_points_world.append(points_world)

        self.publish_centroid_point(detected_cubes, header.stamp)
        self.publish_mapped_point_cloud(all_points_world, stamp=header.stamp)

    def transform_to_world(self, points, transform):
        """
        Vectorized transformation: much faster and more accurate for point clouds.
        """
        from tf.transformations import quaternion_matrix
        
        # 1. Build a 4x4 Transformation Matrix
        t = transform.transform.translation
        q = transform.transform.rotation
        matrix = quaternion_matrix([q.x, q.y, q.z, q.w])
        matrix[0:3, 3] = [t.x, t.y, t.z]

        # 2. Convert points to homogeneous coordinates (Nx4)
        num_pts = points.shape[0]
        points_homo = np.hstack([points, np.ones((num_pts, 1))])

        # 3. Multiply matrix by points (Transposed)
        transformed_pts_homo = np.dot(matrix, points_homo.T).T
        
        return transformed_pts_homo[:, :3]

    def update_planning_scene(self, idx, pos, q):
        ps = PoseStamped()
        ps.header.frame_id = self.world_frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = pos
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = q
        self.scene.add_box(f"cube_{idx}", ps, [self.known_cube_size - 0.002]*3)

    def publish_centroid_point(self, cubes, stamp):
        if not cubes: return
        pts = [[c.position[0], c.position[1], c.position[2], c.confidence] for c in cubes]
        fields = [PointField('x',0,7,1), PointField('y',4,7,1), PointField('z',8,7,1), PointField('intensity',12,7,1)]
        msg = pc2.create_cloud(Header(stamp=stamp, frame_id=self.world_frame), fields, pts)
        self.centroid_point_pub.publish(msg)

    def publish_mapped_point_cloud(self, all_points, stamp):
        """
        Publishes all 3D points from the masks, not just the centroids.
        all_points: A numpy array of shape (N, 3)
        """
        all_points = np.vstack(all_points) if all_points else np.empty((0, 3))
        if len(all_points) == 0: return
        
        # Create fields for X, Y, Z
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]
        
        header = Header(stamp=stamp, frame_id=self.world_frame)
        msg = pc2.create_cloud(header, fields, all_points)
        self.mapped_point_cloud.publish(msg)

    def visualize_and_publish(self, result, cv_image, header):
        vis = cv_image.copy()
        for mask in result['segmentation']['masks']:
            vis[mask > 0] = vis[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
        msg = self.bridge.cv2_to_imgmsg(vis, "bgr8")
        msg.header = header
        self.seg_image_pub.publish(msg)

if __name__ == '__main__':
    try:
        detector = SamCubeDetector()
        rospy.spin()
    except rospy.ROSInterruptException: pass