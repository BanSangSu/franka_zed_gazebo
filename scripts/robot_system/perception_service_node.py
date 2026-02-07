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
from geometry_msgs.msg import PointStamped, Pose, PoseArray, PoseStamped, Vector3, Point
from nav_msgs.msg import Odometry

import colorsys
    
from visualization_msgs.msg import Marker, MarkerArray

from sensor_msgs.msg import Image as RosImage, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R

# --- Runtime Path Adjustment ---
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir) 

# # Path for detect_n_segment
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Add path contact_graspnet
# path: parent/contact_graspnet_pytorch/contact_graspnet_pytorch/
grasp_node_path = os.path.join(parent_dir, 'contact_graspnet_pytorch')
if grasp_node_path not in sys.path:
    sys.path.append(grasp_node_path)

from detect_n_segment import MultiDetectorSAM 
# Import the GraspNode class directly if running in same process, 
# or use ServiceProxy if running separately.
# For this file, I will assume we import the logic or have a helper function.
try:
    from grasp_generation_service_node import GraspServiceNode
    # from contact_graspnet_pytorch import GraspServiceNode
    HAS_GRASP_NET = True
    print("GraspServiceNode imported successfully. Grasping enabled.")
except ImportError as e:
    print(f"Import failed: {e}")
    HAS_GRASP_NET = False
    print("Warning: GraspServiceNode not found. Grasping will be disabled.")

class Cube:
    def __init__(self, id, position, orientation, confidence, dimensions, grasp_pose=None, grasp_score=0.0):
        self.id = id
        self.position = position        # [x, y, z] in World
        self.orientation = orientation  # [x, y, z, w] in World
        self.confidence = confidence
        self.dimensions = dimensions
        self.grasp_pose = grasp_pose    # 4x4 Matrix in World Frame
        self.grasp_score = grasp_score

class SamCubeDetector:
    def __init__(self):
        rospy.init_node('sam_cube_detector', anonymous=True)
        
        # 1. Initialize MoveIt
        moveit_commander.roscpp_initialize(sys.argv)
        self.scene = moveit_commander.PlanningSceneInterface()
        
        # 2. Parameters
        self.detector_type = rospy.get_param('~detector_type', 'florence2')
        self.sam_checkpoint = rospy.get_param('~sam_checkpoint', 'sam_vit_b_01ec64.pth')
        self.sam_model_type = rospy.get_param('~sam_model_type', 'vit_b')
        self.prompt = rospy.get_param('~prompt', 'small cube')
        self.world_frame = rospy.get_param('~world_frame', 'world')
        self.known_cube_size = 0.045 
        
        self.table_x_min, self.table_x_max = 0.3, 1.0
        self.table_y_min, self.table_y_max = -0.4, 0.4

        self.bridge = CvBridge()
        self.camera_K = None
        self.camera_frame_id = None
        
        # 3. Initialize Detectors
        detector_config = {"model_name": "microsoft/Florence-2-base"}
        self.pipeline = MultiDetectorSAM(
            detector_type=self.detector_type,
            detector_config=detector_config,
            sam_checkpoint=self.sam_checkpoint,
            sam_model_type=self.sam_model_type
        )
        
        # Initialize Grasp Net (If running in same process for simplicity)
        # In a real heavy setup, use rospy.ServiceProxy('detect_grasps', ...)
        if HAS_GRASP_NET:
            self.grasp_detector = GraspServiceNode(init_node=False)
            # We override the node init inside GraspServiceNode to avoid double init
            # or simply use the helper methods.
        
        # 4. ROS Topics
        image_topic = rospy.get_param('~image_topic', "static_zed2_camera/static_zed2/zed_node/left/image_rect_color")
        camera_info_topic = rospy.get_param('~camera_info_topic', "static_zed2_camera/static_zed2/zed_node/left/camera_info")
        depth_topic = rospy.get_param('~depth_topic', "static_zed2_camera/static_zed2/zed_node/depth/depth_registered")
        point_cloud_topic = rospy.get_param('~point_cloud_topic', "static_zed2_camera/static_zed2/zed_node/point_cloud/cloud_registered")
        
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
        self.grasp_vis_pub = rospy.Publisher('/sam_cube_detection/grasp_poses', MarkerArray, queue_size=10)
        
        rospy.loginfo(f"SAM+GraspNode Started. Frame: {self.world_frame}")

    def detection_callback(self, image_msg, camera_info_msg, depth_msg, point_cloud_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            
            if self.camera_K is None:
                self.camera_K = np.array(camera_info_msg.K).reshape(3, 3)
                self.camera_frame_id = camera_info_msg.header.frame_id
            
            # 1. SAM Inference
            result = self.pipeline.detect_and_segment(cv_image, prompt=self.prompt, conf=0.25)

            if result is None or 'detection' not in result or len(result['detection']['scores']) == 0:
                return
            
            # 2. Process Cubes and Grasps
            self.visualize_and_publish(result, cv_image, image_msg.header)
            
            # Prepare Segmap for GraspNet (Combine masks)
            # ContactGraspnet uses a segmap where 0 is background, 1 is obj1, 2 is obj2...
            segmap = np.zeros(cv_image.shape[:2], dtype=np.uint8)
            masks = result['segmentation']['masks']
            
            # To map GraspNet ID back to our loop index
            seg_id_to_index = {} 
            
            valid_mask_count = 0
            for i, mask in enumerate(masks):
                # Simple check to avoid noise
                if np.sum(mask) > 50: 
                    seg_id = valid_mask_count + 1
                    segmap[mask > 0] = seg_id
                    seg_id_to_index[seg_id] = i
                    valid_mask_count += 1

            # 3. Call GraspNet (Simulating Service Call)
            detected_grasps_cam = {} # Dict {seg_id: {pose: 4x4, score: float}}
            if HAS_GRASP_NET and valid_mask_count > 0:
                # In real ROS Service: 
                # req = DetectGraspsRequest(image_msg, depth_msg, bridge.cv2_to_imgmsg(segmap_combined), camera_info_msg)
                # resp = service_proxy(req)
                
                # Direct call for now:
                grasp_results = self.grasp_detector.process_ros_data(image_msg, depth_msg, segmap, point_cloud_msg, camera_info_msg)
                
                # Re-organize results by ID
                for g in grasp_results:
                    detected_grasps_cam[g['id']] = {'pose': g.get('pose', None), 'score': g.get('score', None), 'contact_pts': g.get('contact_pts', None)}

            # 4. Process Geometry + Merge Grasp Info
            self.process_detections_and_grasps(
                result, depth_image, point_cloud_msg, image_msg.header, 
                detected_grasps_cam, seg_id_to_index
            )
            
        except Exception as e:
            rospy.logerr(f"Callback error: {e}")
            import traceback
            traceback.print_exc()

    def process_detections_and_grasps(self, result, depth_image, point_cloud_msg, header, grasp_data, id_map):
        masks = result['segmentation']['masks']
        scores = result['detection']['scores']
        
        # TF Lookup
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame, self.camera_frame_id, header.stamp, rospy.Duration(0.1))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            return

        detected_cubes = []
        all_points_world = []

        grasp_markers = MarkerArray()
        
        # Iterate over unique IDs we sent to GraspNet
        for seg_id, index in id_map.items():
            mask = masks[index]
            
            # --- Geometric Position Logic (Existing) ---
            rows, cols = np.where(mask > 0)
            uvs = np.stack([cols, rows], axis=1).tolist()
            point_gen = pc2.read_points(point_cloud_msg, field_names=("x", "y", "z"), skip_nans=True, uvs=uvs)
            points_cam = np.array(list(point_gen))
            
            if len(points_cam) < 50: continue

            points_world = self.transform_points_to_world(points_cam, transform)
            w_centroid = np.mean(points_world, axis=0)
            
            if not (self.table_x_min < w_centroid[0] < self.table_x_max and 
                    self.table_y_min < w_centroid[1] < self.table_y_max):
                continue

            # Orientation via PCA
            pca_points = points_world - w_centroid
            pca = PCA(n_components=2).fit(pca_points[:, :2])
            angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
            
            cube_pos = [w_centroid[0], w_centroid[1], self.known_cube_size / 2.0]
            cube_quat = R.from_euler('z', angle).as_quat()

            # --- Grasp Processing ---
            final_grasp_world = None
            grasp_score = 0.0
            
            if seg_id in grasp_data:
                # Grasp is in Camera Frame (4x4 Matrix)
                g_pose_cam = grasp_data[seg_id]['pose']
                grasp_score = grasp_data[seg_id]['score']
                
                # Transform Grasp Matrix: Cam -> World
                # T_world_grasp = T_world_cam * T_cam_grasp

                g_pose_cam = np.array(g_pose_cam)
                
                # Get T_world_cam matrix
                t = transform.transform.translation
                q = transform.transform.rotation
                T_world_cam = self.quaternion_matrix([q.x, q.y, q.z, q.w])
                T_world_cam[0:3, 3] = [t.x, t.y, t.z]
                
                T_world_grasp = np.dot(T_world_cam, g_pose_cam)
                final_grasp_world = T_world_grasp

                delete_all = Marker()
                delete_all.action = Marker.DELETEALL
                grasp_markers.markers.append(delete_all)

                # Add to visualization array as Grasp points
                grasp_marker = self.create_gripper_marker(T_world_grasp, seg_id, header)
                grasp_markers.markers.append(grasp_marker)
                
                m_text = self.create_text_marker(T_world_grasp, seg_id, grasp_score, header)
                grasp_markers.markers.append(m_text)

            # Store Cube Info
            cube = Cube(index, cube_pos, cube_quat, scores[index], [self.known_cube_size]*3, 
                       grasp_pose=final_grasp_world, grasp_score=grasp_score)
            
            detected_cubes.append(cube)
            self.update_planning_scene(index, cube_pos, cube_quat)
            all_points_world.append(points_world)

        self.publish_centroid_point(detected_cubes, header.stamp)
        self.publish_mapped_point_cloud(all_points_world, header.stamp)
        self.grasp_vis_pub.publish(grasp_markers)

    def transform_points_to_world(self, points, transform):
        t = transform.transform.translation
        q = transform.transform.rotation
        matrix = self.quaternion_matrix([q.x, q.y, q.z, q.w])
        matrix[0:3, 3] = [t.x, t.y, t.z]
        points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
        return np.dot(matrix, points_homo.T).T[:, :3]

    def quaternion_matrix(self, quaternion):
        # Helper to avoid importing tf.transformations which might be deprecated in py3
        mat = np.eye(4)
    
        if len(quaternion) == 4:
            # 2. Get the 3x3 rotation matrix from Scipy
            rotation_3x3 = R.from_quat(quaternion).as_matrix()
            
            # 3. Insert the 3x3 rotation into the top-left of the 4x4 matrix
            mat[0:3, 0:3] = rotation_3x3
            
        return mat

    # ... (Keep existing visualization and helper methods: update_planning_scene, visualize_and_publish, etc.) ...
    def update_planning_scene(self, idx, pos, q):
        ps = PoseStamped()
        ps.header.frame_id = self.world_frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = pos
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = q
        self.scene.add_box(f"cube_{idx}", ps, [self.known_cube_size - 0.002]*3)

    def visualize_and_publish(self, result, cv_image, header):
        vis = cv_image.copy()
        for mask in result['segmentation']['masks']:
            vis[mask > 0] = vis[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
        msg = self.bridge.cv2_to_imgmsg(vis, "bgr8")
        msg.header = header
        self.seg_image_pub.publish(msg)

    def publish_centroid_point(self, cubes, stamp):
        if not cubes: return
        pts = [[c.position[0], c.position[1], c.position[2], c.confidence] for c in cubes]
        fields = [PointField('x',0,7,1), PointField('y',4,7,1), PointField('z',8,7,1), PointField('intensity',12,7,1)]
        msg = pc2.create_cloud(Header(stamp=stamp, frame_id=self.world_frame), fields, pts)
        self.centroid_point_pub.publish(msg)

    def publish_mapped_point_cloud(self, all_points, stamp):
        all_points = np.vstack(all_points) if all_points else np.empty((0, 3))
        if len(all_points) == 0: return
        fields = [PointField('x', 0, PointField.FLOAT32, 1), PointField('y', 4, PointField.FLOAT32, 1), PointField('z', 8, PointField.FLOAT32, 1)]
        msg = pc2.create_cloud(Header(stamp=stamp, frame_id=self.world_frame), fields, all_points)
        self.mapped_point_cloud.publish(msg)

    def create_gripper_marker(self, T_world_grasp, grasp_id, header):
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = header.stamp
        marker.ns = "grasps"
        marker.id = grasp_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        
        marker.scale.x = 0.003 # 선 두께
        
        hue = (grasp_id * 0.137) % 1.0  # 고정된 간격으로 색상 변경
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        
        marker.color.r, marker.color.g, marker.color.b = rgb
        marker.color.a = 1.0 # 불투명도

        # 그리퍼 기하학 정의 (GraspNet 표준 축 기준)
        w = 0.04  # 절반 너비
        d = 0.06  # 손가락 길이
        
        # 아래 좌표는 Z축이 정면, X축이 좌우인 일반적인 GraspNet 기준입니다.
        # 만약 방향이 이상하면 이 리스트의 [x, y, z] 순서를 바꿔보세요.
        local_pts = [
            [-w, 0, 0], [w, 0, 0],   # 바닥 가로바 (X축 방향)
            [-w, 0, 0], [-w, 0, d],  # 왼쪽 손가락 (Z축 방향 전진)
            [w, 0, 0],  [w, 0, d]    # 오른쪽 손가락 (Z축 방향 전진)
        ]

        for pt in local_pts:
            p_homo = np.array([pt[0], pt[1], pt[2], 1.0])
            p_world = np.dot(T_world_grasp, p_homo)
            
            ros_pt = Point()
            ros_pt.x, ros_pt.y, ros_pt.z = p_world[0:3]
            marker.points.append(ros_pt)
            
        return marker
    
    def create_text_marker(self, T_world_grasp, grasp_id, score, header):
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = header.stamp
        marker.ns = "grasp_labels"
        marker.id = grasp_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # 그리퍼 위치보다 약간 위(Z축 방향)에 텍스트 배치
        marker.pose.position.x = T_world_grasp[0, 3]
        marker.pose.position.y = T_world_grasp[1, 3]
        marker.pose.position.z = T_world_grasp[2, 3] + 0.04  # 4cm 위
        
        marker.scale.z = 0.025  # 글자 크기
        hue = (grasp_id * 0.137) % 1.0  # 고정된 간격으로 색상 변경
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        
        marker.color.r, marker.color.g, marker.color.b = rgb
        marker.color.a = 1.0 # 불투명도
        
        # 표시할 텍스트 설정 (ID와 점수)
        marker.text = f"ID: {grasp_id} ({score:.2f})"
    
        return marker
if __name__ == '__main__':
    try:
        detector = SamCubeDetector()
        rospy.spin()
    except rospy.ROSInterruptException: pass