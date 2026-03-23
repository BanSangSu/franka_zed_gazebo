#!/usr/bin/env python3

import rospy
import message_filters
import sys
import os
import numpy as np
import cv2
import tf2_ros
import threading
import moveit_commander
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, Pose, PoseStamped, Point, PoseArray, Vector3
from nav_msgs.msg import Odometry
import colorsys
    
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image as RosImage, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2

from sklearn.decomposition import PCA
import open3d as o3d
import copy

from scipy.spatial.transform import Rotation as R

from franka_zed_gazebo.srv import PerceptionService, PerceptionServiceResponse

# --- Runtime Path Adjustment ---
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir) 

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

grasp_node_path = os.path.join(parent_dir, 'contact_graspnet_pytorch')
if grasp_node_path not in sys.path:
    sys.path.append(grasp_node_path)

from detect_n_segment import MultiDetectorSAM 

try:
    from grasp_generation_service_node import GraspServiceNode
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
        self.clear_planning_scene()

        # 2. Parameters
        self.detector_type = rospy.get_param('~detector_type', 'florence2')
        self.sam_checkpoint = rospy.get_param('~sam_checkpoint', 'sam_vit_b_01ec64.pth')
        self.sam_model_type = rospy.get_param('~sam_model_type', 'vit_b')
        # self.prompt = rospy.get_param('~prompt', 'carrot')
        self.prompt = rospy.get_param('~prompt', 'small cube')
        # self.world_frame = rospy.get_param('~world_frame', 'panda_link0')
        self.world_frame = rospy.get_param('~world_frame', 'world')
        self.po_camera_frame = rospy.get_param('~po_camera_frame', 'zedr_base_link') # for poesidon robot
        self.known_cube_size = rospy.get_param('~known_cube_size', 0.045)  # 4.5cm
        
        self.table_x_min, self.table_x_max = 0.3, 1.0
        self.table_y_min, self.table_y_max = -0.4, 0.4
        
        self.create_table()

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
        
        if HAS_GRASP_NET:
            self.grasp_detector = GraspServiceNode(init_node=False)

        # # real zed
        self.topic_image = rospy.get_param('~image_topic', "zedr/zed_node/left/image_rect_color")
        self.topic_info = rospy.get_param('~camera_info_topic', "zedr/zed_node/left/camera_info")
        self.topic_depth = rospy.get_param('~depth_topic', "zedr/zed_node/depth/depth_registered")
        self.topic_cloud = rospy.get_param('~point_cloud_topic', "zedr/zed_node/point_cloud/cloud_registered")
        
        # 4. ROS Topics Configuration (Strings only, subscriber creation happens on demand)
        # self.topic_image = rospy.get_param('~image_topic', "static_zed2_camera/static_zed2/zed_node/left/image_rect_color")
        # self.topic_info = rospy.get_param('~camera_info_topic', "static_zed2_camera/static_zed2/zed_node/left/camera_info")
        # self.topic_depth = rospy.get_param('~depth_topic', "static_zed2_camera/static_zed2/zed_node/depth/depth_registered")
        # self.topic_cloud = rospy.get_param('~point_cloud_topic', "static_zed2_camera/static_zed2/zed_node/point_cloud/cloud_registered")
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Publishers
        self.centroid_point_pub = rospy.Publisher('/sam_cube_detection/cubes_centroid_point', PointCloud2, queue_size=10)
        self.mapped_point_cloud = rospy.Publisher('/sam_cube_detection/cubes_mapped_point_cloud', PointCloud2, queue_size=10)
        self.seg_image_pub = rospy.Publisher('/sam_cube_detection/segmentation_image', RosImage, queue_size=10)
        self.grasp_vis_pub = rospy.Publisher('/sam_cube_detection/grasp_poses', MarkerArray, queue_size=10)
        self.cube_id_vis_pub = rospy.Publisher('/sam_cube_detection/cube_id_markers', MarkerArray, queue_size=10)

        # Service
        self.service = rospy.Service('/perception_service', PerceptionService, self.service_callback)
        
        rospy.loginfo(f"SAM+GraspNode Service Ready. Frame: {self.world_frame}")

    def clear_planning_scene(self):
        """Remove ALL collision objects (table + cubes) from MoveIt scene"""
        self.scene.clear()
        rospy.sleep(1.0)
        rospy.loginfo("✓ Cleared ALL objects from planning scene")


    def create_table(self):
        table_pose = PoseStamped()
        table_pose.header.frame_id = self.world_frame
        table_pose.header.stamp = rospy.Time.now()
        table_pose.pose.position.x = 0.5  # Table center X (matches your table bounds 0.3-1.0)
        table_pose.pose.position.y = 0.0   # Table center Y (matches -0.4 to 0.4)
        table_pose.pose.position.z = -0.25 # -0.3 # Half table height (5cm total height)
        table_pose.pose.orientation.w = 1.0
        
        # Table size: width=0.7m (X), length=0.8m (Y), height=0.05m (Z)
        self.scene.add_box("table", table_pose, size=(0.8, 1.5, 0.5))
        rospy.sleep(2.0)  # Wait for scene update
        rospy.loginfo("✓ Table added to MoveIt planning scene (0.7x0.8x0.05m at [0.65, 0.0, 0.025])")

    def acquire_frames(self, timeout=5.0):
        """
        Temporarily subscribes to topics, waits for ONE synchronized set of messages, 
        and then unregisters. Returns None if timed out.
        """
        captured_data = {}
        capture_event = threading.Event()

        def sync_cb(img, info, depth, cloud):
            captured_data['img'] = img
            captured_data['info'] = info
            captured_data['depth'] = depth
            captured_data['cloud'] = cloud
            capture_event.set()

        # Create temporary subscribers
        sub_img = message_filters.Subscriber(self.topic_image, RosImage)
        sub_info = message_filters.Subscriber(self.topic_info, CameraInfo)
        sub_depth = message_filters.Subscriber(self.topic_depth, RosImage)
        sub_cloud = message_filters.Subscriber(self.topic_cloud, PointCloud2)

        ts = message_filters.ApproximateTimeSynchronizer([sub_img, sub_info, sub_depth, sub_cloud], 10, 0.1)
        ts.registerCallback(sync_cb)

        # Wait for data
        got_data = capture_event.wait(timeout)

        # Cleanup immediately
        sub_img.unregister()
        sub_info.unregister()
        sub_depth.unregister()
        sub_cloud.unregister()
        del ts

        if got_data:
            return captured_data['img'], captured_data['info'], captured_data['depth'], captured_data['cloud']
        else:
            return None

    def service_callback(self, req):
        """
        The main Service Handler.
        """
        rospy.loginfo("Perception Service Called. Acquiring frames...")
        
        # 1. Get Data (One-Shot)
        data = self.acquire_frames(timeout=5.0)
        
        if data is None:
            rospy.logerr("Failed to acquire synchronized frames within timeout.")
            # Depending on your .srv definition, return success=False
            return PerceptionServiceResponse() # success=False implied or check your srv fields
        
        image_msg, camera_info_msg, depth_msg, point_cloud_msg = data
        
        # 2. Process
        try:
            detected_cubes = self.process_pipeline(image_msg, camera_info_msg, depth_msg, point_cloud_msg)
            
            # 3. Construct Response
            response = PerceptionServiceResponse()
            response.success = True
            response.num_cubes = len(detected_cubes)
            response.message = f"Found {len(detected_cubes)} cubes"
            
            response.cube_poses = PoseArray()
            response.cube_poses.header.frame_id = self.world_frame
            response.cube_poses.header.stamp = rospy.Time.now()
            
            for cube in detected_cubes:
                p = Pose()
                p.position.x, p.position.y, p.position.z = cube.position
                p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = cube.orientation
                response.cube_poses.poses.append(p)
                
                response.confidences.append(cube.confidence)
                
                dim = Vector3()
                dim.x, dim.y, dim.z = cube.dimensions
                response.dimensions.append(dim)
                
                response.labels.append("cube")
            
            rospy.loginfo(f"Service completed. Found {len(detected_cubes)} objects.")
            return response
            
        except Exception as e:
            rospy.logerr(f"Error during processing: {e}")
            import traceback
            traceback.print_exc()
            return PerceptionServiceResponse()


    def process_pipeline(self, image_msg, camera_info_msg, depth_msg, point_cloud_msg):
        """
        Core logic separated from ROS communication. Returns list of Cube objects.
        """
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
        
        if self.camera_K is None:
            self.camera_K = np.array(camera_info_msg.K).reshape(3, 3)
            self.camera_frame_id = camera_info_msg.header.frame_id
        
        # 1. SAM Inference
        result = self.pipeline.detect_and_segment(cv_image, prompt=self.prompt, conf=0.25)

        if result is None or 'detection' not in result or len(result['detection']['scores']) == 0:
            rospy.logwarn("No objects detected by SAM.")
            return []
        
        # Visualize Segmentations
        self.visualize_and_publish(result, cv_image, image_msg.header)
        
        # Prepare Segmap for GraspNet
        segmap = np.zeros(cv_image.shape[:2], dtype=np.uint8)
        masks = result['segmentation']['masks']
        seg_id_to_index = {} 
        
        valid_mask_count = 0
        for i, mask in enumerate(masks):
            if np.sum(mask) > 50: 
                seg_id = valid_mask_count + 1
                segmap[mask > 0] = seg_id
                seg_id_to_index[seg_id] = i
                valid_mask_count += 1

        # 2. Call GraspNet
        detected_grasps_cam = {} 
        if HAS_GRASP_NET and valid_mask_count > 0:
            grasp_results = self.grasp_detector.process_ros_data(image_msg, depth_msg, segmap, point_cloud_msg, camera_info_msg)
            for g in grasp_results:
                detected_grasps_cam[g['id']] = {'pose': g.get('pose', None), 'score': g.get('score', None)}
        rospy.loginfo(f"detected_grasps_cam keys: {detected_grasps_cam.keys()}")
        # 3. Process Geometry & Merge
        detected_cubes = self.process_geometry_and_grasps(
            result, depth_image, point_cloud_msg, image_msg.header, 
            detected_grasps_cam, seg_id_to_index
        )
        
        return detected_cubes

    def process_geometry_and_grasps(self, result, depth_image, point_cloud_msg, header, grasp_data, id_map):
        masks = result['segmentation']['masks']
        scores = result['detection']['scores']
        print("CameraInfo frame:", self.camera_frame_id)
        print("PointCloud frame:", point_cloud_msg.header.frame_id)

        try:
            transform_cam = self.tf_buffer.lookup_transform(
                self.world_frame, point_cloud_msg.header.frame_id, header.stamp, rospy.Duration(0.5)) # Increased duration for safety
            transform_grasp = self.tf_buffer.lookup_transform(
                self.world_frame, self.camera_frame_id, header.stamp, rospy.Duration(0.5)) # Increased duration for safety
                # self.world_frame, self.camera_frame_id, header.stamp, rospy.Duration(0.5)) # Increased duration for safety
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"TF Error: {e}")
            return []

        detected_cubes = []
        all_points_world = []
        grasp_markers = MarkerArray()
        
        # Clear previous markers
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        grasp_markers.markers.append(delete_all)
        
        for seg_id, index in id_map.items():
            mask = masks[index]
            
            # Read Points
            rows, cols = np.where(mask > 0)
            uvs = np.stack([cols, rows], axis=1).tolist()
            point_gen = pc2.read_points(point_cloud_msg, field_names=("x", "y", "z"), skip_nans=True, uvs=uvs)
            points_cam = np.array(list(point_gen))
            
            if len(points_cam) < 50: continue

            points_world = self.transform_points_to_world(points_cam, transform_cam)
            w_centroid = np.mean(points_world, axis=0)
            
            rospy.loginfo(f"spy1-----------------------------------------------------")
            if not (self.table_x_min < w_centroid[0] < self.table_x_max and 
                    self.table_y_min < w_centroid[1] < self.table_y_max):
                continue

            # ICP Refinement (Using Identity Rotation as starting guess)
            try:
                # Initial transformation: Centroid position with no initial rotation
                initial_trans = np.eye(4)
                initial_trans[:3, 3] = w_centroid

                # Perform ICP refinement
                icp_matrix = self.refine_pose_with_icp(points_world, initial_trans)
                
                # Extract refined pose
                refined_pos = icp_matrix[:3, 3]
                refined_quat = R.from_matrix(icp_matrix[:3, :3]).as_quat()
                
            except Exception as e:
                rospy.logwarn(f"ICP 정밀 보정 실패: {e}. 기본 Centroid 결과로 대체합니다.")
                refined_pos = w_centroid
                refined_quat = [0, 0, 0, 1] # Identity quaternion

            # # PCA Orientation
            # pca_points = points_world - w_centroid
            # pca = PCA(n_components=2).fit(pca_points[:, :2])
            # initial_angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
            
            # # ICP Refinement
            # try:
            #     # 초기 변환 행렬 생성 (Centroid + PCA Angle)
            #     init_rot = R.from_euler('z', initial_angle).as_matrix()
            #     initial_trans = np.eye(4)
            #     initial_trans[:3, :3] = init_rot
            #     initial_trans[:3, 3] = w_centroid

            #     # ICP 수행
            #     icp_matrix = self.refine_pose_with_icp(points_world, initial_trans)
                
            #     # 최종 정밀 포즈 추출
            #     refined_pos = icp_matrix[:3, 3]
            #     refined_quat = R.from_matrix(icp_matrix[:3, :3]).as_quat()
            # except Exception as e:
            #     rospy.logwarn(f"ICP 정밀 보정 실패: {e}. PCA 결과로 대체합니다.")
            #     refined_pos = w_centroid
            #     refined_quat = R.from_euler('z', initial_angle).as_quat()

            # Grasp Processing
            final_grasp_world = None
            grasp_score = 0.0
            
            if seg_id in grasp_data:
                g_pose_cam = np.array(grasp_data[seg_id]['pose'])
                grasp_score = grasp_data[seg_id]['score']
                
                # Transform Cam -> World
                t = transform_grasp.transform.translation
                q = transform_grasp.transform.rotation
                T_world_cam = self.quaternion_matrix([q.x, q.y, q.z, q.w])
                T_world_cam[0:3, 3] = [t.x, t.y, t.z]
                
                T_world_grasp = np.dot(T_world_cam, g_pose_cam)
                final_grasp_world = T_world_grasp

                # Visualization Markers
                grasp_marker = self.create_gripper_marker(T_world_grasp, seg_id, header)
                grasp_markers.markers.append(grasp_marker)
                m_text = self.create_text_marker(T_world_grasp, seg_id, grasp_score, header)
                grasp_markers.markers.append(m_text)
                
            cube = Cube(index, refined_pos.tolist(), refined_quat.tolist(), scores[index], [self.known_cube_size]*3, 
                       grasp_pose=final_grasp_world, grasp_score=grasp_score)
            
            detected_cubes.append(cube)
            self.update_planning_scene(index, refined_pos, refined_quat)
            all_points_world.append(points_world)
        self.publish_centroid_point(detected_cubes, header.stamp)
        self.publish_mapped_point_cloud(all_points_world, header.stamp)
        self.grasp_vis_pub.publish(grasp_markers)
        
        return detected_cubes
    
    def refine_pose_with_icp(self, target_points, initial_trans):
        """ICP를 통해 가상 모델을 실제 점군에 맞춤 (완전 메모리 격리 버전)"""
        try:
            # 1. Target(실제 점군) 물리적 복제
            # NumPy를 거쳐 Python List로 변환하면 모든 메모리 플래그가 소멸됩니다.
            target_list = np.array(target_points, dtype=np.float64).tolist()
            tmp_target = o3d.geometry.PointCloud()
            tmp_target.points = o3d.utility.Vector3dVector(target_list)
            tmp_target.estimate_normals()

            # 2. Source(가상 모델) 즉석 생성 
            # 클래스 변수를 절대 쓰지 않고, 순수 좌표 리스트로부터 매번 새 PointCloud를 만듭니다.
            s = self.known_cube_size / 2.0
            grid = np.linspace(-s, s, 10)
            src_pts = []
            for i in grid:
                for j in grid:
                    src_pts.extend([[i, j, s], [i, j, -s], [i, s, j], [i, -s, j], [s, i, j], [-s, i, j]])
            
            tmp_source = o3d.geometry.PointCloud()
            tmp_source.points = o3d.utility.Vector3dVector(src_pts)
            tmp_source.estimate_normals()

            # 3. ICP 실행
            reg = o3d.pipelines.registration.registration_icp(
                tmp_source, 
                tmp_target, 
                self.known_cube_size * 0.3, 
                initial_trans,
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            
            # 결과값만 복사해서 반환하고, 생성된 tmp 객체들은 함수 종료와 함께 소멸됨
            return np.array(reg.transformation, copy=True)
            
        except Exception as e:
            rospy.logwarn(f"ICP 연산 시도 중 에러: {e}")
            raise e

    def transform_points_to_world(self, points, transform_cam):
        t = transform_cam.transform.translation
        q = transform_cam.transform.rotation
        
        R_wc = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        t_wc = np.array([t.x, t.y, t.z])

        # Apply transform: p_world = R * p_cam + t
        points_world = (R_wc @ points.T).T + t_wc

        return points_world
    

    def quaternion_matrix(self, quaternion):
        mat = np.eye(4)
        if len(quaternion) == 4:
            rotation_3x3 = R.from_quat(quaternion).as_matrix()
            mat[0:3, 0:3] = rotation_3x3
        return mat

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
        marker.scale.x = 0.003 
        hue = (grasp_id * 0.137) % 1.0  
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        marker.color.r, marker.color.g, marker.color.b = rgb
        marker.color.a = 1.0 
        w = 0.04  
        d = 0.06  
        local_pts = [
            [-w, 0, 0], [w, 0, 0],   
            [-w, 0, 0], [-w, 0, d],  
            [w, 0, 0],  [w, 0, d]    
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
        marker.pose.position.x = T_world_grasp[0, 3]
        marker.pose.position.y = T_world_grasp[1, 3]
        marker.pose.position.z = T_world_grasp[2, 3] + 0.04  
        marker.scale.z = 0.025  
        hue = (grasp_id * 0.137) % 1.0  
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        marker.color.r, marker.color.g, marker.color.b = rgb
        marker.color.a = 1.0 
        marker.text = f"ID: {grasp_id} ({score:.2f})"
        return marker
    
    def create_cube_label_marker(self, pos, cube_id, header):
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = header.stamp
        marker.ns = "cube_labels"
        marker.id = cube_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Position the marker slightly above the cube center (+0.05m)
        marker.pose.position.x = pos[0]
        marker.pose.position.y = pos[1]
        marker.pose.position.z = pos[2] + 0.05 
        marker.pose.orientation.w = 1.0
        
        marker.scale.z = 0.03  # Text size (3cm)
        
        # Set color to yellow or white for better visibility
        marker.color.r, marker.color.g, marker.color.b = (1.0, 1.0, 0.0) # Yellow
        marker.color.a = 1.0
        
        marker.text = f"ID: {cube_id}"
        return marker

if __name__ == '__main__':
    try:
        detector = SamCubeDetector()
        rospy.spin()
    except rospy.ROSInterruptException: pass
