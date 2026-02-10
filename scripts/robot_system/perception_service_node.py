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
        
        if HAS_GRASP_NET:
            self.grasp_detector = GraspServiceNode(init_node=False)

        # # real zed
        # image_topic = rospy.get_param('~image_topic', "zed2/zed_node/left/image_rect_color")
        # camera_info_topic = rospy.get_param('~camera_info_topic', "zed2/zed_node/left/camera_info")
        # depth_topic = rospy.get_param('~depth_topic', "zed2/zed_node/depth/depth_registered")
        # point_cloud_topic = rospy.get_param('~point_cloud_topic', "zed2/zed_node/point_cloud/cloud_registered")
        
        # 4. ROS Topics Configuration (Strings only, subscriber creation happens on demand)
        self.topic_image = rospy.get_param('~image_topic', "static_zed2_camera/static_zed2/zed_node/left/image_rect_color")
        self.topic_info = rospy.get_param('~camera_info_topic', "static_zed2_camera/static_zed2/zed_node/left/camera_info")
        self.topic_depth = rospy.get_param('~depth_topic', "static_zed2_camera/static_zed2/zed_node/depth/depth_registered")
        self.topic_cloud = rospy.get_param('~point_cloud_topic', "static_zed2_camera/static_zed2/zed_node/point_cloud/cloud_registered")
        
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

        # 3. Process Geometry & Merge
        detected_cubes = self.process_geometry_and_grasps(
            result, depth_image, point_cloud_msg, image_msg.header, 
            detected_grasps_cam, seg_id_to_index
        )
        
        return detected_cubes

    def process_geometry_and_grasps(self, result, depth_image, point_cloud_msg, header, grasp_data, id_map):
        masks = result['segmentation']['masks']
        scores = result['detection']['scores']
        
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame, self.camera_frame_id, header.stamp, rospy.Duration(0.5)) # Increased duration for safety
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

            points_world = self.transform_points_to_world(points_cam, transform)
            w_centroid = np.mean(points_world, axis=0)
            
            if not (self.table_x_min < w_centroid[0] < self.table_x_max and 
                    self.table_y_min < w_centroid[1] < self.table_y_max):
                continue

            # PCA Orientation
            pca_points = points_world - w_centroid
            pca = PCA(n_components=2).fit(pca_points[:, :2])
            angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
            
            cube_pos = [w_centroid[0], w_centroid[1], w_centroid[2]]
            cube_quat = R.from_euler('z', angle).as_quat()

            # Grasp Processing
            final_grasp_world = None
            grasp_score = 0.0
            
            if seg_id in grasp_data:
                g_pose_cam = np.array(grasp_data[seg_id]['pose'])
                grasp_score = grasp_data[seg_id]['score']
                
                # Transform Cam -> World
                t = transform.transform.translation
                q = transform.transform.rotation
                T_world_cam = self.quaternion_matrix([q.x, q.y, q.z, q.w])
                T_world_cam[0:3, 3] = [t.x, t.y, t.z]
                
                T_world_grasp = np.dot(T_world_cam, g_pose_cam)
                final_grasp_world = T_world_grasp

                # Visualization Markers
                grasp_marker = self.create_gripper_marker(T_world_grasp, seg_id, header)
                grasp_markers.markers.append(grasp_marker)
                m_text = self.create_text_marker(T_world_grasp, seg_id, grasp_score, header)
                grasp_markers.markers.append(m_text)

            cube = Cube(index, cube_pos, cube_quat, scores[index], [self.known_cube_size]*3, 
                       grasp_pose=final_grasp_world, grasp_score=grasp_score)
            
            detected_cubes.append(cube)
            self.update_planning_scene(index, cube_pos, cube_quat)
            all_points_world.append(points_world)

        self.publish_centroid_point(detected_cubes, header.stamp)
        self.publish_mapped_point_cloud(all_points_world, header.stamp)
        self.grasp_vis_pub.publish(grasp_markers)
        
        return detected_cubes

    def transform_points_to_world(self, points, transform):
        t = transform.transform.translation
        q = transform.transform.rotation
        matrix = self.quaternion_matrix([q.x, q.y, q.z, q.w])
        matrix[0:3, 3] = [t.x, t.y, t.z]
        points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
        return np.dot(matrix, points_homo.T).T[:, :3]

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
