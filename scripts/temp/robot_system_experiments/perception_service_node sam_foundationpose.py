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

foundationpose_node_path = os.path.join(parent_dir, 'FoundationPose')
if foundationpose_node_path not in sys.path:
    sys.path.append(foundationpose_node_path)

from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
from lightning_modules.utils import *
import nvdiffrast.torch as dr


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

class FoundationSamDetector:
    def __init__(self):
        rospy.init_node('foundation_sam_detector', anonymous=True)
        
        # 1. Load Parameters & Mesh
        self.mesh_path = rospy.get_param('~mesh_file', 'path/to/your/cube_mesh.obj')
        self.world_frame = rospy.get_param('~world_frame', 'world')
        self.known_cube_size = 0.045
        
        # 2. Initialize FoundationPose
        rospy.loginfo("Initializing FoundationPose...")
        self.mesh = trimesh.load(self.mesh_path)
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        
        self.est = FoundationPose(
            model_pts=self.mesh.vertices, 
            model_normals=self.mesh.vertex_normals, 
            mesh=self.mesh, 
            scorer=self.scorer, 
            refiner=self.refiner, 
            glctx=self.glctx
        )
        
        # To handle coordinate offsets if mesh is not centered
        self.to_origin, _ = trimesh.bounds.oriented_bounds(self.mesh)
        
        # 3. Initialize SAM & MoveIt
        self.bridge = CvBridge()
        self.pipeline = MultiDetectorSAM(detector_type='florence2') # Simplified for brevity
        self.scene = moveit_commander.PlanningSceneInterface()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 4. ROS Topics
        self.topic_image = rospy.get_param('~image_topic', "zedr/zed_node/left/image_rect_color")
        self.topic_info = rospy.get_param('~camera_info_topic', "zedr/zed_node/left/camera_info")
        self.topic_depth = rospy.get_param('~depth_topic', "zedr/zed_node/depth/depth_registered")
        self.topic_cloud = rospy.get_param('~point_cloud_topic', "zedr/zed_node/point_cloud/cloud_registered")

        self.seg_image_pub = rospy.Publisher('/perception/foundation_vis', RosImage, queue_size=10)
        self.service = rospy.Service('/perception_service', PerceptionService, self.service_callback)
        
        rospy.loginfo("FoundationPose + SAM Node Ready.")

    def acquire_frames(self, timeout=5.0):
        captured_data = {}
        event = threading.Event()

        def sync_cb(img, info, depth, cloud):
            captured_data.update({'img': img, 'info': info, 'depth': depth, 'cloud': cloud})
            event.set()

        sub_img = message_filters.Subscriber(self.topic_image, RosImage)
        sub_info = message_filters.Subscriber(self.topic_info, CameraInfo)
        sub_depth = message_filters.Subscriber(self.topic_depth, RosImage)
        sub_cloud = message_filters.Subscriber(self.topic_cloud, PointCloud2)

        ts = message_filters.ApproximateTimeSynchronizer([sub_img, sub_info, sub_depth, sub_cloud], 10, 0.1)
        ts.registerCallback(sync_cb)
        
        if event.wait(timeout):
            for s in [sub_img, sub_info, sub_depth, sub_cloud]: s.unregister()
            return captured_data['img'], captured_data['info'], captured_data['depth'], captured_data['cloud']
        return None

    def service_callback(self, req):
        data = self.acquire_frames()
        if not data: return PerceptionServiceResponse(success=False)
        
        img_msg, info_msg, depth_msg, cloud_msg = data
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, "rgb8") # FoundationPose expects RGB
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        K = np.array(info_msg.K).reshape(3, 3)

        # 1. Run SAM to get the mask (Segmented Data)
        result = self.pipeline.detect_and_segment(cv_img, prompt="small cube", conf=0.25)
        if not result or len(result['segmentation']['masks']) == 0:
            return PerceptionServiceResponse(success=False, message="No masks found")

        detected_cubes = []
        
        # 2. Use the first mask for FoundationPose registration
        # Note: FoundationPose usually tracks one object at a time. 
        # For multiple cubes, you'd loop or use multiple estimator instances.
        for i, mask in enumerate(result['segmentation']['masks']):
            mask = mask.astype(bool)
            
            # FoundationPose Registration (Initial Pose Estimation)
            # This uses the segmented data you requested
            pose_cam = self.est.register(K=K, rgb=cv_img, depth=depth_img, ob_mask=mask, iteration=5)
            
            # Transform to World Frame
            try:
                t_stamped = self.tf_buffer.lookup_transform(self.world_frame, img_msg.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
                T_world_cam = self.tf_to_matrix(t_stamped)
                pose_world = T_world_cam @ pose_cam
                
                pos = pose_world[:3, 3]
                quat = R.from_matrix(pose_world[:3, :3]).as_quat()
                
                cube = Cube(i, pos, quat, result['detection']['scores'][i], [self.known_cube_size]*3)
                detected_cubes.append(cube)
                
                # Update Planning Scene
                self.add_to_planning_scene(f"cube_{i}", pos, quat)
                
            except Exception as e:
                rospy.logerr(f"TF/Pose conversion error: {e}")

        # Construct Response
        resp = PerceptionServiceResponse()
        resp.success = True
        for c in detected_cubes:
            p = Pose()
            p.position.x, p.position.y, p.position.z = c.position
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = c.orientation
            resp.cube_poses.poses.append(p)
        
        return resp

    def tf_to_matrix(self, transform):
        t = transform.transform.translation
        q = transform.transform.rotation
        mat = np.eye(4)
        mat[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        mat[:3, 3] = [t.x, t.y, t.z]
        return mat

    def add_to_planning_scene(self, name, pos, quat):
        ps = PoseStamped()
        ps.header.frame_id = self.world_frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = pos
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = quat
        self.scene.add_box(name, ps, size=(self.known_cube_size, self.known_cube_size, self.known_cube_size))

if __name__ == '__main__':
    try:
        node = FoundationSamDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass