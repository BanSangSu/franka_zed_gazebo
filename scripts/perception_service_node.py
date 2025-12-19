#!/usr/bin/env python3


import rospy
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from sensor_msgs.msg import Image as RosImage, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import PoseArray, Pose, Vector3
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree
from scipy.linalg import svd


from geometry_msgs.msg import PoseStamped
from tf2_geometry_msgs import do_transform_pose


import tf2_ros
from geometry_msgs.msg import PointStamped


from detect_n_segment import MultiDetectorSAM


# MoveIt Imports
import moveit_commander
from geometry_msgs.msg import PoseStamped


# Import your custom service
from franka_zed_gazebo.srv import PerceptionService, PerceptionServiceResponse



class PerceptionServiceNode:
    def __init__(self):
        rospy.init_node('perception_service_node')


        # Initialize MoveIt Planning Scene
        moveit_commander.roscpp_initialize([])
        self.scene = moveit_commander.PlanningSceneInterface()
        rospy.sleep(1.0) # Wait for scene to initialize
        
        # Params
        self.detector_type = rospy.get_param('~detector_type', 'florence2')
        self.sam_checkpoint = rospy.get_param('~sam_checkpoint', 'sam_vit_b_01ec64.pth')
        self.sam_model_type = rospy.get_param('~sam_model_type', 'vit_b')
        self.prompt = rospy.get_param('~prompt', 'small cube')
        self.use_sam = rospy.get_param('~use_sam', False)  # Ensure SAM is disabled by default
        
        self.camera_frame = rospy.get_param('~camera_frame', 'static_zed2_left_camera_optical_frame')
        self.world_frame = rospy.get_param('~world_frame', 'world')
        self.min_points_for_pca = rospy.get_param('~min_points_for_pca', 50)
        self.depth_scale = rospy.get_param('~depth_scale', 1.0)
        
        # Known object size (cube side length in meters)
        self.known_cube_size = rospy.get_param('~known_cube_size', 0.045)  # 4.5cm
        
        # Debug mode
        self.debug_mode = rospy.get_param('~debug_mode', True)
        
        self.bridge = CvBridge()
        self.camera_K = None
        self.camera_frame_id = None
        self.latest_image = None
        self.latest_pc = None
        self.latest_header = None  # Store latest image header
        
        # Initialize Pipeline
        detector_config = {"model_name": "microsoft/Florence-2-base"} if self.detector_type == 'florence2' else {}
        self.pipeline = MultiDetectorSAM(
            detector_type=self.detector_type,
            detector_config=detector_config,
            sam_checkpoint=self.sam_checkpoint,
            sam_model_type=self.sam_model_type,
            use_sam=self.use_sam
        )
        
        # Subscribe to camera topics
        image_topic = rospy.get_param('~image_topic', "/static_zed2/zed_node/left/image_rect_color")
        camera_info_topic = rospy.get_param('~camera_info_topic', "/static_zed2/zed_node/left/camera_info")
        depth_topic = rospy.get_param('~depth_topic', "/static_zed2/zed_node/depth/depth_registered")
        
        image_sub = message_filters.Subscriber(image_topic, RosImage)
        camera_info_sub = message_filters.Subscriber(camera_info_topic, CameraInfo)
        pc_sub = message_filters.Subscriber("/static_zed2/zed_node/point_cloud/cloud_registered", PointCloud2)
        
        ts = message_filters.ApproximateTimeSynchronizer([image_sub, camera_info_sub, pc_sub], 10, 0.1)
        ts.registerCallback(self.camera_callback)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)


        # Visualization publishers
        self.point_cloud_pub = rospy.Publisher('/cube_detection/cubes_pointcloud', PointCloud2, queue_size=10)
        self.seg_image_pub = rospy.Publisher('/cube_detection/segmentation_image', RosImage, queue_size=10)
        
        # Debug publishers
        if self.debug_mode:
            self.raw_pc_pub = rospy.Publisher('/debug/raw_points', PointCloud2, queue_size=10)
            self.table_removed_pub = rospy.Publisher('/debug/table_removed', PointCloud2, queue_size=10)
            self.clustered_pc_pub = rospy.Publisher('/debug/clustered_points', PointCloud2, queue_size=10)
            self.plane_marker_pub = rospy.Publisher('/debug/table_plane', Marker, queue_size=10)
            rospy.loginfo("Debug mode enabled - publishing to /debug/* topics")
        
        # Create service server
        self.service = rospy.Service('/perception_service', PerceptionService, self.handle_perception_request)
        
        rospy.loginfo("Perception Service Node Ready")


    def camera_callback(self, image_msg, camera_info_msg, pc_msg):
        """Cache latest sensor data"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
            self.latest_pc = pc_msg # Store PointCloud2
            self.latest_header = image_msg.header
            
            if self.camera_K is None:
                self.camera_K = np.array(camera_info_msg.K).reshape(3, 3)
                self.camera_frame_id = camera_info_msg.header.frame_id
                rospy.loginfo(f"Camera intrinsics initialized")
        except CvBridgeError as e:
            rospy.logerr(f"Conversion error: {e}")


    def visualize_and_publish(self, result, cv_image, header):
        """Create and publish visualization image with masks and bounding boxes"""
        vis_img = cv_image.copy()
        masks = result['segmentation']['masks']
        bboxes = result['detection']['bboxes']
        labels = result['detection']['labels']
        scores = result['detection']['scores']
        
        if masks and len(masks) > 0:
            # Draw masks with random colors
            for i, mask in enumerate(masks):
                color = np.random.randint(0, 255, 3).tolist()
                vis_img[mask > 0] = vis_img[mask > 0] * 0.5 + np.array(color) * 0.5
        
        # Draw bounding boxes and labels
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = labels[i] if i < len(labels) else "obj"
            score = scores[i] if i < len(scores) else 0.0
            cv2.putText(vis_img, f"{label} {score:.2f}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Publish visualization
        try:
            msg = self.bridge.cv2_to_imgmsg(vis_img, "bgr8")
            msg.header = header
            self.seg_image_pub.publish(msg)
        except CvBridgeError as e:
            rospy.logwarn(f"Failed to publish visualization: {e}")


    def publish_point_cloud(self, cubes_data):
        """Publish detected cubes as PointCloud2"""
        if len(cubes_data) == 0:
            return
        
        timestamp = rospy.Time.now()
        points = [[c['position'][0], c['position'][1], c['position'][2], c['confidence']] 
                  for c in cubes_data]
        
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
        ]
        
        pc_msg = pc2.create_cloud(
            Header(stamp=timestamp, frame_id=self.world_frame), 
            fields, 
            points
        )
        self.point_cloud_pub.publish(pc_msg)


    def publish_debug_cloud(self, points, topic_name, color):
        """Publish colored point cloud for debugging"""
        if not self.debug_mode or len(points) == 0:
            return
        
        # RGB colors (0-1 range for RViz)
        colors = np.tile(color, (len(points), 1))
        
        # Combine points and colors
        points_with_color = np.hstack([points, colors])
        
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('r', 12, PointField.FLOAT32, 1),
            PointField('g', 16, PointField.FLOAT32, 1),
            PointField('b', 20, PointField.FLOAT32, 1),
        ]
        
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.world_frame
        
        pc_msg = pc2.create_cloud(header, fields, points_with_color.tolist())
        
        # Select correct publisher
        if 'raw' in topic_name:
            self.raw_pc_pub.publish(pc_msg)
        elif 'table_removed' in topic_name:
            self.table_removed_pub.publish(pc_msg)
        elif 'clustered' in topic_name:
            self.clustered_pc_pub.publish(pc_msg)


    def visualize_plane(self, normal, d, cube_idx=0):
        """Visualize detected table plane as RViz marker"""
        if not self.debug_mode:
            return
            
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "table_plane"
        marker.id = cube_idx
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        # Plane center (on the table surface)
        marker.pose.position.x = -normal[0] * d
        marker.pose.position.y = -normal[1] * d
        marker.pose.position.z = -normal[2] * d
        
        # Orientation (align with plane normal)
        # For simplicity, just use identity orientation
        marker.pose.orientation.w = 1.0
        
        # Large flat rectangle
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 0.005  # Very thin
        
        # Semi-transparent red
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.3
        
        marker.lifetime = rospy.Duration(10.0)
        
        self.plane_marker_pub.publish(marker)


    def handle_perception_request(self, req):
        """Service callback for perception requests"""
        response = PerceptionServiceResponse()
        
        while True:
            if self.latest_image is None or self.latest_pc is None:
                rospy.loginfo("Waiting for camera data...")
                rospy.sleep(1.0)
            else:
                rospy.loginfo("Camera data received, proceeding to detection...")
                break
        
        try:
            # Run detection + segmentation
            result = self.pipeline.detect_and_segment(
                self.latest_image, 
                prompt=self.prompt if self.detector_type != 'yolov11' else None,
                conf=0.25
            )
            
            # Publish visualization
            if self.latest_header is not None:
                self.visualize_and_publish(result, self.latest_image, self.latest_header)
            
            # Process detections using PointCloud2
            cubes_data = self.process_detections(result, self.latest_image, self.latest_pc)
            
            # Publish point cloud
            self.publish_point_cloud(cubes_data)
            
            # Populate response
            response.success = True
            response.num_cubes = len(cubes_data)
            response.message = f"Detected {len(cubes_data)} cubes"
            
            pose_array = PoseArray()
            pose_array.header.frame_id = self.world_frame
            pose_array.header.stamp = rospy.Time.now()
            
            for cube in cubes_data:
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = cube['position']
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = cube['orientation']
                pose_array.poses.append(pose)
                
                response.confidences.append(cube['confidence'])
                dim = Vector3()
                dim.x, dim.y, dim.z = cube['dimensions']
                response.dimensions.append(dim)
            
            response.cube_poses = pose_array
            
            # --- UPDATE COLLISION OBJECTS ---
            try:
                # Remove old objects
                old_objects = self.scene.get_known_object_names()
                if old_objects:
                    self.scene.remove_world_object(old_objects)
                
                rospy.sleep(0.1)
                
                # Add new objects
                for i, cube_pose in enumerate(response.cube_poses.poses):
                    cube_name = f"cube_{i}"
                    
                    p = PoseStamped()
                    p.header.frame_id = self.world_frame
                    p.pose = cube_pose
                    
                    # Use detected dimensions with sanity checks
                    dims = response.dimensions[i]
                    
                    # Clamp dimensions to reasonable values (e.g., 2cm to 10cm)
                    safe_x = max(0.02, min(0.10, dims.x))
                    safe_y = max(0.02, min(0.10, dims.y))
                    safe_z = max(0.02, min(0.10, dims.z))
                    
                    size = (safe_x, safe_y, safe_z) 
                    
                    self.scene.add_box(cube_name, p, size)
                    rospy.loginfo(f"Added collision object: {cube_name} with size {size}")
                    
            except Exception as e:
                rospy.logwarn(f"Failed to update planning scene: {e}")
            
            rospy.loginfo(f"Perception completed: {response.num_cubes} cubes detected")
            
        except Exception as e:
            response.success = False
            response.message = f"Perception failed: {str(e)}"
            rospy.logerr(response.message)
        
        return response


    def process_detections(self, result, cv_image, pc_msg):
        """Extract 3D cube information using ROI extraction, RANSAC, and Clustering"""
        cubes_data = []
        bboxes = result['detection']['bboxes']
        scores = result['detection']['scores']
        
        rospy.loginfo(f"--- Process Detections Start: {len(bboxes)} bboxes ---")
        
        try:
            # Explicitly wait for transform
            transform = self.tf_buffer.lookup_transform(self.world_frame, pc_msg.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
        except Exception as e:
            rospy.logwarn(f"TF Lookup failed: {e}")
            return []
        
        camera_pos_world = np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
        
        for idx, bbox in enumerate(bboxes):
            rospy.loginfo(f"\n=== Processing BBox[{idx}] ===")
            
            # 1. Extract 3D points from PointCloud2 within the 2D BBox
            points_cam = self.extract_points_from_cloud(pc_msg, bbox)
            
            rospy.loginfo(f"BBox[{idx}]: Extracted {len(points_cam)} points from PC2")
            
            if len(points_cam) < 10:
                rospy.logwarn(f"BBox[{idx}]: Too few points ({len(points_cam)})")
                continue
            
            # Transform to World Frame
            points_world = self.transform_points_to_world(points_cam, transform)
            
            # Debug: Publish raw points (RED)
            self.publish_debug_cloud(points_world, 'raw', [1.0, 0.0, 0.0])
            
            # 2. RANSAC to remove the table surface
            plane_model, inliers = self.fit_plane_ransac(points_world)
            
            if plane_model is not None:
                normal, d = plane_model
                if normal[2] < 0: 
                    normal, d = -normal, -d
                    
                # Visualize detected plane
                self.visualize_plane(normal, d, idx)
                
                dist_to_plane = np.dot(points_world, normal) + d
                cube_candidate_points = points_world[dist_to_plane > 0.008]
                
                num_removed = len(points_world) - len(cube_candidate_points)
                rospy.loginfo(f"BBox[{idx}]: RANSAC - Before={len(points_world)}, After={len(cube_candidate_points)}, Removed={num_removed} (table points)")
                
                # Debug: Publish table-removed points (GREEN)
                self.publish_debug_cloud(cube_candidate_points, 'table_removed', [0.0, 1.0, 0.0])
            else:
                cube_candidate_points = points_world
                rospy.logwarn(f"BBox[{idx}]: RANSAC failed to find plane")

            if len(cube_candidate_points) < 5:
                rospy.logwarn(f"BBox[{idx}]: Too few points after RANSAC ({len(cube_candidate_points)})")
                continue

            # 3. Clustering (DBSCAN)
            clustering = DBSCAN(eps=0.05, min_samples=3).fit(cube_candidate_points)
            labels = clustering.labels_
            
            unique_labels = set(labels)
            if -1 in unique_labels: 
                unique_labels.remove(-1)
            
            if not unique_labels:
                rospy.logwarn(f"BBox[{idx}]: No clusters found (points too sparse)")
                continue
                
            # Pick the largest cluster
            largest_label = max(unique_labels, key=lambda l: np.sum(labels == l))
            cube_points = cube_candidate_points[labels == largest_label]
            
            rospy.loginfo(f"BBox[{idx}]: DBSCAN - Found {len(unique_labels)} clusters, Selected cluster with {len(cube_points)} points")
            
            # Debug: Publish clustered points (BLUE)
            self.publish_debug_cloud(cube_points, 'clustered', [0.0, 0.0, 1.0])

            # 4. Initial Pose Estimation
            centroid, orientation, dims = self.compute_upright_box(cube_points, camera_pos_world)
            
            if centroid is not None:
                # 5. ICP Refinement (optional)
                try:
                    yaw_init = np.arctan2(2.0*(orientation[3]*orientation[2]), 1.0 - 2.0*(orientation[2]**2))
                    refined_centroid, refined_orientation = self.refine_with_icp(cube_points, centroid, yaw_init)
                    
                    if refined_centroid is not None:
                        rospy.loginfo(f"BBox[{idx}]: ICP refinement successful")
                        centroid, orientation = refined_centroid, refined_orientation
                except Exception as e:
                    rospy.logwarn(f"BBox[{idx}]: ICP refinement failed: {e}")

                confidence = scores[idx] if idx < len(scores) else 1.0
                cubes_data.append({
                    'position': centroid,
                    'orientation': orientation,
                    'confidence': confidence,
                    'dimensions': dims
                })
                
                rospy.loginfo(f"BBox[{idx}]: SUCCESS - Cube detected at [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]")
        
        rospy.loginfo(f"=== Process Detections Complete: {len(cubes_data)} cubes detected ===\n")
        return cubes_data


    def extract_points_from_cloud(self, pc_msg, bbox):
        """Read points from PointCloud2 message within 2D BBox"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # Clamp coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(pc_msg.width-1, x2), min(pc_msg.height-1, y2)
        
        # Create UV coordinates for sampling
        uvs = []
        for v in range(y1, y2, 2):  # Step 2 to speed up
            for u in range(x1, x2, 2):
                uvs.append((u, v))
        
        # Read points
        points = list(pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True, uvs=uvs))
        
        if not points:
            return np.empty((0, 3))
        
        # Filter outliers (too far or too close)
        pts = np.array(points)
        valid_range = np.all(np.abs(pts) < 5.0, axis=1)
        return pts[valid_range]


    def fit_plane_ransac(self, points, threshold=0.008, iterations=100):
        """RANSAC plane fitting to detect table"""
        if len(points) < 3: 
            return None, None
            
        best_plane = None
        max_inliers = -1
        
        for _ in range(iterations):
            # Randomly sample 3 points
            idx = np.random.choice(len(points), 3, replace=False)
            p1, p2, p3 = points[idx]
            
            # Compute plane normal
            v1, v2 = p2 - p1, p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            
            if norm < 1e-6: 
                continue
                
            normal /= norm
            d = -np.dot(normal, p1)
            
            # Count inliers
            dist = np.abs(np.dot(points, normal) + d)
            inliers = np.sum(dist < threshold)
            
            if inliers > max_inliers:
                max_inliers = inliers
                best_plane = (normal, d)
        
        return best_plane, None


    def refine_with_icp(self, source_points, init_centroid, init_yaw, max_iterations=30):
        """ICP refinement matching to synthetic cube model"""
        cube_size = self.known_cube_size
        model_points = self._generate_cube_model(cube_size, density=20)
        
        c, s = np.cos(init_yaw), np.sin(init_yaw)
        R = np.array([[c, -s, 0], [s,  c, 0], [0,  0, 1]])
        t = np.array(init_centroid)
        
        tree = KDTree(model_points)
        prev_rmse = float('inf')
        
        for i in range(max_iterations):
            # Transform to model frame
            source_local = (source_points - t) @ R
            distances, indices = tree.query(source_local)
            
            # Keep only close points
            mask = distances < (cube_size * 0.4)
            if np.sum(mask) < 10: 
                break
            
            m_matched = model_points[indices[mask]]
            s_matched = source_points[mask]
            
            # Compute transformation
            mu_s, mu_m = np.mean(s_matched, axis=0), np.mean(m_matched, axis=0)
            H = (m_matched - mu_m).T @ (s_matched - mu_s)
            U, _, Vt = svd(H)
            R_new = Vt.T @ U.T
            
            if np.linalg.det(R_new) < 0:
                Vt[2, :] *= -1
                R_new = Vt.T @ U.T
                
            t_new = mu_s - R_new @ mu_m
            
            R, t = R_new, t_new
            rmse = np.sqrt(np.mean(distances[mask]**2))
            
            if abs(prev_rmse - rmse) < 1e-6: 
                break
            prev_rmse = rmse

        yaw = np.arctan2(R[1, 0], R[0, 0])
        orientation = np.array([0.0, 0.0, np.sin(yaw/2.0), np.cos(yaw/2.0)])
        return t, orientation


    def _generate_cube_model(self, size, density=15):
        """Generate synthetic cube surface points"""
        points = []
        h = size / 2.0
        lin = np.linspace(-h, h, density)
        
        # Top and bottom faces
        for z in [-h, h]:
            xx, yy = np.meshgrid(lin, lin)
            points.append(np.stack([xx.ravel(), yy.ravel(), np.full_like(xx.ravel(), z)], axis=1))
        
        # Left and right faces
        for x in [-h, h]:
            yy, zz = np.meshgrid(lin, lin)
            points.append(np.stack([np.full_like(yy.ravel(), x), yy.ravel(), zz.ravel()], axis=1))
        
        # Front and back faces
        for y in [-h, h]:
            xx, zz = np.meshgrid(lin, lin)
            points.append(np.stack([xx.ravel(), np.full_like(xx.ravel(), y), zz.ravel()], axis=1))
        
        return np.vstack(points)


    def transform_points_to_world(self, points, transform):
        """Transform points from camera frame to world frame"""
        trans = transform.transform.translation
        rot = transform.transform.rotation
        
        t = np.array([trans.x, trans.y, trans.z])
        q = np.array([rot.x, rot.y, rot.z, rot.w])
        
        R = self.quaternion_to_matrix(q)
        
        points_world = points @ R.T + t
        return points_world


    def quaternion_to_matrix(self, q):
        """Convert quaternion [x, y, z, w] to rotation matrix 3x3"""
        x, y, z, w = q
        return np.array([
            [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w],
            [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w],
            [2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x - 2*y*y]
        ])


    def compute_upright_box(self, points, camera_pos_world):
        """
        Compute oriented bounding box with perspective correction
        """
        if len(points) < 20:
            return None, None, None
        
        cube_size = self.known_cube_size
        
        # 1. Statistical outlier removal
        mean = np.mean(points, axis=0)
        std = np.std(points, axis=0)
        mask = np.all(np.abs(points - mean) < 2.0 * std, axis=1)
        clean_points = points[mask]
        
        if len(clean_points) < 20:
            clean_points = points

        # 2. Filter: Use upper portion only (avoid table/shadow points)
        z_min_raw = np.min(clean_points[:, 2])
        z_max_raw = np.max(clean_points[:, 2])
        z_threshold = z_min_raw + (z_max_raw - z_min_raw) * 0.2
        
        upper_points = clean_points[clean_points[:, 2] > z_threshold]
        if len(upper_points) < 10:
            upper_points = clean_points
            
        # 3. Robust XY centroid using MEDIAN
        perceived_x = np.median(upper_points[:, 0])
        perceived_y = np.median(upper_points[:, 1])
        
        # 4. Perspective correction (shift away from camera)
        dx = perceived_x - camera_pos_world[0]
        dy = perceived_y - camera_pos_world[1]
        dist = np.sqrt(dx**2 + dy**2)
        
        if dist > 0.01:
            ux = dx / dist
            uy = dy / dist
            
            radius_shift = cube_size / 2.0
            corrected_x = perceived_x + ux * radius_shift
            corrected_y = perceived_y + uy * radius_shift
            
            rospy.loginfo(f"Perspective Correction: Perceived=[{perceived_x:.3f}, {perceived_y:.3f}], "
                         f"Corrected=[{corrected_x:.3f}, {corrected_y:.3f}] (shift={radius_shift:.3f}m)")
        else:
            corrected_x = perceived_x
            corrected_y = perceived_y
        
        # 5. Z position: top surface detected
        corrected_z = z_max_raw - cube_size / 2.0
        
        # Prevent cube from sinking into table
        min_z = -0.05 + cube_size / 2.0
        corrected_z = max(corrected_z, min_z)
        
        centroid = np.array([corrected_x, corrected_y, corrected_z])
        
        # 6. Orientation from MinAreaRect
        points_xy = clean_points[:, :2].astype(np.float32)
        rect = cv2.minAreaRect(points_xy)
        (_, _), (w, h), angle_deg = rect
        
        yaw = np.radians(angle_deg)
        if w < h:
            yaw += np.pi / 2
            
        # Normalize yaw to ±45 degrees
        while yaw > np.pi/4: 
            yaw -= np.pi/2
        while yaw < -np.pi/4: 
            yaw += np.pi/2
        
        orientation = np.array([0.0, 0.0, np.sin(yaw/2.0), np.cos(yaw/2.0)])
        
        rospy.loginfo(f"Final Cube Pose: Centroid=[{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}], "
                     f"Yaw={np.degrees(yaw):.1f}°")
        
        return centroid, orientation, np.array([cube_size, cube_size, cube_size])


if __name__ == '__main__':
    try:
        node = PerceptionServiceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
