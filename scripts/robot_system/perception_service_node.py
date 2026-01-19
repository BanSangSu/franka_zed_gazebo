#!/usr/bin/env python3
import rospy
import message_filters
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseArray, Pose, Vector3
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import copy

# Open3D imports
import open3d as o3d

# TF imports
import tf2_ros

# MoveIt Imports
import moveit_commander
from geometry_msgs.msg import PoseStamped

# Custom service
from franka_zed_gazebo.srv import PerceptionService, PerceptionServiceResponse


class Open3DPerceptionNode:
    def __init__(self):
        rospy.init_node('open3d_perception_node')

        # Initialize MoveIt Planning Scene
        moveit_commander.roscpp_initialize([])
        self.scene = moveit_commander.PlanningSceneInterface()
        rospy.sleep(1.0)
        
        # Parameters
        self.world_frame = rospy.get_param('~world_frame', 'world')
        self.known_cube_size = rospy.get_param('~known_cube_size', 0.045)  # 4.5cm
        
        # Point cloud processing parameters
        self.voxel_size = rospy.get_param('~voxel_size', 0.002)
        self.z_threshold = rospy.get_param('~z_threshold', 1.5)  # Remove world boundaries
        self.ransac_distance = rospy.get_param('~ransac_distance', 0.01)
        self.ransac_iterations = rospy.get_param('~ransac_iterations', 1000)
        self.dbscan_eps = rospy.get_param('~dbscan_eps', 0.015)
        self.dbscan_min_points = rospy.get_param('~dbscan_min_points', 30)
        self.min_cluster_points = rospy.get_param('~min_cluster_points', 50)
        self.icp_max_correspondence = rospy.get_param('~icp_max_correspondence', 0.01)
        self.icp_max_iterations = rospy.get_param('~icp_max_iterations', 30)
        
        # Debug mode
        self.debug_mode = rospy.get_param('~debug_mode', True)
        
        self.latest_pc = None
        
        # Generate synthetic cube model for ICP (once)
        rospy.loginfo("Generating synthetic cube model...")
        self.target_cube_mesh = o3d.geometry.TriangleMesh.create_box(
            width=self.known_cube_size, 
            height=self.known_cube_size, 
            depth=self.known_cube_size
        )
        self.target_cube_mesh.translate(-np.array([self.known_cube_size] * 3) / 2)
        self.target_cube_pcd = self.target_cube_mesh.sample_points_uniformly(number_of_points=1000)
        
        # Subscribe to point cloud
        pc_topic = rospy.get_param('~pointcloud_topic', "/static_zed2/zed_node/point_cloud/cloud_registered")
        rospy.Subscriber(pc_topic, PointCloud2, self.pointcloud_callback)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Visualization publishers
        self.point_cloud_pub = rospy.Publisher('/cube_detection/cubes_pointcloud', PointCloud2, queue_size=10)
        
        # Debug publishers
        if self.debug_mode:
            self.raw_pc_pub = rospy.Publisher('/debug/raw_cloud', PointCloud2, queue_size=10)
            self.table_removed_pub = rospy.Publisher('/debug/table_removed', PointCloud2, queue_size=10)
            self.clustered_pc_pub = rospy.Publisher('/debug/clustered_points', PointCloud2, queue_size=10)
            self.plane_marker_pub = rospy.Publisher('/debug/table_plane', Marker, queue_size=10)
            rospy.loginfo("Debug mode enabled - publishing to /debug/* topics")
        
        # Service server
        self.service = rospy.Service('/perception_service', PerceptionService, self.handle_perception_request)
        
        rospy.loginfo("Open3D Perception Node Ready")

    def pointcloud_callback(self, pc_msg):
        """Cache latest point cloud"""
        self.latest_pc = pc_msg

    def ros_to_open3d(self, ros_cloud):
        """Convert ROS PointCloud2 to Open3D PointCloud"""
        points = list(pc2.read_points(ros_cloud, field_names=("x", "y", "z"), skip_nans=True))
        
        if not points:
            return o3d.geometry.PointCloud()
        
        pts = np.array(points)
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        
        return pcd

    def transform_open3d_cloud(self, pcd, transform):
        """Transform Open3D cloud using TF transform"""
        trans = transform.transform.translation
        rot = transform.transform.rotation
        
        t = np.array([trans.x, trans.y, trans.z])
        q = np.array([rot.x, rot.y, rot.z, rot.w])
        
        # Build transformation matrix
        R = self.quaternion_to_matrix(q)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        
        # Transform point cloud
        pcd.transform(T)
        return pcd

    def quaternion_to_matrix(self, q):
        """Convert quaternion [x, y, z, w] to rotation matrix"""
        x, y, z, w = q
        return np.array([
            [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w],
            [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w],
            [2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x - 2*y*y]
        ])

    def publish_debug_cloud(self, pcd, publisher, color):
        """Publish Open3D cloud as colored ROS PointCloud2 for debugging"""
        if not self.debug_mode or pcd is None or len(pcd.points) == 0:
            return
        
        points = np.asarray(pcd.points)
        colors = np.tile(color, (len(points), 1))
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
        publisher.publish(pc_msg)

    def visualize_plane(self, normal, d):
        """Visualize detected table plane as RViz marker"""
        if not self.debug_mode:
            return
            
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = rospy.Time.now()
        marker.ns = "table_plane"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        
        marker.pose.position.x = -normal[0] * d
        marker.pose.position.y = -normal[1] * d
        marker.pose.position.z = -normal[2] * d
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 0.005
        
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.3
        
        marker.lifetime = rospy.Duration(10.0)
        
        self.plane_marker_pub.publish(marker)

    def publish_point_cloud(self, cubes_data):
        """Publish detected cubes as PointCloud2"""
        if len(cubes_data) == 0:
            return
        
        timestamp = rospy.Time.now()
        points = [[c['position'][0], c['position'][1], c['position'][2], 1.0] 
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

    def handle_perception_request(self, req):
        """Service callback for perception requests"""
        response = PerceptionServiceResponse()
        
        # Wait for point cloud data
        while self.latest_pc is None:
            rospy.loginfo("Waiting for point cloud data...")
            rospy.sleep(1.0)
        
        rospy.loginfo("Point cloud received, starting detection...")
        
        try:
            # Process point cloud using Open3D
            cubes_data = self.process_point_cloud(self.latest_pc)
            
            # Filter already placed cubes
            cubes_data = self.filter_placed_cubes(cubes_data)
            
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
                
                response.confidences.append(1.0)  # No vision model, so confidence = 1.0
                dim = Vector3()
                dim.x, dim.y, dim.z = cube['dimensions']
                response.dimensions.append(dim)
                if hasattr(response, 'labels'):
                    response.labels.append('cube')
            
            response.cube_poses = pose_array
            
            # Update MoveIt planning scene
            self.update_planning_scene(response)
            
            rospy.loginfo(f"Perception completed: {response.num_cubes} cubes detected")
            
        except Exception as e:
            response.success = False
            response.message = f"Perception failed: {str(e)}"
            rospy.logerr(response.message)
            import traceback
            rospy.logerr(traceback.format_exc())
        
        return response

    def process_point_cloud(self, pc_msg):
        """
        Process point cloud using Open3D pipeline:
        1. Convert to Open3D
        2. Transform to world frame
        3. Remove world boundaries (z-threshold)
        4. RANSAC plane segmentation (table removal)
        5. DBSCAN clustering (find individual cubes)
        6. ICP refinement (pose estimation)
        """
        rospy.loginfo(f"\n{'='*60}")
        rospy.loginfo("Starting Open3D Point Cloud Processing")
        rospy.loginfo(f"{'='*60}")
        
        # 1. Convert to Open3D
        pcd = self.ros_to_open3d(pc_msg)
        rospy.loginfo(f"Converted PointCloud2 to Open3D: {len(pcd.points)} points")
        
        if len(pcd.points) == 0:
            rospy.logwarn("Empty point cloud!")
            return []
        
        # 2. Transform to world frame
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame, 
                pc_msg.header.frame_id, 
                rospy.Time(0), 
                rospy.Duration(1.0)
            )
            self.transform_open3d_cloud(pcd, transform)
            rospy.loginfo(f"Transformed to {self.world_frame} frame")
        except Exception as e:
            rospy.logwarn(f"TF lookup failed: {e}, using original frame")
        
        # Debug: Publish raw cloud (WHITE)
        if self.debug_mode:
            self.publish_debug_cloud(pcd, self.raw_pc_pub, [1.0, 1.0, 1.0])
        
        # 3. Remove world boundaries (z-threshold)
        points = np.asarray(pcd.points)
        mask = points[:, 2] < self.z_threshold
        pcd = pcd.select_by_index(np.where(mask)[0])
        rospy.loginfo(f"After z-threshold ({self.z_threshold}m): {len(pcd.points)} points")
        
        # 4. Voxel Downsampling
        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        rospy.loginfo(f"After voxel downsampling ({self.voxel_size}m): {len(pcd_down.points)} points")
        
        # 5. RANSAC Plane Segmentation (Table Removal)
        plane_model, inliers = pcd_down.segment_plane(
            distance_threshold=self.ransac_distance,
            ransac_n=3,
            num_iterations=self.ransac_iterations
        )
        
        table_z = None
        if plane_model is not None:
            a, b, c, d = plane_model
            normal = np.array([a, b, c])
            
            # Ensure normal points up
            if c < 0:
                normal = -normal
                d = -d
            
            # Calculate table height
            if abs(c) > 0.8:  # Horizontal plane
                table_z = -d / c
                rospy.loginfo(f"RANSAC detected table at z={table_z:.3f}m")
            
            # Visualize plane
            self.visualize_plane(normal, d)
            
            # Extract object points (above table)
            objects_cloud = pcd_down.select_by_index(inliers, invert=True)
            
            rospy.loginfo(f"RANSAC: Removed {len(inliers)} table points, {len(objects_cloud.points)} remain")
            
            # Debug: Table-removed points (GREEN)
            if self.debug_mode:
                self.publish_debug_cloud(objects_cloud, self.table_removed_pub, [0.0, 1.0, 0.0])
        else:
            objects_cloud = pcd_down
            rospy.logwarn("RANSAC failed to find plane")
        
        if len(objects_cloud.points) < self.min_cluster_points:
            rospy.logwarn(f"Too few object points ({len(objects_cloud.points)})")
            return []
        
        # 6. DBSCAN Clustering
        labels = np.array(objects_cloud.cluster_dbscan(
            eps=self.dbscan_eps, 
            min_points=self.dbscan_min_points, 
            print_progress=False
        ))
        
        max_label = labels.max()
        if max_label < 0:
            rospy.logwarn("No clusters found")
            return []
        
        rospy.loginfo(f"DBSCAN: Found {max_label + 1} clusters")
        
        # 7. Process each cluster with ICP
        cubes_data = []
        
        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            
            if len(cluster_indices) < self.min_cluster_points:
                rospy.loginfo(f"Cluster {i}: Skipping (only {len(cluster_indices)} points)")
                continue
            
            cube_cloud = objects_cloud.select_by_index(cluster_indices)
            rospy.loginfo(f"\n--- Processing Cluster {i} ({len(cube_cloud.points)} points) ---")
            
            # Debug: Publish this cluster (BLUE)
            if self.debug_mode:
                self.publish_debug_cloud(cube_cloud, self.clustered_pc_pub, [0.0, 0.0, 1.0])
            
            # ICP Registration
            center = cube_cloud.get_center()
            trans_init = np.identity(4)
            trans_init[:3, 3] = center
            
            try:
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    self.target_cube_pcd,  # source (template)
                    cube_cloud,             # target (detected)
                    max_correspondence_distance=self.icp_max_correspondence,
                    init=trans_init,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=self.icp_max_iterations
                    )
                )
                
                cube_pose = reg_p2p.transformation
                rospy.loginfo(f"Cluster {i}: ICP fitness={reg_p2p.fitness:.3f}, RMSE={reg_p2p.inlier_rmse:.4f}")
                
            except Exception as e:
                rospy.logwarn(f"Cluster {i}: ICP failed: {e}, using centroid")
                cube_pose = trans_init
            
            # Extract position
            centroid = cube_pose[:3, 3]
            
            # Adjust Z using table height if available
            if table_z is not None:
                centroid[2] = table_z + self.known_cube_size / 2.0
            
            # Extract orientation (yaw only for upright cubes)
            R = cube_pose[:3, :3]
            yaw = np.arctan2(R[1, 0], R[0, 0])
            
            # Normalize yaw to Â±45 degrees (cubes have 90Â° symmetry)
            while yaw > np.pi/4: 
                yaw -= np.pi/2
            while yaw < -np.pi/4: 
                yaw += np.pi/2
            
            # Convert to quaternion
            orientation = np.array([0.0, 0.0, np.sin(yaw/2.0), np.cos(yaw/2.0)])
            
            cubes_data.append({
                'position': centroid.tolist(),
                'orientation': orientation.tolist(),
                'dimensions': [self.known_cube_size] * 3,
            })
            
            rospy.loginfo(f"Cluster {i}: Cube at [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}], yaw={np.degrees(yaw):.1f}Â°")
        
        rospy.loginfo(f"\n{'='*60}")
        rospy.loginfo(f"Detection Complete: {len(cubes_data)} cubes detected")
        rospy.loginfo(f"{'='*60}\n")
        
        return cubes_data

    def filter_placed_cubes(self, cubes_data):
        """Filter out cubes that are already in planning scene as 'placed_cube_*'"""
        try:
            known_objects = self.scene.get_known_object_names()
            placed_cube_names = [name for name in known_objects if name.startswith("placed_cube_")]
            
            if not placed_cube_names or not cubes_data:
                return cubes_data
            
            placed_poses = self.scene.get_object_poses(placed_cube_names)
            filtered_cubes = []
            
            for cube in cubes_data:
                pos = np.array(cube['position'])
                is_already_placed = False
                
                for p_name in placed_cube_names:
                    p_pose = placed_poses[p_name]
                    p_pos = np.array([p_pose.position.x, p_pose.position.y, p_pose.position.z])
                    
                    dist = np.linalg.norm(pos - p_pos)
                    if dist < 0.04:  # 4cm threshold
                        rospy.loginfo(f"Filtering detection at {pos} - too close to {p_name} (dist={dist:.3f}m)")
                        is_already_placed = True
                        break
                
                if not is_already_placed:
                    filtered_cubes.append(cube)
            
            return filtered_cubes
            
        except Exception as e:
            rospy.logwarn(f"Failed to filter placed cubes: {e}")
            return cubes_data

    def update_planning_scene(self, response):
        """Update MoveIt planning scene with detected cubes"""
        try:
            # Remove old 'cube_*' objects (not 'placed_cube_*')
            old_objects = self.scene.get_known_object_names()
            cubes_to_remove = [obj for obj in old_objects if obj.startswith('cube_') and not obj.startswith('placed_cube_')]
            
            if cubes_to_remove:
                for obj in cubes_to_remove:
                    self.scene.remove_world_object(obj)
                rospy.sleep(0.2)
            
            # Add new detections
            for i, cube_pose in enumerate(response.cube_poses.poses):
                cube_name = f"cube_{i}"
                
                p = PoseStamped()
                p.header.frame_id = self.world_frame
                p.pose = cube_pose
                
                # Use detected dimensions with safety checks
                dims = response.dimensions[i]
                safe_x = max(0.02, min(0.10, dims.x))
                safe_y = max(0.02, min(0.10, dims.y))
                safe_z = max(0.02, min(0.10, dims.z))
                
                # Shrink collision box slightly to avoid phantom collisions
                shrink = 0.005
                size = (max(0.01, safe_x - shrink), 
                        max(0.01, safe_y - shrink), 
                        max(0.01, safe_z - shrink))
                
                self.scene.add_box(cube_name, p, size)
                rospy.loginfo(f"Added collision object: {cube_name} with size {size}")
                
        except Exception as e:
            rospy.logwarn(f"Failed to update planning scene: {e}")


if __name__ == '__main__':
    try:
        node = Open3DPerceptionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

