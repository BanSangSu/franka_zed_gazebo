#!/usr/bin/env python3
import rospy
import numpy as np
import copy
import open3d as o3d
import tf2_ros
import moveit_commander
import sensor_msgs.point_cloud2 as pc2

from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from franka_zed_gazebo.srv import PerceptionService, PerceptionServiceResponse

class Open3DPerceptionNode:
    def __init__(self):
        rospy.init_node('open3d_perception_node')

        # 1. Initialize MoveIt Planning Scene
        moveit_commander.roscpp_initialize([])
        self.scene = moveit_commander.PlanningSceneInterface()
        rospy.sleep(1.0)
        
        # 2. Parameters & Configuration
        self.world_frame = rospy.get_param('~world_frame', 'world')
        self.known_cube_size = rospy.get_param('~known_cube_size', 0.045)
        self.target_z_ground_truth = 0.022  # Reference Z from Gazebo
        self.debug_mode = rospy.get_param('~debug_mode', True)
        
        # 3. Setup ICP Template (Synthetic Cube)
        self.target_cube_mesh = o3d.geometry.TriangleMesh.create_box(
            width=self.known_cube_size, height=self.known_cube_size, depth=self.known_cube_size)
        self.target_cube_mesh.translate(-np.array([self.known_cube_size] * 3) / 2)
        self.target_cube_pcd = self.target_cube_mesh.sample_points_uniformly(number_of_points=1000)
        
        # 4. Publishers & Subscribers
        pc_topic = "/static_zed2_camera/static_zed2/zed_node/point_cloud/cloud_registered"
        self.pc_sub = rospy.Subscriber(pc_topic, PointCloud2, self.pointcloud_callback)
        
        # Debug Publishers
        self.point_cloud_pub = rospy.Publisher('/cube_detection/cubes_pointcloud', PointCloud2, queue_size=10)
        self.table_removed_pub = rospy.Publisher('/debug/table_removed', PointCloud2, queue_size=10)
        self.plane_marker_pub = rospy.Publisher('/debug/table_plane', Marker, queue_size=10)
        
        # TF Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Service
        self.service = rospy.Service('/perception_service', PerceptionService, self.handle_perception_request)
        
        self.latest_pc = None
        rospy.loginfo("Advanced Open3D Perception Node Started")

    def pointcloud_callback(self, pc_msg):
        self.latest_pc = pc_msg

    def align_pose_to_normal(self, pose_matrix, plane_normal):
        """Aligns the cube's Z-axis to the table normal to fix tilting."""
        R_current = pose_matrix[:3, :3]
        t_current = pose_matrix[:3, 3]
        
        # Force Z-axis to match table normal
        z_new = plane_normal / np.linalg.norm(plane_normal)
        if np.dot(z_new, R_current[:, 2]) < 0: z_new = -z_new
        
        # Maintain Yaw by projecting current X onto the plane
        x_curr = R_current[:, 0]
        x_projected = x_curr - np.dot(x_curr, z_new) * z_new
        if np.linalg.norm(x_projected) < 1e-6: 
            x_new = np.cross(z_new, np.array([1, 0, 0]))
        else: 
            x_new = x_projected / np.linalg.norm(x_projected)
        
        y_new = np.cross(z_new, x_new)
        
        new_pose = np.eye(4)
        new_pose[:3, :3] = np.column_stack((x_new, y_new, z_new))
        new_pose[:3, 3] = t_current
        return new_pose

    def publish_debug_cloud(self, pcd, publisher):
        """Converts Open3D PCD back to ROS PointCloud2 for visualization."""
        if not self.debug_mode or pcd is None: return
        points = np.asarray(pcd.points)
        header = Header(stamp=rospy.Time.now(), frame_id=self.world_frame)
        pc_msg = pc2.create_cloud_xyz32(header, points.tolist())
        publisher.publish(pc_msg)

    def visualize_plane(self, normal, d):
        """Publishes a red plane marker in RViz."""
        if not self.debug_mode: return
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = -normal * d
        marker.pose.orientation.w = 1.0
        marker.scale.x, marker.scale.y, marker.scale.z = 1.0, 1.0, 0.005
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.0, 0.0, 0.3
        self.plane_marker_pub.publish(marker)

    def process_point_cloud(self, pc_msg):
        # 1. Transform Cloud to World Frame
        points = list(pc2.read_points(pc_msg, skip_nans=True, field_names=("x", "y", "z")))
        if not points: return []
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))

        try:
            trans = self.tf_buffer.lookup_transform(self.world_frame, pc_msg.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            t = [trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]
            q = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
            # Use Open3D's efficient matrix conversion
            r_mat = pcd.get_rotation_matrix_from_quaternion(q)
            transform_mat = np.eye(4)
            transform_mat[:3, :3] = r_mat
            transform_mat[:3, 3] = t
            pcd.transform(transform_mat)
        except: return []

        # 2. Table Removal (RANSAC)
        pcd_down = pcd.voxel_down_sample(0.002)
        plane_model, inliers = pcd_down.segment_plane(0.01, 3, 1000)
        table_normal = np.array(plane_model[:3])
        self.visualize_plane(table_normal, plane_model[3])
        
        objects_cloud = pcd_down.select_by_index(inliers, invert=True)
        self.publish_debug_cloud(objects_cloud, self.table_removed_pub)

        # 3. Clustering (DBSCAN)
        labels = np.array(objects_cloud.cluster_dbscan(eps=0.015, min_points=30))
        raw_poses = []

        # 4. ICP & Normal Alignment
        for i in range(labels.max() + 1):
            indices = np.where(labels == i)[0]
            if len(indices) < 50: continue
            
            cluster = objects_cloud.select_by_index(indices)
            trans_init = np.eye(4); trans_init[:3, 3] = cluster.get_center()
            
            reg = o3d.pipelines.registration.registration_icp(
                self.target_cube_pcd, cluster, 0.01, trans_init)
            
            # Apply normal alignment to fix tilts
            aligned_pose = self.align_pose_to_normal(reg.transformation, table_normal)
            raw_poses.append(aligned_pose)

        # 5. Automatic Z-Calibration & Result Collection
        if not raw_poses: return []
        avg_z = np.median([p[2, 3] for p in raw_poses])
        z_offset = avg_z - self.target_z_ground_truth
        
        results = []
        for p in raw_poses:
            pos = p[:3, 3]
            pos[2] -= z_offset # Calibration
            
            R = p[:3, :3]
            yaw = (np.arctan2(R[1, 0], R[0, 0]) + np.pi/4) % (np.pi/2) - np.pi/4
            
            results.append({
                'position': pos.tolist(),
                'orientation': [0, 0, np.sin(yaw/2), np.cos(yaw/2)],
                'dimensions': [self.known_cube_size] * 3
            })
        return results

    def handle_perception_request(self, req):
        res = PerceptionServiceResponse()
        if self.latest_pc is None: return res

        cubes_data = self.process_point_cloud(self.latest_pc)
        self.update_planning_scene(cubes_data)
        
        res.success = True
        res.num_cubes = len(cubes_data)
        res.cube_poses = PoseArray()
        res.cube_poses.header.frame_id = self.world_frame
        res.cube_poses.header.stamp = rospy.Time.now()
        
        for cube in cubes_data:
            p = Pose()
            p.position.x, p.position.y, p.position.z = cube['position']
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = cube['orientation']
            res.cube_poses.poses.append(p)
        return res

    def update_planning_scene(self, cubes_data):
        # Remove old objects
        for obj in self.scene.get_known_object_names():
            if obj.startswith("cube_"): self.scene.remove_world_object(obj)
        rospy.sleep(0.2)

        # Add new objects
        for i, cube in enumerate(cubes_data):
            ps = PoseStamped()
            ps.header.frame_id = self.world_frame
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = cube['position']
            ps.pose.orientation.z, ps.pose.orientation.w = cube['orientation'][2:]
            
            size = [s - 0.002 for s in cube['dimensions']] # Shrink for safety
            self.scene.add_box(f"cube_{i}", ps, size)

if __name__ == '__main__':
    try:
        node = Open3DPerceptionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
