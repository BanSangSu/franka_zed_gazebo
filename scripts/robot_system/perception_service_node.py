#!/usr/bin/env python3
import rospy
import numpy as np
import copy
import open3d as o3d
import tf2_ros
import moveit_commander
import sensor_msgs.point_cloud2 as pc2


from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, Vector3
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
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
      
       # 3. Table ROI Bounds (World Frame)
       self.table_x_min = rospy.get_param('~table_x_min', 0.3)
       self.table_x_max = rospy.get_param('~table_x_max', 0.8)
       self.table_y_min = rospy.get_param('~table_y_min', -0.4)
       self.table_y_max = rospy.get_param('~table_y_max', 0.4)
      
       # 4. Setup ICP Template (Synthetic Cube)
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
       self.marker_pub = rospy.Publisher('/cube_detection/markers', MarkerArray, queue_size=10)
      
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
       # 1. Prepare Open3D Point Cloud in CAMERA FRAME (Same as final_poses.py)
       points = list(pc2.read_points(pc_msg, skip_nans=True, field_names=("x", "y", "z")))
       if not points: return []
       pcd = o3d.geometry.PointCloud()
       pcd.points = o3d.utility.Vector3dVector(np.array(points))


       # 2. Pre-processing (Voxel Downsample)
       pcd_down = pcd.voxel_down_sample(0.002)
      
       # Filter floor/distance (Camera Frame Z is depth)
       points_np = np.asarray(pcd_down.points)
       mask = points_np[:, 2] < 2.5 # Keep points within 2.5m
       pcd_down = pcd_down.select_by_index(np.where(mask)[0])


       # 3. Table Removal (RANSAC) in Camera Frame
       plane_model, inliers = pcd_down.segment_plane(0.01, 3, 1000)
       table_normal = np.array(plane_model[:3])
       objects_cloud = pcd_down.select_by_index(inliers, invert=True)
      
       # 4. Clustering (DBSCAN) - Thresholds same as final_poses.py
       labels = np.array(objects_cloud.cluster_dbscan(eps=0.015, min_points=30))
      
       # TF Lookup (World to Camera)
       try:
           trans = self.tf_buffer.lookup_transform(self.world_frame, pc_msg.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
           t_vec = [trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]
           q_vec = [trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w]
           # Convert TF to 4x4 Matrix
           q_o3d = [q_vec[3], q_vec[0], q_vec[1], q_vec[2]]
           r_mat = o3d.geometry.get_rotation_matrix_from_quaternion(q_o3d)
           camera_to_world = np.eye(4)
           camera_to_world[:3, :3] = r_mat
           camera_to_world[:3, 3] = t_vec
       except: return []


       raw_poses_world = []
       all_clusters_cloud = o3d.geometry.PointCloud()


       # 5. ICP & Normal Alignment in Camera Frame, then Transform to World
       for i in range(labels.max() + 1):
           indices = np.where(labels == i)[0]
           if len(indices) < 50: continue
          
           cluster = objects_cloud.select_by_index(indices)
          
           # --- ROI & Noise Filter (Crucial to prevent strange cubes at edges) ---
           # Transform cluster center to world just to check if it's on the table
           center_cam = cluster.get_center()
           center_world = np.dot(camera_to_world, np.append(center_cam, 1.0))[:3]
          
           # Table bounds ROI using class parameters
           if not (self.table_x_min < center_world[0] < self.table_x_max and
                   self.table_y_min < center_world[1] < self.table_y_max):
               continue
          
           # Check cluster proportions (Prevent flat table edges from being cubes)
           min_bound = cluster.get_min_bound()
           max_bound = cluster.get_max_bound()
           dims = max_bound - min_bound
           if max(dims) > 0.08 or min(dims) < 0.01: # Too big or too flat
               continue
           # -----------------------------------------------------------------------


           all_clusters_cloud += cluster
           trans_init = np.eye(4); trans_init[:3, 3] = center_cam
          
           # ICP (Camera Frame)
           reg = o3d.pipelines.registration.registration_icp(
               self.target_cube_pcd, cluster, 0.01, trans_init)
          
           # Align Pose to Table Normal (Camera Frame)
           aligned_pose_cam = self.align_pose_to_normal(reg.transformation, table_normal)
          
           # Transform to World Frame
           world_pose = np.dot(camera_to_world, aligned_pose_cam)
           raw_poses_world.append(world_pose)


       # Publish debug cloud (transformed for visualization)
       all_clusters_cloud.transform(camera_to_world)
       self.publish_debug_cloud(all_clusters_cloud, self.point_cloud_pub)
      
       # Also published transformed table-removed cloud for consistency
       objects_cloud_world = copy.deepcopy(objects_cloud)
       objects_cloud_world.transform(camera_to_world)
       self.publish_debug_cloud(objects_cloud_world, self.table_removed_pub)


       # 6. Automatic Z-Calibration (Matching final_poses.py but targeting CENTER)
       if not raw_poses_world: return []
      
       # Final target center Z should be 0.022 (approx half of 0.045 cube)
       raw_center_zs = [p[2, 3] for p in raw_poses_world]
       avg_center_z = np.median(raw_center_zs)
       z_offset = avg_center_z - 0.022
      
       results = []
       for p in raw_poses_world:
           pos = p[:3, 3]
           # Correct only Z (Apply offset to bring median center to 0.022)
           pos[2] -= z_offset
          
           R = p[:3, :3]
           # Handle cube 90-degree symmetry for Yaw
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
       self.publish_markers(cubes_data)
      
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
          
           # Populate dimensions and labels
           dim = Vector3()
           dim.x, dim.y, dim.z = cube['dimensions']
           res.dimensions.append(dim)
           res.labels.append("cube")
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


   def publish_markers(self, cubes_data):
       """Publishes 3D bounding boxes and labels for each cube."""
       marker_array = MarkerArray()
       for i, cube in enumerate(cubes_data):
           # Cube Box Marker
           m = Marker()
           m.header.frame_id = self.world_frame
           m.header.stamp = rospy.Time.now()
           m.ns = "cubes"
           m.id = i
           m.type = Marker.CUBE
           m.action = Marker.ADD
           m.pose.position.x, m.pose.position.y, m.pose.position.z = cube['position']
           m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = cube['orientation']
           m.scale.x, m.scale.y, m.scale.z = cube['dimensions']
           m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 0.6
           marker_array.markers.append(m)
          
           # Label Marker
           l = Marker()
           l.header.frame_id = self.world_frame
           l.header.stamp = rospy.Time.now()
           l.ns = "labels"
           l.id = i + 100
           l.type = Marker.TEXT_VIEW_FACING
           l.action = Marker.ADD
           l.pose.position.x, l.pose.position.y, l.pose.position.z = cube['position']
           l.pose.position.z += 0.05
           l.scale.z = 0.02
           l.color.r, l.color.g, l.color.b, l.color.a = 1.0, 1.0, 1.0, 1.0
           l.text = f"Cube_{i}"
           marker_array.markers.append(l)
          
       self.marker_pub.publish(marker_array)


if __name__ == '__main__':
   try:
       node = Open3DPerceptionNode()
       rospy.spin()
   except rospy.ROSInterruptException:
       pass





