# import open3d as o3d
# import numpy as np
# import copy

# # --- 1. CONFIGURATION ---
# CAMERA_TO_ROBOT = np.array([
#     [-0.000000, -1.000000, 0.000796, 0.500000],
#     [-1.000000, -0.000000, 0.000000, 0.060000],
#     [-0.000000, -0.000796, -1.000000, 0.600000],
#     [0.000000, 0.000000, 0.000000, 1.000000],
# ])

# CUBE_SIZE = 0.045

# # Ground Truth from spawn_cubes.py (X, Y) and Gazebo physics (Z ~ 0.022)
# IDEAL_GRID = np.array([
#     [0.38, -0.12, 0.022], [0.50, -0.12, 0.022], [0.62, -0.12, 0.022],
#     [0.38,  0.00, 0.022], [0.50,  0.00, 0.022], [0.62,  0.00, 0.022],
#     [0.38,  0.12, 0.022], [0.50,  0.12, 0.022], [0.62,  0.12, 0.022]
# ])

# def calculate_error(detected_pos):
#     """
#     Finds the closest 'Ideal Cube' and calculates the error.
#     Returns: (error_vector, distance_error, closest_truth)
#     """
#     differences = IDEAL_GRID[:, :2] - detected_pos[:2]
#     distances = np.linalg.norm(differences, axis=1)
    
#     best_idx = np.argmin(distances)
#     closest_truth = IDEAL_GRID[best_idx]
    
#     # 3. Calculate Error
#     total_dist_error = distances[best_idx]
#     error_vector = differences[best_idx] # [dx, dy]
    
#     return error_vector, total_dist_error, closest_truth

# # --- 2. SETUP (Standard) ---
# target_cube = o3d.geometry.TriangleMesh.create_box(width=CUBE_SIZE, height=CUBE_SIZE, depth=CUBE_SIZE)
# target_cube.translate(-np.array([CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]) / 2)
# target_pcd = target_cube.sample_points_uniformly(number_of_points=1000)

# print("Loading Point Cloud...")
# pcd = o3d.io.read_point_cloud("debug_standard.ply")
# if pcd.is_empty(): exit()
# pcd_down = pcd.voxel_down_sample(voxel_size=0.002)

# #removing world boundaries
# points = np.asarray(pcd_down.points)
# z_threshold = 1.5
# mask = points[:, 2] < z_threshold
# pcd_down = pcd_down.select_by_index(np.where(mask)[0])


# plane_model, inliers = pcd_down.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
# objects_cloud = pcd_down.select_by_index(inliers, invert=True)

# labels = np.array(objects_cloud.cluster_dbscan(eps=0.015, min_points=30, print_progress=False))
# max_label = labels.max()
# visual_list = [objects_cloud]

# print(f"\nAnalyzing {max_label + 1} detections vs Ground Truth...\n")

# # --- 3. MAIN LOOP ---
# for i in range(max_label + 1):
#     cluster_indices = np.where(labels == i)[0]
#     if len(cluster_indices) < 50: continue

#     real_cube = objects_cloud.select_by_index(cluster_indices)
#     center = real_cube.get_center()
    
#     # ICP
#     trans_init = np.identity(4)
#     trans_init[:3, 3] = center
#     reg_p2p = o3d.pipelines.registration.registration_icp(
#         target_pcd, real_cube, 0.01, trans_init,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint()
#     )
    
#     # Transforms
#     camera_pose = reg_p2p.transformation
#     robot_pose = np.dot(CAMERA_TO_ROBOT, camera_pose)
#     rx, ry, rz = robot_pose[:3, 3]
#     vision_center_pos = np.array([rx, ry, rz - (CUBE_SIZE/2)]) 
    
#     err_vec, dist_err, truth = calculate_error(vision_center_pos)
    
#     print(f"Cube {i} Validation:")
#     print(f"  Detected (Center): [{vision_center_pos[0]:.3f}, {vision_center_pos[1]:.3f}, {vision_center_pos[2]:.3f}]")
#     print(f"  Closest Truth:     [{truth[0]:.3f}, {truth[1]:.3f}, {truth[2]:.3f}]")
    
#     # Color code the error output
#     status = "PASS" if dist_err < 0.02 else "FAIL"
#     print(f"  Error: {dist_err*1000:.1f} mm  ({status})")
#     print("-" * 40)

#     # Visualization
#     fitted_box = copy.deepcopy(target_cube)
#     fitted_box.transform(camera_pose)
#     fitted_box.paint_uniform_color([0, 1, 0] if dist_err < 0.02 else [1, 0, 0])
#     visual_list.append(fitted_box)

# o3d.visualization.draw_geometries(visual_list)


import open3d as o3d
import numpy as np
import copy
import rospy
import tf
from tf.transformations import quaternion_matrix, translation_matrix
from gazebo_msgs.srv import GetWorldProperties, GetModelState

def get_camera_transform():
    if rospy.get_name() == '/unnamed':
        rospy.init_node('tf_listener', anonymous=True)

    listener = tf.TransformListener()
    rospy.sleep(1.0)

    try:
        # (trans, rot) = listener.lookupTransform("panda_link0", "left_camera_link_optical", rospy.Time(0))
        (trans, rot) = listener.lookupTransform("world", "static_zed2_left_camera_link_optical", rospy.Time(0))
        print("Live Transform Acquired")
        return np.dot(translation_matrix(trans), quaternion_matrix(rot))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        print("TF Failed. Using Identity Matrix.")
        return np.eye(4)

def align_pose_to_normal(pose_matrix, plane_normal):
    """Aligns Z-axis to the table normal while keeping Yaw."""
    R_current = pose_matrix[:3, :3]
    t_current = pose_matrix[:3, 3]
    z_new = plane_normal / np.linalg.norm(plane_normal)
    if np.dot(z_new, R_current[:, 2]) < 0: z_new = -z_new
    x_curr = R_current[:, 0]
    x_projected = x_curr - np.dot(x_curr, z_new) * z_new
    if np.linalg.norm(x_projected) < 1e-6: x_new = np.cross(z_new, np.array([1, 0, 0]))
    else: x_new = x_projected / np.linalg.norm(x_projected)
    y_new = np.cross(z_new, x_new)
    new_pose = np.eye(4)
    new_pose[:3, :3] = np.column_stack((x_new, y_new, z_new))
    new_pose[:3, 3] = t_current
    return new_pose

def get_ideal_grid_from_gazebo():
    """
    Asks Gazebo for the true positions of all objects named 'cube*'.
    Returns: Numpy array of shape (N, 3) -> [[x, y, z], ...]
    """
    # Ensure node is running
    if rospy.get_name() == '/unnamed':
        rospy.init_node('cube_validator', anonymous=True)

    print("Listening to Gazebo for Ground Truth...")
    rospy.wait_for_service('/gazebo/get_world_properties')
    rospy.wait_for_service('/gazebo/get_model_state')

    try:
        # 1. Get list of all models
        get_world = rospy.ServiceProxy('/gazebo/get_world_properties', GetWorldProperties)
        get_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        
        model_names = get_world().model_names
        cube_list = []

        # 2. Loop through and find cubes
        for name in model_names:
            if "cube" in name.lower():  # Case insensitive check
                resp = get_state(name, "world") # Get pose relative to world origin
                if resp.success:
                    pos = resp.pose.position
                    # Save [x, y, z]
                    cube_list.append([pos.x, pos.y, pos.z])

        if not cube_list:
            print("⚠️ WARNING: No cubes found in Gazebo! Using default fallback grid.")
            return None

        print(f"✅ Found {len(cube_list)} cubes in simulation.")
        return np.array(cube_list)

    except rospy.ServiceException as e:
        print(f"❌ Gazebo Error: {e}")
        return None

def calculate_yaw_error(pose_matrix):
    """
    Extracts the Z-rotation (Yaw) error considering 90-degree symmetry.
    Returns: Error in degrees (float)
    """
    R = pose_matrix[:3, :3]
    yaw_rad = np.arctan2(R[1, 0], R[0, 0])
    yaw_deg = np.degrees(yaw_rad)
    symmetry_deg = yaw_deg % 90

    if symmetry_deg > 45:
        error_deg = symmetry_deg - 90
    else:
        error_deg = symmetry_deg
        
    return error_deg

def calculate_error(detected_pos):
    """
    Finds the closest 'Ideal Cube' and calculates the error.
    Returns: (error_vector, distance_error, closest_truth)
    """
    differences = IDEAL_GRID[:, :2] - detected_pos[:2]
    distances = np.linalg.norm(differences, axis=1)
    
    best_idx = np.argmin(distances)
    closest_truth = IDEAL_GRID[best_idx]
    
    #Calculate Error
    total_dist_error = distances[best_idx]
    error_vector = differences[best_idx]
    
    return error_vector, total_dist_error, closest_truth


# --- 1. CONFIGURATION (FINAL CALIBRATION) ---
# We adjusted the Z-Translation from 0.426 -> 0.703 to fix the -27cm error.
# We also tweaked Y slightly to center the table.
# CAMERA_TO_ROBOT = np.array([
#     [-0.000000, -0.734190, -0.678944, 0.800000],
#     [-1.000000, -0.000000, 0.000000, 0.060000],
#     [-0.000000, 0.678944, -0.734190, 0.600000],
#     [0.000000, 0.000000, 0.000000, 1.000000],
# ])
CAMERA_TO_ROBOT = get_camera_transform()
IDEAL_GRID = get_ideal_grid_from_gazebo()
# fetched_grid = get_ideal_grid_from_gazebo()

# if fetched_grid is not None:
#     IDEAL_GRID = fetched_grid
# else:
#     IDEAL_GRID = np.array([
#         [0.38, -0.12, 0.022], [0.50, -0.12, 0.022], [0.62, -0.12, 0.022],
#         [0.38,  0.00, 0.022], [0.50,  0.00, 0.022], [0.62,  0.00, 0.022],
#         [0.38,  0.12, 0.022], [0.50,  0.12, 0.022], [0.62,  0.12, 0.022]
#     ])
CUBE_SIZE = 0.045

# Ground Truth
# IDEAL_GRID = np.array([
#     [0.38, -0.12, 0.022], [0.50, -0.12, 0.022], [0.62, -0.12, 0.022],
#     [0.38,  0.00, 0.022], [0.50,  0.00, 0.022], [0.62,  0.00, 0.022],
#     [0.38,  0.12, 0.022], [0.50,  0.12, 0.022], [0.62,  0.12, 0.022]
# ])

VALID_Z_MIN = -0.01  
VALID_Z_MAX =  0.06 

# --- 2. SETUP ---
target_cube = o3d.geometry.TriangleMesh.create_box(width=CUBE_SIZE, height=CUBE_SIZE, depth=CUBE_SIZE)
target_cube.translate(-np.array([CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]) / 2)
target_pcd = target_cube.sample_points_uniformly(number_of_points=1000)

print("Loading Point Cloud...")
pcd = o3d.io.read_point_cloud("./debug_standard.ply")
if pcd.is_empty(): exit()
pcd_down = pcd.voxel_down_sample(voxel_size=0.002)

# Filter floor (Camera Frame) - Keep points up to 2.5m away
points = np.asarray(pcd_down.points)
mask = points[:, 2] < 2.5
pcd_down = pcd_down.select_by_index(np.where(mask)[0])

# Plane Segmentation
plane_model, inliers = pcd_down.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
table_normal = np.array(plane_model[:3])
objects_cloud = pcd_down.select_by_index(inliers, invert=True)

labels = np.array(objects_cloud.cluster_dbscan(eps=0.015, min_points=30, print_progress=False))
max_label = labels.max()
visual_list = [objects_cloud]

# --- 3. COLLECT RAW DATA ---
raw_cubes = []

for i in range(max_label + 1):
    cluster_indices = np.where(labels == i)[0]
    if len(cluster_indices) < 50: continue

    real_cube = objects_cloud.select_by_index(cluster_indices)
    center = real_cube.get_center()
    
    # Run ICP
    trans_init = np.identity(4); trans_init[:3, 3] = center
    reg_p2p = o3d.pipelines.registration.registration_icp(
        target_pcd, real_cube, 0.01, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    #ironing to table by forcing z axis to match the table normal
    camera_pose_raw = reg_p2p.transformation
    camera_pose_aligned = align_pose_to_normal(camera_pose_raw, table_normal)
    # Calculate pose
    robot_pose = np.dot(CAMERA_TO_ROBOT, camera_pose_aligned)
    rx, ry, rz = robot_pose[:3, 3]
    
    # z-filter
    raw_z = rz - (CUBE_SIZE/2)
    if -0.5 < raw_z < 0.5:
        raw_cubes.append({
            'pos': np.array([rx, ry, raw_z]), 
            'draw_pose': camera_pose_aligned,   #for visualization
            'robot_matrix': robot_pose          #for angle calculation
        })
if not raw_cubes: exit()

# Calibrate Z: median z of all detected cubes and shift it to match actual z
avg_z = np.median([c['pos'][2] for c in raw_cubes])
z_offset = avg_z - 0.022
print(f"Auto-Calibrated Z-Offset: {z_offset:.4f} m")

# --- 4. MATCH AND VALIDATE ---
for i, cube in enumerate(raw_cubes):
    #Apply Offset
    corrected_pos = cube['pos'].copy()
    corrected_pos[2] -= z_offset
    
    #Nearest Neighbor Match (Robust to random order)
    dists = np.linalg.norm(IDEAL_GRID[:, :2] - corrected_pos[:2], axis=1)
    best_idx = np.argmin(dists)
    best_truth = IDEAL_GRID[best_idx]
    dist_err = dists[best_idx]
    yaw_err = calculate_yaw_error(cube['robot_matrix'])
    
    #Print
    status = "PASS" if dist_err < 0.02 else "FAIL"
    ang_status = "PASS" if abs(yaw_err) < 5.0 else "FAIL"
    print(f"Cube {i} (Matches Truth {best_idx}):")
    print(f"  Detected:  [{corrected_pos[0]:.3f}, {corrected_pos[1]:.3f}, {corrected_pos[2]:.3f}]")
    print(f"  Truth:     [{best_truth[0]:.3f}, {best_truth[1]:.3f}, 0.022]")
    print(f"  Pos Error: {dist_err*1000:.1f} mm  [{status}]")
    print(f"  Ang Error: {yaw_err:.1f} deg     [{ang_status}]")
    print("-" * 30)

    #VISUALIZATION
    fitted_box = copy.deepcopy(target_cube)
    fitted_box.transform(cube['draw_pose']) 
    fitted_box.paint_uniform_color([0, 1, 0] if dist_err < 0.02 else [1, 0, 0])
    visual_list.append(fitted_box)

o3d.visualization.draw_geometries(visual_list)