import open3d as o3d
import numpy as np
import copy

# --- 1. CONFIGURATION ---
CAMERA_TO_ROBOT = np.array([
    [-0.000000, -1.000000, 0.000796, 0.500000],
    [-1.000000, -0.000000, 0.000000, 0.060000],
    [-0.000000, -0.000796, -1.000000, 0.600000],
    [0.000000, 0.000000, 0.000000, 1.000000],
])

CUBE_SIZE = 0.045

# Ground Truth from spawn_cubes.py (X, Y) and Gazebo physics (Z ~ 0.022)
IDEAL_GRID = np.array([
    [0.38, -0.12, 0.022], [0.50, -0.12, 0.022], [0.62, -0.12, 0.022],
    [0.38,  0.00, 0.022], [0.50,  0.00, 0.022], [0.62,  0.00, 0.022],
    [0.38,  0.12, 0.022], [0.50,  0.12, 0.022], [0.62,  0.12, 0.022]
])

def calculate_error(detected_pos):
    """
    Finds the closest 'Ideal Cube' and calculates the error.
    Returns: (error_vector, distance_error, closest_truth)
    """
    differences = IDEAL_GRID[:, :2] - detected_pos[:2]
    distances = np.linalg.norm(differences, axis=1)
    
    best_idx = np.argmin(distances)
    closest_truth = IDEAL_GRID[best_idx]
    
    # 3. Calculate Error
    total_dist_error = distances[best_idx]
    error_vector = differences[best_idx] # [dx, dy]
    
    return error_vector, total_dist_error, closest_truth

# --- 2. SETUP (Standard) ---
target_cube = o3d.geometry.TriangleMesh.create_box(width=CUBE_SIZE, height=CUBE_SIZE, depth=CUBE_SIZE)
target_cube.translate(-np.array([CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]) / 2)
target_pcd = target_cube.sample_points_uniformly(number_of_points=1000)

print("Loading Point Cloud...")
pcd = o3d.io.read_point_cloud("debug_standard.ply")
if pcd.is_empty(): exit()
pcd_down = pcd.voxel_down_sample(voxel_size=0.002)

#removing world boundaries
points = np.asarray(pcd_down.points)
z_threshold = 1.5
mask = points[:, 2] < z_threshold
pcd_down = pcd_down.select_by_index(np.where(mask)[0])


plane_model, inliers = pcd_down.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
objects_cloud = pcd_down.select_by_index(inliers, invert=True)

labels = np.array(objects_cloud.cluster_dbscan(eps=0.015, min_points=30, print_progress=False))
max_label = labels.max()
visual_list = [objects_cloud]

print(f"\nAnalyzing {max_label + 1} detections vs Ground Truth...\n")

# --- 3. MAIN LOOP ---
for i in range(max_label + 1):
    cluster_indices = np.where(labels == i)[0]
    if len(cluster_indices) < 50: continue

    real_cube = objects_cloud.select_by_index(cluster_indices)
    center = real_cube.get_center()
    
    # ICP
    trans_init = np.identity(4)
    trans_init[:3, 3] = center
    reg_p2p = o3d.pipelines.registration.registration_icp(
        target_pcd, real_cube, 0.01, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    
    # Transforms
    camera_pose = reg_p2p.transformation
    robot_pose = np.dot(CAMERA_TO_ROBOT, camera_pose)
    rx, ry, rz = robot_pose[:3, 3]
    vision_center_pos = np.array([rx, ry, rz - (CUBE_SIZE/2)]) 
    
    err_vec, dist_err, truth = calculate_error(vision_center_pos)
    
    print(f"Cube {i} Validation:")
    print(f"  Detected (Center): [{vision_center_pos[0]:.3f}, {vision_center_pos[1]:.3f}, {vision_center_pos[2]:.3f}]")
    print(f"  Closest Truth:     [{truth[0]:.3f}, {truth[1]:.3f}, {truth[2]:.3f}]")
    
    # Color code the error output
    status = "PASS" if dist_err < 0.02 else "FAIL"
    print(f"  Error: {dist_err*1000:.1f} mm  ({status})")
    print("-" * 40)

    # Visualization
    fitted_box = copy.deepcopy(target_cube)
    fitted_box.transform(camera_pose)
    fitted_box.paint_uniform_color([0, 1, 0] if dist_err < 0.02 else [1, 0, 0])
    visual_list.append(fitted_box)

o3d.visualization.draw_geometries(visual_list)