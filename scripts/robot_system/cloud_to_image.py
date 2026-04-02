from turtle import width
import open3d as o3d
import numpy as np
import copy
import cv2
import os
import config

class Cloud_to_Image:
    def __init__(self):
        self.pcd_flat = None
        self.main_image = None
        self.table_image = None
        
        # Store transformation math here
        self.transform_data = {
            "R": np.eye(3), 
            "min_x": 0.0, 
            "min_y": 0.0, 
            "table_z_flat": 0.0,
            "fixed_z_cam": 0.0 
        }

    def load_and_align(self):
        """Original Method: Loads from Disk (Keep for offline testing)"""
        if not os.path.exists(config.PLY_FILE_PATH):
            raise FileNotFoundError(f"PLY file not found: {config.PLY_FILE_PATH}")

        # 1. Load & Downsample
        pcd = o3d.io.read_point_cloud(config.PLY_FILE_PATH)
        pcd = pcd.remove_non_finite_points()
        
        # Call the shared alignment logic
        self._align_cloud(pcd)

    def process_live(self, pcd_input):
        """NEW Method: Accepts PointCloud directly from ROS"""
        # Ensure we work on a copy so we don't mess up the original ROS data
        pcd = copy.deepcopy(pcd_input)
        pcd = pcd.remove_non_finite_points()
        
        # Call the shared alignment logic
        self._align_cloud(pcd)
        
        # Automatically generate projections after aligning
        self.generate_projections()

    def _align_cloud(self, pcd):
        o3d.utility.random.seed(42) 

        pcd_down = pcd.voxel_down_sample(voxel_size=config.VOXEL_SIZE)

        # 1. Rough Fit
        plane_model, inliers = pcd_down.segment_plane(distance_threshold=0.015, ransac_n=6, num_iterations=1000)
        
        # 2. Refined Fit
        table_cloud = pcd_down.select_by_index(inliers)
        refined_plane, _ = table_cloud.segment_plane(distance_threshold=0.01, ransac_n=6, num_iterations=1000)
        
        [a, b, c, d] = refined_plane
        
        # --- FIX: SAVE THE Z-HEIGHT NOW (Deterministic) ---
        # The equation is ax + by + cz + d = 0.
        # After we rotate 'normal' to (0,0,1), the equation becomes: 1*z + d = 0  ->  z = -d
        self.transform_data["table_z_flat"] = -d 
        # --------------------------------------------------

        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        
        target_up = np.array([0, 0, 1])
        axis = np.cross(normal, target_up)
        
        if np.linalg.norm(axis) < 0.001:
            R = np.identity(3)
        else:
            axis = axis / np.linalg.norm(axis)
            angle_val = np.arccos(np.clip(np.dot(normal, target_up), -1.0, 1.0))
            R = pcd_down.get_rotation_matrix_from_axis_angle(axis * angle_val)
        
        self.transform_data["R"] = R

        self.pcd_flat = copy.deepcopy(pcd_down)
        self.pcd_flat.rotate(R, center=(0,0,0))

    def generate_projections(self):
        if self.pcd_flat is None:
            print("Warning: No cloud aligned yet.")
            return

        # 1. Get Table Z
        table_z_flat = self.transform_data["table_z_flat"]
        
        # 2. Slice
        pts = np.asarray(self.pcd_flat.points)
        
        mask = (pts[:, 2] < (table_z_flat + config.CUT_START)) & (pts[:, 2] > (table_z_flat + config.CUT_END))
        objects_points = pts[mask]
        
        if len(objects_points) == 0:
            self.main_image = np.full((500, 500), 255, dtype=np.uint8)
            self.table_image = np.full((500, 500), 255, dtype=np.uint8)
            return

        # 3. Project
        x_vals = objects_points[:, 0]
        y_vals = objects_points[:, 1]
        
        x_min, y_min = np.min(x_vals), np.min(y_vals)
        self.transform_data["min_x"] = x_min
        self.transform_data["min_y"] = y_min

        res = config.RESOLUTION if config.RESOLUTION > 0 else 0.002
        
        px = ((x_vals - x_min) / res).astype(int)
        py = ((y_vals - y_min) / res).astype(int)

        width = np.max(px) + (config.PADDING * 2) + 10 # Added small safety buffer
        height = np.max(py) + (config.PADDING * 2) + 10

        self.main_image = np.full((height, width), 255, dtype=np.uint8)
        self.table_image = np.full((height, width), 255, dtype=np.uint8)
        
        # 4. Draw (Vectorized - Faster & Safer than try/except)
        draw_x = px + config.PADDING
        draw_y = py + config.PADDING
        
        # Keep points inside canvas
        valid = (draw_x >= 0) & (draw_x < width) & (draw_y >= 0) & (draw_y < height)
        
        draw_x = draw_x[valid]
        draw_y = draw_y[valid]
        
        # Loop is fine, but now it won't crash
        for i in range(len(draw_x)):
            # Note: cv2 uses (x=col, y=row)
            cv2.circle(self.main_image, (draw_x[i], draw_y[i]), config.SPLAT_RADIUS_MAIN, 0, -1)
            cv2.circle(self.table_image, (draw_x[i], draw_y[i]), config.SPLAT_RADIUS_TABLE, 0, -1)

    # --- UPDATED: Return BOTH images for the pipeline ---
    def get_result_images(self):
        """Returns (main_image, table_image) directly for RAM pipeline"""
        return self.main_image, self.table_image

    def save_images(self):
        """Disabled to prevent ghost files"""
        pass 
        # path_main = os.path.join(config.OUTPUT_FOLDER, "main_image.png")
        # path_table = os.path.join(config.OUTPUT_FOLDER, "table_image.png")
        # cv2.imwrite(path_main, self.main_image)
        # cv2.imwrite(path_table, self.table_image)
        # print(f"Generated Fresh Images in: {config.OUTPUT_FOLDER}")