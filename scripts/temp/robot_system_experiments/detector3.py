import cv2
import numpy as np
import config
from scipy.spatial.transform import Rotation as R_scipy

class CubeDetector:
    def __init__(self):
        self.clean_img = None
        self.output_img = None
        
        # --- DIMENSIONS ---
        self.CUBE_SIDE_M = 0.045
        self.PIXEL_SIZE = int(self.CUBE_SIDE_M / config.RESOLUTION)
        if self.PIXEL_SIZE > 1000: self.PIXEL_SIZE = int(self.PIXEL_SIZE / 1000)
        self.EXPECTED_AREA = self.PIXEL_SIZE * self.PIXEL_SIZE
        
        # --- FILTERS ---
        # self.MIN_AREA = self.EXPECTED_AREA * 0.5 
        self.MIN_AREA = 0.7
        self.MAX_AREA = self.EXPECTED_AREA * 1.5
        print(f"Detector Ready. Target Area: {self.EXPECTED_AREA} px")

    def set_image(self, clean_img):
        """Accepts image from RAM (from Processor)"""
        if clean_img is None:
            print("Error: Detector received None image")
            return

        if len(clean_img.shape) == 3:
            self.clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
        else:
            self.clean_img = clean_img.copy()
        self.output_img = cv2.cvtColor(self.clean_img, cv2.COLOR_GRAY2BGR)

    def detect_pose(self):
        """ Returns list of ((cx, cy), angle) for valid cubes """
        if self.clean_img is None: return []

        # print("--- RUNNING DETECTOR ---")
        contours, _ = cv2.findContours(self.clean_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_poses = []
        
        for i, cnt in enumerate(contours):
            # 1. Size Filter
            area = cv2.contourArea(cnt)
            # if area < self.MIN_AREA or area > self.MAX_AREA: continue
            if area < self.MIN_AREA: continue

            # 2. Geometry Filter
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            if w == 0 or h == 0: continue

            # # 3. Aspect Ratio Filter (Squareness)
            # aspect_ratio = min(w, h) / max(w, h)
            # if aspect_ratio < 0.60: continue

            # Fix Angle (OpenCV returns -90 to 0 or 0 to 90 depending on version)
            if w < h: angle += 90
            
            detected_poses.append(((int(cx), int(cy)), angle))
            self._draw_result(int(cx), int(cy), angle, color=(0, 255, 0))

        return detected_poses

    def _draw_result(self, cx, cy, angle, color):
        if self.output_img is None: return
        cv2.circle(self.output_img, (cx, cy), 5, color, -1)
        rect = ((cx, cy), (self.PIXEL_SIZE, self.PIXEL_SIZE), angle)
        box = np.int32(cv2.boxPoints(rect))
        cv2.drawContours(self.output_img, [box], 0, color, 2)

    def calculate_camera_coordinates(self, pixel_poses, math_data):
        """
        Calculates 3D coordinates in the CAMERA FRAME.
        """
        # Get Rotation (Camera -> Flat)
        R_cam_to_flat = math_data["R"]
        # Invert it (Flat -> Camera)
        R_flat_to_cam = R_cam_to_flat.T 
        
        min_x = math_data["min_x"]
        min_y = math_data["min_y"]
        table_z_flat = math_data["table_z_flat"] 
        
        # --- FIX SETUP: Define 180-degree flip matrix ---
        # We will apply this LOCALLY to the cube. 
        # Rotating 180 around X keeps the X-axis (Red) fixed but flips Z (Blue) and Y (Green).
        R_flip_local = R_scipy.from_euler('x', 180, degrees=True).as_matrix()
        # ------------------------------------------------

        results = []
        
        for i, ((u, v), angle_deg) in enumerate(pixel_poses):
            # 1. Convert Pixels to Meters (Table Frame)
            x_flat = ((u - config.PADDING) * config.RESOLUTION) + min_x
            y_flat = ((v - config.PADDING) * config.RESOLUTION) + min_y
            
            # Z is height relative to table 
            z_flat = table_z_flat - 0.0225 
            
            point_flat = np.array([x_flat, y_flat, z_flat])
            
            # 2. Rotate Position to Camera Frame
            point_camera = R_flat_to_cam @ point_flat 
            
            # 3. Orientation Calculation
            # Step A: Create the base rotation in the Flat Frame (just the Z-angle)
            # This creates a cube sitting 'normal' on the table.
            r_cube_flat = R_scipy.from_euler('z', -angle_deg, degrees=True).as_matrix()
            
            # Step B: Apply the FLIP locally!             # This flips the Blue arrow (Z) to point the other way, 
            # while keeping the Red arrow (X) aligned with the cube edge.
            r_cube_flipped = r_cube_flat @ R_flip_local
            
            # Step C: Rotate the whole thing into the Camera Frame
            total_rotation = R_flat_to_cam @ r_cube_flipped
            
            quat = R_scipy.from_matrix(total_rotation).as_quat()
            
            results.append((point_camera, quat, angle_deg))
            
        return results

    def get_debug_image(self):
        return self.output_img