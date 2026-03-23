import cv2
import numpy as np
import os
import config
import matplotlib.pyplot as plt
import open3d as o3d

class CubeDetector:
    def __init__(self):
        self.clean_img = None
        self.output_img = None

    def load_image(self):
        """Offline Mode: Loads from disk"""
        path = os.path.join(config.OUTPUT_FOLDER, "final_clean_cubes.png")
        self.clean_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if self.clean_img is None:
            raise FileNotFoundError(f"Could not find {path}. Run Processor.py first!")
        
        self.output_img = cv2.cvtColor(self.clean_img, cv2.COLOR_GRAY2BGR)

    def set_image(self, clean_img):
        """ROS Mode: Accepts image from memory"""
        # Ensure it's grayscale
        if len(clean_img.shape) == 3:
            self.clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
        else:
            self.clean_img = clean_img.copy()
            
        # Create output image for visualization overlays
        self.output_img = cv2.cvtColor(self.clean_img, cv2.COLOR_GRAY2BGR)

    def detect_pose(self):
        if self.clean_img is None:
            print("Error: No image loaded in Detector.")
            return []

        print("--- DETECTING POSES ---")
        contours, _ = cv2.findContours(self.clean_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_poses = []

        for i, cnt in enumerate(contours):
            if cv2.contourArea(cnt) < 5: continue

            # 1. Get Pose
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect

            # Normalize Angle
            if w < h:
                w, h = h, w
                angle += 90
            
            detected_poses.append(((cx, cy), angle))

            # 2. Visualization (Green Box)
            box = cv2.boxPoints(rect)
            box = np.int32(box) 
            cv2.drawContours(self.output_img, [box], 0, (0, 255, 0), 2)
            
            # Draw ID
            cv2.putText(self.output_img, str(i+1), (int(cx), int(cy)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return detected_poses

    def calculate_world_coordinates(self, pixel_poses, math_data, pcd):
        # PASS 'pcd' (High Res) here!
        
        debug_geometries = [pcd]  # Start with the cloud so we see the table too
        results = []
        
        # print("\n--- CALCULATING REAL WORLD COORDINATES ---")
        
        R = math_data["R"]
        min_x = math_data["min_x"]
        min_y = math_data["min_y"]
        
        table_z_flat = math_data["table_z_flat"] 
        
        for i, ((u, v), angle) in enumerate(pixel_poses):
            # 1. Pixel -> Flat 2D
            x_flat = ((u - config.PADDING) * config.RESOLUTION) + min_x
            y_flat = ((v - config.PADDING) * config.RESOLUTION) + min_y
            
            # 2. Assign Z in Flat Frame (Table - Half Cube)
            z_flat = table_z_flat - 0.022

            # 3. Un-Rotate (Flat -> Camera Frame)
            point_flat = np.array([x_flat, y_flat, z_flat])
            point_camera = R.T @ point_flat 

            final_x = point_camera[0]
            final_y = point_camera[1]
            final_z = point_camera[2]

            # print(f"Cube {i+1}: World X={final_x:.4f}, Y={final_y:.4f}, Z={final_z:.4f} | Angle={angle:.1f}")
            results.append(((final_x, final_y, final_z), angle))

            # Visualization Sphere
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            sphere.translate([final_x, final_y, final_z])
            sphere.paint_uniform_color([1, 0, 0]) # Red
            debug_geometries.append(sphere)

        # VISUALIZATION (Blocks script until window closes)
        # print("DEBUG: Check the Red Spheres. Close window to continue...")
        # o3d.visualization.draw_geometries(debug_geometries)
        # -------------------------------------------

        return results

    def show_results(self):
        if self.output_img is None: return
        plt.figure(figsize=(10, 10))
        plt.title("Detected Cubes with IDs")
        plt.imshow(cv2.cvtColor(self.output_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()