import os

# --- PATHS ---
PLY_FILE_PATH = r"C:\Users\dheer\OneDrive\Desktop\franka_vision_experiments\data\burst_capture_1.ply"
OUTPUT_FOLDER = "Images"

# --- 3D PROCESSING ---
RESOLUTION = 0.0005      # Meters per pixel
CUT_START  = 0.04       
CUT_END    = -0.04
VOXEL_SIZE = 0.002
 
# --- IMAGING ---
SPLAT_RADIUS_MAIN  = 7
SPLAT_RADIUS_TABLE = 14
PADDING = 25             

# --- VISION PROCESSING ---
MIN_AREA_SIZE = 2000

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)