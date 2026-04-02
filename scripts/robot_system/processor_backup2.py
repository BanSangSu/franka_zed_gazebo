import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import config

class CubeSegmenter:
    def __init__(self):
        self.table_img = None
        self.main_img = None
        self.final_result = None
        self.debug_steps = []

    def load_images(self):
        """Offline Mode: Loads from Disk (For manual testing only)"""
        main_path = os.path.join(config.OUTPUT_FOLDER, "main_image.png")
        table_path = os.path.join(config.OUTPUT_FOLDER, "table_image.png")

        if not os.path.exists(main_path) or not os.path.exists(table_path):
             print("Warning: Images not found on disk. (This is expected in RAM mode)")
             return

        self.main_img = cv2.imread(main_path, cv2.IMREAD_GRAYSCALE)
        self.table_img = cv2.imread(table_path, cv2.IMREAD_GRAYSCALE)
        self._preprocess_images()

    def set_images(self, main_img, table_img):
        """ROS/RAM Mode: Accepts images directly from memory"""
        if main_img is None or table_img is None:
            print("Error: Received empty images in set_images")
            return
            
        self.main_img = main_img.copy()
        self.table_img = table_img.copy()
        self.debug_steps = []
        self._preprocess_images()

    def _preprocess_images(self):
        """Shared logic: Fills gaps in the point cloud"""
        if self.main_img is None or self.table_img is None: return

        kernel = np.ones((5,5), np.uint8)
        
        # Erode to expand black dots
        self.main_img = cv2.erode(self.main_img, kernel, iterations=1)
        self.table_img = cv2.erode(self.table_img, kernel, iterations=1)

        self.debug_steps.append(("1. Input: Eroded Main", self.main_img.copy()))
        self.debug_steps.append(("2. Input: Eroded Table", self.table_img.copy()))

    def _create_raw_table_mask(self):
        # Inverted Threshold for black dots
        _, binary = cv2.threshold(self.table_img, 200, 255, cv2.THRESH_BINARY_INV)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(binary)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
        return mask

    def process_segmentation(self):
        if self.main_img is None: return

        # 1. Get Initial Mask
        raw_mask = self._create_raw_table_mask()
        cleanup_kernel = np.ones((5,5), np.uint8)
        eroded_mask = cv2.erode(raw_mask, cleanup_kernel, iterations=25)
        self.debug_steps.append(("3. Mask (Eroded)", eroded_mask.copy()))

        # 2. Get Rectangular Mask
        rect_mask = np.zeros_like(eroded_mask)
        contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest)
            (center, (w, h), angle) = rect
            scale = 0.80
            new_size = (w * scale, h * scale)
            box = cv2.boxPoints((center, new_size, angle))
            box = np.int32(box) 
            cv2.drawContours(rect_mask, [box], 0, 255, -1)
            
        self.debug_steps.append(("4. Mask (Rect)", rect_mask.copy()))

        # Combine Masks
        final_table_mask = cv2.bitwise_and(eroded_mask, rect_mask)
        self.debug_steps.append(("5. Mask (Final)", final_table_mask.copy()))

        # 3. Normal Threshold (Find White Holes)
        _, bright_objects = cv2.threshold(self.main_img, 127, 255, cv2.THRESH_BINARY)
        self.debug_steps.append(("6. Bright Objects", bright_objects.copy()))

        # 4. Intersection (Candidates)
        candidates = cv2.bitwise_and(bright_objects, final_table_mask)
        self.debug_steps.append(("7. Intersection", candidates.copy()))

        # 5. Filter & Splat
        contours, _ = cv2.findContours(candidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_candidates = np.zeros_like(candidates)
        
        for cnt in contours:
            if cv2.contourArea(cnt) >= config.MIN_AREA_SIZE:
                cv2.drawContours(clean_candidates, [cnt], -1, 255, thickness=cv2.FILLED)
        
        self.debug_steps.append(("8. Clean Candidates", clean_candidates.copy()))

        # 6. Final Splatting
        y_coords, x_coords = np.where(clean_candidates > 0)
        final_splatted = clean_candidates.copy()
        
        if len(x_coords) > 0:
            for x, y in zip(x_coords, y_coords):
                cv2.circle(final_splatted, (x, y), 4, 255, -1)

        self.final_result = final_splatted
        self.debug_steps.append(("9. Result", self.final_result.copy()))

    def get_clean_image(self):
        """Returns the result directly"""
        return self.final_result

    def save_result(self):
        """Disabled to prevent ghost files"""
        pass
        # if self.final_result is None: return
        # save_path = os.path.join(config.OUTPUT_FOLDER, "final_clean_cubes.png")
        # cv2.imwrite(save_path, self.final_result)
        # print(f"Cleaned cubes saved to: {save_path}")

    def show_debug_window(self):
        if not self.debug_steps: return
        num_images = len(self.debug_steps)
        cols = 3
        rows = (num_images + cols - 1) // cols 
        plt.figure(figsize=(15, 5 * rows))
        for i, (title, img) in enumerate(self.debug_steps):
            plt.subplot(rows, cols, i + 1)
            plt.title(title)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def run(self):
        """Standard run for testing (Offline)"""
        self.load_images()
        self.process_segmentation()
        # self.save_result() # Disabled
        self.show_debug_window()

if __name__ == "__main__":
    cleaner = CubeSegmenter()
    cleaner.run()