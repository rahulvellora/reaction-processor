import cv2
import numpy as np
import os
from decimer_segmentation import segment_chemical_structures
import matplotlib.pyplot as plt


class ReactionProcessor:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def load_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")
        return cv2.imread(image_path)

    def extract_reaction_region(self, image_path):
        image = self.load_image(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4)
        kernel = np.ones((3, 3), np.uint8)
        enhanced = cv2.dilate(enhanced, kernel, iterations=2)
        edges = cv2.Canny(enhanced, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=150, maxLineGap=15)
        
        horizontal_lines = sorted(set(y1 for line in lines for x1, y1, x2, y2 in line if abs(y1 - y2) < 5)) if lines is not None else []
        bottom_line_y = horizontal_lines[1] if len(horizontal_lines) >= 2 else (horizontal_lines[0] if horizontal_lines else None)
        
        if bottom_line_y is None:
            print("No valid horizontal line detected!")
            return None
        
        reaction_region = image[:bottom_line_y + 2, :]
        output_path = os.path.join(self.output_folder, os.path.basename(image_path).replace(".png", "_reaction.png"))
        cv2.imwrite(output_path, reaction_region)
        print(f"Extracted reaction region saved to {output_path}")
        return output_path

    def detect_arrow_and_split(self, image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = 0, 0

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                if line_length > 100 and abs(y2 - y1) < 10:  # Horizontal line check
                    min_x, min_y = min(min_x, x1, x2), min(min_y, y1, y2)
                    max_x, max_y = max(max_x, x1, x2), max(max_y, y1, y2)

        if min_x < max_x and min_y < max_y:
            left_part = image[:, :min_x]  
            right_part = image[:, max_x:]  
        else:
            return None, None 
        
        reactants_path = image_path.replace("_reaction.png", "_reactants.png")
        products_path = image_path.replace("_reaction.png", "_products.png")
        cv2.imwrite(reactants_path, left_part)
        cv2.imwrite(products_path, right_part)

        return reactants_path, products_path

    def apply_decimer_segmentation(self, image_path, output_folder):
        os.makedirs(self.output_folder, exist_ok=True)
        image = self.load_image(image_path)
        segmented_images, _ = segment_chemical_structures(image, output_folder)
        for idx, img in enumerate(segmented_images):
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            plt.imsave(os.path.join(output_folder, f"segmented_{idx}.png"), img, cmap="gray")
        print(f"Segmented structures saved in {output_folder}")
        return segmented_images

    def process_batch(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    
        for file in os.listdir(self.input_folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(self.input_folder, file)
                base_name = os.path.splitext(file)[0]
                
                extracted_path = self.extract_reaction_region(input_path)
                if extracted_path:
                    reactants_path, products_path = self.detect_arrow_and_split(extracted_path)
                    if reactants_path and products_path:
                        self.apply_decimer_segmentation(reactants_path, os.path.join(self.output_folder, f"{base_name}_reactants_segments"))
                        self.apply_decimer_segmentation(products_path, os.path.join(self.output_folder, f"{base_name}_products_segments"))


if __name__ == "__main__":
    processor = ReactionProcessor("input_images_new", "output_results")
    processor.process_batch()
