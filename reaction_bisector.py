import argparse
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from decimer_segmentation import segment_chemical_structures


class ReactionProcessor:
    def __init__(self, input_folder, output_folder="out"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def load_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")
        return cv2.imread(image_path)

    def extract_reaction_region(
        self, image_path
    ):  # TODO: Replace this logic with a more robust method, like extract entire reaction region using YOLO, instead of opencv
        image = self.load_image(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
        )
        kernel = np.ones((3, 3), np.uint8)
        enhanced = cv2.dilate(enhanced, kernel, iterations=5)
        edges = cv2.Canny(enhanced, 50, 150, apertureSize=5)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=80, minLineLength=150, maxLineGap=15
        )

        horizontal_lines = (
            sorted(
                set(y1 for line in lines for x1, y1, x2, y2 in line if abs(y1 - y2) < 5)
            )
            if lines is not None
            else []
        )
        bottom_line_y = (
            horizontal_lines[1]
            if len(horizontal_lines) >= 2
            else (horizontal_lines[0] if horizontal_lines else None)
        )

        if bottom_line_y is None:
            print("No valid horizontal line detected!")
            return None

        reaction_region = image[: bottom_line_y + 2, :]
        output_path = os.path.join(
            self.output_folder,
            os.path.basename(image_path).replace(".png", "_reaction.png"),
        )
        cv2.imwrite(output_path, reaction_region)
        print(f"Extracted reaction region saved to {output_path}")
        return output_path

    def infer_coordinates(self, image_path):
        model = YOLO("reaction_region_yolo/runs/detect/train2/weights/best.pt")
        results = model(image_path, conf=0.1)
        return results[0].boxes.xywh.mean(dim=0)

    def detect_arrow_and_split(self, image_path):
        img = cv2.imread(image_path)
        try:
            x, _, w, _ = self.infer_coordinates(image_path)
            x = math.floor(x.item())
            w_half = math.ceil(w.item() / 2)
            left_part = img[:, : x - w_half]
            right_part = img[:, x + w_half :]

            reactants_path = image_path.replace("_reaction.png", "_reactants.png")
            products_path = image_path.replace("_reaction.png", "_products.png")
            cv2.imwrite(reactants_path, left_part)
            cv2.imwrite(products_path, right_part)
        except Exception as e:
            print(f"Error during arrow detection and splitting: {e}")
            reactants_path, products_path = None, None

        return reactants_path, products_path

    def apply_decimer_segmentation(self, image_path, output_folder):
        os.makedirs(self.output_folder, exist_ok=True)
        image = self.load_image(image_path)
        segmented_images, _ = segment_chemical_structures(image, output_folder)
        for idx, img in enumerate(segmented_images):
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            plt.imsave(
                os.path.join(output_folder, f"segmented_{idx}.png"), img, cmap="gray"
            )
        print(f"Segmented structures saved in {output_folder}")
        return segmented_images

    def process_batch(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        for file in os.listdir(self.input_folder):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                input_path = os.path.join(self.input_folder, file)
                base_name = os.path.splitext(file)[0]

                extracted_path = self.extract_reaction_region(input_path)
                if extracted_path:
                    reactants_path, products_path = self.detect_arrow_and_split(
                        extracted_path
                    )
                    if reactants_path and products_path:
                        self.apply_decimer_segmentation(
                            reactants_path,
                            os.path.join(
                                self.output_folder, f"{base_name}_reactants_segments"
                            ),
                        )
                        self.apply_decimer_segmentation(
                            products_path,
                            os.path.join(
                                self.output_folder, f"{base_name}_products_segments"
                            ),
                        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process reaction images.")
    parser.add_argument(
        "--input", required=True, help="Path to the input folder containing images."
    )
    parser.add_argument(
        "--output",
        nargs="?",
        default="out",
        help="Path to the output folder (default: out).",
    )
    args = parser.parse_args()

    processor = ReactionProcessor(args.input, args.output)
    processor.process_batch()
