# Reaction Processor

<!-- ENV: reaction-extraction -->

## Overview

The **Reaction Processor** extracts reaction regions from input images, detects reaction arrows, and splits the image into reactants and products.

## Installation

Ensure you have Python (3.10+) installed, then install the required dependencies:

```sh
pip install -r requirements.txt
```

## Usage

Run the script with the required `--

input` argument and an optional `--output` argument:

```sh
python reaction_bisector.py --input input_images [--output output_results]
```

### Arguments

- `--input` (**required**) : Path to the input folder containing reaction images.
- `--output` (**optional**) : Path to the output folder (default: `output_results`).

## Example Usage

### Default output folder

```sh
python reaction_bisector.py --input input_images
```

### Custom output folder

```sh
python reaction_bisector.py --input input_images --output my_output_folder
```

## Output Structure

After processing, the output directory will contain:

- Extracted reaction regions
- Separated reactants and products
- Segmented chemical structures

## YOLO training

- Change diretory to reaction_arrow_detection_yolo and run train.py to train or infer.py for inferencing.
