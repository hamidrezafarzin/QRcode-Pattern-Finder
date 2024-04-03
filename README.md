# Finder Pattern Detector

This Python code implements a finder pattern detector used for detecting patterns in images, particularly suitable for applications like QR code detection.

## Table of Contents

1. [Overview](#overview)
2. [Usage](#usage)
    1. [Clone Repository](#clone-repository)
    2. [Installation](#installation)
    3. [Run the Code](#run-the-code)
    4. [Input Images](#input-images)
    5. [Output](#output)
3. [Example](#example)
4. [License](#license)

## Overview

The Finder Pattern Detector detects patterns within images, specifically designed for identifying finder patterns such as those used in QR codes. It performs image preprocessing, pattern detection, and draws rectangles around the detected finder patterns.

## Usage

### Clone Repository

To clone the repository, use the following command:

```bash
git clone https://github.com/hamidrezafarzin/QRcode-Pattern-Finder.git
```

### Installation

Before running the code, ensure you have Python installed on your system along with the necessary dependencies, including OpenCV (cv2) and NumPy (numpy). You can install the required dependencies using the provided requirements.txt file. Navigate to the project directory and execute the following command:
```bash 
pip install -r requirements.txt
```

### Run the Code:
    - Import the `FinderPatternDetector` class from the provided Python script.
    - Create an instance of the `FinderPatternDetector` class.
    - Specify the path of the input image.
    - Read the input image using the `read_image()` method.
    - Detect finder patterns in the image using the `detector()` method.
    - Optionally, create a result image with detected patterns drawn using the `create_result_image()` method.

4. **Input Images**: Provide images containing finder patterns as input. The code supports various image formats, including JPEG, PNG, etc.

5. **Output**:
    - The code generates a result image with detected patterns outlined in rectangles. The output image is saved with "_result" appended to the original image filename.

## Example

```python
from finder_pattern_detector import FinderPatternDetector

# Create an instance of FinderPatternDetector class
detector = FinderPatternDetector()

# Specify the path of the input image
path = "path/to/input/image.jpg"

# Read the input image
detector.read_image(path)

# Detect finder patterns in the image
detector.detector()

# Create a result image with detected patterns drawn
detector.create_result_image()
```

## License
This code is provided under the MIT License.
