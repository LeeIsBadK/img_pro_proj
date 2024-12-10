# 2301368 Image processing term project


## Tools

* OpenCV
* numpy

## Overview
The pipeline processes grayscale medical X-ray images to detect and segment vertebrae regions. The core steps in the pipeline include:
1. **Preprocessing**: Masking the image in 30%-70% of x coordinate, applying power rules, and performing segmentation.
2. **Thresholding**: Segmenting the image into regions using height-based segmentation.
3. **Connected Components**: Detecting and processing connected components in the thresholded image.
4. **Region Growing**: Enhancing regions of interest using region growing techniques.
5. **Colormap Application**: Applying distinct colormaps to the segmented regions and visualizing the result.
6. **Saving Results**: Saving various stages of the processed image to output directories for further analysis.

The pipeline outputs images at each step, including masked images, segmented regions, connected components, and final overlays.

## Requirements
To run this project, ensure you have the following Python packages installed:

- numpy
- opencv-python
- matplotlib (optional, for colormap generation)