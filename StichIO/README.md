# Homework #5
# StichIO

![Kalibri AR Screenshot](https://github.com/felioncactus/cv_class26/blob/main/StichIO/result_page.png?raw=true)

## Description
This project is an automatic image stitching program made with Python, OpenCV, and NumPy.

It can:
- load 3 or more overlapping images from the same directory
- detect and describe feature points automatically
- match features between overlapping images
- estimate homography transformations using RANSAC
- align multiple images into one common coordinate system
- generate a large panorama image without using `cv2.Stitcher`
- support both planar and cylindrical projection
- blend overlapping regions smoothly using feather blending
- crop unnecessary black borders from the final result
- save the stitched panorama as an output image

## Functions
- Image loading using `cv2.imread`
- Directory file collection using `glob.glob`
- Feature detection using `cv2.SIFT_create` or `cv2.ORB_create`
- Feature description using SIFT or ORB descriptors
- Feature matching using `cv2.BFMatcher`
- Homography estimation using `cv2.findHomography`
- Perspective warping using `cv2.warpPerspective`
- Cylindrical warping using `cv2.remap`
- Corner transformation using `cv2.perspectiveTransform`
- Distance-based feather blending using `cv2.distanceTransform`
- Black border cropping using `cv2.boundingRect`
- Output panorama saving using `cv2.imwrite`

## Algorithm
- Load all input images from the target directory
- Optionally apply cylindrical projection to each image
- Convert each image to grayscale
- Detect feature points and compute descriptors
- Match descriptors between image pairs
- Estimate homographies using RANSAC to reject outliers
- Build pairwise image connection scores from inlier counts
- Choose a center image automatically
- Estimate the best stitching order from matching scores
- Compute global transforms for all images relative to the center image
- Warp all images into one panorama canvas
- Blend overlapping areas smoothly with feather blending
- Crop black outer borders from the stitched result
- Save the final panorama image

## Variables
- `input_dir` : directory containing the input images
- `patterns` : file patterns used to collect image files such as `*.png`, `*.jpg`, `*.jpeg`, `*.bmp`
- `output` : output filename for the stitched panorama
- `projection` : projection mode, either `planar` or `cylindrical`
- `show_order` : prints the estimated stitching order when enabled
- `ImageFeatures` : data structure storing image, grayscale image, keypoints, descriptors, and optional cylindrical results
- `focal_length` : estimated focal length used for cylindrical projection
- `scores` : pairwise image matching inlier score matrix
- `homographies` : pairwise homography transformation matrix table
- `order` : estimated image stitching order
- `transforms` : global projective transforms for each image
- `panorama` : final stitched output image

## Example Run
```bash
python stichio.py --input_dir . --output panorama.png --projection cylindrical --show_order
