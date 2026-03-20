# Homework #2
# Cartoon Rendering

## Description
This project is a simple Cartoon Rendering program made with Python, OpenCV, and NumPy.

It can:
- load an image from a manually set file path
- convert the image into a cartoon-style image
- smooth colors while keeping strong edges
- reduce the number of colors for a cartoon effect
- save the output image
- show the original image and cartoon result in a preview window

## Functions
- Image loading using `cv2.imread`
- Image saving using `cv2.imwrite`
- Manual input/output path setting with variables
- Edge extraction using grayscale + median blur + adaptive threshold
- Color smoothing using `cv2.bilateralFilter`
- Color quantization using `cv2.kmeans`
- Cartoon rendering by combining flat colors with edge mask
- Preview window using `cv2.imshow`
- Resize for large images
- Error handling for invalid image path

## Algorithm
- Convert the input image to grayscale
- Apply median blur to reduce noise
- Detect edges using adaptive threshold
- Smooth the original image with bilateral filtering
- Reduce the number of colors using k-means clustering
- Combine the simplified color image with the edge mask
- Save and optionally display the final cartoon image

## Variables
- `INPUT_IMAGE_PATH` : path to the input image
- `OUTPUT_IMAGE_PATH` : path to save the result image
- `SHOW_PREVIEW` : `True` or `False`
- `MAX_SIDE` : maximum image size before resizing
- `K_COLORS` : number of colors used in quantization

## Good Result
![Kartoonika Screenshot](https://github.com/felioncactus/cv_class26/blob/main/Kartoonika/cartoon_BADoutput.png)
## Bad Result
![Kartoonika Screenshot](https://github.com/felioncactus/cv_class26/blob/main/Kartoonika/cartoon_GOODoutput.png)

## How to use it

### 1. Install OpenCV and NumPy
```bash
pip install opencv-python numpy
