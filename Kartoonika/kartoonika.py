#!/usr/bin/env python3
import cv2
import numpy as np

# Set image paths manually here
INPUT_IMAGE_PATH = "media/gerodot.png"
OUTPUT_IMAGE_PATH = "cartoon_output.png"

# Optional settings
SHOW_PREVIEW = True
MAX_SIDE = 1600
K_COLORS = 8
BILATERAL_PASSES = 2
BILATERAL_DIAMETER = 9
SIGMA_COLOR = 120
SIGMA_SPACE = 120
MEDIAN_BLUR_SIZE = 7
ADAPTIVE_BLOCK_SIZE = 9
ADAPTIVE_C = 2


def resize_if_needed(image, max_side=1600):
    height, width = image.shape[:2]
    longest = max(height, width)

    if longest <= max_side:
        return image

    scale = max_side / float(longest)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def make_edge_mask(image_bgr, blur_ksize=7, block_size=9, c_value=2):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, blur_ksize)

    if block_size % 2 == 0:
        block_size += 1
    if block_size < 3:
        block_size = 3

    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_value,
    )
    return edges


def quantize_colors_kmeans(image_bgr, k=8, attempts=10):
    pixels = image_bgr.reshape((-1, 3)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.2,
    )

    _, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        attempts,
        cv2.KMEANS_PP_CENTERS,
    )

    centers = np.uint8(centers)
    quantized = centers[labels.flatten()].reshape(image_bgr.shape)
    return quantized


def cartoonize(image_bgr):
    filtered = image_bgr.copy()

    for _ in range(max(1, BILATERAL_PASSES)):
        filtered = cv2.bilateralFilter(
            filtered,
            d=BILATERAL_DIAMETER,
            sigmaColor=SIGMA_COLOR,
            sigmaSpace=SIGMA_SPACE,
        )

    flat_colors = quantize_colors_kmeans(filtered, k=K_COLORS)
    edges = make_edge_mask(
        image_bgr,
        blur_ksize=MEDIAN_BLUR_SIZE,
        block_size=ADAPTIVE_BLOCK_SIZE,
        c_value=ADAPTIVE_C,
    )

    cartoon = cv2.bitwise_and(flat_colors, flat_colors, mask=edges)
    return cartoon


def main():
    image = cv2.imread(INPUT_IMAGE_PATH)

    if image is None:
        raise FileNotFoundError("Failed to read image: " + INPUT_IMAGE_PATH)

    image = resize_if_needed(image, max_side=MAX_SIDE)
    cartoon = cartoonize(image)

    success = cv2.imwrite(OUTPUT_IMAGE_PATH, cartoon)
    if not success:
        raise RuntimeError("Failed to save output image: " + OUTPUT_IMAGE_PATH)

    print("Saved cartoon image to:", OUTPUT_IMAGE_PATH)

    if SHOW_PREVIEW:
        combined = np.hstack([image, cartoon])
        cv2.imshow("Original / Cartoon", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
