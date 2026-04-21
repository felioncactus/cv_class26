#!/usr/bin/env python3
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


INPUT_IMAGE = "image.png"
OUTPUT_PANORAMA = "stitched_panorama.png"
OUTPUT_FIGURE = "result_page.png"


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


def crop_image_into_three_parts(image: np.ndarray):
    h, w = image.shape[:2]

    overlap = int(w * 0.15)
    crop_w = int(w * 0.45)

    x1 = 0
    x2 = crop_w - overlap
    x3 = w - crop_w

    part1 = image[:, x1:x1 + crop_w].copy()
    part2 = image[:, x2:x2 + crop_w].copy()
    part3 = image[:, x3:x3 + crop_w].copy()

    return [part1, part2, part3]


def save_crops(parts):
    for i, part in enumerate(parts, start=1):
        cv2.imwrite(f"part{i}.png", part)


def detect_and_match_features(img1: np.ndarray, img2: np.ndarray):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=4000)
        norm = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(nfeatures=6000)
        norm = cv2.NORM_HAMMING

    k1, d1 = detector.detectAndCompute(gray1, None)
    k2, d2 = detector.detectAndCompute(gray2, None)

    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        raise RuntimeError("Not enough features detected.")

    matcher = cv2.BFMatcher(norm, crossCheck=False)
    knn = matcher.knnMatch(d1, d2, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        raise RuntimeError("Not enough good matches.")

    pts1 = np.float32([k1[m.queryIdx].pt for m in good])
    pts2 = np.float32([k2[m.trainIdx].pt for m in good])

    return pts1, pts2


def estimate_translation(img_left: np.ndarray, img_right: np.ndarray):
    pts_left, pts_right = detect_and_match_features(img_left, img_right)

    shifts = pts_left - pts_right
    dx, dy = np.median(shifts, axis=0)

    return float(dx), float(dy)


def warp_image_translation(image: np.ndarray, dx: float, dy: float, out_shape):
    h_out, w_out = out_shape
    H = np.array([
        [1.0, 0.0, dx],
        [0.0, 1.0, dy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    warped = cv2.warpPerspective(
        image,
        H,
        (w_out, h_out),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(
        mask,
        H,
        (w_out, h_out),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return warped, warped_mask


def feather_blend(base, base_mask, overlay, overlay_mask):
    base_f = base.astype(np.float32)
    overlay_f = overlay.astype(np.float32)

    base_w = cv2.distanceTransform((base_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    overlay_w = cv2.distanceTransform((overlay_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)

    base_w = np.where(base_mask > 0, np.maximum(base_w, 1.0), 0.0).astype(np.float32)
    overlay_w = np.where(overlay_mask > 0, np.maximum(overlay_w, 1.0), 0.0).astype(np.float32)

    weight_sum = base_w + overlay_w
    weight_sum = np.where(weight_sum == 0, 1.0, weight_sum)

    blended = (
        base_f * base_w[:, :, None] +
        overlay_f * overlay_w[:, :, None]
    ) / weight_sum[:, :, None]

    blended_mask = np.where((base_mask > 0) | (overlay_mask > 0), 255, 0).astype(np.uint8)
    return blended.astype(np.uint8), blended_mask


def crop_non_black(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero((gray > 0).astype(np.uint8))
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y + h, x:x + w]


def stitch_translation(parts):
    img1, img2, img3 = parts

    dx12, dy12 = estimate_translation(img1, img2)
    dx23, dy23 = estimate_translation(img2, img3)

    pos1 = np.array([0.0, 0.0], dtype=np.float32)
    pos2 = np.array([-dx12, -dy12], dtype=np.float32)
    pos3 = pos2 + np.array([-dx23, -dy23], dtype=np.float32)

    positions = np.stack([pos1, pos2, pos3], axis=0)

    min_x = int(np.floor(np.min(positions[:, 0])))
    min_y = int(np.floor(np.min(positions[:, 1])))

    shifted_positions = positions - np.array([min_x, min_y], dtype=np.float32)

    heights = [p.shape[0] for p in parts]
    widths = [p.shape[1] for p in parts]

    max_x = 0
    max_y = 0
    for (x, y), w, h in zip(shifted_positions, widths, heights):
        max_x = max(max_x, int(np.ceil(x + w)))
        max_y = max(max_y, int(np.ceil(y + h)))

    canvas = np.zeros((max_y, max_x, 3), dtype=np.uint8)
    canvas_mask = np.zeros((max_y, max_x), dtype=np.uint8)

    for image, (x, y) in zip(parts, shifted_positions):
        warped, warped_mask = warp_image_translation(image, float(x), float(y), (max_y, max_x))
        canvas, canvas_mask = feather_blend(canvas, canvas_mask, warped, warped_mask)

    return crop_non_black(canvas)


def make_result_page(original, parts, panorama, output_path):
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    parts_rgb = [cv2.cvtColor(p, cv2.COLOR_BGR2RGB) for p in parts]
    panorama_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(18, 10))

    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(original_rgb)
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(parts_rgb[0])
    ax2.set_title("Cropped Part 1")
    ax2.axis("off")

    ax3 = plt.subplot(2, 3, 3)
    ax3.imshow(parts_rgb[1])
    ax3.set_title("Cropped Part 2")
    ax3.axis("off")

    ax4 = plt.subplot(2, 3, 4)
    ax4.imshow(parts_rgb[2])
    ax4.set_title("Cropped Part 3")
    ax4.axis("off")

    ax5 = plt.subplot(2, 1, 2)
    ax5.imshow(panorama_rgb)
    ax5.set_title("Stitched Panorama")
    ax5.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()


def main():
    image = load_image(INPUT_IMAGE)

    parts = crop_image_into_three_parts(image)
    save_crops(parts)

    panorama = stitch_translation(parts)
    cv2.imwrite(OUTPUT_PANORAMA, panorama)

    make_result_page(image, parts, panorama, OUTPUT_FIGURE)

    print(f"Saved cropped images: part1.png, part2.png, part3.png")
    print(f"Saved stitched panorama: {OUTPUT_PANORAMA}")
    print(f"Saved result page: {OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()