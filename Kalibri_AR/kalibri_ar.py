import math
from pathlib import Path

import cv2
import numpy as np

VIDEO_PATH = "sample.mp4"
CAR_IMAGE_PATH = "car.png"
OUTPUT_VIDEO_PATH = "ar_car_slow_turns.mp4"
CALIBRATION_CACHE_PATH = "camera_calibration_result.npz"

PATTERN_SIZE = (10, 7)   # checkerboard inner corners: columns, rows
SQUARE_SIZE = 25.0       # millimeters

FRAME_STEP_FOR_CALIBRATION = 10
MIN_VALID_FRAMES = 12

SHOW_PREVIEW = False
DRAW_AXES = True
DRAW_TRACK = False
DRAW_CORNERS = False
DRAW_INFO = True

# Car dimensions on the checkerboard plane.
CAR_LENGTH_CELLS = 2.10
CAR_WIDTH_CELLS = 1.05

# Slower motion than before.
PATH_SPEED_CELLS_PER_FRAME = 0.032

# Sprite heading correction.
# The provided car sprite points upward in image coordinates.
# This offset makes the car face the actual direction of travel.
SPRITE_FORWARD_OFFSET_RAD = math.pi / 2.0


def load_rgba_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to load image: {path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    elif image.shape[2] == 3:
        alpha = np.full((image.shape[0], image.shape[1], 1), 255, dtype=np.uint8)
        image = np.concatenate([image, alpha], axis=2)

    return image


def build_object_points(pattern_size, square_size):
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    return objp


def collect_calibration_points(video_path, pattern_size, square_size, frame_step):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )

    object_points = []
    image_points = []
    image_size = None
    objp = build_object_points(pattern_size, square_size)

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % frame_step != 0:
            frame_index += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if found:
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria,
            )
            object_points.append(objp.copy())
            image_points.append(corners)

        frame_index += 1

    cap.release()

    if image_size is None:
        raise RuntimeError("No frames were read from the video.")

    return object_points, image_points, image_size


def load_or_calibrate(video_path):
    cache_path = Path(CALIBRATION_CACHE_PATH)
    if cache_path.exists():
        data = np.load(str(cache_path))
        return data["camera_matrix"], data["dist_coeffs"]

    object_points, image_points, image_size = collect_calibration_points(
        video_path,
        PATTERN_SIZE,
        SQUARE_SIZE,
        FRAME_STEP_FOR_CALIBRATION,
    )

    if len(image_points) < MIN_VALID_FRAMES:
        raise RuntimeError(
            f"Not enough valid calibration detections: found {len(image_points)}, "
            f"need at least {MIN_VALID_FRAMES}."
        )

    _, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    np.savez(
        str(cache_path),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )

    return camera_matrix, dist_coeffs


def detect_pose(frame, camera_matrix, dist_coeffs):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )

    found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, flags)
    if not found:
        return None, None, None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        criteria,
    )

    objp = build_object_points(PATTERN_SIZE, SQUARE_SIZE)
    success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
    if not success:
        return None, None, None

    return rvec, tvec, corners


def project_points(points_3d, rvec, tvec, camera_matrix, dist_coeffs):
    projected, _ = cv2.projectPoints(
        np.asarray(points_3d, dtype=np.float32),
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    return projected.reshape(-1, 2)


def alpha_blend_rgba_onto_bgr(dst_bgr, src_rgba):
    alpha = src_rgba[:, :, 3:4].astype(np.float32) / 255.0
    src = src_rgba[:, :, :3].astype(np.float32)
    dst = dst_bgr.astype(np.float32)
    blended = src * alpha + dst * (1.0 - alpha)
    np.copyto(dst_bgr, np.clip(blended, 0, 255).astype(np.uint8))


def warp_rgba_to_quad(frame, rgba, quad_2d):
    h, w = rgba.shape[:2]
    src = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )
    dst = np.asarray(quad_2d, dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        rgba,
        H,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    alpha_blend_rgba_onto_bgr(frame, warped)


def rotate_sprite_keep_alpha(sprite_rgba, angle_deg):
    h, w = sprite_rgba.shape[:2]
    center = (w * 0.5, h * 0.5)

    rotation = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos_v = abs(rotation[0, 0])
    sin_v = abs(rotation[0, 1])

    new_w = int(math.ceil((h * sin_v) + (w * cos_v)))
    new_h = int(math.ceil((h * cos_v) + (w * sin_v)))

    rotation[0, 2] += (new_w * 0.5) - center[0]
    rotation[1, 2] += (new_h * 0.5) - center[1]

    rotated = cv2.warpAffine(
        sprite_rgba,
        rotation,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return rotated


def draw_axes(frame, rvec, tvec, camera_matrix, dist_coeffs, axis_len=70.0):
    axis = np.float32([
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, -axis_len],
    ])
    imgpts = project_points(axis, rvec, tvec, camera_matrix, dist_coeffs).astype(np.int32)
    o, x, y, z = imgpts
    cv2.line(frame, tuple(o), tuple(x), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.line(frame, tuple(o), tuple(y), (0, 255, 0), 3, cv2.LINE_AA)
    cv2.line(frame, tuple(o), tuple(z), (255, 0, 0), 3, cv2.LINE_AA)


class RoundedRectangleTrack:
    def __init__(self, xmin, ymin, xmax, ymax, radius):
        if xmax <= xmin or ymax <= ymin:
            raise ValueError("Invalid track bounds.")
        width = xmax - xmin
        height = ymax - ymin
        max_radius = 0.5 * min(width, height) - 1e-6
        self.r = max(0.01, min(radius, max_radius))
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.width = width
        self.height = height

        self.top_len = self.width - 2.0 * self.r
        self.right_len = self.height - 2.0 * self.r
        self.bottom_len = self.top_len
        self.left_len = self.right_len
        self.arc_len = 0.5 * math.pi * self.r

        self.segment_lengths = [
            self.top_len,
            self.arc_len,
            self.right_len,
            self.arc_len,
            self.bottom_len,
            self.arc_len,
            self.left_len,
            self.arc_len,
        ]
        self.total_len = float(sum(self.segment_lengths))

    def sample(self, s):
        s = float(s % self.total_len)
        x0, y0, x1, y1, r = self.xmin, self.ymin, self.xmax, self.ymax, self.r

        if s < self.top_len:
            u = 0.0 if self.top_len == 0 else s / self.top_len
            pos = np.array([x0 + r + u * (self.width - 2.0 * r), y0], dtype=np.float32)
            tangent = np.array([1.0, 0.0], dtype=np.float32)
            return pos, tangent
        s -= self.top_len

        if s < self.arc_len:
            angle = -0.5 * math.pi + s / r
            center = np.array([x1 - r, y0 + r], dtype=np.float32)
            pos = center + r * np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
            tangent = np.array([-math.sin(angle), math.cos(angle)], dtype=np.float32)
            return pos, tangent
        s -= self.arc_len

        if s < self.right_len:
            u = 0.0 if self.right_len == 0 else s / self.right_len
            pos = np.array([x1, y0 + r + u * (self.height - 2.0 * r)], dtype=np.float32)
            tangent = np.array([0.0, 1.0], dtype=np.float32)
            return pos, tangent
        s -= self.right_len

        if s < self.arc_len:
            angle = 0.0 + s / r
            center = np.array([x1 - r, y1 - r], dtype=np.float32)
            pos = center + r * np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
            tangent = np.array([-math.sin(angle), math.cos(angle)], dtype=np.float32)
            return pos, tangent
        s -= self.arc_len

        if s < self.bottom_len:
            u = 0.0 if self.bottom_len == 0 else s / self.bottom_len
            pos = np.array([x1 - r - u * (self.width - 2.0 * r), y1], dtype=np.float32)
            tangent = np.array([-1.0, 0.0], dtype=np.float32)
            return pos, tangent
        s -= self.bottom_len

        if s < self.arc_len:
            angle = 0.5 * math.pi + s / r
            center = np.array([x0 + r, y1 - r], dtype=np.float32)
            pos = center + r * np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
            tangent = np.array([-math.sin(angle), math.cos(angle)], dtype=np.float32)
            return pos, tangent
        s -= self.arc_len

        if s < self.left_len:
            u = 0.0 if self.left_len == 0 else s / self.left_len
            pos = np.array([x0, y1 - r - u * (self.height - 2.0 * r)], dtype=np.float32)
            tangent = np.array([0.0, -1.0], dtype=np.float32)
            return pos, tangent
        s -= self.left_len

        angle = math.pi + s / r
        center = np.array([x0 + r, y0 + r], dtype=np.float32)
        pos = center + r * np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        tangent = np.array([-math.sin(angle), math.cos(angle)], dtype=np.float32)
        return pos, tangent

    def polyline(self, samples=240):
        pts = []
        for i in range(samples):
            p, _ = self.sample(self.total_len * i / samples)
            pts.append(p)
        return np.asarray(pts, dtype=np.float32)


def create_track():
    cols, rows = PATTERN_SIZE
    margin = 0.75
    xmin = margin
    ymin = margin
    xmax = (cols - 1.0) - margin
    ymax = (rows - 1.0) - margin
    radius = 1.20
    return RoundedRectangleTrack(xmin, ymin, xmax, ymax, radius)


def build_car_quad_world(center_xy_mm, heading_rad, car_length_mm, car_width_mm, z_lift_mm=0.0):
    half_length = 0.5 * car_length_mm
    half_width = 0.5 * car_width_mm

    # Local sprite frame:
    # - x = across the car width
    # - y = along the car length
    # The front of the sprite is at negative y because the car image points upward.
    local = np.array([
        [-half_width, -half_length, 0.0],   # top-left
        [ half_width, -half_length, 0.0],   # top-right
        [ half_width,  half_length, 0.0],   # bottom-right
        [-half_width,  half_length, 0.0],   # bottom-left
    ], dtype=np.float32)

    cos_a = math.cos(heading_rad)
    sin_a = math.sin(heading_rad)
    rotation = np.array([
        [cos_a, -sin_a, 0.0],
        [sin_a,  cos_a, 0.0],
        [0.0,    0.0,   1.0],
    ], dtype=np.float32)

    world = (rotation @ local.T).T
    world[:, 0] += center_xy_mm[0]
    world[:, 1] += center_xy_mm[1]
    world[:, 2] += -z_lift_mm
    return world


def draw_track(frame, track, rvec, tvec, camera_matrix, dist_coeffs):
    pts_cells = track.polyline(samples=240)
    pts_world = np.column_stack([
        pts_cells[:, 0] * SQUARE_SIZE,
        pts_cells[:, 1] * SQUARE_SIZE,
        np.zeros(len(pts_cells), dtype=np.float32),
    ])
    pts_img = project_points(pts_world, rvec, tvec, camera_matrix, dist_coeffs).astype(np.int32)
    cv2.polylines(frame, [pts_img], True, (0, 220, 255), 2, cv2.LINE_AA)


def add_shadow(frame, quad_2d):
    shadow = frame.copy()
    pts = np.asarray(quad_2d, dtype=np.int32)
    cv2.fillConvexPoly(shadow, pts, (20, 20, 20), lineType=cv2.LINE_AA)
    cv2.addWeighted(shadow, 0.18, frame, 0.82, 0.0, dst=frame)


def main():
    video_path = Path(VIDEO_PATH)
    car_path = Path(CAR_IMAGE_PATH)

    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")
    if not car_path.exists():
        raise FileNotFoundError(f"Missing car image: {car_path}")

    camera_matrix, dist_coeffs = load_or_calibrate(str(video_path))
    car_rgba = load_rgba_image(str(car_path))
    track = create_track()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    car_length_mm = CAR_LENGTH_CELLS * SQUARE_SIZE
    car_width_mm = CAR_WIDTH_CELLS * SQUARE_SIZE

    progress = 0.0
    last_rvec = None
    last_tvec = None

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rvec, tvec, corners = detect_pose(frame, camera_matrix, dist_coeffs)
        if rvec is not None:
            last_rvec = rvec
            last_tvec = tvec
        elif last_rvec is not None:
            rvec = last_rvec
            tvec = last_tvec

        if rvec is not None:
            center_cells, tangent = track.sample(progress)
            tangent = tangent / max(np.linalg.norm(tangent), 1e-8)

            # Because the sprite points upward, add the offset so heading matches travel.
            heading_rad = math.atan2(float(tangent[1]), float(tangent[0])) + SPRITE_FORWARD_OFFSET_RAD

            center_mm = center_cells * SQUARE_SIZE
            car_quad_world = build_car_quad_world(
                center_mm,
                heading_rad,
                car_length_mm,
                car_width_mm,
                z_lift_mm=1.0,
            )
            car_quad_2d = project_points(car_quad_world, rvec, tvec, camera_matrix, dist_coeffs)

            # Rotate the sprite in its own canvas first, then project as a rigid rectangle.
            # This keeps turns visually correct and avoids the squashed look.
            sprite_rotation_deg = -math.degrees(heading_rad - SPRITE_FORWARD_OFFSET_RAD)
            rotated_car = rotate_sprite_keep_alpha(car_rgba, sprite_rotation_deg)

            shadow_world = build_car_quad_world(
                center_mm + np.array([1.5, 1.5], dtype=np.float32),
                heading_rad,
                car_length_mm * 1.04,
                car_width_mm * 1.04,
                z_lift_mm=0.0,
            )
            shadow_2d = project_points(shadow_world, rvec, tvec, camera_matrix, dist_coeffs)

            add_shadow(frame, shadow_2d)
            warp_rgba_to_quad(frame, rotated_car, car_quad_2d)

            if DRAW_TRACK:
                draw_track(frame, track, rvec, tvec, camera_matrix, dist_coeffs)

            if DRAW_AXES:
                draw_axes(frame, rvec, tvec, camera_matrix, dist_coeffs)

            if DRAW_CORNERS and corners is not None:
                cv2.drawChessboardCorners(frame, PATTERN_SIZE, corners, True)

            if DRAW_INFO:
                cv2.putText(
                    frame,
                    f"car speed: {PATH_SPEED_CELLS_PER_FRAME:.3f} cells/frame",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "turning aligned to motion direction",
                    (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            progress += PATH_SPEED_CELLS_PER_FRAME
        else:
            cv2.putText(
                frame,
                "checkerboard not found",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)

        if SHOW_PREVIEW:
            cv2.imshow("AR Car", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

        frame_index += 1

    cap.release()
    writer.release()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

    print(f"Saved: {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()
