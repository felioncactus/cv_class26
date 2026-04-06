import cv2
import numpy as np

VIDEO_PATH = r"./sample.mp4"

# From your checkerboard image:
# "10x7 vertices" => 10 columns x 7 rows of INNER corners
PATTERN_SIZE = (10, 7)

# 25 mm from the checkerboard generator screenshot.
# For camera intrinsics, the exact unit is not critical as long as it is consistent.
SQUARE_SIZE = 25.0

# Sampling interval for video frames
FRAME_STEP = 10

# Minimum number of successful detections required
MIN_VALID_FRAMES = 12

# Visualization / output
SHOW_DETECTIONS = True
SAVE_UNDISTORTED_VIDEO = True
UNDISTORTED_VIDEO_PATH = "undistorted_output.mp4"


def build_object_points(pattern_size, square_size):
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = grid * square_size
    return objp


def collect_calibration_points(video_path, pattern_size, square_size, frame_step=10, show=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    objp = build_object_points(pattern_size, square_size)
    object_points = []
    image_points = []

    frame_index = 0
    valid_count = 0
    image_size = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % frame_step != 0:
            frame_index += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])

        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK
        )

        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        vis = frame.copy()

        if found:
            corners_refined = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                criteria,
            )

            object_points.append(objp.copy())
            image_points.append(corners_refined)
            valid_count += 1

            if show:
                cv2.drawChessboardCorners(vis, pattern_size, corners_refined, found)
                cv2.putText(
                    vis,
                    f"Detected frames: {valid_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
        else:
            if show:
                cv2.putText(
                    vis,
                    "Chessboard not found",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        if show:
            cv2.imshow("Calibration Detection", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

        frame_index += 1

    cap.release()
    if show:
        cv2.destroyAllWindows()

    if image_size is None:
        raise RuntimeError("No frames were read from the video.")

    return object_points, image_points, image_size


def calibrate_camera(object_points, image_points, image_size):
    if len(object_points) == 0:
        raise RuntimeError("No valid chessboard detections found.")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    return ret, camera_matrix, dist_coeffs, rvecs, tvecs


def compute_mean_reprojection_error(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    total_error = 0.0
    total_points = 0

    for objp, imgp, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(imgp, projected, cv2.NORM_L2)
        total_error += error * error
        total_points += len(objp)

    rmse = np.sqrt(total_error / total_points) if total_points > 0 else 0.0
    return rmse


def undistort_video(input_path, output_path, camera_matrix, dist_coeffs):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (width, height),
        1,
        (width, height),
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

        x, y, w, h = roi
        if w > 0 and h > 0:
            cropped = undistorted[y:y + h, x:x + w]
            result = np.zeros_like(frame)
            resized = cv2.resize(cropped, (width, height))
            result[:] = resized
        else:
            result = undistorted

        writer.write(result)

        cv2.imshow("Undistorted Video", result)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


def show_before_after(input_path, camera_matrix, dist_coeffs):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_path}")

    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to read first frame from the video.")

    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (w, h),
        1,
        (w, h),
    )

    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted = undistorted[y:y + rh, x:x + rw]
        undistorted = cv2.resize(undistorted, (w, h))

    combined = np.hstack([frame, undistorted])

    while True:
        cv2.imshow("Original (left) | Undistorted (right)", combined)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q"), ord(" ")):
            break

    cv2.destroyAllWindows()


def main():
    object_points, image_points, image_size = collect_calibration_points(
        VIDEO_PATH,
        PATTERN_SIZE,
        SQUARE_SIZE,
        frame_step=FRAME_STEP,
        show=SHOW_DETECTIONS,
    )

    print(f"Valid chessboard detections: {len(image_points)}")

    if len(image_points) < MIN_VALID_FRAMES:
        raise RuntimeError(
            f"Not enough valid detections for stable calibration. "
            f"Found {len(image_points)}, need at least {MIN_VALID_FRAMES}."
        )

    reproj_error, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(
        object_points,
        image_points,
        image_size,
    )

    rmse = compute_mean_reprojection_error(
        object_points,
        image_points,
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs,
    )

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    print("\n=== Calibration Result ===")
    print(f"fx   = {fx}")
    print(f"fy   = {fy}")
    print(f"cx   = {cx}")
    print(f"cy   = {cy}")
    print(f"rmse = {rmse}")

    print("\nCamera Matrix:")
    print(camera_matrix)

    print("\nDistortion Coefficients:")
    print(dist_coeffs.ravel())

    np.savez(
        "camera_calibration_result.npz",
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        rmse=rmse,
    )

    show_before_after(VIDEO_PATH, camera_matrix, dist_coeffs)

    if SAVE_UNDISTORTED_VIDEO:
        undistort_video(
            VIDEO_PATH,
            UNDISTORTED_VIDEO_PATH,
            camera_matrix,
            dist_coeffs,
        )
        print(f"\nSaved undistorted video to: {UNDISTORTED_VIDEO_PATH}")


if __name__ == "__main__":
    main()