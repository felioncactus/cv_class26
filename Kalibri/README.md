# Homework #3
# Kalibri

## Description
This project is a simple camera calibration and lens distortion correction program made with Python, OpenCV, and NumPy.

It can:
- load a prerecorded calibration video from a manually set file path
- detect a chessboard pattern from sampled video frames
- estimate camera intrinsic parameters
- estimate lens distortion coefficients
- compute reprojection error
- save calibration results
- undistort video frames using the estimated camera parameters
- show original and corrected results for visual comparison

## Functions
- Video loading using `cv2.VideoCapture`
- Chessboard corner detection using `cv2.findChessboardCorners`
- Corner refinement using `cv2.cornerSubPix`
- 3D checkerboard point generation using NumPy
- Camera calibration using `cv2.calibrateCamera`
- Reprojection error computation using `cv2.projectPoints`
- Lens distortion correction using `cv2.undistort`
- Optimal camera matrix estimation using `cv2.getOptimalNewCameraMatrix`
- Undistorted video saving using `cv2.VideoWriter`
- Preview window using `cv2.imshow`
- Calibration result saving using `np.savez`

## Algorithm
- Load the prerecorded chessboard video
- Sample frames at a fixed interval
- Convert each sampled frame to grayscale
- Detect chessboard inner corners
- Refine detected corner locations to subpixel accuracy
- Build corresponding 3D world points for the checkerboard
- Accumulate 2D image points and 3D object points from valid frames
- Estimate camera matrix and distortion coefficients with calibration
- Compute reprojection RMSE to evaluate calibration quality
- Undistort frames and compare the result with the original video

## Variables
- `VIDEO_PATH` : path to the input calibration video
- `PATTERN_SIZE` : checkerboard inner corner size `(10, 7)`
- `SQUARE_SIZE` : checker square size in millimeters
- `FRAME_STEP` : frame sampling interval
- `MIN_VALID_FRAMES` : minimum number of valid chessboard detections
- `SHOW_DETECTIONS` : `True` or `False`
- `SAVE_UNDISTORTED_VIDEO` : `True` or `False`
- `UNDISTORTED_VIDEO_PATH` : path to save the corrected video

## Calibration Result
![Kalibri Screenshot](https://github.com/felioncactus/cv_class26/blob/main/media/kalibri%20result.png?raw=true)
### Intrinsic Parameters
- `fx` : `392.11556377559646`
- `fy` : `393.91126500614456`
- `cx` : `421.3397406173853`
- `cy` : `236.55035803151193`

### RMSE
- `rmse` : `0.2419601499399567`

### Camera Matrix
```text
[[392.11556378   0.         421.33974062]
 [  0.         393.91126501 236.55035803]
 [  0.           0.           1.        ]]

### Distortion Coefficients
[ 0.04804938 -0.12412666 -0.00376381 -0.00086057  0.12046412]
