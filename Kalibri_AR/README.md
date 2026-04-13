# Homework #4
# Kalibiri AR

## Description
This project is a simple augmented reality camera pose estimation program made with Python, OpenCV, and NumPy.

It can:
- load a prerecorded checkerboard video from a manually set file path
- use previously calibrated camera parameters
- detect a chessboard pattern in video frames
- estimate camera pose for each valid frame
- place a 2D car image onto the checkerboard plane as an AR object
- animate the car moving around the table surface
- rotate the car correctly while turning left or right
- render the final AR result as a processed video

## Functions
- Video loading using `cv2.VideoCapture`
- Chessboard corner detection using `cv2.findChessboardCorners`
- Corner refinement using `cv2.cornerSubPix`
- Camera pose estimation using `cv2.solvePnP`
- 3D checkerboard point generation using NumPy
- 3D point projection using `cv2.projectPoints`
- Perspective warping using `cv2.getPerspectiveTransform`
- AR image overlay using `cv2.warpPerspective`
- Frame rendering and preview using `cv2.imshow`
- Output video saving using `cv2.VideoWriter`

## Algorithm
- Load the prerecorded checkerboard video
- Load the car image with transparency
- Define checkerboard 3D object points
- Detect checkerboard inner corners in each frame
- Refine detected corner positions to subpixel accuracy
- Estimate the camera pose using the calibrated camera matrix and distortion coefficients
- Define a path on the checkerboard plane for the AR car
- Move the car along that path with a slower animation speed
- Change the car orientation according to the current movement direction
- Warp the car image onto the board without incorrect squishing during turns
- Blend the warped car into the frame
- Save the final AR animation as an output video

## Variables
- `VIDEO_PATH` : path to the input checkerboard video
- `CAR_IMAGE_PATH` : path to the AR car image
- `PATTERN_SIZE` : checkerboard inner corner size `(10, 7)`
- `SQUARE_SIZE` : checker square size in millimeters
- `CAMERA_MATRIX` : intrinsic camera matrix from calibration
- `DIST_COEFFS` : distortion coefficients from calibration
- `CAR_SCALE` : size of the rendered car on the checkerboard
- `MOVE_SPEED` : movement speed of the car
- `TURN_SMOOTHING` : controls smoother direction changes while turning
- `OUTPUT_VIDEO_PATH` : path to save the AR result video
- `SHOW_PREVIEW` : `True` or `False`