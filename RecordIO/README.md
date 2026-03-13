# Homework #1
# RecordIO (Video Recorder)
![RecordIO Screenshot](https://raw.githubusercontent.com/felioncactus/cv_class26/main/media/recordio_screenshot.png)
## Description
This project is a simple Video Recorder made with Python and OpenCV.

It can:
- show live camera video
- switch between Preview mode and Record mode
- save recorded video as an MP4 file
- show a recording indicator on the screen
- apply different filters using keyboard keys

## Functions
- Live camera preview using `cv.VideoCapture`
- Video recording using `cv.VideoWriter`
- Preview / Record mode
- Recording indicator (`REC` + red circle)
- Non-mirrored camera view
- Filters with keys `1 2 3 4 5`
- FPS display
- Output filename display
- Exit with `ESC`

## Filters
- `0` : Normal
- `1` : Grayscale
- `2` : Blur
- `3` : Edge
- `4` : Brightness
- `5` : Contrast

## How to use it

### 1. Install OpenCV
```bash
pip install opencv-python
