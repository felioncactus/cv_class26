import cv2 as cv
import time
from datetime import datetime
from pathlib import Path


def create_output_path() -> str:
    output_dir = Path(__file__).resolve().parent / "recordings"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"recorded_video_{timestamp}.avi"
    return str(output_path)


def apply_filter(frame, filter_mode):
    if filter_mode == 0:
        return frame
    elif filter_mode == 1:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        return cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    elif filter_mode == 2:
        return cv.GaussianBlur(frame, (15, 15), 0)
    elif filter_mode == 3:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        edges = cv.Canny(gray, 100, 200)
        return cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    elif filter_mode == 4:
        return cv.convertScaleAbs(frame, alpha=1.0, beta=50)
    elif filter_mode == 5:
        return cv.convertScaleAbs(frame, alpha=1.5, beta=0)
    return frame


def get_filter_name(filter_mode):
    names = {
        0: "Normal",
        1: "Grayscale",
        2: "Blur",
        3: "Edge",
        4: "Bright",
        5: "Contrast"
    }
    return names.get(filter_mode, "Normal")


def main():
    camera_source = 0
    cap = cv.VideoCapture(camera_source)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        cap.release()
        return

    frame = cv.flip(frame, 1)
    height, width = frame.shape[:2]

    camera_fps = cap.get(cv.CAP_PROP_FPS)
    if camera_fps <= 1 or camera_fps > 120:
        camera_fps = 20.0

    fourcc = cv.VideoWriter_fourcc(*"XVID")
    writer = None
    output_filename = "Not recording"

    is_recording = False
    filter_mode = 0
    prev_time = time.time()
    record_start_time = None

    print("======================================")
    print("Video Recorder started")
    print("Space : Toggle Preview / Record")
    print("0     : Normal filter")
    print("1     : Grayscale filter")
    print("2     : Blur filter")
    print("3     : Edge filter")
    print("4     : Brightness filter")
    print("5     : Contrast filter")
    print("ESC   : Exit program")
    print("======================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Failed to read frame.")
            break

        frame = cv.flip(frame, 1)
        processed_frame = apply_filter(frame, filter_mode)
        display_frame = processed_frame.copy()

        current_time = time.time()
        elapsed = current_time - prev_time
        prev_time = current_time

        fps_text_value = 0.0
        if elapsed > 0:
            fps_text_value = 1.0 / elapsed

        if is_recording and writer is not None:
            writer.write(processed_frame)

            cv.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
            cv.putText(
                display_frame,
                "REC",
                (50, 38),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            if record_start_time is not None:
                recording_seconds = int(time.time() - record_start_time)
                hours = recording_seconds // 3600
                minutes = (recording_seconds % 3600) // 60
                seconds = recording_seconds % 60
                timer_text = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                cv.putText(
                    display_frame,
                    f"Time: {timer_text}",
                    (20, 80),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

        mode_text = "Mode: RECORD" if is_recording else "Mode: PREVIEW"
        mode_color = (0, 0, 255) if is_recording else (0, 255, 0)

        cv.putText(
            display_frame,
            mode_text,
            (20, height - 120),
            cv.FONT_HERSHEY_SIMPLEX,
            0.8,
            mode_color,
            2
        )

        cv.putText(
            display_frame,
            f"File: {output_filename}",
            (20, height - 90),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv.putText(
            display_frame,
            f"FPS: {fps_text_value:.1f}",
            (20, height - 60),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv.putText(
            display_frame,
            f"Filter: {get_filter_name(filter_mode)}",
            (20, height - 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv.imshow("Video Recorder", display_frame)

        key = cv.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key == 32:
            is_recording = not is_recording

            if is_recording:
                output_filename = create_output_path()
                writer = cv.VideoWriter(
                    output_filename,
                    fourcc,
                    camera_fps,
                    (width, height)
                )

                if not writer.isOpened():
                    print("Error: Could not open VideoWriter.")
                    print("Try installing codecs or changing FOURCC / file extension.")
                    writer = None
                    is_recording = False
                    output_filename = "Not recording"
                else:
                    record_start_time = time.time()
                    print("Recording started.")
                    print(f"Saving to: {output_filename}")
            else:
                if writer is not None:
                    writer.release()
                    writer = None
                record_start_time = None
                print(f"Recording stopped. File saved to: {output_filename}")
                output_filename = "Not recording"
        elif key == ord("0"):
            filter_mode = 0
        elif key == ord("1"):
            filter_mode = 1
        elif key == ord("2"):
            filter_mode = 2
        elif key == ord("3"):
            filter_mode = 3
        elif key == ord("4"):
            filter_mode = 4
        elif key == ord("5"):
            filter_mode = 5

    cap.release()

    if writer is not None:
        writer.release()

    cv.destroyAllWindows()
    print("Program terminated.")


if __name__ == "__main__":
    main()