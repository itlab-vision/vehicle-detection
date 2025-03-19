"""
Computer Vision CLI Visualization Module
Provides functionality for visualizing object detection results alongside ground truth data.
Handles both image sequences and video inputs with the following components:
- Frame data reading (images/video)
- Object detection implementation
Dependencies:
- OpenCV (cv2) for image processing and display
- time and sys for output progressing in console
- FrameDataReader for getting images
- Writer for write detections in file
- Detector for getting detections in image
"""
import time
import sys

class CLIVisualize:
    """
    Visualization controller for detections.
    Handles frame iteration and custom progress bar, print all detections.
    """

    def __init__(self):
        """
        Initialize visualization components with data sources.
        :param datareader: Input source for frames (video/images)
        :param detector: Detection component
        """
        self.start_time = time.time()
        self.last_update = 0

    def update_status(self, frame_idx: int):
        """
        Update string status
        :param frame_idx: current id of frame
        """
        elapsed = time.time() - self.start_time
        fps = frame_idx / elapsed if elapsed > 0 else 0
        status = (
            f"Processing frame: {frame_idx}"
            f"[{elapsed:.1f}s, {fps:.1f}fps]"
        )
        sys.stdout.write("\r\033[K" + status)
        sys.stdout.flush()

    def print_detections(self, detections: list[tuple]):
        """
        print info about detections
        
        :param detections: list of tuples that got by detector
        """
        if not detections:
            print("\n  No detections")
            return

        for box in detections:
            label = box[0]
            coords = tuple(map(int, box[1:5]))
            conf = f", Confidence: {box[5]:.2f}" if len(box) > 5 else ""
            detection_str = (
                f"\n  {label} {coords[0]},{coords[1]}-{coords[2]},{coords[3]}{conf}"
            )

            sys.stdout.write(detection_str)
            sys.stdout.flush()

    def print_final_status(self, frame_idx: int):
        """
        Print final result
        :param frame_idx: current id of frame
        """
        elapsed = time.time() - self.start_time
        fps = frame_idx / elapsed if elapsed > 0 else 0
        print(
            f"\nProcessing completed: {frame_idx} frames "
            f"in {elapsed:.1f}s ({fps:.1f} fps)"
        )
