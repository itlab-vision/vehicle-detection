"""
Frame Data Reading Module

Provides abstract and concrete implementations for reading frame data from different sources.
Supports video files and image directories with OpenCV integration.

Classes:
    :FrameDataReader: Abstract base class for frame readers
    :VideoDataReader: Concrete implementation for video files
    :ImgDataReader: Concrete implementation for image directories

Dependencies:
    :OpenCV (cv2): for image reading
    :pathlib: module for file operations
"""
from pathlib import Path
from abc import ABC, abstractmethod
import cv2 as cv


class FrameDataReader(ABC):
    """
    Abstract Base Class for frame data readers.
    
    Defines the interface for iterating through frames from different sources.
    
    Methods:
        create: Factory method to instantiate appropriate reader (static)
        __enter__: Context manager entry (abstract)
        __exit__: Context manager exit with resource cleanup (abstract)
        __iter__: Returns iterator object (abstract)
        __next__: Returns next frame (abstract)
    """
    @abstractmethod
    def __enter__(self):
        """
        Context manager entry point.

        :return self: Object instance
        """

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release resources on exit."""

    @abstractmethod
    def __iter__(self):
        """
        :return self: Iterator instance
        """

    @abstractmethod
    def __next__(self):
        """
        Get next frame in sequence.
        
        :return ndarray: Next frame as numpy array
        :raise StopIteration: When no more frames available
        """
    @abstractmethod
    def get_total_batches(self):
        """
        Get number of batches

        :return int: total batches
        """
    @staticmethod
    def create(mode: str, dir_path: str, batch_size: int):
        """
        Factory method to create appropriate reader instance.
        
        :param mode: Source type - 'video' or 'image'
        :param dir_path: Path to video file or image directory
        :param batch_size: Size of image batch
        :return FrameDataReader: Concrete reader instance
        :raise ValueError: For unsupported modes or invalid paths
        """
        if mode == "video":
            return VideoDataReader(dir_path)
        if mode == "image":
            return ImgDataReader(dir_path, batch_size)
        raise ValueError(f"Unsupported mode: {mode}")


class VideoDataReader(FrameDataReader):
    """
    Video file frame reader using OpenCV VideoCapture.
    
    Iterates through frames of a video file. Automatically handles
    video resource cleanup on exhaustion.
    """

    def __init__(self, video_path: str):
        """
        Initialize video capture and validate path.

        :param video_path: Path to video file

        :var cap: OpenCV VideoCapture object.
            Used internally for video frame reading and video properties access

        :raise ValueError: If video file cannot be opened
        """
        self.video_path = video_path
        self._cap = None

    def get_total_batches(self):
        """
        :return int: number of batches
        """
        return int(self._cap.get(cv.CAP_PROP_FRAME_COUNT))

    def __enter__(self):
        """Initialize video capture and return iterator."""
        self._cap = cv.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            raise IOError(f"Could not open video file: {self.video_path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release video resources on exit."""
        if self._cap and self._cap.isOpened():
            self._cap.release()

    def __iter__(self):
        """
        :return self: Iterator instance
        """
        return self

    def __next__(self):
        """
        Get next video frame.

        :return ndarray: Next video frame as numpy array
        :raise StopIteration: When video ends or is closed
        """
        if self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                return [frame]
            raise StopIteration
        raise StopIteration


class ImgDataReader(FrameDataReader):
    """
    Image directory frame reader.

    - Iterates through image files in directory sorted alphabetically.
    - Returns images in batches of specified size.
    - If last batch is incomplete, it is padded with copies of the last image.
    - Supports common image formats: PNG, JPG, JPEG, BMP, TIFF.
    """

    def __init__(self, dir_path: str, batch_size: int = 1):
        """
        :param dir_path: Path to image directory
        :param batch_size: Number of images per batch
        :raise ValueError: For invalid directory path or no valid images
        """
        self.index = 0
        self.batch_size = batch_size
        self.batches = []
        self.images_paths = []
        self.dir_path = Path(dir_path)
        self._validate_directory()
        self._prepare_file_list()
        self._prepare_batches()

    def _validate_directory(self):
        """Ensure directory exists and is accessible."""
        if not self.dir_path.exists():
            raise ValueError(f"Invalid image directory: {self.dir_path}")
        if not self.dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {self.dir_path}")

    def _prepare_file_list(self):
        """Create sorted list of valid image files."""
        valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        self.image_paths = [
            str(file) for file in self.dir_path.iterdir()
            if file.is_file() and file.suffix.lower() in valid_extensions
        ]

        if not self.image_paths:
            raise ValueError(f"No valid images found in: {self.dir_path}")

    def _prepare_batches(self):
        """
        Loads images and forms fixed-size batches in one pass through the directory.

        :raise ValueError: if no valid images are found or an image can't be read.
        """
        batch = []
        last_img = None

        for path in self.image_paths:
            img = cv.imread(str(path))
            if img is None:
                raise ValueError(f"Cannot read image file: {path}")

            batch.append(img)
            last_img = img

            if len(batch) == self.batch_size:
                self.batches.append(batch)
                batch = []

        if last_img is not None:
            while len(batch) < self.batch_size:
                batch.append(last_img.copy())

        if batch is not None:
            self.batches.append(batch)

    def get_total_batches(self):
        """
        :return int: number of batches
        """
        return len(self.batches)

    def __enter__(self):
        """
        Context manager entry point.
        
        :return self: Object instanse
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release resources on exit."""

    def __iter__(self):
        """
        :return self: Iterator instance
        """
        return self

    def __next__(self):
        """
        Load the next batch of images.

        :return List[np.ndarray]: Batch of images
        :raise StopIteration: When all batches are processed
        """
        if self.index < len(self.batches):
            batch = self.batches[self.index]
            self.index += 1
            return batch
        raise StopIteration
