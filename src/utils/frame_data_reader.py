"""
Frame Data Reading Module

Provides abstract and concrete implementations for reading frame data from different sources.
Supports video files and image directories with OpenCV integration.

Classes:
    :FrameDataReader: Abstract base class for frame readers
    :VideoDataReader: Concrete implementation for video files
    :ImgDataReader: Concrete implementation for image directories

Dependencies:
    - :OpenCV (cv2): for image reading
    - :os: module for file operations
"""
import os
from abc import ABC, abstractmethod
import cv2 as cv

class FrameDataReader(ABC):
    """Abstract Base Class for frame data readers.
    
    Defines the interface for iterating through frames from different sources.
    
    Methods:
    
        create: Factory method to instantiate appropriate reader
        __iter__: Returns iterator object (abstract)
        __next__: Returns next frame (abstract)
    """
    @abstractmethod
    def __iter__(self):
        """Return iterator object.
        
        Returns:
            self: Iterator instance
        """
        pass

    @abstractmethod
    def __next__(self):
        """Get next frame in sequence.
        
        Returns:
            ndarray: Next frame as numpy array
            
        Raises:
            StopIteration: When no more frames available
        """
        pass

    @staticmethod
    def create(mode, dir_path):
        """Factory method to create appropriate reader instance.
        
        Args:
            mode (str): Source type - 'video' or 'image'
            dir_path (str): Path to video file or image directory
            
        Returns:
            FrameDataReader: Concrete reader instance
            
        Raises:
            ValueError: For unsupported modes or invalid paths
        """
        if mode == "video":
            return VideoDataReader(dir_path)
        if mode == "image":
            return ImgDataReader(dir_path)
        raise ValueError(f"Unsupported mode: {mode}")

class VideoDataReader(FrameDataReader):
    """Video file frame reader using OpenCV VideoCapture.
    
    Iterates through frames of a video file. Automatically handles
    video resource cleanup on exhaustion.
    
    Args:
        video_path (str): Path to video file
        
    Raises:
        ValueError: If video file cannot be opened
    """

    def __init__(self, video_path):
        """Initialize video capture and validate path."""
        self.video_path = video_path
        self.cap = cv.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

    def __iter__(self):
        """Return iterator object.
        
        Returns:
            self: Iterator instance
        """
        return self

    def __next__(self):
        """Get next video frame.
        
        Returns:
            ndarray: Next video frame as numpy array
            
        Raises:
            StopIteration: When video ends or is closed
        """
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
            self.cap.release()
            raise StopIteration
        raise StopIteration

class ImgDataReader(FrameDataReader):
    """Image directory frame reader.
    
    Iterates through image files in directory sorted alphabetically.
    Supports common image formats: PNG, JPG, JPEG, BMP, TIFF.
    
    Args:
        dir_path (str): Path to image directory
        
    Raises:
        ValueError: For invalid directory path
    """

    def __init__(self, dir_path):
        """Validate directory and prepare image file list."""
        self.index = 0
        self.directory_path = dir_path
        if not os.path.exists(dir_path):
            raise ValueError(f"Directory does not exist: {dir_path}")
        self.image_files = [
            os.path.join(dir_path, f) for f in os.listdir(dir_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))] 

    def __iter__(self):
        """Return iterator object.
        
        Returns:
            self: Iterator instance
        """
        return self

    def __next__(self):
        """Load next image in directory.
        
        Returns:
            ndarray: Image data as numpy array
            
        Raises:
            StopIteration: When all images processed
            ValueError: If image file cannot be read
        """
        if self.index < len(self.image_files):
            img_path = self.image_files[self.index]
            self.index += 1
            img = cv.imread(img_path)
            if img is None:
                raise ValueError(f"Cannot read image file: {img_path}")
            return img
        raise StopIteration
