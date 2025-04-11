"""
Performance Metrics Calculation Module

Provides tools for collecting and analyzing vehicle detection pipeline performance metrics.
Handles timing data aggregation and key performance indicators calculation.
"""

import time
from typing import List, Dict


class PerformanceCalculator:
    """
    Collects and processes timing data from detection pipeline batches.

    Attributes:
        total_frames:       Total processed frames
        batch_size:         Expected number of frames per full batch
        valid_batches:      Count of complete batches processed
        start_time:         Timestamp of calculator initialization
        preproc_times:      List of preprocessing durations (seconds)
        inference_times:    List of inference durations (seconds)
        postproc_times:     List of postprocessing durations (seconds)
    """

    def __init__(self,
                 total_frames: int,
                 batch_size: int):
        """Initialize performance calculator with batch configuration."""
        self.total_frames = total_frames
        self.batch_size = batch_size

        self.valid_batches = 0
        self.start_time = time.time()
        self.preproc_times:     List[float] = []
        self.inference_times:   List[float] = []
        self.postproc_times:    List[float] = []

    def reset(self):
        """Reset calculator state while preserving total frames and batch size configuration."""
        self.start_time = time.time()
        self.preproc_times:     List[float] = []
        self.inference_times:   List[float] = []
        self.postproc_times:    List[float] = []

    def add_batch(self,
                  preproc: float,
                  inference: float,
                  postproc: float):
        """
        Record timing data for a processed batch.

        :param preproc:     Preprocessing time in seconds
        :param inference:   Model inference time in seconds
        :param postproc:    Postprocessing time in seconds
        """
        self.preproc_times.append(preproc)
        self.inference_times.append(inference)
        self.postproc_times.append(postproc)
        self.valid_batches += 1

    def calculate(self) -> Dict[str, float]:
        """
        Compute performance metrics from collected timing data.

        :return: Dictionary containing calculated metrics:
            - total_frames:             Total processed frames
            - total_time:               Total wall clock time
            - latency:                  Median inference time
            - avg_time_of_single_pass:  Average time of a single pass
            - batch_fps:                Frames per second (batch)
            - inference_fps:            Frames per second (inference)
        """
        total_frames = self.valid_batches * self.batch_size
        total_time = time.time() - self.start_time
        latency = self._median(sorted(self.inference_times))
        total_batches = len(self.inference_times)
        total_inference = sum(self.inference_times)

        return {
            'total_frames': total_frames,
            'total_time': total_time,
            'latency': latency,
            'avg_time_of_single_pass': total_time / total_batches if total_batches > 0 else 0,
            'batch_fps': self.batch_size / latency if latency > 0 else 0,
            'inference_fps': total_frames / total_inference if total_inference > 0 else 0
        }

    @staticmethod
    def _median(data: List[float]) -> float:
        """
        Calculate median value from sorted data.

        :param data: Pre-sorted list of timing values
        :return: Median value in seconds
        """
        if not data:
            return 0.0

        total_elems = len(data)
        mid = total_elems // 2
        if total_elems % 2 == 0:
            return (data[mid] + data[mid - 1]) / 2
        return data[mid]
