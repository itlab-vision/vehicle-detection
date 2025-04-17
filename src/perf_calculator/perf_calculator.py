"""
Performance Metrics Calculation Module

Provides tools for collecting and analyzing vehicle detection pipeline performance metrics.
Handles timing data aggregation and key performance indicators calculation.
"""

from typing import List, Dict
from src.detector_pipeline.detector_pipeline import BatchesTimings


class PerformanceCalculator:
    """Collects and processes timing data from detection pipeline batches."""

    def calculate(self, total_frames: int, batch_size: int, time_data: BatchesTimings) -> (
            Dict)[str, float]:
        """
        Compute performance metrics from collected timing data.

        :param total_frames:    Total processed frames
        :param batch_size:      Number of frames per full batch
        :param time_data:        All batch processing timing metrics

        :return: Dictionary containing calculated metrics:
            - total_frames:             Total processed frames
            - total_time:               Total wall clock time
            - latency:                  Median inference time
            - avg_time_of_single_pass:  Average time of a single pass
            - batch_fps:                Frames per second (batch)
            - inference_fps:            Frames per second (inference)
        """
        total_time = (sum(time_data.preprocess_time) + sum(time_data.inference_time) +
                      sum(time_data.postprocess_time))
        latency = self._median(sorted(time_data.inference_time))
        total_batches = len(time_data.inference_time)
        total_inference = sum(time_data.inference_time)

        return {
            'total_frames': total_frames,
            'total_time': total_time,
            'latency': latency,
            'avg_time_of_single_pass': total_time / total_batches if total_batches > 0 else 0,
            'batch_fps': batch_size / latency if latency > 0 else 0,
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
