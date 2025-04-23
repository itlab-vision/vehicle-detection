"""
Performance Metrics Calculation Module

Provides tools for collecting and analyzing vehicle detection pipeline performance metrics.
Handles timing data aggregation and key performance indicators calculation.
"""

from typing import Dict
from src.detector_pipeline.detector_pipeline import BatchesTimings


class PerformanceCalculator:
    """Collects and processes timing data from detection pipeline batches."""

    @staticmethod
    def calculate(total_frames: int, batch_size: int, time_data: BatchesTimings) -> (
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
        total_batches = len(time_data.inference_time)
        total_inference_time = sum(time_data.inference_time)

        if total_batches == 0:
            latency = 0.0
        else:
            latency = sorted(time_data.inference_time)[total_batches // 2]

        return {
            'total_frames': total_frames,
            'total_time': total_time,
            'latency': latency,
            'avg_time_of_single_pass': total_time / total_batches if total_batches > 0 else 0,
            'batch_fps': batch_size / latency if latency > 0 else 0,
            'inference_fps': total_frames / total_inference_time if total_inference_time > 0 else 0
        }
