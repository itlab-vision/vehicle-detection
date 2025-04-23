"""
Experiment Runner Script

Executes batch detection experiments with different models and batch sizes.
Collects performance metrics and quality indicators, saves results to CSV,
and generates analysis plots.
"""
import os
from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm

from samples import config_parser
from src.detector_pipeline.detector_pipeline import DetectionPipeline
from src.accuracy_checker.accuracy_checker import AccuracyCalculator
from src.perf_calculator.perf_calculator import PerformanceCalculator
from src.benchmark.generate_plots import generate_perf_plots, generate_quality_plot
from src.benchmark.config_benchmark import config_pipeline_components, ExperimentParameters


def experiment_argument_parser():
    """
    Parses command-line arguments for the detection experiment.

    :return: Parsed arguments including input data path, groundtruth path, output path, and mode.
    """
    parser = argparse.ArgumentParser(description='Run detection experiments')
    parser.add_argument('-in', '--input_data_path',
                        type=str,
                        help='Path to images or video',
                        required=True)
    parser.add_argument('-gt', '--groundtruth_path',
                        type=str,
                        help='Path to groundtruth',
                        required=True)
    parser.add_argument('-out', '--output_path',
                        type=str,
                        help='Output directory for results',
                        default='./results')
    parser.add_argument('-m', '--mode',
                        type=str,
                        choices=['image', 'video'],
                        default='image')

    return parser.parse_args()


def run_experiments(
        model_config_paths: list[str],
        batch_sizes: list[int],
        params: ExperimentParameters):
    """
    Executes batch detection experiments with given model configurations and batch sizes.
    Runs detection pipelines, collects performance and accuracy metrics, saves results to CSV,
    and generates analysis plots.

    :param model_config_paths: List of paths to model configuration YAML files.
    :param batch_sizes: List of batch sizes to be tested.
    :param params: Experiment parameters including input/output paths and mode.
    """
    # Create output directory
    os.makedirs(params.output_path, exist_ok=True)
    os.makedirs(Path('./src/benchmark/tmp').absolute(), exist_ok=True)
    template_file = Path('./src/benchmark/tmp/tmp_output.csv').absolute()

    results = []
    # Main experiment loop
    for model_config in tqdm(model_config_paths, desc='Models configs'):
        for batch_size in tqdm(batch_sizes, desc='Batch sizes', leave=False):
            # Setup pipeline
            config_params = config_parser.parse_yaml_file(model_config)
            components = config_pipeline_components(config_params,
                                                    batch_size,
                                                    template_file,
                                                    params)
            # Run detection pipeline
            pipeline = DetectionPipeline(components)
            pipeline.run()

            # Collect metrics
            perf_metrics = PerformanceCalculator.calculate(
                components.reader.get_total_images(),
                batch_size,
                pipeline.batches_timings
            )

            accuracy_calculator = AccuracyCalculator()
            accuracy_calculator.load_detections(template_file)
            accuracy_calculator.load_groundtruths(params.groundtruth_path)

            accuracy_map = accuracy_calculator.calc_map()

            # Store results
            results.append({
                'model': config_params['model_name'],
                'batch_size': batch_size,
                **perf_metrics,
                'accuracy_map': accuracy_map
            })

    # Save and analyze results
    df = pd.DataFrame(results)
    results_file = os.path.join(params.output_path, 'benchmark_results.csv')
    df.to_csv(results_file, index=False)

    # Generate plots
    generate_perf_plots(df, params.output_path)
    generate_quality_plot(df, params.output_path)


if __name__ == '__main__':
    DEFAULT_MODEL_CONFIGS = [
        './configs/detector_config_file_fasterRCNN.yaml',
        './configs/yolo/detector_config_yolov3_tinyu.yaml',
        './configs/yolo/detector_config_yolov11s.yaml',
        './configs/yolo/detector_config_yolov12s.yaml',
        './configs/rtdetr/detector_config_rtdetr-l.yaml',
    ]
    DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16]

    try:
        args = experiment_argument_parser()
        data_params = ExperimentParameters(
            args.input_data_path,
            args.groundtruth_path,
            args.output_path,
            args.mode
        )
        run_experiments(
            DEFAULT_MODEL_CONFIGS,
            DEFAULT_BATCH_SIZES,
            data_params,
        )

    except Exception as e:
        print(e)
