"""
Experiment Runner Script

Executes batch detection experiments with different models and batch sizes.
Collects performance metrics and quality indicators, saves results to CSV,
and generates analysis plots.
"""
import os
import traceback
from pathlib import Path
import argparse
import multiprocessing as mp

import pandas as pd
from tqdm import tqdm

from samples.config_parser import parse_yaml_file
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


def run_single_experiment(model_config: str, batch_size: int, params: ExperimentParameters,
                          template_file: Path):
    """
    Runs a single experiment for the given model with the specified batch size.

    :param model_config: Path to the YAML model configuration file.
    :param batch_size: The batch size to be used in the experiment.
    :param params: Experiment parameters, including paths to data, output files, and other settings.
    :param template_file: Path to the temporary CSV file for storing data.

    :return: A dictionary containing the experiment results:
             model name, batch size, performance metrics, and accuracy.
    """
    try:
        config_params = parse_yaml_file(model_config)
        components = config_pipeline_components(config_params,
                                                batch_size,
                                                template_file,
                                                params)
        pipeline = DetectionPipeline(components)
        pipeline.run()

        perf_metrics = PerformanceCalculator.calculate(
            components.reader.get_total_images(),
            batch_size,
            pipeline.batches_timings
        )

        accuracy_calculator = AccuracyCalculator()
        accuracy_calculator.load_detections(template_file)
        accuracy_calculator.load_groundtruths(params.groundtruth_path)

        accuracy_map = accuracy_calculator.calc_map()

        return {
            'model': config_params['model_name'],
            'batch_size': batch_size,
            **perf_metrics,
            'accuracy_map': accuracy_map
        }

    except Exception as e:
        traceback.print_exc()
        return {
            'model': model_config,
            'batch_size': batch_size,
            'error': str(e)
        }


def run_experiment_in_pool(experiment_args: tuple):
    """
    Runs an experiment in the pool of processes. This function is used to pass arguments
    to the process pool.

    :param experiment_args: tuple: A tuple containing the parameters for running the experiment
                 (model, batch size, parameters, file path).

    :return: The results of the single experiment obtained from run_single_experiment.
    """
    return run_single_experiment(*experiment_args)


def run_experiments_pool(
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
    tmp_dir = Path('./src/benchmark/tmp').absolute()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    experiment_args = [
        (
            model_config,
            batch_size,
            params,
            tmp_dir / f"{Path(model_config).stem}_bs{batch_size}.csv"
        )
        for model_config in model_config_paths
        for batch_size in batch_sizes
    ]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(run_experiment_in_pool, experiment_args)

    # Save and analyze results
    df = pd.DataFrame(results)
    results_file = os.path.join(params.output_path, 'benchmark_results.csv')
    df.to_csv(results_file, index=False)

    # Generate plots
    generate_perf_plots(df, params.output_path)
    generate_quality_plot(df, params.output_path)


def run_experiments_mp(
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
    tmp_dir = Path('./src/benchmark/tmp').absolute()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for model_config in tqdm(model_config_paths, desc='Models configs'):
        for batch_size in tqdm(batch_sizes, desc='Batch sizes', leave=False):
            experiment_args = (
                    model_config,
                    batch_size,
                    params,
                    tmp_dir / f"{Path(model_config).stem}_bs{batch_size}.csv"
            )
            p = mp.Process(target=run_single_experiment, args=experiment_args)
            p.start()
            p.join()

    # Save and analyze results
    df = pd.DataFrame(results)
    results_file = os.path.join(params.output_path, 'benchmark_results.csv')
    df.to_csv(results_file, index=False)

    # Generate plots
    generate_perf_plots(df, params.output_path)
    generate_quality_plot(df, params.output_path)


def run_experiments_shared(
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
    tmp_dir = Path('./src/benchmark/tmp').absolute()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for model_config in tqdm(model_config_paths, desc='Models configs'):
        for batch_size in tqdm(batch_sizes, desc='Batch sizes', leave=False):
            experiment_args = (
                model_config,
                batch_size,
                params,
                tmp_dir / f"{Path(model_config).stem}_bs{batch_size}.csv"
            )
            results.append(run_single_experiment(*experiment_args))

    # Save and analyze results
    df = pd.DataFrame(results)
    results_file = os.path.join(params.output_path, 'benchmark_results.csv')
    df.to_csv(results_file, index=False)

    # Generate plots
    generate_perf_plots(df, params.output_path)
    generate_quality_plot(df, params.output_path)


if __name__ == '__main__':
    DEFAULT_MODEL_CONFIGS = [
        './configs/torchvision/detector_config_fasterRCNN.yaml',
        './configs/torchvision/detector_config_FCOS.yaml',
        './configs/torchvision/detector_config_RetinaNet.yaml',
        './configs/torchvision/detector_config_SSD.yaml',
        './configs/torchvision/detector_config_SSDlite.yaml',

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
        run_experiments_shared(
            DEFAULT_MODEL_CONFIGS,
            DEFAULT_BATCH_SIZES,
            data_params,
        )

    except Exception as e:
        print(e)
