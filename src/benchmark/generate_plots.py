"""
Plot Generation Module

Generates performance and quality visualizations for detection experiments.
Includes:
- Line plots comparing inference FPS across batch sizes for different models.
- Bar plots showing mAP (mean Average Precision) accuracy comparison between models.

Used to visually analyze trade-offs between speed and detection quality.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt


def generate_perf_plots(df: pd.DataFrame, output_dir: str):
    """
    Generate performance comparison plots.

    :param df: Experiment results dataframe
    :param output_dir: Output directory for plots
    """
    plt.figure(figsize=(12, 6))

    # FPS vs Batch Size plot
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.plot(model_data['batch_size'], model_data['inference_fps'],
                 marker='o', linestyle='--', label=model)

    plt.title('Inference FPS vs Batch Size')
    plt.xlabel('Batch Size')
    plt.ylabel('FPS')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'fps_vs_batch.png'))
    plt.close()


def generate_quality_plot(df: pd.DataFrame, output_dir: str):
    """
    Generate quality comparison histogram.

    :param df: Experiment results dataframe
    :param output_dir: Output directory for plots
    """
    plt.figure(figsize=(10, 6))

    # Aggregate accuracy by model
    accuracy_data = df.groupby('model')['accuracy_map'].mean()

    accuracy_data.plot(kind='bar', color='skyblue')
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    plt.close()
