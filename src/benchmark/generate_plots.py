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
from adjustText import adjust_text


def generate_perf_plots(df: pd.DataFrame, output_dir: str):
    """
    Generate performance comparison plots.

    :param df: Experiment results dataframe
    :param output_dir: Output directory for plots
    """
    plt.figure(figsize=(14, 6), constrained_layout=True)

    # FPS vs Batch Size plot
    texts = []
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        plt.plot(model_data['batch_size'], model_data['inference_fps'],
                 marker='o', linestyle='--', label=model)

        for x, y in zip(model_data['batch_size'], model_data['inference_fps']):
            texts.append(plt.text(x, y, f'{y:.1f}', fontsize=8))

    adjust_text(texts, arrowprops={"arrowstyle": '-',
                                   "color": 'gray'})

    plt.xlabel('Batch Size')
    plt.ylabel('FPS')
    plt.grid(True)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

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
    models = accuracy_data.index.tolist()
    accuracies = accuracy_data.values

    bars = plt.bar(range(len(models)), accuracies, color='skyblue')

    plt.ylabel('mAP, %')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.xticks(ticks=range(len(models)), labels=models, rotation=45, ha='center')

    # Add value labels above each bar
    for elem in bars:
        height = elem.get_height()
        plt.text(elem.get_x() + elem.get_width() / 2, height + 0.03,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9, color='black')

    plt.subplots_adjust(bottom=0.15)
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    plt.close()
