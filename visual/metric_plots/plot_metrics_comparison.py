"""
Metric Comparison Visualization

Creates bar charts comparing model performance metrics across different prediction horizons.
Generates 5 stacked charts: RMSE, MAE, MAPE, R2, and Improvement Rate.

Input CSV format:
    Model,Horizon,RMSE,MAE,MAPE,R2
    DLinear,6,4.125197,1.943053,0.919952,0.993273
    ...

Usage:
    python plot_metrics_comparison.py --input metrics.csv [--output OUTPUT_DIR] [--baseline DLinear]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import argparse


# Project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent


# Distinct colors using matplotlib's tableau palette for maximum contrast
DISTINCT_COLORS = [
    '#1f77b4',  # Blue
    '#ff7f0e',  # Orange
    '#2ca02c',  # Green
    '#d62728',  # Red
    '#9467bd',  # Purple
    '#8c564b',  # Brown
    '#e377c2',  # Pink
    '#7f7f7f',  # Gray
    '#bcbd22',  # Olive
    '#17becf',  # Cyan
]

# Default model color assignments - VERY DISTINCT colors for each model
DEFAULT_MODEL_COLORS = {
    # Baseline linear models
    'DLinear': '#1f77b4',       # Blue
    'NLinear': '#17becf',       # Cyan

    # Deep learning models - distinct colors
    'lstm': '#9467bd',          # Purple
    'LSTM': '#9467bd',          # Purple
    'patchtst': '#2ca02c',      # Green
    'PatchTST': '#2ca02c',      # Green
    'transformer': '#d62728',   # Red
    'Transformer': '#d62728',   # Red

    # EMD variants - Orange/Yellow tones
    'EMD-DLinear': '#ff7f0e',   # Orange
    'EMD-NLinear': '#ffbb78',   # Light Orange

    # EEMD variants - Brown tones
    'EEMD-DLinear': '#8c564b',  # Brown
    'EEMD-NLinear': '#c49c94',  # Light Brown

    # CEEMD variants - Green tones
    'CEEMD-DLinear': '#2ca02c', # Green
    'CEEMD-NLinear': '#98df8a', # Light Green
}


def get_model_colors(models: List[str]) -> Dict[str, str]:
    """
    Assign DISTINCT colors to models based on their names.

    Args:
        models: List of model names

    Returns:
        Dict mapping model names to colors
    """
    colors = {}
    used_colors = set()
    fallback_idx = 0

    for model in models:
        # Check both original and lowercase versions
        if model in DEFAULT_MODEL_COLORS:
            colors[model] = DEFAULT_MODEL_COLORS[model]
            used_colors.add(DEFAULT_MODEL_COLORS[model])
        elif model.lower() in DEFAULT_MODEL_COLORS:
            colors[model] = DEFAULT_MODEL_COLORS[model.lower()]
            used_colors.add(DEFAULT_MODEL_COLORS[model.lower()])
        else:
            # Use distinct fallback colors, avoiding already used ones
            while fallback_idx < len(DISTINCT_COLORS):
                color = DISTINCT_COLORS[fallback_idx]
                fallback_idx += 1
                if color not in used_colors:
                    colors[model] = color
                    used_colors.add(color)
                    break
            else:
                # If all colors used, start cycling with modifications
                colors[model] = DISTINCT_COLORS[len(colors) % len(DISTINCT_COLORS)]

    return colors


def load_metrics(csv_path: str) -> pd.DataFrame:
    """
    Load metrics from CSV file and standardize column names.

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame with standardized columns
    """
    df = pd.read_csv(csv_path)

    # Standardize column names (handle different orderings)
    col_mapping = {
        'model': 'Model',
        'horizon': 'Horizon',
        'rmse': 'RMSE',
        'mae': 'MAE',
        'mape': 'MAPE',
        'r2': 'R2',
        'mse': 'MSE',
        'mae_sudden': 'MAE_Sudden',
    }

    # Rename columns (case-insensitive)
    df.columns = [col_mapping.get(c.lower(), c) for c in df.columns]

    # Ensure required columns exist
    required = ['Model', 'Horizon', 'RMSE', 'MAE', 'MAPE', 'R2']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort by model and horizon
    df = df.sort_values(['Model', 'Horizon']).reset_index(drop=True)

    return df


def calculate_improvements(
    df: pd.DataFrame,
    baseline_model: str = "DLinear"
) -> pd.DataFrame:
    """
    Calculate improvement rates relative to baseline model.

    Args:
        df: DataFrame with metrics
        baseline_model: Model to use as baseline

    Returns:
        DataFrame with improvement columns added
    """
    if baseline_model not in df['Model'].unique():
        print(f"Warning: Baseline model '{baseline_model}' not found. Using first model.")
        baseline_model = df['Model'].unique()[0]

    baseline = df[df['Model'] == baseline_model].set_index('Horizon')

    improvements = []
    for _, row in df.iterrows():
        horizon = row['Horizon']
        model = row['Model']

        if horizon in baseline.index:
            base = baseline.loc[horizon]

            # For RMSE, MAE, MAPE: lower is better, so improvement = (base - current) / base * 100
            rmse_imp = (base['RMSE'] - row['RMSE']) / base['RMSE'] * 100 if base['RMSE'] != 0 else 0
            mae_imp = (base['MAE'] - row['MAE']) / base['MAE'] * 100 if base['MAE'] != 0 else 0
            mape_imp = (base['MAPE'] - row['MAPE']) / base['MAPE'] * 100 if base['MAPE'] != 0 else 0

            # For R2: higher is better, so improvement = (current - base) / (1 - base) * 100
            # But simpler: just show relative improvement
            r2_imp = (row['R2'] - base['R2']) / (1 - base['R2']) * 100 if base['R2'] != 1 else 0

            improvements.append({
                'Model': model,
                'Horizon': horizon,
                'RMSE_Improve': rmse_imp,
                'MAE_Improve': mae_imp,
                'MAPE_Improve': mape_imp,
                'R2_Improve': r2_imp,
            })

    return pd.DataFrame(improvements)


def plot_metric_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    models: List[str],
    horizons: List[int],
    colors: Dict[str, str],
    y_label: str,
    show_values: bool = True,
    value_fontsize: int = 7,
) -> None:
    """
    Plot grouped bar chart for a single metric.

    Args:
        ax: Matplotlib axes
        df: DataFrame with metrics
        metric: Column name for the metric
        models: List of models to plot
        horizons: List of horizons
        colors: Dict mapping models to colors
        y_label: Y-axis label
        show_values: Whether to show values on bars
        value_fontsize: Font size for bar values
    """
    n_models = len(models)
    n_horizons = len(horizons)
    bar_width = 0.8 / n_models
    x = np.arange(n_horizons)

    for i, model in enumerate(models):
        model_data = df[df['Model'] == model].set_index('Horizon')
        values = [model_data.loc[h, metric] if h in model_data.index else 0 for h in horizons]

        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width, label=model, color=colors[model], edgecolor='white', linewidth=0.5)

        # Add value labels on bars
        if show_values:
            for bar, val in zip(bars, values):
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{val:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 2),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=value_fontsize,
                                rotation=90 if n_models > 5 else 0)

    ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'+{h}h' for h in horizons], fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    # Set y-axis to start from 0
    if metric != 'R2':
        ax.set_ylim(bottom=0)


def plot_metrics_comparison(
    csv_path: str,
    output_dir: Optional[str] = None,
    baseline_model: str = "DLinear",
    models_to_plot: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (14, 22),
    dpi: int = 150,
    show_plot: bool = False,
    output_filename: str = "metrics_comparison.png",
) -> Path:
    """
    Create comprehensive metric comparison visualization.

    Args:
        csv_path: Path to input CSV file
        output_dir: Output directory (default: same as input)
        baseline_model: Model to use as baseline for improvements
        models_to_plot: List of models to include (default: all)
        figsize: Figure size (width, height)
        dpi: Resolution for saved figure
        show_plot: Whether to display the plot
        output_filename: Name of output file

    Returns:
        Path to saved figure
    """
    # Load data
    print(f"Loading metrics from: {csv_path}")
    df = load_metrics(csv_path)

    # Get unique models and horizons
    all_models = df['Model'].unique().tolist()
    horizons = sorted(df['Horizon'].unique().tolist())

    # Filter models if specified
    if models_to_plot:
        models = [m for m in models_to_plot if m in all_models]
    else:
        models = all_models

    print(f"Models: {models}")
    print(f"Horizons: {horizons}")

    # Get colors
    colors = get_model_colors(models)

    # Calculate improvements
    improvements = calculate_improvements(df, baseline_model)

    # Create figure with 5 subplots
    fig, axes = plt.subplots(5, 1, figsize=figsize)
    fig.suptitle(f"Model Performance Comparison\n(Baseline: {baseline_model})",
                 fontsize=14, fontweight='bold', y=0.995)

    # 1. RMSE
    plot_metric_bars(axes[0], df, 'RMSE', models, horizons, colors, 'RMSE')

    # 2. MAE
    plot_metric_bars(axes[1], df, 'MAE', models, horizons, colors, 'MAE')

    # 3. MAPE
    plot_metric_bars(axes[2], df, 'MAPE', models, horizons, colors, 'MAPE')

    # 4. R2
    plot_metric_bars(axes[3], df, 'R2', models, horizons, colors, 'R2')
    axes[3].set_ylim(0, 1.1)  # R2 is bounded [0, 1]

    # 5. Improvement Rates
    ax = axes[4]
    metrics_improve = ['RMSE_Improve', 'MAE_Improve', 'MAPE_Improve', 'R2_Improve']
    metric_labels = ['RMSE Improves', 'MAE Improves', 'MAPE Improves', 'R2 Improves']
    improve_colors = ['#5b9bd5', '#f4b183', '#70ad47', '#7d3c98']

    n_metrics = len(metrics_improve)
    bar_width = 0.8 / n_metrics
    x = np.arange(len(horizons))

    # Filter improvements to non-baseline models
    non_baseline_models = [m for m in models if m != baseline_model]

    if non_baseline_models:
        # Average improvement across non-baseline models
        for i, (metric, label, color) in enumerate(zip(metrics_improve, metric_labels, improve_colors)):
            avg_improvements = []
            for h in horizons:
                h_data = improvements[(improvements['Horizon'] == h) &
                                      (improvements['Model'].isin(non_baseline_models))]
                if len(h_data) > 0:
                    avg_improvements.append(h_data[metric].mean())
                else:
                    avg_improvements.append(0)

            offset = (i - n_metrics / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, avg_improvements, bar_width, label=label, color=color, edgecolor='white')

            # Add value labels
            for bar, val in zip(bars, avg_improvements):
                height = bar.get_height()
                ax.annotate(f'{val:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 2),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=7)

    ax.set_ylabel('Improvement Rate (%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Prediction Step (Hr)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f'+{h}h' for h in horizons], fontsize=10)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    # Add legend for models (shared across top 4 plots)
    handles = [mpatches.Patch(color=colors[m], label=m) for m in models]
    axes[0].legend(handles=handles, loc='upper right', fontsize=9, ncol=min(4, len(models)))

    # Tight layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.96, hspace=0.25)

    # Save figure
    if output_dir is None:
        output_dir = Path(csv_path).parent
    else:
        output_dir = Path(output_dir)

    output_path = output_dir / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return output_path


def plot_individual_metrics(
    csv_path: str,
    output_dir: Optional[str] = None,
    baseline_model: str = "DLinear",
    models_to_plot: Optional[List[str]] = None,
    dpi: int = 150,
) -> List[Path]:
    """
    Create individual plots for each metric (separate files).

    Args:
        csv_path: Path to input CSV file
        output_dir: Output directory
        baseline_model: Model to use as baseline
        models_to_plot: List of models to include

    Returns:
        List of paths to saved figures
    """
    df = load_metrics(csv_path)

    all_models = df['Model'].unique().tolist()
    horizons = sorted(df['Horizon'].unique().tolist())
    models = models_to_plot if models_to_plot else all_models
    models = [m for m in models if m in all_models]

    colors = get_model_colors(models)

    if output_dir is None:
        output_dir = Path(csv_path).parent
    else:
        output_dir = Path(output_dir)

    saved_paths = []
    metrics = [('RMSE', 'RMSE'), ('MAE', 'MAE'), ('MAPE', 'MAPE (%)'), ('R2', 'R2')]

    for metric, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_metric_bars(ax, df, metric, models, horizons, colors, ylabel)
        ax.set_xlabel('Prediction Step (Hr)', fontsize=11)
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')

        # Add legend
        handles = [mpatches.Patch(color=colors[m], label=m) for m in models]
        ax.legend(handles=handles, loc='upper left', fontsize=9, ncol=min(3, len(models)))

        plt.tight_layout()
        output_path = output_dir / f"{metric.lower()}_comparison.png"
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        saved_paths.append(output_path)
        print(f"Saved: {output_path}")

    return saved_paths


def main():
    parser = argparse.ArgumentParser(description="Plot metric comparison charts")
    parser.add_argument('--input', '-i', type=str, required=True,
                        help="Path to input CSV file with metrics")
    parser.add_argument('--output', '-o', type=str, default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument('--baseline', '-b', type=str, default="DLinear",
                        help="Baseline model for improvements (default: DLinear)")
    parser.add_argument('--models', '-m', type=str, nargs='+', default=None,
                        help="Models to include (default: all)")
    parser.add_argument('--dpi', type=int, default=150,
                        help="DPI for saved figures (default: 150)")
    parser.add_argument('--show', action='store_true',
                        help="Show plot instead of just saving")
    parser.add_argument('--individual', action='store_true',
                        help="Also create individual metric plots")

    args = parser.parse_args()

    # Main combined plot
    plot_metrics_comparison(
        csv_path=args.input,
        output_dir=args.output,
        baseline_model=args.baseline,
        models_to_plot=args.models,
        dpi=args.dpi,
        show_plot=args.show,
    )

    # Individual plots if requested
    if args.individual:
        plot_individual_metrics(
            csv_path=args.input,
            output_dir=args.output,
            baseline_model=args.baseline,
            models_to_plot=args.models,
            dpi=args.dpi,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
