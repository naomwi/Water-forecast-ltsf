"""
CEEMDAN IMF Decomposition Visualization

Plots the CEEMDAN decomposition results showing:
- Original signal (EC or pH)
- IMF1 through IMF12 (or IMF13 if exists)
- Residual component

Usage:
    python plot_imf_decomposition.py [--target ec|ph] [--output OUTPUT_DIR]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Optional, List, Tuple
import argparse


# Project paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent

# Default cache locations (priority order)
CEEMDAN_CACHE_DIRS = [
    PROJECT_DIR / "CEEMDAN_models" / "decomposed_imfs",
    PROJECT_DIR / "CEEMD_Baselines" / "EC" / "cache",
    PROJECT_DIR / "CEEMD_Baselines" / "pH" / "cache",
]

# Data location
DATA_DIR = PROJECT_DIR / "Baselines_model" / "data" / "USGs"


def load_raw_data(target: str = "EC") -> Tuple[pd.DataFrame, np.ndarray, pd.DatetimeIndex]:
    """
    Load raw data from CSV file.

    Args:
        target: 'EC' or 'pH'

    Returns:
        Tuple of (DataFrame, target array, datetime index)
    """
    # Try different possible data file names
    possible_files = [
        DATA_DIR / "usgs_wq_data.csv",
        DATA_DIR / "water_data_2021_2025_clean.csv",
        DATA_DIR / "usgs_data.csv",
    ]

    data_file = None
    for f in possible_files:
        if f.exists():
            data_file = f
            break

    if data_file is None:
        raise FileNotFoundError(f"No data file found in {DATA_DIR}. Tried: {[f.name for f in possible_files]}")

    df = pd.read_csv(data_file)

    # Filter by site_no (same as models use)
    site_no = 1463500
    if 'site_no' in df.columns:
        df = df[df['site_no'] == site_no].copy()

    # Handle datetime
    if 'Time' in df.columns:
        df['datetime'] = pd.to_datetime(df['Time'])
    elif 'date' in df.columns:
        df['datetime'] = pd.to_datetime(df['date'])

    df = df.sort_values('datetime').reset_index(drop=True)

    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear').bfill().ffill()

    target_col = target.upper() if target.upper() in df.columns else target
    data = df[target_col].values.astype(np.float64)
    dates = df['datetime']

    return df, data, dates


def load_imfs(prefix: str = "ec", cache_dir: Optional[Path] = None) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load cached IMF decomposition results.

    Args:
        prefix: 'ec' or 'ph'
        cache_dir: Optional specific cache directory

    Returns:
        Tuple of (list of IMF arrays, residue array)
    """
    # Try cache directories in order
    if cache_dir is not None:
        search_dirs = [cache_dir]
    else:
        search_dirs = CEEMDAN_CACHE_DIRS

    for cache_path in search_dirs:
        if not cache_path.exists():
            continue

        # Check for IMF files
        imf_files = sorted(cache_path.glob(f"{prefix}_imf_*.npy"))
        residue_file = cache_path / f"{prefix}_residue.npy"

        if imf_files and residue_file.exists():
            print(f"Loading IMFs from: {cache_path}")

            # Load IMFs
            imfs = []
            for imf_file in imf_files:
                imfs.append(np.load(imf_file))

            # Load residue
            residue = np.load(residue_file)

            return imfs, residue

    raise FileNotFoundError(f"No cached IMFs found for prefix '{prefix}'")


def plot_imf_decomposition(
    target: str = "EC",
    output_path: Optional[Path] = None,
    figsize: Tuple[float, float] = (14, 24),
    dpi: int = 150,
    show_plot: bool = False,
) -> Path:
    """
    Plot CEEMDAN decomposition for EC or pH.

    Args:
        target: 'EC' or 'pH'
        output_path: Output file path (default: same directory as script)
        figsize: Figure size (width, height) in inches
        dpi: Resolution for saved figure
        show_plot: Whether to display the plot

    Returns:
        Path to saved figure
    """
    prefix = target.lower()

    # Load data
    print(f"Loading {target} data...")
    df, original_signal, dates = load_raw_data(target)

    print(f"Loading CEEMDAN IMFs for {target}...")
    imfs, residue = load_imfs(prefix)

    n_imfs = len(imfs)
    n_rows = 1 + n_imfs + 1  # Original + IMFs + Residual

    print(f"Found {n_imfs} IMFs + residue")
    print(f"Data length: {len(original_signal)} samples")
    print(f"Date range: {dates.iloc[0]} to {dates.iloc[-1]}")

    # Create figure
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
    fig.suptitle(f"CEEMDAN Decomposition - {target}", fontsize=14, fontweight='bold', y=0.995)

    # Color
    line_color = '#1f77b4'  # Blue

    # Convert dates for plotting
    plot_dates = pd.to_datetime(dates)

    # Plot original signal
    ax = axes[0]
    ax.plot(plot_dates, original_signal, color=line_color, linewidth=0.5)
    ax.set_ylabel(target, fontsize=10, fontweight='bold')
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, alpha=0.3)

    # Set y-axis limits for original
    y_margin = (original_signal.max() - original_signal.min()) * 0.1
    ax.set_ylim(original_signal.min() - y_margin, original_signal.max() + y_margin)

    # Plot IMFs
    for i, imf in enumerate(imfs):
        ax = axes[i + 1]
        ax.plot(plot_dates[:len(imf)], imf, color=line_color, linewidth=0.5)
        ax.set_ylabel(f"IMF{i+1}", fontsize=10, fontweight='bold')
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True, alpha=0.3)

        # Set symmetric y-axis for IMFs
        y_max = max(abs(imf.min()), abs(imf.max())) * 1.1
        ax.set_ylim(-y_max, y_max)

    # Plot residue
    ax = axes[-1]
    ax.plot(plot_dates[:len(residue)], residue, color=line_color, linewidth=0.5)
    ax.set_ylabel("Residual", fontsize=10, fontweight='bold')
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%d/%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=0, fontsize=9)

    # Add x-axis label at bottom
    ax.set_xlabel(f"{dates.iloc[0].strftime('%b/%d/%Y %H:%M')} ~ {dates.iloc[-1].strftime('%b/%d/%Y %H:%M')}",
                  fontsize=10)

    # Tight layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.98, hspace=0.1)

    # Save figure
    if output_path is None:
        output_path = SCRIPT_DIR / f"{prefix}_imf_decomposition.png"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return output_path


def plot_ec_decomposition(output_path: Optional[Path] = None, **kwargs) -> Path:
    """Plot EC CEEMDAN decomposition."""
    return plot_imf_decomposition(target="EC", output_path=output_path, **kwargs)


def plot_ph_decomposition(output_path: Optional[Path] = None, **kwargs) -> Path:
    """Plot pH CEEMDAN decomposition."""
    return plot_imf_decomposition(target="pH", output_path=output_path, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Plot CEEMDAN IMF decomposition")
    parser.add_argument('--target', '-t', choices=['ec', 'ph', 'both'], default='both',
                        help="Target variable to plot (default: both)")
    parser.add_argument('--output', '-o', type=str, default=None,
                        help="Output directory (default: same as script)")
    parser.add_argument('--dpi', type=int, default=150,
                        help="DPI for saved figures (default: 150)")
    parser.add_argument('--show', action='store_true',
                        help="Show plot instead of just saving")

    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else SCRIPT_DIR

    if args.target in ['ec', 'both']:
        plot_ec_decomposition(
            output_path=output_dir / "ec_imf_decomposition.png",
            dpi=args.dpi,
            show_plot=args.show
        )

    if args.target in ['ph', 'both']:
        plot_ph_decomposition(
            output_path=output_dir / "ph_imf_decomposition.png",
            dpi=args.dpi,
            show_plot=args.show
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
