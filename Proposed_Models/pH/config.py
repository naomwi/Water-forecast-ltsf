"""
Configuration for Proposed Models
Target: pH
"""

import sys
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ROOT_DIR.parent.parent
DATA_DIR = ROOT_DIR.parent / "data" / "USGs"
MODEL_DIR = ROOT_DIR / "saved_models"
RESULTS_DIR = ROOT_DIR / "results"

# Add project root to path for importing config_global
sys.path.insert(0, str(PROJECT_DIR))
from config_global import (
    DEVICE, NUM_WORKERS, PIN_MEMORY,
    DATA_CONFIG as GLOBAL_DATA_CONFIG,
    TRAIN_CONFIG as GLOBAL_TRAIN_CONFIG,
    HORIZONS,
)

# =============================================================================
# DATA CONFIGURATION - RTX 3090 Optimized
# =============================================================================
DATA_CONFIG = {
    'data_file': GLOBAL_DATA_CONFIG['data_file'],
    'target_col': 'pH',
    'train_ratio': GLOBAL_DATA_CONFIG['train_ratio'],
    'val_ratio': GLOBAL_DATA_CONFIG['val_ratio'],
    'test_ratio': GLOBAL_DATA_CONFIG['test_ratio'],
    'seq_len': GLOBAL_DATA_CONFIG['seq_len'],
    'batch_size': 256,  # Boosted for RTX 3090
    'batch_size_eval': GLOBAL_DATA_CONFIG['batch_size_eval'],
}

# Re-export HORIZONS
HORIZONS = HORIZONS

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
MODEL_CONFIG = {
    'SpikeDLinear': {},
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
TRAIN_CONFIG = GLOBAL_TRAIN_CONFIG.copy()
TRAIN_CONFIG['epochs'] = 150
TRAIN_CONFIG['learning_rate'] = 0.001
TRAIN_CONFIG['early_stopping_patience'] = 15

# =============================================================================
# EXPERIMENTS
# =============================================================================
EXPERIMENTS = {
    'SpikeDLinear': {'description': 'SpikeDLinear (Proposed)'},
}

def get_config_summary():
    print("\n" + "=" * 60)
    print("PROPOSED MODELS (robust-wq-hybrid)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Target: {DATA_CONFIG['target_col']}")
    print(f"Batch Size: {DATA_CONFIG['batch_size']}")
    print(f"Epochs: {TRAIN_CONFIG['epochs']}")
    print("=" * 60)

if __name__ == "__main__":
    get_config_summary()


