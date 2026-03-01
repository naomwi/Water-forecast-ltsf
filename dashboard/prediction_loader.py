"""
Prediction Loader — Detects user intent and loads model predictions for Gemini analysis.

Flow:
  1. detect_intent(message) → {"features": [...], "horizon": int, "is_prediction": bool}
  2. load_predictions(feature, horizon, site) → last N predicted values
  3. build_prediction_context(features, horizon, site) → formatted string for Gemini
"""

import re
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent

# ============================================================================
# FEATURE & HORIZON MAPPINGS
# ============================================================================
FEATURE_KEYWORDS = {
    'EC': ['ec', 'electrical conductivity', 'conductivity', 'điện dẫn', 'độ dẫn điện', 'salinity', 'mặn'],
    'pH': ['ph', 'độ ph', 'acid', 'axit'],
    'Temp': ['temp', 'temperature', 'nhiệt độ', 'nhiet do'],
    'Flow': ['flow', 'lưu lượng', 'luu luong', 'dòng chảy', 'dong chay'],
    'DO': ['do', 'dissolved oxygen', 'oxy hòa tan', 'oxy'],
    'Turbidity': ['turbidity', 'độ đục', 'do duc', 'đục'],
}

# "water quality" triggers ALL features
WATER_QUALITY_KEYWORDS = [
    'chất lượng nước', 'chat luong nuoc', 'water quality',
    'nước sạch', 'nuoc sach', 'clean water', 'ô nhiễm', 'o nhiem',
    'pollution', 'contamination', 'tổng hợp', 'tong hop', 'overall',
    'all features', 'tất cả', 'tat ca', 'toàn bộ', 'toan bo',
    'predict all', 'dự đoán nước', 'du doan nuoc',
]

# Time horizon mapping
HORIZON_PATTERNS = {
    6:   [r'6\s*(?:h|giờ|gio|hour)', r'6\s*tiếng'],
    12:  [r'12\s*(?:h|giờ|gio|hour)', r'12\s*tiếng', r'nửa ngày', r'nua ngay'],
    24:  [r'24\s*(?:h|giờ|gio|hour)', r'1\s*(?:ngày|ngay|day)', r'một ngày', r'mot ngay'],
    48:  [r'48\s*(?:h|giờ|gio|hour)', r'2\s*(?:ngày|ngay|day)', r'hai ngày', r'hai ngay'],
    96:  [r'96\s*(?:h|giờ|gio|hour)', r'4\s*(?:ngày|ngay|day)', r'bốn ngày', r'bon ngay'],
    168: [r'168\s*(?:h|giờ|gio|hour)', r'7\s*(?:ngày|ngay|day)', r'(?:1|một|mot)\s*tuần',
          r'(?:1|một|mot)\s*tuan', r'one week', r'a week'],
}

# Fallback: map N days → nearest horizon
DAY_PATTERN = re.compile(r'(\d+)\s*(?:ngày|ngay|day|days)', re.IGNORECASE)
HOUR_PATTERN = re.compile(r'(\d+)\s*(?:giờ|gio|h(?:our)?s?|tiếng|tieng)', re.IGNORECASE)

VALID_HORIZONS = [6, 12, 24, 48, 96, 168]

DEFAULT_SITE = 1463500


def _nearest_horizon(hours: int) -> int:
    """Find the nearest valid horizon."""
    return min(VALID_HORIZONS, key=lambda h: abs(h - hours))


def detect_intent(message: str) -> dict:
    """
    Analyze the user's message to detect prediction intent.
    
    Returns:
        {
            "is_prediction": bool,
            "features": list[str],      # e.g. ['EC'] or ['EC','pH','Temp','Flow','DO','Turbidity']
            "horizon": int,             # e.g. 24
            "raw_query": str,
        }
    """
    msg_lower = message.lower()
    
    # --- Detect features ---
    detected_features = []
    
    # Check for "water quality" (all features)
    is_water_quality = any(kw in msg_lower for kw in WATER_QUALITY_KEYWORDS)
    
    if is_water_quality:
        detected_features = list(FEATURE_KEYWORDS.keys())  # All 6
    else:
        for feature, keywords in FEATURE_KEYWORDS.items():
            for kw in keywords:
                if kw in msg_lower:
                    if feature not in detected_features:
                        detected_features.append(feature)
                    break
    
    # --- Detect horizon ---
    detected_horizon = None
    
    # Check explicit patterns first
    for horizon_val, patterns in HORIZON_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, msg_lower):
                detected_horizon = horizon_val
                break
        if detected_horizon:
            break
    
    # Fallback: extract N days / N hours
    if not detected_horizon:
        day_match = DAY_PATTERN.search(msg_lower)
        hour_match = HOUR_PATTERN.search(msg_lower)
        
        if day_match:
            days = int(day_match.group(1))
            detected_horizon = _nearest_horizon(days * 24)
        elif hour_match:
            hours = int(hour_match.group(1))
            detected_horizon = _nearest_horizon(hours)
    
    # Default horizon if prediction intent but no horizon specified
    if not detected_horizon and detected_features:
        detected_horizon = 24  # Default to 24h
    
    is_prediction = len(detected_features) > 0
    
    return {
        "is_prediction": is_prediction,
        "features": detected_features,
        "horizon": detected_horizon,
        "raw_query": message,
    }


def load_predictions(feature: str, horizon: int, site: int = DEFAULT_SITE, n_last: int = None) -> dict:
    """
    Load forecasted values for a specific feature from SpikeDLinear results.
    
    Returns dict with predicted values summary, or None if file not found.
    """
    if n_last is None:
        n_last = horizon  # Show N predictions matching the horizon
    
    series_path = (
        PROJECT_DIR / "Proposed_Models" / feature / "results" 
        / f"site_{site}" / "series" / f"series_SpikeDLinear_P{horizon}_{feature}.csv"
    )
    
    if not series_path.exists():
        return None
    
    try:
        df = pd.read_csv(series_path)
        tail = df.tail(n_last)
        
        predicted_vals = tail['Predicted'].values
        actual_vals = tail['Actual'].values
        
        return {
            "feature": feature,
            "horizon": horizon,
            "site": site,
            "predicted_mean": float(predicted_vals.mean()),
            "predicted_min": float(predicted_vals.min()),
            "predicted_max": float(predicted_vals.max()),
            "predicted_last": float(predicted_vals[-1]),
            "actual_last": float(actual_vals[-1]),
            "n_points": len(predicted_vals),
            "predicted_values": [round(v, 2) for v in predicted_vals.tolist()],
        }
    except Exception as e:
        print(f"Error loading predictions for {feature}: {e}")
        return None


def build_prediction_context(features: list, horizon: int, site: int = DEFAULT_SITE) -> str:
    """
    Load predictions for multiple features and build a formatted context string for Gemini.
    """
    lines = [
        f"[PREDICTION DATA — Site {site}, Horizon {horizon}h ({horizon // 24:.0f} days)]" 
        if horizon >= 24 else
        f"[PREDICTION DATA — Site {site}, Horizon {horizon}h]"
    ]
    
    available = []
    unavailable = []
    
    for feature in features:
        result = load_predictions(feature, horizon, site)
        if result:
            available.append(result)
            lines.append(
                f"\n  {feature}:"
                f"\n    Predicted Mean: {result['predicted_mean']:.2f}"
                f"\n    Predicted Range: [{result['predicted_min']:.2f} — {result['predicted_max']:.2f}]"
                f"\n    Latest Predicted: {result['predicted_last']:.2f}"
                f"\n    Latest Actual: {result['actual_last']:.2f}"
            )
        else:
            unavailable.append(feature)
    
    if unavailable:
        lines.append(f"\n  ⚠ No trained results available for: {', '.join(unavailable)}")
    
    if available:
        lines.append(
            "\n\n[INSTRUCTION] Based on the predicted values above, provide a comprehensive "
            "water quality assessment for a WATER TREATMENT PLANT (Nhà máy xử lý nước), NOT a river or natural ecosystem. "
            "Evaluate if the water is ready for the next treatment phase or safe for distribution. "
            "Referencing standard treatment thresholds (e.g., EC, pH 6.5-8.5, DO, Turbidity < 1 NTU for drinking, < 5 NTU max). "
            "Do NOT talk about fish or aquatic life. "
            "If any metric is concerning, explain what operational actions the plant manager should take "
            "(e.g., adjust aeration, add coagulants, backwash filters, modify pH with chemicals). "
            "Respond in Vietnamese. Format your response clearly with emojis, bold text, and bullet points."
        )
    
    return "\n".join(lines)
