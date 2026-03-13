import sys
import os
import json

# Add parent directory to path so we can import from the existing python files
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def get_system_context():
    try:
        from data_loader import get_global_kpi_summary
        global_data_context = get_global_kpi_summary()
        
        report_path = os.path.join(os.path.dirname(__file__), "..", "documents", "report_context.txt")
        report_context = ""
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report_context = f.read()

        system_instruction = (
            "You are HydroBot, an expert AI Water Quality Data Analyst designed for the FPT University Capstone Project.\n"
            "Your role is to help users understand water quality metrics (EC, pH, Temp, Flow, DO, Turbidity) and the performance of your group's predictive models.\n\n"
            "I am providing you with the ENTIRE BENCHMARK RESULTS of all models built in this project across multiple sites and horizons. "
            "You must use this data to confidently answer any question about which model is best, what the MSE/R2 is, and how the models compare.\n\n"
            f"--- BENCHMARK RESULTS ---\n{global_data_context}\n\n"
            f"--- PROJECT REPORT CONTEXT ---\n{report_context}\n\n"
            "Speak professionally but be helpful. Provide concise, bolded, and data-backed answers based tightly on the provided report context and benchmark data."
        )
        print(json.dumps({"success": True, "context": system_instruction}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

def get_live_context(prompt):
    try:
        from prediction_loader import detect_intent, build_prediction_context
        intent = detect_intent(prompt)
        if intent.get("is_prediction"):
            pred_context = build_prediction_context(
                intent["features"], intent["horizon"]
            )
            print(json.dumps({"success": True, "is_prediction": True, "context": pred_context}))
        else:
            print(json.dumps({"success": True, "is_prediction": False, "context": ""}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

def get_raw_data(site, target):
    try:
        import pandas as pd
        csv_path = os.path.join(os.path.dirname(__file__), "..", "Deep_Baselines", "data", "USGs", "water_data_sample.csv.gz")
        df = pd.read_csv(csv_path, compression='gzip')
        
        # Filter by site
        df = df[df['site_no'] == int(site)]
        
        if df.empty:
            print(json.dumps({"success": False, "error": f"No data found for site {site}"}))
            return
            
        # Select target and time, handle missing target
        if target not in df.columns:
            print(json.dumps({"success": False, "error": f"Target {target} not found in dataset"}))
            return
            
        # Sort by Time to be safe
        if 'Time' in df.columns:
            df = df.sort_values('Time')
            
        # Calculate stats before slicing
        target_series = df[target].dropna()
        stats = {
            "mean": float(target_series.mean()) if not target_series.empty else 0,
            "min": float(target_series.min()) if not target_series.empty else 0,
            "max": float(target_series.max()) if not target_series.empty else 0,
            "std": float(target_series.std()) if not target_series.empty else 0,
            "count": int(target_series.count())
        }
        
        # We will downsample the data to reduce the payload size while maintaining the shape
        # For a dataset of ~22k rows, taking every Nth row (e.g. N=20) or using a rolling mean 
        # is a good strategy for visualization. Let's take every 15th row which gives ~1400 points.
        df_sample = df.iloc[::15].copy()
        
        data = []
        for i, row in enumerate(df_sample.itertuples()):
            val = getattr(row, target)
            if pd.notna(val):
                # Extract year-month from Time string (e.g., "2021-01-01 05:00:00+00:00" -> "2021-01")
                time_str = getattr(row, 'Time', str(i))
                display_time = time_str[:7] if isinstance(time_str, str) and len(time_str) >= 7 else str(i)
                
                data.append({
                    "time": display_time,
                    "actual": val
                })
                
        print(json.dumps({
            "success": True, 
            "data": data,
            "stats": stats
        }))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

def get_series_data(site, target, horizon):
    try:
        import pandas as pd
        # Load the metrics to get R2 and MAPE
        metrics_path = os.path.join(os.path.dirname(__file__), "..", "Proposed_Models", target, "results", f"site_{site}", "metrics", f"SpikeDLinear_h{horizon}.csv")
        r2 = 0
        mape = 0
        if os.path.exists(metrics_path):
            metrics_df = pd.read_csv(metrics_path)
            if not metrics_df.empty:
                r2 = metrics_df['R2'].iloc[0] if 'R2' in metrics_df.columns else 0
                mape = metrics_df['MAPE'].iloc[0] if 'MAPE' in metrics_df.columns else 0

        # Load the series
        series_path = os.path.join(os.path.dirname(__file__), "..", "Proposed_Models", target, "results", f"site_{site}", "series", f"series_SpikeDLinear_P{horizon}_{target}.csv")
        if not os.path.exists(series_path):
            print(json.dumps({"success": False, "error": f"Series file not found: {series_path}"}))
            return

        series_df = pd.read_csv(series_path)
        
        # Load raw data to get timestamps
        raw_path = os.path.join(os.path.dirname(__file__), "..", "Deep_Baselines", "data", "USGs", "water_data_sample.csv.gz")
        raw_df = pd.read_csv(raw_path, compression='gzip')
        raw_df = raw_df[raw_df['site_no'] == int(site)]
        if 'Time' in raw_df.columns:
            raw_df = raw_df.sort_values('Time')
            # The series corresponds to the test set, which is the last len(series_df) rows
            if len(raw_df) >= len(series_df):
                timestamps = raw_df['Time'].tail(len(series_df)).values
                series_df['Time'] = timestamps
            else:
                series_df['Time'] = [str(i) for i in range(len(series_df))]
        else:
            series_df['Time'] = [str(i) for i in range(len(series_df))]

        # Downsample the series_df for visualization (e.g., max 500 points) to avoid lagging
        step = max(1, len(series_df) // 800)
        df_sample = series_df.iloc[::step].copy()

        data = []
        for i, row in enumerate(df_sample.itertuples()):
            # format time to YYYY-MM-DD
            time_str = str(row.Time)[:10] if hasattr(row, 'Time') else str(i)
            data.append({
                "time": time_str,
                "actual": float(row.Actual),
                "predicted": float(row.Predicted)
            })

        print(json.dumps({
            "success": True, 
            "data": data,
            "metrics_extra": {"r2": float(r2), "mape": float(mape)}
        }))

    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))

if __name__ == "__main__":
    action = sys.argv[1]
    if action == "get_system_context":
        get_system_context()
    elif action == "get_live_context":
        get_live_context(sys.argv[2])
    elif action == "get_raw_data":
        get_raw_data(sys.argv[2], sys.argv[3])
    elif action == "get_series_data":
        get_series_data(sys.argv[2], sys.argv[3], sys.argv[4])
