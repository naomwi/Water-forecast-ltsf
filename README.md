# AquaPredict: Water Quality Forecasting Framework

**AquaPredict** is a comprehensive, rigorous benchmarking framework developed as part of our group project for predicting critical water quality metrics (EC and pH) of a water treatment factory. 

The project aims to provide strong baseline models and introduces **SpikeDLinear**, an advanced Proposed Model that leverages spiking neural network principles and DLinear structures to handle non-stationary and volatile water quality data effectively.

## 👥 Team Members
- Member 1: [Trần Quang Thái] - [0009-0000-4674-4388]
- Member 2: [Nguyễn Tấn Khôi Nguyên] - [0009-0009-7970-9883]
- Member 3: [Trịnh Tiến Khải] - [0009-0002-0447-5743]
- Member 4: [Phan Tấn Phước] - [0009-0003-7078-9406]

## 📁 Project Structure

- **`Proposed_Models/`**: Source code, checkpoints, and experimental results for the advanced `SpikeDLinear` model.
- **`CEEMD_Baselines/`**: Implementations of hybrid `CEEMD-DLinear` and `CEEMD-NLinear` baseline models.
- **`Deep_Baselines/`**: Standard deep learning baselines for time-series forecasting (`LSTM`, `PatchTST`, `Transformer`).
- **`dashboard/`**: A Streamlit interactive web dashboard for real-time visualization of predictions and an integrated AI chatbot (AquaBot).
- **`scripts/`**: Auxiliary training, hyperparameter tuning, and evaluation scripts.
- **`visual/`**: Scripts and exported graphics for data visualization and report materials.

---

## 🚀 Quick Start

### 1. Installation Environment
It is highly recommended to use a virtual environment before installing the dependencies.
- **Windows**: Run `install.bat`
- **Linux/Mac**: Run `bash install.sh`

Alternatively, manually install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (Optional but Recommended)
For the interactive AI Chatbot (**AquaBot**) to function inside the dashboard, configure your Gemini API Key.
Create a `.env` file in the root directory and add:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 3. Launch the Dashboard
To start the Analytical Dashboard and explore the pre-computed benchmarking results on your `localhost`:
- **Windows**: Run `run_local_dashboard.bat`
- **Manual Command**: 
  ```bash
  python -m streamlit run dashboard/app.py
  ```

This will open the dashboard in your default browser at `http://localhost:8501`.

## 📈 Experiments & Results
All pre-run benchmarking results are organized inside their respective models' `results/` folders. 
If you wish to re-run experiments from scratch, you can use the generalized runners available in the `scripts/` directory (e.g., `python scripts/run_all_experiments.py --quick`).

### 4. Run All Experiments
To run all experiments, execute the `run_all_experiments.py` script:
- **Windows**: Run `python run_all_experiments.py`
- **Manual Command**: 
  ```bash
  python run_all_experiments.py
  ```

This will run all experiments and save the results in their respective models' `results/` folders.

## Have a beautiful day - naomwi
