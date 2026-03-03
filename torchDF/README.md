# torchDF: DeepFilterNet in Pure PyTorch

This folder contains a **standalone PyTorch implementation** of DeepFilterNet (DF3). It eliminates the need for custom C++/Rust extensions (`libdf`), making it easy to deploy, research, and export to ONNX/WASM.

Inspired by the work in: [grazder/DeepFilterNet - torchDF-changes](https://github.com/grazder/DeepFilterNet/tree/torchDF-changes/torchDF)

---

## 🚀 Quick Start with [uv](https://github.com/astral-sh/uv)

We recommend using **uv** for fast and reproducible environment management.

### 1. Setup Environment
```bash
# Create and activate a virtual environment
uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install onnxruntime onnx tqdm numpy loguru pandas psutil soundfile requests onnx-tool
```

### 2. Basic Denoising (Offline)
Process a folder of noisy audio files:
```bash
uv run torchDF/torch_df_offline.py --input-folder assets/ --output-folder outputs/
```

### 3. Real-time / Streaming (Single File)
Process an audio file frame-by-frame (simulating real-time):
```bash
uv run torchDF/torch_df_streaming.py --audio-path assets/noisy_snr0.wav --output-path outputs/denoised.wav
```

---

## 🧪 Testing & Validation

To verify that the PyTorch implementation matches the original DeepFilterNet output:
```bash
uv run torchDF/test_torchdf.py
```

---

## ⚙️ Quantization & Performance (DF3)

Convert models to **INT8** for significantly smaller sizes (ideal for browser/WASM/mobile).

### Step 1: Download Calibration Data
```bash
uv run torchDF/down_data.py --calib-dir ./calibration_data --max-files 20
```

### Step 2: Export to ONNX & Quantize
Export the standard model and apply static INT8 quantization:
```bash
uv run torchDF/quantize_model.py --model-base-dir DeepFilterNet3 --calib-dir ./calibration_data --output-dir ./exports
```

### Step 3: Optimization Grid Search
Find the best balance between speed and quality by running a grid search:
```bash
uv run torchDF/quantize_gridsearch.py --calib-dir ./calibration_data --output-dir ./quantize/gridsearch
```

### Step 4: Massive Benchmarking
Benchmark speed, CPU usage, and MACs across all generated models:
```bash
uv run torchDF/test_ort_perf_type.py --eval-folder ./quantize/gridsearch --num-runs 10000
```
Detailed results will be saved in `quantize/gridsearch/detailed_ort_benchmark.csv`.

---

## 📂 File Structure
- `torch_df_offline.py`: Standard offline inference.
- `torch_df_streaming.py`: Streaming implementation (ONNX exportable).
- `model_onnx_export.py`: Tool to export Torch model to ONNX.
- `quantize_model.py`: Script for INT8 quantization.
- `test_torchdf.py`: Functional tests comparing Torch vs Original DF.
- `test_ort_perf_type.py`: Advanced performance benchmarking tool.

---

## 📚 Attribution & Training
This implementation focuses on **inference**. For training or detailed architecture info, please visit the official [DeepFilterNet repository](https://github.com/Rikorose/DeepFilterNet).