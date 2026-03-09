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
uv pip install onnxruntime onnx tqdm soundfile numpy loguru pandas psutil soundfile requests onnx-tool
```

### 2. Download & Extract Models
Ensure the model files are extracted:
```bash
unzip models/DeepFilterNet3.zip -d models/
```

### 3. Basic Usage (PyTorch)
```bash
# Offline Batch Processing
uv run torchDF/torch_df_offline.py --input-folder assets/ --output-folder outputs/

# Frame-by-frame Streaming Simulation
uv run torchDF/torch_df_streaming.py --audio-path assets/noisy_snr0.wav --output-path outputs/denoised.wav
```

---

## 🛠️ Model Export & Quantization Pipeline

This toolset provides a complete pipeline to convert PyTorch models into highly optimized **INT8 ONNX/ORT** models for web or mobile deployment.

### 1. Exporting to ONNX (`model_onnx_export.py`)
Convert the streaming PyTorch model to ONNX format. This script handles the complex state-management logic required for streaming.

```bash
uv run torchDF/model_onnx_export.py \
    --output-path ./exports/df3_fp32.onnx \
    --test \
    --performance \
    --ort
```
*   `--ort`: Additionally saves the model in the `.ort` format (optimized for runtime).
*   `--test`: Verifies that ORT output matches PyTorch output exactly (within tolerance).
*   `--performance`: Measures the median iteration time.
*   `--inference-path`: Runs the exported model on a specific file to check audio quality.

### 2. Static INT8 Quantization (`quantize_model.py`)
Perform static quantization using calibration data to achieve minimal quality loss.

```bash
# 1. First, download some calibration data (clean speech preferably)
uv run torchDF/down_data.py --calib-dir ./calibration_data --max-files 20

# 2. Run quantization (includes export if --onnx-path is omitted)
uv run torchDF/quantize_model.py \
    --model-base-dir DeepFilterNet3 \
    --calib-dir ./calibration_data \
    --output-dir ./export_quantized \
    --test-audio assets/noisy_snr0.wav
```
This script handles the full workflow: **Export → Calibration → Quantization → ORT Conversion → Functional Test.**

### 3. Optimization Grid Search (`quantize_gridsearch.py`)
Different hardware treats INT8 operations differently. Use this script to find the absolute best combination of calibration methods and quantized operators.

```bash
uv run torchDF/quantize_gridsearch.py \
    --calib-dir ./calibration_data \
    --output-dir ./quantize/gridsearch \
    --test-audio assets/noisy_snr0.wav
```
It evaluates permutations of:
*   **Calibration Methods**: `MinMax`, `Entropy`, `Percentile`, `Distribution`.
*   **Operator Selection**: Quantizing `Conv` only, `Conv+Gemm`, `Conv+Gemm+Add`, etc.

### 4. Detailed Performance Analysis (`test_ort_perf_type.py`)
The ultimate benchmarking tool to analyze models produced by the grid search or individual exports.

```bash
uv run torchDF/test_ort_perf_type.py \
    --eval-folder ./quantize/gridsearch \
    --num-runs 10000
```
It generates a comprehensive report (`detailed_ort_benchmark.csv`) including:
*   **Mean/99th Percentile Latency**: Precise timing for real-time viability.
*   **Compute (MACs)**: Calculated using `onnx-tool`.
*   **CPU & Memory Usage**: Runtime resource footprint.
*   **Model Stats**: ONNX/ORT file sizes and QDQ node counts.

---

## 📂 File Summary
- `torch_df_offline.py`: Standard offline inference (PyTorch).
- `torch_df_streaming.py`: Streaming implementation (PyTorch, ONNX-ready).
- `model_onnx_export.py`: Handles PyTorch → ONNX conversion logic.
- `quantize_model.py`: Easy, single-command static quantization.
- `quantize_gridsearch.py`: Exhaustive search for optimal INT8 parameters.
- `test_ort_perf_type.py`: Advanced benchmarking and MACs estimation.
- `down_data.py`: Helper to download calibration datasets.
- `test_torchdf.py`: Validation tests for the PyTorch implementation.

---

## 📚 Attribution
This implementation focuses on **inference optimization**. For training or detailed architecture info, please visit the official [DeepFilterNet repository](https://github.com/Rikorose/DeepFilterNet).