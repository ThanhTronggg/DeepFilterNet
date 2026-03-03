import os
import argparse
import subprocess
import glob
import itertools
import pandas as pd
import time
import numpy as np
import onnxruntime as ort
import onnx

def benchmark_model_internal(model_path, num_runs=5000):
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    input_names = [i.name for i in session.get_inputs()]
    
    dummy_inputs = {}
    if 'input_frame' in input_names:
        dummy_inputs['input_frame'] = np.random.randn(480).astype(np.float32)
    if 'states' in input_names:
        dummy_inputs['states'] = np.zeros(45304, dtype=np.float32)
    if 'atten_lim_db' in input_names:
        dummy_inputs['atten_lim_db'] = np.array([0.0], dtype=np.float32)
        
    for _ in range(100):
        session.run(None, dummy_inputs)
        
    run_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, dummy_inputs)
        end = time.perf_counter()
        run_times.append((end - start) * 1000)
        
    return np.mean(run_times), np.std(run_times), np.var(run_times), np.percentile(run_times, 99)

def run_quantize(calib_method_name, ops_list, base_dir, calib_dir, test_audio, output_dir):
    ops_str = "_".join(ops_list) if ops_list else "None"
    config_name = f"df3_{calib_method_name}_{ops_str}"
    
    # We use a temporary script to run quantization cleanly without modifying the main script iteratively
    wrapper_script_path = os.path.join(output_dir, f"tmp_{config_name}_runner.py")
    
    with open(wrapper_script_path, "w") as f:
        f.write(f"""import os
import sys
# Insert torchDF path directly
sys.path.insert(0, "/home/trong/code/DeepFilterNet/torchDF")

import onnx
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationMethod
from quantize_model import DeepFilterNetCalibrationReader

class TruncatedReader(DeepFilterNetCalibrationReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Cap at 5 files instead of default 400 for speed
        self.wav_files = self.wav_files[:5]
        print(f"Truncated calibration files to {{len(self.wav_files)}}")

def do_quantize():
    fp32_model = os.path.join("{output_dir}", "df3_fp32.onnx")
    int8_model = os.path.join("{output_dir}", "{config_name}.onnx")
    
    if not os.path.exists(fp32_model):
        from torch_df_streaming import TorchDFPipeline
        from quantize_model import export_to_onnx
        pipeline = TorchDFPipeline(model_base_dir="{base_dir}", device='cpu')
        export_to_onnx(pipeline, fp32_model)
        
    dr = TruncatedReader("{calib_dir}", fp32_model_path=fp32_model)
    onnx_model = onnx.load(fp32_model)
    dr.input_names = [inp.name for inp in onnx_model.graph.input]
    
    op_types = {ops_list}
    if "{calib_method_name}" == "MinMax":
        method = CalibrationMethod.MinMax
    elif "{calib_method_name}" == "Percentile":
        method = CalibrationMethod.Percentile
    elif "{calib_method_name}" == "Distribution":
        method = CalibrationMethod.Distribution
    else:
        method = CalibrationMethod.Entropy
        
    quantize_static(
        fp32_model,
        int8_model,
        dr,
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=method,
        op_types_to_quantize=op_types
    )
if __name__ == '__main__':
    do_quantize()
""")

    try:
        # Capture output so we can print it on failure
        result = subprocess.run(
            ["/home/trong/code/DeepFilterNet/.venv/bin/python3.10", wrapper_script_path], 
            check=True, 
            capture_output=True, 
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"[{config_name}] Failed to quantize.")
        print("--- ERROR OUTPUT ---")
        print(e.stdout)
        print(e.stderr)
        print("--------------------")
        return None
        
    int8_onnx = os.path.join(output_dir, f"{config_name}.onnx")
    int8_ort = os.path.join(output_dir, f"{config_name}.ort")
    
    # Convert to ORT format
    subprocess.run(["/home/trong/code/DeepFilterNet/.venv/bin/python3.10", "-m", "onnxruntime.tools.convert_onnx_models_to_ort", int8_onnx, "--output_dir", output_dir], check=True, capture_output=True)
    
    return int8_onnx, int8_ort

def generate_grid_search(model_base_dir, calib_dir, test_audio, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    methods = ["MinMax", "Entropy", "Percentile", "Distribution"]
    
    # Define combinations
    op_combinations = [
        ["Conv"],
        ["Conv", "MatMul"],
        ["Conv", "Add"],
        ["Conv", "MatMul", "Add"]
    ]
    
    results = []
    csv_path = os.path.join(output_dir, "gridsearch_results.csv")
    print(f"\n--- [GridSearch Baseline] Original FP32 ---")
    fp32_onnx = os.path.join(output_dir, "df3_fp32.onnx")
    if not os.path.exists(fp32_onnx):
        from torch_df_streaming import TorchDFPipeline
        from quantize_model import export_to_onnx
        pipeline = TorchDFPipeline(model_base_dir=model_base_dir, device='cpu')
        export_to_onnx(pipeline, fp32_onnx)
        
    fp32_ort = os.path.join(output_dir, "df3_fp32.ort")
    if not os.path.exists(fp32_ort):
        subprocess.run(["/home/trong/code/DeepFilterNet/.venv/bin/python3.10", "-m", "onnxruntime.tools.convert_onnx_models_to_ort", fp32_onnx, "--output_dir", output_dir], check=True, capture_output=True)
        
    fp32_onnx_size = os.path.getsize(fp32_onnx) / (1024 * 1024)
    fp32_ort_size = os.path.getsize(fp32_ort) / (1024 * 1024)
    
    try:
        avg_ms, std_ms, var_ms, pct_99 = benchmark_model_internal(fp32_ort, num_runs=25000)
    except Exception as e:
        print(f"Benchmark failed: {e}")
        avg_ms, std_ms, var_ms, pct_99 = (-1, -1, -1, -1)
        
    results.append({
        "Method": "None",
        "Ops Quantized": "Original_FP32",
        "ONNX Size (MB)": round(fp32_onnx_size, 2),
        "ORT Size (MB)": round(fp32_ort_size, 2),
        "QDQ Nodes": 0,
        "Avg MS": round(avg_ms, 4) if avg_ms != -1 else None,
        "Std Dev MS": round(std_ms, 4) if std_ms != -1 else None,
        "Variance MS": round(var_ms, 6) if var_ms != -1 else None,
        "99th Percentile MS": round(pct_99, 4) if pct_99 != -1 else None,
        "Filename": "df3_fp32.onnx"
    })
    
    total = len(methods) * len(op_combinations)
    current = 0
    
    for method in methods:
        for ops in op_combinations:
            current += 1
            print(f"\\n--- [GridSearch {current}/{total}] Method: {method} | Ops: {ops} ---")
            
            paths = run_quantize(method, ops, model_base_dir, calib_dir, test_audio, output_dir)
            if not paths:
                continue
                
            onnx_path, ort_path = paths
            
            onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
            ort_size = os.path.getsize(ort_path) / (1024 * 1024)
            
            model = onnx.load(onnx_path)
            qdq_nodes = [n.op_type for n in model.graph.node if 'Quantize' in n.op_type or 'Dequantize' in n.op_type]
            
            print(f"Running Benchmark on {ort_path}...")
            try:
                avg_ms, std_ms, var_ms, pct_99 = benchmark_model_internal(ort_path, num_runs=25000)
            except Exception as e:
                print(f"Benchmark failed: {e}")
                avg_ms, std_ms, var_ms, pct_99 = (-1, -1, -1, -1)
                
            results.append({
                "Method": method,
                "Ops Quantized": "+".join(ops),
                "ONNX Size (MB)": round(onnx_size, 2),
                "ORT Size (MB)": round(ort_size, 2),
                "QDQ Nodes": len(qdq_nodes),
                "Avg MS": round(avg_ms, 4) if avg_ms != -1 else None,
                "Std Dev MS": round(std_ms, 4) if std_ms != -1 else None,
                "Variance MS": round(var_ms, 6) if var_ms != -1 else None,
                "99th Percentile MS": round(pct_99, 4) if pct_99 != -1 else None,
                "Filename": os.path.basename(onnx_path)
            })
            
            df = pd.DataFrame(results)
            df.to_csv(csv_path, index=False)
            print(f"Result saved to CSV at {csv_path}")
            
    if 'df' in locals():
        print(df.to_string())
        print(f"\\nGridSearch completed successfully! Saved to: {csv_path}")
    else:
        print("No results generated.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-base-dir', type=str, default='/home/trong/code/DeepFilterNet/models/DeepFilterNet3/DeepFilterNet3')
    parser.add_argument('--calib-dir', type=str, default='/home/trong/code/DeepFilterNet_quantize/noisy_wav')
    parser.add_argument('--test-audio', type=str, default='/home/trong/code/DeepFilterNet_quantize/test_wav/audio-file.wav')
    parser.add_argument('--output-dir', type=str, default='/home/trong/code/DeepFilterNet/quantize/gridsearch')
    args = parser.parse_args()
    
    generate_grid_search(args.model_base_dir, args.calib_dir, args.test_audio, args.output_dir)