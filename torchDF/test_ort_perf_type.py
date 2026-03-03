import argparse
import time
import numpy as np
import onnxruntime as ort
import os
import onnx
import psutil

FP32_PATH = '/home/trong/code/DeepFilterNet/quantize/df3_fp32.onnx'
INT8_PATH = '/home/trong/code/DeepFilterNet/quantize/df3_int8.onnx'
FP32_ORT_PATH = '/home/trong/code/DeepFilterNet/quantize/df3_fp32.ort'
INT8_ORT_PATH = '/home/trong/code/DeepFilterNet/quantize/df3_int8.ort'


# ---------------------------------------------------------------------------
# MACs via onnx_tool — runtime input shapes (per-layer accurate)
# ---------------------------------------------------------------------------
# Actual input shapes used at inference time for DeepFilterNet
_RUNTIME_INPUTS = {
    'input_frame':  np.random.randn(480).astype(np.float32),
    'states':       np.zeros(45304, dtype=np.float32),
    'atten_lim_db': np.array([0.0], dtype=np.float32),
}


def estimate_macs_from_onnx(onnx_path: str, input_dict: dict | None = None) -> int:
    """Compute total MACs using onnx_tool, propagating actual runtime input shapes.

    - Loads the ONNX model via onnx_tool.Model
    - Calls graph.shape_infer(actual_inputs) so every intermediate tensor
      gets its true runtime shape (e.g. Conv output after 480-sample frame)
    - Calls graph.profile() to accumulate MACs per layer
    - Returns graph.macs[0]  (integer dense-MACs, excludes sparsity discount)
    """
    if not os.path.exists(onnx_path):
        return 0
    if input_dict is None:
        input_dict = _RUNTIME_INPUTS
    try:
        import onnx_tool
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            # onnx_tool.Model wraps the ONNX graph
            m = onnx_tool.Model(onnx_path, mcfg={'verbose': False})
            # Filter inputs to only those actually present in the model
            sess_inputs = {inp.name for inp in
                           __import__('onnxruntime').InferenceSession(
                               onnx_path, providers=['CPUExecutionProvider']
                           ).get_inputs()}
            actual = {k: v for k, v in input_dict.items() if k in sess_inputs}
            m.graph.shape_infer(actual)
            m.graph.profile()
        macs_list = getattr(m.graph, 'macs', [0, 0])
        # macs[0] = dense MACs, macs[1] = sparse-adjusted MACs
        return int(macs_list[0]) if macs_list else 0
    except ImportError:
        print("  [WARN] onnx_tool not installed – uv pip install onnx-tool")
        return 0
    except Exception as e:
        print(f"  [WARN] onnx_tool profiling failed: {e}")
        return 0


def macs_to_str(macs: int) -> str:
    if macs == 0:
        return "N/A"
    if macs >= 1_000_000_000:
        return f"{macs/1e9:.2f} GMACs"
    if macs >= 1_000_000:
        return f"{macs/1e6:.2f} MMACs"
    if macs >= 1_000:
        return f"{macs/1e3:.2f} KMACs"
    return f"{macs} MACs"


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------
def check_model_info(onnx_path, ort_path):
    print(f"\n{'='*50}")
    print(f"--- MODEL INFO: {os.path.basename(ort_path)} ---")

    onnx_size = 0
    qdq_nodes_count = 0
    int8_ops_count = 0
    ort_size = 0
    macs = 0

    # Check File Sizes
    if os.path.exists(onnx_path):
        onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"ONNX Size: {onnx_size:.2f} MB")

        # Parse ONNX to count quantize nodes if available
        model = onnx.load(onnx_path)
        qdq_nodes = [n.op_type for n in model.graph.node if 'Quantize' in n.op_type or 'Dequantize' in n.op_type]
        int8_ops = [n.op_type for n in model.graph.node if 'QLinear' in n.op_type]

        qdq_nodes_count = len(qdq_nodes)
        int8_ops_count = len(int8_ops)

        print(f"Quantization Nodes: {qdq_nodes_count} QDQ nodes, {int8_ops_count} QLinear ops")
        if qdq_nodes_count > 0 or int8_ops_count > 0:
            print("Status: Verified INT8/Quantized Structure")
        else:
            print("Status: Verified Standard FP32 Structure")

        # MACs
        macs = estimate_macs_from_onnx(onnx_path)
        print(f"MACs (est.)  : {macs_to_str(macs)}")

    if os.path.exists(ort_path):
        ort_size = os.path.getsize(ort_path) / (1024 * 1024)
        print(f"ORT Size : {ort_size:.2f} MB")

        session = ort.InferenceSession(ort_path, providers=['CPUExecutionProvider'])
        print(f"Loaded successfully with {session.get_providers()}")

    return onnx_size, ort_size, qdq_nodes_count, int8_ops_count, macs


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def benchmark_model(model_path, num_runs=10000):
    print(f"\n--- Benchmarking: {os.path.basename(model_path)} ({num_runs} runs) ---")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    input_names = [i.name for i in session.get_inputs()]

    dummy_inputs = {}
    if 'input_frame' in input_names:
        dummy_inputs['input_frame'] = np.random.randn(480).astype(np.float32)
    if 'states' in input_names:
        dummy_inputs['states'] = np.zeros(45304, dtype=np.float32)
    if 'atten_lim_db' in input_names:
        dummy_inputs['atten_lim_db'] = np.array([0.0], dtype=np.float32)

    print("Warming up...")
    for _ in range(100):
        session.run(None, dummy_inputs)

    print("Running...")
    run_times = []
    cpu_samples = []
    proc = psutil.Process()

    for _ in range(num_runs):
        cpu_before = proc.cpu_percent(interval=None)
        start = time.perf_counter()
        session.run(None, dummy_inputs)
        end = time.perf_counter()
        cpu_after = proc.cpu_percent(interval=None)
        run_times.append((end - start) * 1000)   # ms
        # average of before/after to reduce noise
        cpu_samples.append((cpu_before + cpu_after) / 2.0)

    run_times = np.array(run_times)
    cpu_samples = np.array(cpu_samples)

    avg_ms    = np.mean(run_times)
    std_ms    = np.std(run_times)
    var_ms    = np.var(run_times)
    min_ms    = np.min(run_times)
    max_ms    = np.max(run_times)
    pct_99    = np.percentile(run_times, 99)
    pct_95    = np.percentile(run_times, 95)

    avg_cpu   = np.mean(cpu_samples)
    max_cpu   = np.max(cpu_samples)

    print(f"Average CPU  : {avg_ms:.4f} ms")
    print(f"Std Dev      : ±{std_ms:.4f} ms")
    print(f"Variance     : {var_ms:.6f}")
    print(f"Min / Max    : {min_ms:.4f} ms / {max_ms:.4f} ms")
    print(f"95th Pct     : {pct_95:.4f} ms")
    print(f"99th Pct     : {pct_99:.4f} ms")
    print(f"CPU Avg %%    : {avg_cpu:.1f}%%")
    print(f"CPU Max %%    : {max_cpu:.1f}%%")

    deadline_ms = 10.0
    rtf = avg_ms / deadline_ms
    print(f"Mean RTF     : {rtf:.4f}x (lower is better, < 1.0 is real-time)")

    return avg_ms, std_ms, var_ms, pct_99, avg_cpu, max_cpu


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp32-ort', type=str, default='/home/trong/code/DeepFilterNet/quantize/df3_fp32.ort')
    parser.add_argument('--int8-ort', type=str, default='/home/trong/code/DeepFilterNet/quantize/df3_int8.ort')
    parser.add_argument('--eval-folder', type=str, help='Folder containing ORT models to benchmark')
    parser.add_argument('--num-runs', type=int, default=25000)
    args = parser.parse_args()

    print(f"OS Available Providers: {ort.get_available_providers()}")

    if args.eval_folder and os.path.exists(args.eval_folder):
        import glob
        import pandas as pd
        ort_files = glob.glob(os.path.join(args.eval_folder, "*.ort"))

        # Filter out temporary ORT Optimizer files (containing 'with_runtime' or 'without')
        ort_files = [f for f in ort_files if "runtime_opt" not in f]

        print(f"Found {len(ort_files)} ORT models in {args.eval_folder}. Starting massive benchmark...")
        results = []

        for ort_file in sorted(ort_files):
            # Check if there is a matching .onnx file to get node context
            base_name = os.path.splitext(ort_file)[0]
            onnx_file = base_name + ".onnx"
            filename = os.path.basename(ort_file)

            onnx_sz, ort_sz, qdq_nodes, int8_ops, macs = check_model_info(onnx_file, ort_file)

            # Split filename to extract Method and Operations
            # Example: df3_MinMax_Conv_Gemm.ort
            parts = filename.replace("df3_", "").replace(".ort", "").split("_")
            method = parts[0] if len(parts) > 0 else "Unknown"
            ops_quant = parts[1:] if len(parts) > 1 else ["Unknown"]

            if "fp32" in filename:
                method = "None"
                ops_quant = ["Original_FP32"]
            elif "int8" in filename and method == "int8":
                method = "Entropy"
                ops_quant = ["Static"]

            try:
                avg, std, var, pct99, avg_cpu, max_cpu = benchmark_model(ort_file, args.num_runs)
                results.append({
                    "Method": method,
                    "Ops Quantized": "+".join(ops_quant),
                    "Num Runs": args.num_runs,
                    "ONNX Size (MB)": round(onnx_sz, 2),
                    "ORT Size (MB)": round(ort_sz, 2),
                    "QDQ Nodes": qdq_nodes,
                    "MACs": macs,
                    "Avg Mean (ms)": round(avg, 4),
                    "Std Dev (ms)": round(std, 4),
                    "Variance (ms²)": round(var, 6),
                    "99th Percentile MS": round(pct99, 4),
                    "CPU Avg %": round(avg_cpu, 1),
                    "CPU Max %": round(max_cpu, 1),
                    "Filename": filename,
                })
            except Exception as e:
                print(f"Benchmark ERROR on {ort_file}: {e}")

        # Export to Grid Search CSV
        if results:
            df = pd.DataFrame(results)
            csv_path = os.path.join(args.eval_folder, "detailed_ort_benchmark.csv")
            # Save to CSV
            df.to_csv(csv_path, mode='a', header=not file_exists, index=False)
            print(f"\nGridSearch benchmarking results saved/appended to: {csv_path}")
            print(df.to_string())
        return

    print("Beginning scalar tests...\n")

    if os.path.exists(args.fp32_ort):
        fp32_onnx = args.fp32_ort.replace(".ort", ".onnx")
        check_model_info(fp32_onnx, args.fp32_ort)

    if os.path.exists(args.int8_ort):
        int8_onnx = args.int8_ort.replace(".ort", ".onnx")
        check_model_info(int8_onnx, args.int8_ort)

    print("\n" + "="*50)

    t_fp32_res = benchmark_model(args.fp32_ort, args.num_runs) if os.path.exists(args.fp32_ort) else None
    t_int8_res = benchmark_model(args.int8_ort, args.num_runs) if os.path.exists(args.int8_ort) else None

    if t_fp32_res and t_int8_res:
        t_fp32, std_fp32, var_fp32, pct99_fp32, cpu_avg_fp32, cpu_max_fp32 = t_fp32_res
        t_int8, std_int8, var_int8, pct99_int8, cpu_avg_int8, cpu_max_int8 = t_int8_res

        print("\n" + "="*50 + "\nSUMMARY COMPARISON")
        print(f"{'Metric':<15} | {'FP32 Model':<25} | {'INT8 Model':<25}")
        print("-" * 70)
        print(f"{'Mean Speed':<15} | {t_fp32:.4f} ± {std_fp32:.4f} ms    | {t_int8:.4f} ± {std_int8:.4f} ms")
        print(f"{'CPU Avg %':<15} | {cpu_avg_fp32:.1f}%                   | {cpu_avg_int8:.1f}%")
        print(f"{'CPU Max %':<15} | {cpu_max_fp32:.1f}%                   | {cpu_max_int8:.1f}%")

        speedup = t_fp32 / t_int8
        diff_str = f"{abs(1-speedup)*100:.1f}% " + ("faster" if speedup > 1.0 else "slower")
        print(f"\nRESULT: INT8 is {diff_str} than FP32 (Speedup ratio: {speedup:.2f}x)")

        if speedup < 1.0:
            print("\nNOTE:")
            print("INT8 x86 CPU emulation without hardware AVX512-VNNI or equivalent extensions")
            print("typically requires unpacking float vectors from uint8 memory formats back and forth,")
            print("which can result in equal or slightly slower raw speed than pure FP32.")
            print("The primary benefit of INT8 streaming models on browsers/WASM is the RAM size reduction.")


if __name__ == '__main__':
    main()