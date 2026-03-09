import os
import time
import argparse
import numpy as np
import onnxruntime as ort
import torchaudio
import csv
import psutil


def benchmark_model(onnx_path, input_wav, num_runs=1):
    if not os.path.exists(onnx_path):
        return None

    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    noisy, sr = torchaudio.load(input_wav)
    noisy = noisy.mean(0).unsqueeze(0).numpy()

    hop = 480
    chunks = [noisy[0, i : i + hop] for i in range(0, noisy.shape[1], hop)]
    if len(chunks[-1]) < hop:
        chunks[-1] = np.pad(chunks[-1], (0, hop - len(chunks[-1])))

    input_info = {i.name: i.shape for i in session.get_inputs()}

    # Warm up to avoid cold start impact
    states = np.zeros(45304, dtype=np.float32)
    atten_lim = np.array([0.0], dtype=np.float32)
    for chunk in chunks:  # warmup 1 full pass over all chunks
        inputs = {
            "input_frame": chunk.astype(np.float32),
            "states": states,
            "atten_lim_db": atten_lim,
        }
        inputs = {k: v for k, v in inputs.items() if k in input_info}
        try:
            out = session.run(None, inputs)
            out_names = [o.name for o in session.get_outputs()]
            states = (
                out[out_names.index("out_states")]
                if "out_states" in out_names
                else out[1]
            )
        except Exception as e:
            print(f"Error during warmup: {e}")
            return None

    frame_times = []

    process = psutil.Process()
    process.cpu_percent(interval=None)  # Initialize CPU tracking
    start_mem = process.memory_info().rss
    max_rss = start_mem

    for run in range(num_runs):
        states = np.zeros(45304, dtype=np.float32)
        atten_lim = np.array([0.0], dtype=np.float32)
        for chunk in chunks:
            inputs = {
                "input_frame": chunk.astype(np.float32),
                "states": states,
                "atten_lim_db": atten_lim,
            }
            inputs = {k: v for k, v in inputs.items() if k in input_info}

            start_time = time.perf_counter()
            out = session.run(None, inputs)
            end_time = time.perf_counter()

            frame_times.append((end_time - start_time) * 1000)  # time in milliseconds

            # Record max memory over inference
            current_mem = process.memory_info().rss
            if current_mem > max_rss:
                max_rss = current_mem

            # Update states for next iteration
            out_names = [o.name for o in session.get_outputs()]
            states = (
                out[out_names.index("out_states")]
                if "out_states" in out_names
                else out[1]
            )

    cpu_percent = round(process.cpu_percent(interval=0.1), 2)

    avg_time = np.mean(frame_times)
    var_time = np.var(frame_times)
    min_time = np.min(frame_times)
    max_time = np.max(frame_times)

    max_ram_mb = (max_rss - start_mem) / (1024 * 1024)

    return {
        "model": os.path.basename(onnx_path),
        "size_mb": round(file_size_mb, 2),
        "avg_time_ms": round(avg_time, 5),
        "min_time_ms": round(min_time, 5),
        "max_time_ms": round(max_time, 5),
        "variance": round(var_time, 5),
        "cpu_percent": cpu_percent,
        "max_ram_mb": round(max_ram_mb, 2),
        "num_iterations": len(frame_times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-dir",
        type=str,
        default="/home/trong/code/DeepFilterNet_4/torchDF",
        help="Directory with ONNX models",
    )
    parser.add_argument(
        "--test-audio", type=str, required=True, help="Input wav file for testing"
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="benchmark_onnx_models.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of full inference passes over the audio",
    )
    args = parser.parse_args()

    # The 4 models exported by our quantize_model.py
    model_names = [
        "df3_fp32.onnx",
        "df3_fp32_simp.onnx",
        "df3_int8.onnx",
        "df3_int8_simp.onnx",
    ]

    results = []
    for m_name in model_names:
        m_path = os.path.join(args.models_dir, m_name)
        print(f"Benchmarking {m_name}...")
        res = benchmark_model(m_path, args.test_audio, num_runs=args.runs)
        if res is not None:
            results.append(res)
            print(
                f"  -> Avg: {res['avg_time_ms']} ms | Min: {res['min_time_ms']} ms | Max: {res['max_time_ms']} ms | Var: {res['variance']}"
            )
            print(
                f"  -> CPU: {res['cpu_percent']}% | RAM (Max): {res['max_ram_mb']} MB | Size: {res['size_mb']} MB"
            )
        else:
            print(f"  -> Skipping {m_name}, not found.")

    if results:
        keys = [
            "model",
            "size_mb",
            "avg_time_ms",
            "min_time_ms",
            "max_time_ms",
            "variance",
            "cpu_percent",
            "max_ram_mb",
            "num_iterations",
        ]
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {args.out_csv}")
    else:
        print("No models found or benchmarked.")


if __name__ == "__main__":
    main()
