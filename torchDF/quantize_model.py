#!/usr/bin/env python3
"""
DeepFilterNet3 ONNX Export and Static INT8 Quantization Script
"""

import os
import argparse
import torch
import torchaudio
import onnx
import onnx.helper
# Fix for onnx 1.14+ compatibility with onnxruntime 1.14 symbolic shape inference
if not hasattr(onnx.helper, 'make_sequence_value_info'):
    onnx.helper.make_sequence_value_info = onnx.helper.make_tensor_sequence_value_info

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod
from onnxruntime.quantization.shape_inference import quant_pre_process
import soundfile as sf
import glob
from torch_df_streaming import TorchDFPipeline, ExportableStreamingTorchDF

class DeepFilterNetCalibrationReader(CalibrationDataReader):
    def __init__(self, calibration_dir: str, hop_size: int = 480, sample_rate: int = 48000, fp32_model_path: str = None):
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.input_names = None
        self.wav_files = sorted(glob.glob(os.path.join(calibration_dir, "**/*.wav"), recursive=True))[:400]
        print(f"Found {len(self.wav_files)} calibration files")
        
        self.current_file_idx = 0
        self.current_audio = None
        self.current_pos = 0
        
        # We need an inference session to generate the correct states for each step
        self.session = None
        self.states_size = 45304 # Default
        if fp32_model_path:
            # Disable mem_pattern to fix "Shape mismatch attempting to re-use buffer"
            sess_options = ort.SessionOptions()
            sess_options.enable_mem_pattern = False
            self.session = ort.InferenceSession(fp32_model_path, sess_options, providers=['CPUExecutionProvider'])
            self.input_names = [i.name for i in self.session.get_inputs()]
            # Detect state size dynamically from model
            for inp in self.session.get_inputs():
                if inp.name == 'states':
                    if isinstance(inp.shape[0], int) and inp.shape[0] > 0:
                        self.states_size = inp.shape[0]
                        print(f"Detected model state size: {self.states_size}")
                    break
        
        self.states = np.zeros(self.states_size, dtype=np.float32)

    def get_next(self):
        if self.current_file_idx >= len(self.wav_files):
            return None

        if self.current_audio is None or self.current_pos >= len(self.current_audio):
            wav_path = self.wav_files[self.current_file_idx]
            audio, sr = sf.read(wav_path)
            if sr != self.sample_rate:
                self.current_audio = None
                self.current_file_idx += 1
                return self.get_next()
            
            audio = audio.astype(np.float32)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            self.current_audio = audio
            self.current_pos = 0
            self.current_file_idx += 1
            # Reset state on new file
            self.states = np.zeros(self.states_size, dtype=np.float32)

        end = min(self.current_pos + self.hop_size, len(self.current_audio))
        chunk = self.current_audio[self.current_pos:end]
        
        if len(chunk) < self.hop_size:
            chunk = np.pad(chunk, (0, self.hop_size - len(chunk)), mode='constant')
        
        self.current_pos = end

        # Model input: [frame_size]
        input_frame = chunk.astype(np.float32)

        feed = {
            'input_frame': input_frame,
            'states': self.states,
        }
        
        # Fallback if self.input_names wasn't set through InferenceSession
        if self.input_names is None:
            self.input_names = ['input_frame', 'states', 'atten_lim_db']
            
        if 'atten_lim_db' in self.input_names:
            feed['atten_lim_db'] = np.array([0.0], dtype=np.float32)

        # Run one pass to get the state for the next step exactly like inference
        if self.session is not None:
            out = self.session.run(None, feed)
            out_names = [o.name for o in self.session.get_outputs()]
            states_idx = out_names.index('out_states') if 'out_states' in out_names else 1
            self.states = out[states_idx]

        return feed

def export_to_onnx(model: TorchDFPipeline, output_path: str = "df3_torchdf.onnx"):
    torch_df = model.torch_streaming_model
    states = model.states
    atten_lim_db = model.atten_lim_db

    input_frame = torch.randn(480)
    input_features = (
        input_frame, states, atten_lim_db
    )
    
    # Warm up / check model as in model_onnx_export.py
    torch_df(*input_features)

    print("Exporting to ONNX using torch.jit.script...")
    torch_df_script = torch.jit.script(torch_df)
    torch.onnx.export(
        torch_df_script,
        input_features,
        output_path,
        verbose=False,
        input_names=['input_frame', 'states', 'atten_lim_db'],
        output_names=['enhanced_audio_frame', 'out_states', 'lsnr'],
        opset_version=14
    )
    print(f"Exported to: {output_path}")

def quantize_model(fp32_model_path: str, int8_model_path: str, calib_dir: str):
    # Use standard ONNX shape inference (more stable than ORT's symbolic for sequences)
    print(f"Applying shape inference: {fp32_model_path}")
    model = onnx.load(fp32_model_path)
    model_inferred = onnx.shape_inference.infer_shapes(model)
    preprocessed_path = fp32_model_path.replace(".onnx", "_inferred.onnx")
    onnx.save(model_inferred, preprocessed_path)
    fp32_model_path = preprocessed_path

    dr = DeepFilterNetCalibrationReader(calib_dir, fp32_model_path=fp32_model_path)
    
    onnx_model = onnx.load(fp32_model_path)
    dr.input_names = [inp.name for inp in onnx_model.graph.input]
    print("Calibration using inputs:", dr.input_names)

    print("Starting static quantization (MinMax)...")
    quantize_static(
        fp32_model_path,
        int8_model_path,
        dr,
        quant_format=QuantFormat.QDQ,
        per_channel=False,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize = [
        "Conv",
        "Gemm",
    ]
    )
    print(f"Quantized model saved to: {int8_model_path}")

def test_inference(onnx_path: str, input_wav: str, output_wav: str):
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    noisy, sr = torchaudio.load(input_wav)
    noisy = noisy.mean(0).unsqueeze(0).numpy()
    
    hop = 480
    chunks = [noisy[0, i:i+hop] for i in range(0, noisy.shape[1], hop)]
    if len(chunks[-1]) < hop:
        chunks[-1] = np.pad(chunks[-1], (0, hop - len(chunks[-1])))
    
    enhanced_chunks = []
    states = np.zeros(45304, dtype=np.float32)
    atten_lim = np.array([0.0], dtype=np.float32)
    
    for chunk in chunks:
        inputs = {
            'input_frame': chunk.astype(np.float32),
            'states': states,
            'atten_lim_db': atten_lim
        }
        
        # Ensure we only pass inputs that the model expects
        input_info = {i.name: i.shape for i in session.get_inputs()}
        inputs = {k: v for k, v in inputs.items() if k in input_info}
        
        out = session.run(None, inputs)
        # Map outputs safely
        out_names = [o.name for o in session.get_outputs()]
        enhanced = out[out_names.index('enhanced_audio_frame')] if 'enhanced_audio_frame' in out_names else out[0]
        states = out[out_names.index('out_states')] if 'out_states' in out_names else out[1]
        if 'lsnr' in out_names:
            lsnr = out[out_names.index('lsnr')]
        elif len(out) > 2:
            lsnr = out[2]
        else:
            lsnr = np.zeros(1)
            
        enhanced_chunks.append(enhanced)
    
    enhanced_audio = np.concatenate(enhanced_chunks)
    sf.write(output_wav, enhanced_audio, sr)
    print(f"Enhanced audio saved to: {output_wav}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-base-dir', type=str, default='DeepFilterNet3', help='Path to pretrained DeepFilterNet3')
    parser.add_argument('--calib-dir', type=str, required=True, help='Directory containing calibration wav files')
    parser.add_argument('--onnx-path', type=str, help='Path to an existing FP32 ONNX model (skips export)')
    parser.add_argument('--test-audio', type=str, help='Input wav file for testing')
    parser.add_argument('--output-dir', type=str, default='./export_quantized', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    int8_onnx  = os.path.join(args.output_dir, "df3_int8.onnx")

    if args.onnx_path:
        fp32_onnx = args.onnx_path
        print(f"Using existing ONNX model: {fp32_onnx}")
    else:
        fp32_onnx = os.path.join(args.output_dir, "df3_fp32.onnx")
        # 1. Export to ONNX
        pipeline = TorchDFPipeline(model_base_dir=args.model_base_dir, device='cpu')
        export_to_onnx(pipeline, fp32_onnx)

    # 2. Quantize
    quantize_model(fp32_onnx, int8_onnx, args.calib_dir)

    # 3. Convert to ORT format
    import subprocess
    print("\nConverting to ORT format...")
    subprocess.run([
        "python", "-m", "onnxruntime.tools.convert_onnx_models_to_ort",
        args.output_dir,
        "--optimization_style", "Fixed"
    ], check=True)
    print("Models converted to ORT format!")

    # 4. Test inference if test audio provided
    if args.test_audio and os.path.exists(args.test_audio):
        print("\nTesting FP32 model...")
        test_inference(fp32_onnx, args.test_audio, os.path.join(args.output_dir, "enhanced_fp32.wav"))
        
        print("\nTest INT8 model...")
        test_inference(int8_onnx, args.test_audio, os.path.join(args.output_dir, "enhanced_int8.wav"))


if __name__ == "__main__":
    main()