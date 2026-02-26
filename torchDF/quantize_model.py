#!/usr/bin/env python3
"""
DeepFilterNet3 ONNX Export and Static INT8 Quantization Script
"""

import os
import argparse
import torch
import torchaudio
import onnx
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat, CalibrationMethod
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
        
        # State tracking for recursive calibration
        self.states = np.zeros(45304, dtype=np.float32)
        
        # We need an inference session to generate the correct states for each step
        self.session = None
        if fp32_model_path:
            self.session = ort.InferenceSession(fp32_model_path, providers=['CPUExecutionProvider'])
            self.input_names = [i.name for i in self.session.get_inputs()]

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
            self.states = np.zeros(45304, dtype=np.float32)

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

    dummy_input = (
        torch.randn(480),           # input_frame
        states.clone(),             # states
        atten_lim_db.clone()        # atten_lim_db
    )

    print("Exporting to ONNX...")
    torch.onnx.export(
        torch_df,
        dummy_input,
        output_path,
        verbose=False,
        input_names=['input_frame', 'states', 'atten_lim_db'],
        output_names=['enhanced_audio_frame', 'out_states', 'lsnr'],
        opset_version=14,
        do_constant_folding=True,
        export_params=True
    )
    print(f"Exported to: {output_path}")

def quantize_model(fp32_model_path: str, int8_model_path: str, calib_dir: str):
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
        
        # Ensure we only pass inputs that the model expects to avoid INVALID_ARGUMENT error
        input_names = [i.name for i in session.get_inputs()]
        inputs = {k: v for k, v in inputs.items() if k in input_names}
        
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
    parser.add_argument('--test-audio', type=str, help='Input wav file for testing')
    parser.add_argument('--output-dir', type=str, default='./export_quantized', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fp32_onnx = os.path.join(args.output_dir, "df3_fp32.onnx")
    int8_onnx  = os.path.join(args.output_dir, "df3_int8.onnx")

    # 1. Export to ONNX
    pipeline = TorchDFPipeline(model_base_dir=args.model_base_dir, device='cpu')
    export_to_onnx(pipeline, fp32_onnx)

    # 2. Quantize
    quantize_model(fp32_onnx, int8_onnx, args.calib_dir)

    # 3. Test inference if test audio provided
    if args.test_audio and os.path.exists(args.test_audio):
        print("\nTesting FP32 model...")
        test_inference(fp32_onnx, args.test_audio, os.path.join(args.output_dir, "enhanced_fp32.wav"))
        
        print("\nTest INT8 model...")
        test_inference(int8_onnx, args.test_audio, os.path.join(args.output_dir, "enhanced_int8.wav"))


if __name__ == "__main__":
    main()