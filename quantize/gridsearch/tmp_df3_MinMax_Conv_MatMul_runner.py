import os
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
        print(f"Truncated calibration files to {len(self.wav_files)}")

def do_quantize():
    fp32_model = os.path.join("/home/trong/code/DeepFilterNet/quantize/gridsearch", "df3_fp32.onnx")
    int8_model = os.path.join("/home/trong/code/DeepFilterNet/quantize/gridsearch", "df3_MinMax_Conv_MatMul.onnx")
    
    if not os.path.exists(fp32_model):
        from torch_df_streaming import TorchDFPipeline
        from quantize_model import export_to_onnx
        pipeline = TorchDFPipeline(model_base_dir="/home/trong/code/DeepFilterNet/models/DeepFilterNet3/DeepFilterNet3", device='cpu')
        export_to_onnx(pipeline, fp32_model)
        
    dr = TruncatedReader("/home/trong/code/DeepFilterNet_quantize/noisy_wav", fp32_model_path=fp32_model)
    onnx_model = onnx.load(fp32_model)
    dr.input_names = [inp.name for inp in onnx_model.graph.input]
    
    op_types = ['Conv', 'MatMul']
    if "MinMax" == "MinMax":
        method = CalibrationMethod.MinMax
    elif "MinMax" == "Percentile":
        method = CalibrationMethod.Percentile
    elif "MinMax" == "Distribution":
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
