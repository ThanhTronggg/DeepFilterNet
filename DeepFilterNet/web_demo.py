import gradio as gr
import torch
import os
import sys
from loguru import logger
import numpy as np

# Adjust path to ensure we can import from df and libdf
sys.path.append(os.getcwd())

# Import DeepFilterNet modules
# Assuming this script is run from inside DeepFilterNet/DeepFilterNet/ or similar where 'df' package is available
try:
    from libdf import Agc, DF
    from df.enhance import init_df, enhance, save_audio
    from df.io import load_audio, resample
    from df.model import ModelParams
except ImportError as e:
    print(f"Error importing DeepFilterNet modules: {e}")
    print("Please make sure you are in the correct directory (e.g. DeepFilterNet/DeepFilterNet)")
    exit(1)

# Global variables for model
model = None
df_state = None
suffix = None
df_sr = 48000

def load_model_global():
    global model, df_state, suffix, df_sr
    print("Loading DeepFilterNet3 Model...")
    # Initialize the model (DeepFilterNet3 by default)
    # We use 'mask_only=False' to enable full enhancement
    model, df_state, suffix, epoch = init_df(
        model_base_dir=None, # None loads default (DeepFilterNet3)
        post_filter=False,
        log_level="INFO"
    )
    model.eval()
    p = ModelParams()
    df_sr = p.sr
    print(f"Model loaded. SR: {df_sr}")

# Initialize model on startup
load_model_global()

def process_audio(input_file, enable_agc=True, desired_output_rms=0.02, distortion_factor=1e-4, snr_thresh=0.5, atten_lim_db=0):
    if input_file is None:
        return None, None, None

    try:
        # Load Audio
        audio, meta = load_audio(input_file, df_sr, "cpu")
        
        # Helper function for AGC
        def apply_agc(audio_tensor):
            agc = Agc(float(desired_output_rms), float(distortion_factor), float(snr_thresh)) 
            audio_np = audio_tensor.numpy()
            
            chunk_dur = 0.02 # 20ms
            chunk_size = int(df_sr * chunk_dur)
            
            if audio_np.shape[-1] > 0:
                for i_chunk in range(0, audio_np.shape[-1], chunk_size):
                    chunk = audio_np[..., i_chunk : i_chunk + chunk_size]
                    if chunk.size == 0: continue
                    
                    frame_energy = (chunk ** 2).mean()
                    if frame_energy < 1e-7:
                        snr = 0.0 # Freeze gain
                    else:
                        snr = 100.0 # Adapt gain
                    
                    agc.process(chunk, snr)
            return torch.from_numpy(audio_np)

        # 1. Generate AGC Only (Intermediate)
        # We always compute this for visualization/debugging if AGC is enabled
        if enable_agc:
            audio_agc_only = apply_agc(audio.clone())
            fn_agc_only = "output_agc_only.wav"
            save_audio(fn_agc_only, audio_agc_only, sr=df_sr, log=False)
        else:
            fn_agc_only = None

        # 2. Option A: Pre-AGC (AGC -> Model)
        # Use the already computed AGC audio if available, else use original
        input_to_model_a = audio_agc_only if enable_agc else audio.clone()
        enhanced_a = enhance(model, df_state, input_to_model_a, pad=True, atten_lim_db=None if atten_lim_db == 0 else atten_lim_db)
        fn_option_a = "output_pre_agc_model.wav"
        save_audio(fn_option_a, enhanced_a, sr=df_sr, log=False)

        # 3. Option B: Post-AGC (Model -> AGC)
        # Run model on ORIGINAL audio
        enhanced_b_raw = enhance(model, df_state, audio.clone(), pad=True, atten_lim_db=None if atten_lim_db == 0 else atten_lim_db)
        if enable_agc:
            enhanced_b = apply_agc(enhanced_b_raw)
        else:
            enhanced_b = enhanced_b_raw
        fn_option_b = "output_model_post_agc.wav"
        save_audio(fn_option_b, enhanced_b, sr=df_sr, log=False)

        # 4. Option C: Dual AGC (AGC -> Model -> AGC)
        # Use Output of Option A (AGC -> Model) as input, then apply AGC again
        if enable_agc:
            enhanced_c = apply_agc(enhanced_a.clone())
        else:
            enhanced_c = enhanced_a # Fallback to Option A if AGC disabled
        fn_option_c = "output_dual_agc.wav"
        save_audio(fn_option_c, enhanced_c, sr=df_sr, log=False)

        return fn_agc_only, fn_option_a, fn_option_b, fn_option_c
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise gr.Error(f"Processing failed: {e}")

# Gradio Interface
with gr.Blocks(title="DeepFilterNet Demo") as demo:
    gr.Markdown("# DeepFilterNet Enhancement Demo")
    gr.Markdown("Upload a noisy audio file to enhance it using DeepFilterNet3.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_audio = gr.Audio(type="filepath", label="Input Noisy Audio")
            
            with gr.Accordion("Advanced Settings", open=True):
                gr.Markdown("### Automatic Gain Control (AGC) Settings")
                enable_agc_chk = gr.Checkbox(label="Enable AGC", value=True, 
                                        info="Normalize volume.")

                with gr.Row():
                    desired_output_rms_slider = gr.Slider(minimum=0.001, maximum=0.1, value=0.02, step=0.001, 
                                                label="Desired Output RMS")
                    distortion_factor_slider = gr.Slider(minimum=1e-6, maximum=1e-2, value=1e-4, step=1e-5, 
                                                label="Distortion Factor (Adaptation Speed)")
                with gr.Row():
                    snr_thresh_slider = gr.Slider(minimum=0.0, maximum=10.0, value=0.5, step=0.1, 
                                                label="SNR Threshold (Internal)") 
                
                gr.Markdown("### Enhancement Settings")
                atten_lim_db_slider = gr.Slider(minimum=0, maximum=100, value=0, step=1, 
                                              label="Attenuation Limit [dB] (0 = No limit)")
            
            btn_process = gr.Button("Enhance Audio (Compare Options)", variant="primary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 1. AGC Only (Diagnostic)")
            output_agc_only = gr.Audio(label="AGC Only", interactive=False)
            
            gr.Markdown("### 2. Option A: AGC → Model")
            gr.Markdown("*Standard Approach.* Normalizes input volume -> Cleans noise. Safe for model inputs.")
            output_pre = gr.Audio(label="Pre-AGC", interactive=False)
            
            gr.Markdown("### 3. Option B: Model → AGC")
            gr.Markdown("*Post-Processing.* Cleans noise -> Normalizes output volume. Ensures output is loud.")
            output_post = gr.Audio(label="Post-AGC", interactive=False)

            gr.Markdown("### 4. Option C: AGC → Model → AGC")
            gr.Markdown("*Dual Normalization.* Normalizes input -> Cleans noise -> Normalizes output again. Maximum loudness control.")
            output_dual = gr.Audio(label="Dual-AGC", interactive=False)
            
    btn_process.click(
        fn=process_audio,
        inputs=[
            input_audio, 
            enable_agc_chk,
            desired_output_rms_slider, 
            distortion_factor_slider,
            snr_thresh_slider,
            atten_lim_db_slider
        ],
        outputs=[output_agc_only, output_pre, output_post, output_dual]
    )

if __name__ == "__main__":
    # Launch on 0.0.0.0 to accessible from host
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
