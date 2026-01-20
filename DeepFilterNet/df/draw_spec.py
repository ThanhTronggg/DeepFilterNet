import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_comparison(noisy_path, enhanced_path, save_name="comparison.png"):
    # 1. Load âm thanh
    y_noisy, sr = librosa.load(noisy_path, sr=None)
    y_enhanced, _ = librosa.load(enhanced_path, sr=sr)

    # 2. Tính toán Spectrogram (Dùng STFT)
    # Chuyển sang đơn vị decibel (dB) để nhìn rõ hơn
    D_noisy = librosa.amplitude_to_db(np.abs(librosa.stft(y_noisy)), ref=np.max)
    D_enhanced = librosa.amplitude_to_db(np.abs(librosa.stft(y_enhanced)), ref=np.max)

    # 3. Vẽ biểu đồ
    plt.figure(figsize=(12, 8))

    # Biểu đồ trên: Noisy
    plt.subplot(2, 1, 1)
    librosa.display.specshow(D_noisy, sr=sr, x_axis='time', y_axis='hz')
    plt.title(f'Noisy Spectrogram: {os.path.basename(noisy_path)}')
    plt.colorbar(format='%+2.0f dB')

    # Biểu đồ dưới: Enhanced
    plt.subplot(2, 1, 2)
    librosa.display.specshow(D_enhanced, sr=sr, x_axis='time', y_axis='hz')
    plt.title(f'Enhanced Spectrogram: {os.path.basename(enhanced_path)}')
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig(save_name)
    print(f"Đã lưu biểu đồ so sánh tại: {save_name}")
    plt.show()

if __name__ == "__main__":
    # Thay đổi đường dẫn file của bạn ở đây
    noisy_file = "/home/trong/code/output_1/Ghi_tieu_chuan_14_DeepFilterNet3_original.wav"
    enhanced_file = "/home/trong/code/output_1/Ghi_tieu_chuan_14_DeepFilterNet3.wav"
    
    plot_comparison(noisy_file, enhanced_file)