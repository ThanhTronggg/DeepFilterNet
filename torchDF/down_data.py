import os
import requests
from tqdm import tqdm

# Directory to store calibration files
calib_dir = "/home/trong/code/DeepFilterNet_quantize/noisy_wav"
os.makedirs(calib_dir, exist_ok=True)

# List of noisy files from DNS-Challenge blind_test_set
file_list = [
    f"datasets/blind_test_set/reverb_fileid_{i}.wav" for i in range(1, 101)
]

downloaded = 0
max_files = 20  # Limit to 20 files for quick testing

with tqdm(total=len(file_list), desc="Downloading noisy wav") as pbar:
    for file_path in file_list:
        if downloaded >= max_files:
            break
        
        local_path = os.path.join(calib_dir, os.path.basename(file_path))
        if os.path.exists(local_path):
            downloaded += 1
            pbar.update(1)
            continue
        
        url = f"https://huggingface.co/datasets/ltnghia/DNS-Challenge/resolve/main/{file_path}"
        try:
            r = requests.get(url, stream=True, timeout=30)
            if r.status_code == 200:
                with open(local_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded: {local_path}")
                downloaded += 1
            else:
                pass
        except Exception as e:
            print(f"Error downloading {file_path}: {e}")
        
        pbar.update(1)

print(f"Downloaded {downloaded} noisy wav files to {calib_dir}")