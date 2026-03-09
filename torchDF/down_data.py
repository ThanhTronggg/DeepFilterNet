import os
import requests
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Download noisy wav files for calibration.")
    parser.add_argument("--calib-dir", type=str, default="./calibration_data", help="Directory to store calibration files")
    parser.add_argument("--max-files", type=int, default=20, help="Maximum number of files to download")
    args = parser.parse_args()

    os.makedirs(args.calib_dir, exist_ok=True)

    # List of noisy files from DNS-Challenge blind_test_set
    file_list = [
        f"datasets/blind_test_set/reverb_fileid_{i}.wav" for i in range(1, 101)
    ]

    downloaded = 0
    with tqdm(total=len(file_list), desc="Downloading noisy wav") as pbar:
        for file_path in file_list:
            if downloaded >= args.max_files:
                break
            
            local_path = os.path.join(args.calib_dir, os.path.basename(file_path))
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
                    # print(f"Downloaded: {local_path}")
                    downloaded += 1
                else:
                    pass
            except Exception as e:
                print(f"Error downloading {file_path}: {e}")
            
            pbar.update(1)

    print(f"Downloaded {downloaded} noisy wav files to {args.calib_dir}")

if __name__ == "__main__":
    main()