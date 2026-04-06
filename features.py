import os
import numpy as np
import librosa
from pathlib import Path
import concurrent.futures

TARGET_WORDS = {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence"}

def process_audio(file_path, sr=16000, max_samples=16000, n_mfcc=40, max_frames=100):
    audio, _ = librosa.load(file_path, sr=sr)

    if len(audio) < max_samples:
        audio = np.pad(audio, (0, max_samples - len(audio)), mode='constant')
    else:
        audio = audio[:max_samples]
        
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=400, hop_length=160)
    
    if mfccs.shape[1] < max_frames:
        pad_width = max_frames - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_frames]
        
    return mfccs.flatten()

def _process_single_file(file_path):
    label = file_path.name.split('_')[0]
    if label not in TARGET_WORDS:
        label = "unknown"
    features = process_audio(file_path)
    return features, label

def build_dataset(data_dir):
    folder_path = Path(data_dir)
    wav_files = list(folder_path.glob("*.wav"))
    total_files = len(wav_files)
    
    print(f"Extracting features from {total_files} files in {data_dir} using ALL CPU cores...")
    X, y = [], []
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(_process_single_file, wav_files, chunksize=100)
        
        for i, (features, label) in enumerate(results, 1):
            X.append(features)
            y.append(label)
            if i % 2000 == 0 or i == total_files:
                percent_done = (i / total_files) * 100
                print(f"   -> Processed {i}/{total_files} files ({percent_done:.1f}%)")
                
    return np.array(X), np.array(y)

def standardize(X, mean=None, std=None):
    if mean is None or std is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1e-10
        
    X_scaled = (X - mean) / std
    return X_scaled, mean, std