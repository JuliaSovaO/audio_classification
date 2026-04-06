import os
import shutil
import random
import numpy as np
import soundfile as sf
from pathlib import Path

def chunk_and_save_silence(base_path, train_dir, val_dir, test_dir):
    """Splits long background noise files into 1-second silence chunks."""
    noise_dir = base_path / "_background_noise_"
    if not noise_dir.exists():
        return 0, 0, 0
        
    silence_files = list(noise_dir.glob("*.wav"))
    chunk_length = 16000
    
    train_count, val_count, test_count = 0, 0, 0
    
    for wav_file in silence_files:
        if wav_file.name == "README.md": continue
        audio, sr = sf.read(wav_file)
        if sr != 16000: continue

        num_chunks = len(audio) // chunk_length
        chunks = np.array_split(audio[:num_chunks * chunk_length], num_chunks)
        
        for i, chunk in enumerate(chunks):
            filename = f"silence_{wav_file.stem}_{i}.wav"

            rand_val = random.random()
            if rand_val < 0.8:
                dest = train_dir / filename
                train_count += 1
            elif rand_val < 0.9:
                dest = val_dir / filename
                val_count += 1
            else:
                dest = test_dir / filename
                test_count += 1
                
            sf.write(dest, chunk, sr)
            
    return train_count, val_count, test_count

def organize_dataset(data_dir="data"):
    base_path = Path(data_dir)
    train_dir = base_path / "train"
    val_dir = base_path / "validation"
    test_dir = base_path / "test"

    for directory in [train_dir, val_dir, test_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    try:
        with open(base_path / "testing_list.txt", "r") as f:
            test_files = set(f.read().splitlines())
        with open(base_path / "validation_list.txt", "r") as f:
            val_files = set(f.read().splitlines())
    except FileNotFoundError:
        print("Error: Could not find testing_list.txt or validation_list.txt")
        return

    moved_count = {"train": 0, "validation": 0, "test": 0}

    print("Processing background noise into 1-second silence chunks...")
    t_sil, v_sil, test_sil = chunk_and_save_silence(base_path, train_dir, val_dir, test_dir)
    moved_count["train"] += t_sil
    moved_count["validation"] += v_sil
    moved_count["test"] += test_sil

    print("Sorting files. Downsampling 'unknown' class to balance dataset...")
    
    target_words = {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"}

    for item in base_path.iterdir():
        if not item.is_dir() or item.name in ["train", "validation", "test", "_background_noise_"]:
            continue
            
        label = item.name
        is_unknown = label not in target_words
        
        for wav_file in item.glob("*.wav"):
            if is_unknown and random.random() > 0.08:
                continue

            rel_path = f"{label}/{wav_file.name}"
            new_filename = f"{label}_{wav_file.name}"
            
            if rel_path in test_files:
                dest = test_dir / new_filename
                moved_count["test"] += 1
            elif rel_path in val_files:
                dest = val_dir / new_filename
                moved_count["validation"] += 1
            else:
                dest = train_dir / new_filename
                moved_count["train"] += 1
                
            shutil.copy2(str(wav_file), str(dest))

    print("\nSorting Complete!")
    print(f"Training files:   {moved_count['train']}")
    print(f"Validation files: {moved_count['validation']}")
    print(f"Testing files:    {moved_count['test']}")

if __name__ == "__main__":
    random.seed(42)
    organize_dataset("data")