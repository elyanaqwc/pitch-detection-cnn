import matplotlib.pyplot as plt
import numpy as np
import librosa
import pretty_midi
import os
import glob
from multiprocessing import Pool, cpu_count

def convert_audio_to_CQT(audio_file, sr=22050):
    """Convert MP3/WAV audio to CQT spectrogram (in dB scale)."""
    y, sr = librosa.load(audio_file, sr=sr)  # Removed backend argument
    C = librosa.cqt(y, sr=sr, fmin=librosa.note_to_hz('A0'), n_bins=88, bins_per_octave=12)
    C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
    return C_db, sr

def create_ground_truth_matrix(midi_file, cqt_frames, sr=22050, hop_length=512):
    """Create ground truth matrix aligned with CQT frames."""
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    fs = sr / hop_length  # frame rate
    total_time = cqt_frames / fs  # duration of audio in seconds
    
    piano_roll = midi_data.get_piano_roll(fs=fs, times=np.linspace(0, total_time, cqt_frames))
    ground_truth = (piano_roll[21:109, :] > 0).astype(int)  # 88 piano keys: A0 to C8
    
    # Ensure shape matches number of CQT frames
    if ground_truth.shape[1] != cqt_frames:
        ground_truth = ground_truth[:, :cqt_frames]
    
    return ground_truth

def chunk_data(cqt, ground_truth, patch_size=128, overlap=64):
    """Split CQT and labels into overlapping patches."""
    patches, labels = [], []
    total_frames = cqt.shape[1]
    
    for start in range(0, total_frames - patch_size + 1, overlap):
        end = start + patch_size
        cqt_patch = cqt[:, start:end]
        label_patch = ground_truth[:, start:end]
        
        patches.append(cqt_patch)
        labels.append(label_patch)
    
    return np.array(patches), np.array(labels)

def save_processed_data(chunks, labels, save_path):
    """Save patches and labels as compressed .npz."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        cqt_patches=chunks,
        label_patches=labels,
        metadata={
            'num_patches': len(chunks),
            'cqt_shape': chunks[0].shape,
            'label_shape': labels[0].shape
        }
    )
    print(f"✅ Saved {len(chunks)} patches to {save_path}")
    print(f"    CQT patch shape: {chunks[0].shape}")
    print(f"    Label patch shape: {labels[0].shape}")

def process_single_audio_and_midi(audio_file, midi_file, save_path, patch_size=128, overlap=64):
    """Full pipeline: audio → CQT → labels → chunks → save"""
    cqt, sr = convert_audio_to_CQT(audio_file)
    
    ground_truth = create_ground_truth_matrix(midi_file, cqt.shape[1], sr)
    
    cqt_chunks, label_chunks = chunk_data(cqt, ground_truth, patch_size, overlap)
    
    save_processed_data(cqt_chunks, label_chunks, save_path)

def process_single_file(args):
    """Wrapper function for multiprocessing."""
    mp3_path, midi_path, save_path = args
    file_name = os.path.basename(mp3_path).replace('.mp3', '')
    
    # Skip if already exists
    if os.path.exists(save_path):
        print(f"⏭️ Skipping {file_name} (already processed)")
        return True
    
    print(f"🔁 Processing: {file_name}")
    try:
        process_single_audio_and_midi(mp3_path, midi_path, save_path)
        return True
    except Exception as e:
        print(f"❌ Failed to process {file_name}: {e}")
        return False

# Parallel processing setup
def run_parallel(num_workers=None):
    if num_workers is None:
        num_workers = min(cpu_count() - 1, 8)
    
    base_path = 'maestro-v2.0.0'
    audio_dir = os.path.join(base_path, '2004')
    result_dir = os.path.join(base_path, 'Result')
    
    # Find all .mp3 files in folder
    mp3_files = sorted(glob.glob(os.path.join(audio_dir, '*.mp3')))
    
    # Prepare arguments for parallel processing
    process_args = []
    for mp3_path in mp3_files:
        file_name = os.path.basename(mp3_path).replace('.mp3', '')
        midi_path = os.path.join(audio_dir, file_name + '.midi')
        
        # Check if corresponding MIDI file exists
        if not os.path.exists(midi_path):
            print(f"⚠️ MIDI not found for {file_name}, skipping...")
            continue
        
        save_path = os.path.join(result_dir, file_name + '.npz')
        process_args.append((mp3_path, midi_path, save_path))
    
    print(f"🚀 Processing {len(process_args)} files with {num_workers} workers...")
    
    # Run in parallel
    with Pool(num_workers) as pool:
        results = pool.map(process_single_file, process_args)
    
    successful = sum(results)
    print(f"✅ Successfully processed {successful}/{len(process_args)} files")

# Original sequential code (unchanged)
base_path = 'maestro-v2.0.0'
audio_dir = os.path.join(base_path, '2004')
result_dir = os.path.join(base_path, 'Result')

# Find all .mp3 files in folder
mp3_files = sorted(glob.glob(os.path.join(audio_dir, '*.mp3')))

# Choose processing method:
# Option 1: Run in parallel (faster)
if __name__ == "__main__":
    run_parallel(num_workers=2)  # Adjust based on your CPU cores

