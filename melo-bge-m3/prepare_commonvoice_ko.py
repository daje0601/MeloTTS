#!/usr/bin/env python3
"""
Prepare Common Voice Korean dataset for MeloTTS BGE-M3 pretraining
Dataset: gglabs/commonvoice_22_ko
"""

import os
import json
import wave
import subprocess
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
import datasets
import soundfile as sf
import torch
import torchaudio
import random

def download_commonvoice():
    """Download Common Voice Korean dataset from HuggingFace"""
    print("=" * 60)
    print("Downloading Common Voice Korean dataset...")
    print("=" * 60)

    # Load dataset (this will download if not cached)
    # Common Voice Korean only has 'train' split
    dataset = load_dataset(
        "gglabs/commonvoice_22_ko",
        split="train"  # Only train split available
    )

    print(f"Total samples: {len(dataset)}")

    # Filter for quality (optional - remove if you want all data)
    # Common Voice has up_votes and down_votes
    if "up_votes" in dataset.column_names:
        dataset = dataset.filter(lambda x: x.get("up_votes", 0) >= x.get("down_votes", 0))
        print(f"After quality filter: {len(dataset)}")

    return dataset

def process_audio_file(audio_array, sample_rate, output_path):
    """Convert audio to 44.1kHz 16-bit mono WAV"""
    # Resample if necessary
    if sample_rate != 44100:
        # Resample using torchaudio
        waveform = torch.from_numpy(audio_array).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=44100
        )
        waveform = resampler(waveform)
        audio_array = waveform.squeeze(0).numpy()
        sample_rate = 44100

    # Ensure mono
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)

    # Normalize to 16-bit range
    if audio_array.dtype != 'int16':
        audio_array = (audio_array * 32767).astype('int16')

    # Save as WAV
    sf.write(output_path, audio_array, sample_rate, subtype='PCM_16')

    return True

def prepare_commonvoice_data(output_dir="data/commonvoice_ko"):
    """Prepare Common Voice Korean data for training"""

    # Create output directories
    output_dir = Path(output_dir)
    wavs_dir = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)

    # Download dataset
    dataset = download_commonvoice()

    # Prepare data lists
    train_list = []
    val_list = []
    test_list = []

    # Process each sample
    print("\n" + "=" * 60)
    print("Processing audio files...")
    print("=" * 60)

    # Track speaker IDs
    speaker_map = {}
    speaker_counter = 0

    # Process with casting to Audio type
    dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=44100))

    for idx, sample in enumerate(tqdm(dataset, desc="Processing")):
        try:
            # Get audio data
            audio = sample["audio"]
            audio_array = audio["array"]
            sample_rate = audio["sampling_rate"]

            # Get text (Common Voice uses 'sentence' field)
            text = sample.get("sentence", "").strip()
            if not text:
                continue

            # Get or assign speaker ID
            # Common Voice has 'client_id' which can be used as speaker ID
            client_id = sample.get("client_id", f"speaker_{idx}")
            if client_id not in speaker_map:
                speaker_map[client_id] = f"CV_SPEAKER_{speaker_counter:04d}"
                speaker_counter += 1
            speaker_id = speaker_map[client_id]

            # Create filename
            filename = f"cv_ko_{idx:08d}.wav"
            wav_path = wavs_dir / filename

            # Process and save audio
            success = process_audio_file(audio_array, sample_rate, wav_path)

            if not success:
                continue

            # Get duration
            with wave.open(str(wav_path), 'r') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)

            # Skip very short or very long utterances
            if duration < 0.5 or duration > 15:
                wav_path.unlink()  # Delete the file
                continue

            # Create data entry
            # Language is always KR for Korean Common Voice
            entry = f"wavs/{filename}|{text}|{speaker_id}|KR"

            # Split into train/val/test (80/10/10)
            rand_num = random.random()
            if rand_num < 0.8:
                train_list.append(entry)
            elif rand_num < 0.9:
                val_list.append(entry)
            else:
                test_list.append(entry)

        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            continue

    # Write data lists
    print("\n" + "=" * 60)
    print("Writing data lists...")
    print("=" * 60)

    with open(output_dir / "train_list.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train_list))
    print(f"Train samples: {len(train_list)}")

    with open(output_dir / "val_list.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(val_list))
    print(f"Validation samples: {len(val_list)}")

    with open(output_dir / "test_list.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(test_list))
    print(f"Test samples: {len(test_list)}")

    # Save speaker mapping
    with open(output_dir / "speakers.json", "w", encoding="utf-8") as f:
        json.dump(speaker_map, f, ensure_ascii=False, indent=2)
    print(f"Total speakers: {len(speaker_map)}")

    # Create config for Common Voice pretraining
    config = {
        "data_dir": str(output_dir),
        "train_list": str(output_dir / "train_list.txt"),
        "val_list": str(output_dir / "val_list.txt"),
        "test_list": str(output_dir / "test_list.txt"),
        "num_speakers": len(speaker_map),
        "total_samples": len(train_list) + len(val_list) + len(test_list),
        "language": "Korean",
        "dataset": "CommonVoice 22.0"
    }

    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("✅ Data preparation completed!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    return output_dir

def create_pretrain_config(data_dir="data/commonvoice_ko"):
    """Create configuration for Common Voice pretraining"""

    data_dir = Path(data_dir)

    # Load dataset info
    with open(data_dir / "dataset_info.json", "r") as f:
        info = json.load(f)

    # Create pretrain config based on the base config
    config = {
        "train": {
            "log_interval": 100,
            "eval_interval": 500,
            "seed": 42,
            "epochs": 10000,
            "learning_rate": 0.0002,  # Slightly lower for pretraining
            "betas": [0.8, 0.99],
            "eps": 1e-09,
            "batch_size": 8,  # Adjust based on GPU memory
            "fp16_run": True,
            "lr_decay": 0.999875,
            "segment_size": 16384,
            "init_lr_ratio": 1,
            "warmup_epochs": 10,  # More warmup for diverse data
            "c_mel": 45,
            "c_kl": 1.0,
            "skip_optimizer": False
        },
        "data": {
            "training_files": str(data_dir / "train_list.txt"),
            "validation_files": str(data_dir / "val_list.txt"),
            "max_wav_value": 32768.0,
            "sampling_rate": 44100,
            "filter_length": 2048,
            "hop_length": 512,
            "win_length": 2048,
            "n_mel_channels": 128,
            "mel_fmin": 0.0,
            "mel_fmax": None,
            "add_blank": True,
            "n_speakers": info["num_speakers"],  # From dataset
            "cleaned_text": True,
            "spk2id": {},
            "disable_bert": False
        },
        "model": {
            "use_spk_conditioned_encoder": True,
            "use_noise_scaled_mas": True,
            "use_mel_posterior_encoder": False,
            "use_duration_discriminator": True,
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "n_layers_trans_flow": 3,
            "kernel_size": 3,
            "p_dropout": 0.1,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [
                [1, 3, 5],
                [1, 3, 5],
                [1, 3, 5]
            ],
            "upsample_rates": [8, 8, 2, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 8, 2, 2],
            "n_layers_q": 3,
            "use_spectral_norm": False,
            "gin_channels": 256,
            "num_languages": None,  # Language-agnostic with BGE-M3
            "num_tones": 16
        }
    }

    # Save config
    config_path = Path("configs/config_commonvoice_pretrain.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"✅ Pretrain config saved to: {config_path}")

    return config_path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Common Voice Korean for MeloTTS")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/commonvoice_ko",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )

    args = parser.parse_args()

    # Prepare data
    data_dir = prepare_commonvoice_data(args.output_dir)

    # Create pretrain config
    config_path = create_pretrain_config(data_dir)

    print("\n" + "=" * 60)
    print("🎉 Ready for pretraining!")
    print("=" * 60)
    print(f"\nTo start training, run:")
    print(f"./train_commonvoice.sh")
    print("=" * 60)