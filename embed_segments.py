#!/usr/bin/env python3
"""
Script to extract audio segments from a WAV file based on CSV sample ranges,
embed them using the WDP Whisper model, and save the embeddings.
"""

import argparse
import csv
import numpy as np
import torch
import scipy.io.wavfile as wavfile
from pathlib import Path

from lib_dolphin import model, processor

# After combining classes: 0=WSTL (1+2), 1=BURST (3+4, includes echo)
COMBINED_LABEL_MAP = {
    0: 'WSTL',
    1: 'BURST'
}

def load_segments_from_csv(csv_path):
    """Load start and end sample indices from CSV file."""
    segments = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = int(row['start_sample'])
            end = int(row['end_sample'])
            segments.append((start, end))
    return segments


def extract_audio_segment(audio_data, start_sample, end_sample):
    """Extract a segment of audio between start and end samples."""
    return audio_data[start_sample:end_sample]


def pad_or_trim_audio(audio, target_length, sample_rate=16000):
    """Pad or trim audio to target length in samples. Default 30 seconds for Whisper."""
    if len(audio) > target_length:
        return audio[:target_length]
    elif len(audio) < target_length:
        padding = np.zeros(target_length - len(audio), dtype=audio.dtype)
        return np.concatenate([audio, padding])
    return audio


def create_input_mask(audio_length, target_length):
    """Create a mask indicating valid audio positions (1) vs padding (0)."""
    mask = np.zeros(target_length)
    mask[:min(audio_length, target_length)] = 1
    return mask


def main():
    parser = argparse.ArgumentParser(
        description='Embed audio segments from WAV file using WDP Whisper model'
    )
    parser.add_argument(
        '--wav', type=str, default='data/l2_large.wav',
        help='Path to input WAV file'
    )
    parser.add_argument(
        '--csv', type=str, default='data/l2_large.csv',
        help='Path to CSV file with start_sample and end_sample columns'
    )
    parser.add_argument(
        '--model-size', type=str, default='small',
        choices=['small', 'base', 'small_augment_L1', 'small_augment_L2'],
        help='Model size to use'
    )
    parser.add_argument(
        '--model-file', type=str, default=None,
        help='Path to model checkpoint file (overrides model-size)'
    )
    parser.add_argument(
        '--output', type=str, default='embeddings.npy',
        help='Path to output embeddings file (.npy format)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=8,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device to use (cuda/cpu). Auto-detect if not specified.'
    )

    args = parser.parse_args()

    # Determine device
    if args.device:
        device = torch.device(args.device)
        cuda = args.device == 'cuda'
    else:
        cuda = torch.cuda.is_available()
        device = torch.device('cuda' if cuda else 'cpu')

    print(f"Using device: {device}")

    # Load model and processor
    print("Loading model...")
    emb_model = model(model_size=args.model_size, model_file=args.model_file, cuda=cuda)
    emb_model.eval()

    print("Loading processor...")
    proc = processor(model_size=args.model_size)

    # Load WAV file
    print(f"Loading WAV file: {args.wav}")
    sample_rate, audio_data = wavfile.read(args.wav)
    print(f"Sample rate: {sample_rate}, Total samples: {len(audio_data)}")

    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)

    # Normalize to float32 in range [-1, 1]
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    elif audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    # Load segment definitions from CSV
    print(f"Loading segments from: {args.csv}")
    segments = load_segments_from_csv(args.csv)
    print(f"Found {len(segments)} segments")

    # Model trained on 44kHz audio, 30 second chunks
    whisper_chunk_samples = 30 * sample_rate  # 30 seconds at native sample rate

    # Process segments in batches
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for batch_start in range(0, len(segments), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(segments))
            batch_segments = segments[batch_start:batch_end]

            print(f"Processing segments {batch_start + 1}-{batch_end} of {len(segments)}...")

            batch_audio = []
            batch_masks = []

            for start, end in batch_segments:
                # Extract segment
                segment = extract_audio_segment(audio_data, start, end)
                original_length = len(segment)

                # Pad or trim to Whisper's expected length
                segment = pad_or_trim_audio(segment, whisper_chunk_samples)

                # Create mask for valid audio positions
                mask = create_input_mask(original_length, whisper_chunk_samples)

                batch_audio.append(segment)
                batch_masks.append(mask)

            # Process through Whisper processor to get mel spectrograms
            inputs = proc(
                batch_audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )

            input_features = inputs.input_features.to(device)

            # Create input mask tensor
            input_mask = torch.tensor(np.array(batch_masks), dtype=torch.float32).to(device)

            # Get embeddings
            outputs = emb_model.forward(input_features, input_mask=input_mask)

            # Sum over time dimension
            class_scores = outputs['classifications'].sum(axis=1).cpu().numpy()  # [batch, num_classes]

            # Combine classes before argmax
            # New layout: 0=WSTL (combined 1+2), 1=BURST (combined 3+4, includes echo)
            combined_scores = np.stack([
                class_scores[:, 1] + class_scores[:, 2],  # WSTL = class 1 + class 2
                class_scores[:, 3] + class_scores[:, 4]   # BURST = class 3 + class 4 (echo)
            ], axis=1)

            labels = [COMBINED_LABEL_MAP[int(i)] for i in np.argmax(combined_scores, axis=1)]
            embeddings = outputs['embeddings']
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels)
    # Concatenate all embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Generated embeddings shape: {all_embeddings.shape}")

    # Flatten labels list
    all_labels = [label for batch_labels in all_labels for label in batch_labels]
    print(f"Generated {len(all_labels)} labels")

    # Save embeddings
    output_path = Path(args.output)
    np.save(output_path, all_embeddings)
    print(f"Saved embeddings to: {output_path}")

    # Save labels
    labels_path = output_path.with_suffix('.labels.npy')
    np.save(labels_path, np.array(all_labels))
    print(f"Saved labels to: {labels_path}")


if __name__ == '__main__':
    main()
