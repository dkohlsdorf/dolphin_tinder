#!/usr/bin/env python3
"""
Extract audio segments from WAV file into individual files.
This allows the frontend to load only the segments it needs.
"""

import argparse
import csv
import os
import scipy.io.wavfile as wavfile
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Extract audio segments from WAV file'
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
        '--output-dir', type=str, default='data/segments',
        help='Directory to save extracted segments'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load WAV file
    print(f"Loading WAV file: {args.wav}")
    sample_rate, audio_data = wavfile.read(args.wav)
    print(f"Sample rate: {sample_rate}, Total samples: {len(audio_data)}")

    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1).astype(audio_data.dtype)

    # Load segment definitions from CSV
    print(f"Loading segments from: {args.csv}")
    segments = []
    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = int(row['start_sample'])
            end = int(row['end_sample'])
            segments.append((start, end))
    print(f"Found {len(segments)} segments")

    # Extract and save each segment
    print(f"Extracting segments to: {args.output_dir}")
    for i, (start, end) in enumerate(segments):
        segment = audio_data[start:end]
        output_path = os.path.join(args.output_dir, f'{i}.wav')
        wavfile.write(output_path, sample_rate, segment)

        if (i + 1) % 100 == 0:
            print(f"  Extracted {i + 1}/{len(segments)} segments...")

    print(f"Done! Extracted {len(segments)} segments to {args.output_dir}")


if __name__ == '__main__':
    main()
