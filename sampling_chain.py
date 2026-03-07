#!/usr/bin/env python3
"""
Sample pairs from embeddings for labeling:
1. Generate all unique pairs (i, j) where i < j
2. Filter to pairs within distance threshold
3. Randomly sample up to N pairs
"""

import argparse
import json
import numpy as np
from collections import defaultdict
from scipy.spatial.distance import cdist
from pathlib import Path


def build_distance_matrix(embeddings, metric='cosine'):
    """Build pairwise distance matrix from embeddings."""
    print(f"Building distance matrix with {metric} metric...")
    distances = cdist(embeddings, embeddings, metric=metric)
    return distances


def sample_pairs(distances, n_pairs=5000, max_distance=None, seed=None):
    """
    Sample unique pairs from all possible pairs.

    Args:
        distances: Pairwise distance matrix
        n_pairs: Maximum number of pairs to sample
        max_distance: If set, only consider pairs within this distance
        seed: Random seed for reproducibility

    Returns:
        pairs: List of (i, j) tuples where i < j
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = len(distances)

    # Get all unique pairs (i, j) where i < j
    i_indices, j_indices = np.triu_indices(n_samples, k=1)

    # Filter by distance if threshold is set
    if max_distance is not None:
        pair_distances = distances[i_indices, j_indices]
        mask = pair_distances <= max_distance
        i_indices = i_indices[mask]
        j_indices = j_indices[mask]
        print(f"  {len(i_indices)} pairs within distance threshold")

    n_available = len(i_indices)
    n_to_sample = min(n_pairs, n_available)

    # Randomly sample pairs
    if n_to_sample < n_available:
        selected = np.random.choice(n_available, size=n_to_sample, replace=False)
        i_indices = i_indices[selected]
        j_indices = j_indices[selected]

    pairs = list(zip(i_indices.tolist(), j_indices.tolist()))
    print(f"  Sampled {len(pairs)} pairs")

    return pairs


def embeddings_by_label(embeddings, labels):
    """Group embeddings by their labels, returning indices for each label."""
    mapping = defaultdict(list)
    for i, label in enumerate(labels):
        mapping[label].append(i)
    return {
        label: (embeddings[indices], indices) for label, indices in mapping.items()
    }


def main():
    parser = argparse.ArgumentParser(
        description='Sample pairs from embeddings for labeling'
    )
    parser.add_argument(
        '--embeddings', type=str, default='embeddings.npy',
        help='Path to embeddings file (.npy format)'
    )
    parser.add_argument(
        '--labels', type=str, default='embeddings.labels.npy',
        help='Path to labels file (.npy format). If provided, generates pairs per label.'
    )
    parser.add_argument(
        '--output', type=str, default='sampling_pairs.json',
        help='Path to output pairs file (.json format)'
    )
    parser.add_argument(
        '--metric', type=str, default='euclidean',
        choices=['cosine', 'euclidean', 'cityblock'],
        help='Distance metric to use'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--n-pairs', type=int, default=5000,
        help='Maximum number of pairs to sample per label (default: 5000)'
    )
    parser.add_argument(
        '--distance-percentile', type=float, default=None,
        help='Percentile of distance distribution to use as threshold (0-1). '
             'Only pairs with distance <= this percentile are included. '
             'If not set, no distance filtering is applied.'
    )

    args = parser.parse_args()

    # Load embeddings
    print(f"Loading embeddings from: {args.embeddings}")
    embeddings = np.load(args.embeddings)
    print(f"Embeddings shape: {embeddings.shape}")

    # Load labels if provided
    if args.labels:
        print(f"Loading labels from: {args.labels}")
        labels = np.load(args.labels)
        print(f"Found {len(labels)} labels")

        # Group embeddings by label
        grouped = embeddings_by_label(embeddings, labels)
        print(f"Found {len(grouped)} unique labels: {list(grouped.keys())}")

        output_path = Path(args.output)

        # Process each label separately
        for label, (label_embeddings, original_indices) in grouped.items():
            print(f"\n=== Processing label: {label} ({len(label_embeddings)} samples) ===")

            if len(label_embeddings) < 2:
                print(f"  Skipping {label}: not enough samples")
                continue

            # Build distance matrix for this label's embeddings
            distances = build_distance_matrix(label_embeddings, metric=args.metric)

            # Compute distance threshold from percentile if set
            max_distance = None
            if args.distance_percentile is not None:
                upper_tri_indices = np.triu_indices(len(distances), k=1)
                upper_tri_distances = distances[upper_tri_indices]
                max_distance = np.percentile(upper_tri_distances, args.distance_percentile * 100)
                print(f"  Distance threshold at {args.distance_percentile:.1%} percentile: {max_distance:.4f}")

            # Sample pairs
            pairs = sample_pairs(
                distances,
                n_pairs=args.n_pairs,
                max_distance=max_distance,
                seed=args.seed
            )

            # Convert local indices back to original indices
            pairs_original = [[int(original_indices[a]), int(original_indices[b])] for a, b in pairs]

            # Save pairs for this label as JSON
            label_output = output_path.with_stem(f"{output_path.stem}_{label}")
            with open(label_output, 'w') as f:
                json.dump(pairs_original, f)
            print(f"  Saved {len(pairs)} pairs to: {label_output}")
    else:
        # Process all embeddings together
        distances = build_distance_matrix(embeddings, metric=args.metric)
        print(f"Distance matrix shape: {distances.shape}")

        # Compute distance threshold from percentile if set
        max_distance = None
        if args.distance_percentile is not None:
            upper_tri_indices = np.triu_indices(len(distances), k=1)
            upper_tri_distances = distances[upper_tri_indices]
            max_distance = np.percentile(upper_tri_distances, args.distance_percentile * 100)
            print(f"Distance threshold at {args.distance_percentile:.1%} percentile: {max_distance:.4f}")

        # Sample pairs
        pairs = sample_pairs(
            distances,
            n_pairs=args.n_pairs,
            max_distance=max_distance,
            seed=args.seed
        )

        # Save as JSON
        pairs_list = [[int(a), int(b)] for a, b in pairs]
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(pairs_list, f)
        print(f"Saved {len(pairs)} pairs to: {output_path}")


if __name__ == '__main__':
    main()
