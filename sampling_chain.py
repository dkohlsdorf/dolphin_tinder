#!/usr/bin/env python3
"""
Build a sampling chain from embeddings using distance-weighted sampling (k-means++ style):
1. Build a distance matrix
2. Take a random example to start
3. Sample close (prob ~ 1/distance) and far (prob ~ distance) examples
4. Save the close examples as pairs: anchor, neighbor
5. Repeat from 3 using the far example without replacement
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


def sample_by_distance(distances, anchor_idx, available_indices, sample_close=True, max_distance=None):
    """
    Sample an example using distance-weighted probability (k-means++ style).

    Args:
        distances: Full distance matrix
        anchor_idx: Current anchor index
        available_indices: Set of indices still available for sampling
        sample_close: If True, prob ~ 1/distance (closer more likely)
                      If False, prob ~ distance (farther more likely)
        max_distance: If set and sample_close=True, only consider candidates
                      within this distance threshold

    Returns:
        Sampled index, or None if no valid candidates
    """
    available_list = list(available_indices)

    if len(available_list) == 0:
        return None
    if len(available_list) == 1:
        return available_list[0]

    # Get distances from anchor to all available indices
    dists = distances[anchor_idx, available_list]

    # Filter by max_distance threshold for close sampling
    if sample_close and max_distance is not None:
        mask = dists <= max_distance
        if not mask.any():
            return None  # No candidates within threshold
        available_list = [available_list[i] for i in range(len(available_list)) if mask[i]]
        dists = dists[mask]

    # Avoid division by zero
    dists = np.maximum(dists, 1e-10)

    if sample_close:
        # Probability proportional to inverse distance (closer = higher prob)
        weights = 1.0 / dists
    else:
        # Probability proportional to distance squared (like k-means++)
        weights = dists ** 2

    # Normalize to probabilities
    probs = weights / weights.sum()

    # Sample
    sampled_idx = np.random.choice(available_list, p=probs)
    return sampled_idx


def build_sampling_chain(embeddings, distances, seed=None, max_distance=None):
    """
    Build the sampling chain by traversing the embedding space.

    Args:
        embeddings: The embedding vectors
        distances: Precomputed distance matrix
        seed: Random seed for reproducibility
        max_distance: Maximum distance threshold for close pairs.
                      Pairs with distance > max_distance are skipped.

    Returns:
        pairs: List of (anchor_idx, neighbor_idx) tuples
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = len(embeddings)
    available = set(range(n_samples))
    pairs = []
    skipped = 0

    # Start with a random example
    anchor_idx = np.random.choice(list(available))
    available.remove(anchor_idx)

    print(f"Starting chain from index {anchor_idx}")
    print(f"Building chain with {n_samples} samples...")
    if max_distance is not None:
        print(f"Using max_distance threshold: {max_distance}")

    iteration = 0
    while len(available) >= 1:
        # Sample close neighbor (prob ~ 1/distance, filtered by max_distance)
        close_idx = sample_by_distance(distances, anchor_idx, available, sample_close=True, max_distance=max_distance)

        if close_idx is None:
            # No candidates within threshold, skip this anchor
            skipped += 1
            # Try to find a new anchor by sampling far
            if len(available) == 0:
                break
            far_idx = sample_by_distance(distances, anchor_idx, available, sample_close=False)
            if far_idx is None:
                break
            anchor_idx = far_idx
            available.discard(far_idx)
            continue

        # Save the anchor-neighbor pair
        pairs.append((anchor_idx, close_idx))
        available.discard(close_idx)

        if len(available) == 0:
            break

        # Sample far example (prob ~ distance^2, k-means++ style)
        far_idx = sample_by_distance(distances, anchor_idx, available, sample_close=False)

        if far_idx is None:
            break

        # Move to the far example for the next iteration
        anchor_idx = far_idx
        available.discard(far_idx)

        iteration += 1
        if iteration % 1000 == 0:
            print(f"  Processed {iteration} iterations, {len(available)} samples remaining...")

    print(f"Generated {len(pairs)} pairs (skipped {skipped} anchors with no close neighbors)")
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
        description='Build sampling chain from embeddings'
    )
    parser.add_argument(
        '--embeddings', type=str, default='embeddings.npy',
        help='Path to embeddings file (.npy format)'
    )
    parser.add_argument(
        '--labels', type=str, default='embeddings.labels.npy',
        help='Path to labels file (.npy format). If provided, generates one chain per label.'
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
        '--distance-percentile', type=float, default=0.1,
        help='Percentile of distance distribution to use as threshold (0-1). '
             'Only pairs with distance <= this percentile are considered close. '
             'Default: 0.1 (10th percentile).'
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

            # Compute distance threshold from percentile
            upper_tri_indices = np.triu_indices(len(distances), k=1)
            upper_tri_distances = distances[upper_tri_indices]
            max_distance = np.percentile(upper_tri_distances, args.distance_percentile * 100)
            print(f"  Distance threshold at {args.distance_percentile:.1%} percentile: {max_distance:.4f}")

            # Build sampling chain
            pairs = build_sampling_chain(
                label_embeddings, distances,
                seed=args.seed,
                max_distance=max_distance
            )

            # Convert local indices back to original indices
            pairs_original = [[int(original_indices[a]), int(original_indices[b])] for a, b in pairs]

            # Save pairs for this label as JSON
            label_output = output_path.with_stem(f"{output_path.stem}_{label}")
            with open(label_output, 'w') as f:
                json.dump(pairs_original, f)
            print(f"  Saved {len(pairs)} pairs to: {label_output}")
    else:
        # Original behavior: process all embeddings together
        # Build distance matrix
        distances = build_distance_matrix(embeddings, metric=args.metric)
        print(f"Distance matrix shape: {distances.shape}")

        # Compute distance threshold from percentile
        upper_tri_indices = np.triu_indices(len(distances), k=1)
        upper_tri_distances = distances[upper_tri_indices]
        max_distance = np.percentile(upper_tri_distances, args.distance_percentile * 100)
        print(f"Distance threshold at {args.distance_percentile:.1%} percentile: {max_distance:.4f}")

        # Build sampling chain
        pairs = build_sampling_chain(
            embeddings, distances,
            seed=args.seed,
            max_distance=max_distance
        )

        # Convert to list and save as JSON
        pairs_list = [[int(a), int(b)] for a, b in pairs]
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(pairs_list, f)
        print(f"Saved {len(pairs)} pairs to: {output_path}")


if __name__ == '__main__':
    main()
