"""
Tapkee Nanobind Python Bindings Example

Demonstrates dimensionality reduction using Tapkee's Python interface.
Run from the repository root: python examples/nanobind/nanobind_example.py
"""

import sys
import os

# Add examples directory to path for utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils import generate_data, embed, plot


def demo_single_method():
    """Demo: Embed Swiss roll with LLE."""
    print("Demo: Swiss roll embedding with LLE")
    print("-" * 40)

    data, colors = generate_data('swissroll', N=1000, random_state=42)
    print(f"Data shape: {data.shape} (features x samples)")

    embedding = embed(data, method='lle', num_neighbors=12)
    print(f"Embedding shape: {embedding.shape}")

    plot(data, embedding.T, colors, method='LLE')


def demo_compare_methods():
    """Demo: Compare different methods on the same dataset."""
    import matplotlib.pyplot as plt

    print("\nDemo: Comparing methods on S-curve")
    print("-" * 40)

    data, colors = generate_data('scurve', N=800, random_state=42)

    methods = ['lle', 'ltsa', 'isomap', 'pca']

    fig = plt.figure(figsize=(12, 3))
    fig.set_facecolor('white')

    for i, method_name in enumerate(methods):
        print(f"Running {method_name}...")
        embedding = embed(data, method=method_name, num_neighbors=12)

        ax = fig.add_subplot(1, 4, i + 1)
        ax.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=plt.cm.Spectral, s=5)
        ax.set_title(method_name.upper())
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def demo_different_datasets():
    """Demo: Apply LLE to different manifolds."""
    import matplotlib.pyplot as plt

    print("\nDemo: LLE on different manifolds")
    print("-" * 40)

    datasets = ['swissroll', 'scurve', 'helix', 'twinpeaks']

    fig = plt.figure(figsize=(12, 6))
    fig.set_facecolor('white')

    for i, dataset_name in enumerate(datasets):
        print(f"Processing {dataset_name}...")
        data, colors = generate_data(dataset_name, N=800, random_state=42)
        embedding = embed(data, method='lle', num_neighbors=12)

        # Original 3D data
        ax1 = fig.add_subplot(2, 4, i + 1, projection='3d')
        ax1.scatter(data[0], data[1], data[2], c=colors, cmap=plt.cm.Spectral, s=3)
        ax1.set_title(dataset_name)
        ax1.axis('off')

        # 2D embedding
        ax2 = fig.add_subplot(2, 4, i + 5)
        ax2.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap=plt.cm.Spectral, s=3)
        ax2.set_title('LLE embedding')
        ax2.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Tapkee nanobind examples')
    parser.add_argument('--demo', choices=['single', 'compare', 'datasets', 'all'],
                        default='single', help='Which demo to run')
    args = parser.parse_args()

    if args.demo == 'single' or args.demo == 'all':
        demo_single_method()

    if args.demo == 'compare' or args.demo == 'all':
        demo_compare_methods()

    if args.demo == 'datasets' or args.demo == 'all':
        demo_different_datasets()
