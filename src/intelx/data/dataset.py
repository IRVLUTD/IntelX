# segmentation_data_loader_chunks.py
"""
Hybrid chunked dataloader optimized for memory budget
Keeps N chunks in memory, streams the rest
"""
import torch
import h5py
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
import json


class HybridChunkedDataset(Dataset):
    """
    Hybrid approach: Keep N chunks in RAM, stream others
    Optimized for configurable memory budget
    """

    def __init__(self, chunks_dir, max_memory_gb=250, transform=None, verbose=True):
        """
        Args:
            chunks_dir: Directory containing chunk_*.h5 files
            max_memory_gb: Maximum RAM to use (in GB)
            transform: Optional transform to apply to images
            verbose: Print detailed loading information
        """
        self.chunks_dir = Path(chunks_dir)
        self.max_memory_gb = max_memory_gb
        self.verbose = verbose

        # Find all chunk files
        self.chunk_files = sorted(list(self.chunks_dir.glob("chunk_*.h5")))
        if not self.chunk_files:
            raise ValueError(f"No chunk files found in {chunks_dir}")

        if verbose:
            print(f"Found {len(self.chunk_files)} chunk files")

        # Load metadata to estimate chunk sizes
        metadata_file = self.chunks_dir / "chunks_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = None
            if verbose:
                print("Warning: chunks_metadata.json not found, will estimate sizes")

        # Build sample index and estimate memory usage
        self.sample_index = []  # [(chunk_idx, local_idx), ...]
        self.chunk_sizes = []  # Number of samples per chunk
        self.chunk_memory_estimate = []  # Estimated GB per chunk
        self.total_samples = 0

        if verbose:
            print("Analyzing chunk sizes...")

        for chunk_idx, chunk_file in enumerate(self.chunk_files):
            with h5py.File(chunk_file, 'r') as f:
                chunk_size = f['origin_images'].shape[0]
                self.chunk_sizes.append(chunk_size)

                # Estimate memory: data size in bytes / (1024^3) for GB
                origin_bytes = f['origin_images'].dtype.itemsize * np.prod(f['origin_images'].shape)
                target_bytes = f['target_heatmaps'].dtype.itemsize * np.prod(f['target_heatmaps'].shape)
                chunk_gb = (origin_bytes + target_bytes) / (1024 ** 3)
                self.chunk_memory_estimate.append(chunk_gb)

                for local_idx in range(chunk_size):
                    self.sample_index.append((chunk_idx, local_idx))

                self.total_samples += chunk_size

        # Determine which chunks to keep in memory
        self._determine_memory_strategy()

        # Initialize chunk cache
        self.cached_chunks = {}  # {chunk_idx: (origin_data, target_data)}
        self.stream_chunk_file = None  # For streaming non-cached chunks
        self.stream_chunk_idx = None

        # Load chunks into memory
        self._load_cached_chunks()

        # Transform for input images
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            self.transform = transform

        if verbose:
            print(f"Dataset ready. Total samples: {self.total_samples:,}")

    def _determine_memory_strategy(self):
        """Determine which chunks to keep in memory based on budget"""
        total_memory_needed = sum(self.chunk_memory_estimate)

        if self.verbose:
            print(f"\nMemory Analysis:")
            print(f"  Total data size: {total_memory_needed:.2f} GB")
            print(f"  Memory budget: {self.max_memory_gb:.2f} GB")

        if total_memory_needed <= self.max_memory_gb:
            # All chunks fit in memory
            self.chunks_to_cache = list(range(len(self.chunk_files)))
            self.chunks_to_stream = []
            if self.verbose:
                print(f"  Strategy: All {len(self.chunk_files)} chunks fit in memory ✓")
        else:
            # Need to cache some, stream others
            # Cache first N chunks that fit in budget
            cumulative_memory = 0
            self.chunks_to_cache = []

            for idx, chunk_gb in enumerate(self.chunk_memory_estimate):
                if cumulative_memory + chunk_gb <= self.max_memory_gb:
                    self.chunks_to_cache.append(idx)
                    cumulative_memory += chunk_gb
                else:
                    break

            self.chunks_to_stream = [i for i in range(len(self.chunk_files))
                                     if i not in self.chunks_to_cache]

            if self.verbose:
                print(f"  Strategy: Hybrid approach")
                print(f"    - Cached chunks: {len(self.chunks_to_cache)} "
                      f"({cumulative_memory:.2f} GB)")
                print(f"    - Streamed chunks: {len(self.chunks_to_stream)}")

                # Calculate what % of samples are cached
                cached_samples = sum(self.chunk_sizes[i] for i in self.chunks_to_cache)
                cache_percent = (cached_samples / self.total_samples) * 100
                print(f"    - Cached samples: {cached_samples:,} ({cache_percent:.1f}%)")

    def _load_cached_chunks(self):
        """Load selected chunks into memory"""
        if not self.chunks_to_cache:
            if self.verbose:
                print("No chunks to cache (pure streaming mode)")
            return

        if self.verbose:
            print(f"\nLoading {len(self.chunks_to_cache)} chunks into memory...")

        from tqdm import tqdm
        for chunk_idx in tqdm(self.chunks_to_cache, desc="Loading chunks",
                              disable=not self.verbose):
            chunk_file = self.chunk_files[chunk_idx]
            with h5py.File(chunk_file, 'r') as f:
                # Load entire chunk into numpy arrays
                origin_data = f['origin_images'][:]
                target_data = f['target_heatmaps'][:]
                self.cached_chunks[chunk_idx] = (origin_data, target_data)

        if self.verbose:
            print(f"✓ {len(self.cached_chunks)} chunks loaded into memory")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        chunk_idx, local_idx = self.sample_index[idx]

        # Check if chunk is in cache
        if chunk_idx in self.cached_chunks:
            # Fast path: data is in memory
            origin_np, target_np = self.cached_chunks[chunk_idx]
            origin_np = origin_np[local_idx]
            target_np = target_np[local_idx]
        else:
            # Slow path: stream from disk
            # Open chunk file if not already open
            if self.stream_chunk_idx != chunk_idx:
                # Close previous file
                if self.stream_chunk_file is not None:
                    self.stream_chunk_file.close()

                # Open new file
                chunk_file = self.chunk_files[chunk_idx]
                self.stream_chunk_file = h5py.File(chunk_file, 'r')
                self.stream_chunk_idx = chunk_idx

            # Read from disk
            origin_np = self.stream_chunk_file['origin_images'][local_idx]
            target_np = self.stream_chunk_file['target_heatmaps'][local_idx]

        # Convert to tensors
        origin_tensor = self.transform(Image.fromarray(origin_np))
        target_tensor = torch.from_numpy(target_np).float().unsqueeze(0)

        return {
            "origin_image": origin_tensor,
            "target_heatmap": target_tensor
        }

    def __del__(self):
        """Cleanup: close any open files"""
        if self.stream_chunk_file is not None:
            self.stream_chunk_file.close()


class FullyStreamedDataset(Dataset):
    """
    Fully streamed version - uses minimal RAM
    Use this if you want to use <10GB RAM
    """

    def __init__(self, chunks_dir, transform=None):
        self.chunks_dir = Path(chunks_dir)
        self.chunk_files = sorted(list(self.chunks_dir.glob("chunk_*.h5")))

        if not self.chunk_files:
            raise ValueError(f"No chunk files found in {chunks_dir}")

        print(f"Found {len(self.chunk_files)} chunk files")

        # Build index
        self.sample_index = []
        self.total_samples = 0

        print("Building sample index...")
        for chunk_idx, chunk_file in enumerate(self.chunk_files):
            with h5py.File(chunk_file, 'r') as f:
                chunk_size = f['origin_images'].shape[0]
                for local_idx in range(chunk_size):
                    self.sample_index.append((chunk_idx, local_idx))
                self.total_samples += chunk_size

        print(f"Dataset loaded. Total samples: {self.total_samples:,}")
        print(f"Mode: Fully streamed (minimal RAM usage)")

        # Cache for currently open chunk
        self._current_chunk_idx = None
        self._current_chunk_file = None

        # Transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        chunk_idx, local_idx = self.sample_index[idx]

        # Open chunk file if needed
        if self._current_chunk_idx != chunk_idx:
            if self._current_chunk_file is not None:
                self._current_chunk_file.close()

            chunk_file = self.chunk_files[chunk_idx]
            self._current_chunk_file = h5py.File(chunk_file, 'r')
            self._current_chunk_idx = chunk_idx

        # Read data
        origin_np = self._current_chunk_file['origin_images'][local_idx]
        target_np = self._current_chunk_file['target_heatmaps'][local_idx]

        # Convert to tensors
        origin_tensor = self.transform(Image.fromarray(origin_np))
        target_tensor = torch.from_numpy(target_np).float().unsqueeze(0)

        return {
            "origin_image": origin_tensor,
            "target_heatmap": target_tensor
        }

    def __del__(self):
        if self._current_chunk_file is not None:
            self._current_chunk_file.close()


class FullyCachedDataset(Dataset):
    """
    Fully cached version - loads ALL data into RAM
    Only use if you have >850GB RAM
    """

    def __init__(self, chunks_dir, transform=None):
        self.chunks_dir = Path(chunks_dir)
        self.chunk_files = sorted(list(self.chunks_dir.glob("chunk_*.h5")))

        if not self.chunk_files:
            raise ValueError(f"No chunk files found in {chunks_dir}")

        print(f"Found {len(self.chunk_files)} chunk files")
        print("Loading ALL data into memory...")

        # Load all data
        self.origin_data = []
        self.target_data = []

        from tqdm import tqdm
        for chunk_file in tqdm(self.chunk_files, desc="Loading chunks"):
            with h5py.File(chunk_file, 'r') as f:
                self.origin_data.append(f['origin_images'][:])
                self.target_data.append(f['target_heatmaps'][:])

        # Concatenate all chunks
        print("Concatenating chunks...")
        self.origin_data = np.concatenate(self.origin_data, axis=0)
        self.target_data = np.concatenate(self.target_data, axis=0)

        self.total_samples = len(self.origin_data)
        print(f"Dataset loaded. Total samples: {self.total_samples:,}")
        print(f"Mode: Fully cached (maximum speed)")

        # Transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        origin_np = self.origin_data[idx]
        target_np = self.target_data[idx]

        origin_tensor = self.transform(Image.fromarray(origin_np))
        target_tensor = torch.from_numpy(target_np).float().unsqueeze(0)

        return {
            "origin_image": origin_tensor,
            "target_heatmap": target_tensor
        }
