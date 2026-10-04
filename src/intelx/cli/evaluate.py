# evaluate_model.py
"""
Evaluation script for trained U-Net model
Visualizes predictions on training and validation samples
Shows: Origin image, Ground truth overlay, Prediction overlay, Inference time
"""
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import os
from torch.utils.data import DataLoader, Subset, random_split
import json

# Import your modules
from intelx.data.dataset import HybridChunkedDataset
from intelx.models.unet import create_unet_resnet34, create_unet_resnet18

from intelx.paths import weights_path, dataset_path, results_path


class EvaluationConfig:
    """Configuration for evaluation"""
    # Model checkpoint
    checkpoint_path = weights_path("checkpoint_best.pth")
    # Data
    chunks_dir = dataset_path("train_chunks")
    max_memory_gb = 50  # Use less memory for evaluation

    # Model architecture (must match training)
    model_type = 'resnet34'
    in_channels = 1
    out_channels = 1

    # Evaluation settings
    num_train_samples = 20  # Number of training samples to visualize
    num_val_samples = 20  # Number of validation samples to visualize
    validation_split_ratio = 0.05  # Same as training

    # Visualization
    output_dir = results_path("evaluation_large")
    overlay_alpha = 0.6  # Transparency for heatmap overlay

    # Inference
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    warmup_iterations = 10  # Warmup GPU before timing


config = EvaluationConfig()


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")

    # Create model
    if config.model_type == 'resnet18':
        model = create_unet_resnet18(
            in_channels=config.in_channels,
            out_channels=config.out_channels
        )
    elif config.model_type == 'resnet34':
        model = create_unet_resnet34(
            in_channels=config.in_channels,
            out_channels=config.out_channels
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle _orig_mod. prefix
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('_orig_mod.', '') if key.startswith('_orig_mod.') else key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    # Print checkpoint info
    epoch = checkpoint.get('epoch', 'unknown')
    val_r2 = checkpoint.get('val_r2', 'unknown')
    val_loss = checkpoint.get('val_loss', 'unknown')

    print(f"✓ Model loaded successfully")
    print(f"  Checkpoint epoch: {epoch}")
    print(f"  Validation R²: {val_r2}")
    print(f"  Validation Loss: {val_loss}\n")

    return model


def create_overlay_visualization(origin_np, heatmap_np, title, inference_time=None):
    """
    Create overlay visualization similar to your inspection script
    Returns figure with origin image and heatmap overlay
    """
    # Origin image is in [-1, 1], denormalize to [0, 1]
    origin_img = (origin_np * 0.5 + 0.5)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Panel 1: Origin image only
    axes[0].imshow(origin_img, cmap='gray')
    axes[0].set_title("Origin Image")
    axes[0].axis('off')

    # Panel 2: Heatmap overlay
    # Find non-zero values for color scaling
    non_zero_values = heatmap_np[heatmap_np > 0]

    if non_zero_values.size > 0:
        vmin = non_zero_values.min()
        vmax = non_zero_values.max()
    else:
        vmin, vmax = 0, 0.1

    # Draw origin image as base
    axes[1].imshow(origin_img, cmap='gray')
    # Overlay heatmap
    im = axes[1].imshow(heatmap_np, cmap='viridis', alpha=config.overlay_alpha, vmin=vmin, vmax=vmax)
    axes[1].set_title(title)
    axes[1].axis('off')

    # Add colorbar
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Add inference time if provided
    if inference_time is not None:
        fig.text(0.99, 0.01, f'Inference time: {inference_time:.2f} ms',
                 ha='right', va='bottom', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return fig


def visualize_sample(model, sample, sample_idx, split_name, device, output_dir):
    """
    Visualize a single sample: origin, ground truth overlay, prediction overlay
    """
    origin = sample["origin_image"].unsqueeze(0).to(device)  # Add batch dim
    target = sample["target_heatmap"].unsqueeze(0).to(device)

    # Warmup GPU
    if device == 'cuda':
        for _ in range(config.warmup_iterations):
            _ = model(origin)
        torch.cuda.synchronize()

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        prediction = model(origin)
    if device == 'cuda':
        torch.cuda.synchronize()
    inference_time_ms = (time.time() - start_time) * 1000

    # Convert to numpy
    origin_np = origin.cpu().squeeze().numpy()
    target_np = target.cpu().squeeze().numpy()
    pred_np = prediction.cpu().squeeze().numpy()

    # Create 3-panel visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: Origin image only
    origin_display = (origin_np * 0.5 + 0.5)
    axes[0].imshow(origin_display, cmap='gray')
    axes[0].set_title("Origin Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Ground truth overlay
    non_zero_target = target_np[target_np > 0]
    if non_zero_target.size > 0:
        vmin_target = non_zero_target.min()
        vmax_target = non_zero_target.max()
    else:
        vmin_target, vmax_target = 0, 0.1

    axes[1].imshow(origin_display, cmap='gray')
    im_target = axes[1].imshow(target_np, cmap='viridis', alpha=config.overlay_alpha,
                               vmin=vmin_target, vmax=vmax_target)
    axes[1].set_title("Ground Truth Overlay", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    fig.colorbar(im_target, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Prediction overlay
    non_zero_pred = pred_np[pred_np > 0]
    if non_zero_pred.size > 0:
        vmin_pred = non_zero_pred.min()
        vmax_pred = non_zero_pred.max()
    else:
        vmin_pred, vmax_pred = 0, 0.1

    axes[2].imshow(origin_display, cmap='gray')
    im_pred = axes[2].imshow(pred_np, cmap='viridis', alpha=config.overlay_alpha,
                             vmin=vmin_pred, vmax=vmax_pred)
    axes[2].set_title("Prediction Overlay", fontsize=14, fontweight='bold')
    axes[2].axis('off')
    fig.colorbar(im_pred, ax=axes[2], fraction=0.046, pad=0.04)

    # Add statistics
    target_mean = target_np[target_np > 0].mean() if (target_np > 0).any() else 0
    pred_mean = pred_np[pred_np > 0].mean() if (pred_np > 0).any() else 0
    target_max = target_np.max()
    pred_max = pred_np.max()

    stats_text = (f"Inference: {inference_time_ms:.2f} ms\n"
                  f"Target: mean={target_mean:.6f}, max={target_max:.6f}\n"
                  f"Pred:   mean={pred_mean:.6f}, max={pred_max:.6f}")

    fig.text(0.99, 0.01, stats_text, ha='right', va='bottom', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             family='monospace')

    # Super title
    plt.suptitle(f"{split_name} Sample {sample_idx}", fontsize=16, fontweight='bold')

    # Save
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{split_name.lower()}_sample_{sample_idx:04d}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return inference_time_ms


def evaluate():
    """Main evaluation function"""

    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    print(f"Checkpoint: {config.checkpoint_path}")
    print(f"Data: {config.chunks_dir}")
    print(f"Device: {config.device}")
    print(f"Output: {config.output_dir}")
    print("=" * 80 + "\n")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Load model
    model = load_model(config.checkpoint_path, config.device)

    # Load dataset
    print("Loading dataset...")
    dataset = HybridChunkedDataset(
        chunks_dir=config.chunks_dir,
        max_memory_gb=config.max_memory_gb,
        verbose=True
    )

    # Split into train/val (same as training)
    val_size = int(len(dataset) * config.validation_split_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\nDataset split:")
    print(f"  Training: {len(train_dataset):,} samples")
    print(f"  Validation: {len(val_dataset):,} samples\n")

    # Sample indices
    train_indices = np.linspace(0, len(train_dataset) - 1, config.num_train_samples, dtype=int)
    val_indices = np.linspace(0, len(val_dataset) - 1, config.num_val_samples, dtype=int)

    # Evaluate training samples
    print("=" * 80)
    print("EVALUATING TRAINING SAMPLES")
    print("=" * 80)

    train_inference_times = []
    for i, idx in enumerate(train_indices):
        print(f"Processing training sample {i + 1}/{config.num_train_samples} (index {idx})...", end='\r')
        sample = train_dataset[idx]
        inference_time = visualize_sample(
            model, sample, idx, "Training", config.device, config.output_dir
        )
        train_inference_times.append(inference_time)

    print(f"\n✓ Processed {config.num_train_samples} training samples")
    print(f"  Average inference time: {np.mean(train_inference_times):.2f} ms")
    print(f"  Min/Max: {np.min(train_inference_times):.2f} / {np.max(train_inference_times):.2f} ms\n")

    # Evaluate validation samples
    print("=" * 80)
    print("EVALUATING VALIDATION SAMPLES")
    print("=" * 80)

    val_inference_times = []
    for i, idx in enumerate(val_indices):
        print(f"Processing validation sample {i + 1}/{config.num_val_samples} (index {idx})...", end='\r')
        sample = val_dataset[idx]
        inference_time = visualize_sample(
            model, sample, idx, "Validation", config.device, config.output_dir
        )
        val_inference_times.append(inference_time)

    print(f"\n✓ Processed {config.num_val_samples} validation samples")
    print(f"  Average inference time: {np.mean(val_inference_times):.2f} ms")
    print(f"  Min/Max: {np.min(val_inference_times):.2f} / {np.max(val_inference_times):.2f} ms\n")

    # Overall statistics
    all_inference_times = train_inference_times + val_inference_times

    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total samples evaluated: {len(all_inference_times)}")
    print(f"Average inference time: {np.mean(all_inference_times):.2f} ms")
    print(f"Std deviation: {np.std(all_inference_times):.2f} ms")
    print(f"Min/Max: {np.min(all_inference_times):.2f} / {np.max(all_inference_times):.2f} ms")
    print(f"FPS (avg): {1000 / np.mean(all_inference_times):.1f}")
    print(f"\nVisualizations saved to: {config.output_dir}/")
    print("=" * 80 + "\n")

    # Save summary
    summary = {
        'checkpoint': config.checkpoint_path,
        'num_train_samples': config.num_train_samples,
        'num_val_samples': config.num_val_samples,
        'train_inference_times_ms': train_inference_times,
        'val_inference_times_ms': val_inference_times,
        'avg_inference_time_ms': float(np.mean(all_inference_times)),
        'std_inference_time_ms': float(np.std(all_inference_times)),
        'min_inference_time_ms': float(np.min(all_inference_times)),
        'max_inference_time_ms': float(np.max(all_inference_times)),
        'avg_fps': float(1000 / np.mean(all_inference_times))
    }

    summary_file = os.path.join(config.output_dir, 'evaluation_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Summary saved to: {summary_file}")


if __name__ == '__main__':
    evaluate()
