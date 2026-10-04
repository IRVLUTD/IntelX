# train_flexible_unet.py
"""
Training script for Flexible U-Net on chunked segmentation data
Supports both validation runs (small data) and full training
FIXED: Smart data loading + removed verbose parameter
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend, no Tkinter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import torch.nn.functional as F
# Hugging Face Accelerate for easy, high-performance training
from accelerate import Accelerator

# Import custom modules
from intelx.data.dataset import HybridChunkedDataset
from intelx.models.unet import create_unet_resnet34, create_unet_resnet18, FlexibleUNet
from torchmetrics import R2Score

from intelx.paths import weights_path, dataset_path


# --- 1. Configuration ---
class TrainingConfig:
    """Configuration for training"""

    # ============================================================
    # MODE SELECTION
    # ============================================================
    # Set to True for quick validation, False for full training
    validation_mode = False

    # ============================================================
    # DATA PARAMETERS
    # ============================================================
    chunks_dir = dataset_path("train_chunks")
    max_memory_gb = 200  # RAM budget for HybridChunkedDataset

    # Validation mode: use subset of data
    if validation_mode:
        use_data_fraction = 0.05  # Use 5% of data (~50k samples)
        validation_split_ratio = 0.05  # 5% of subset for validation
    else:
        use_data_fraction = 1.0  # Use all data
        validation_split_ratio = 0.05  # 1% for validation (~10k samples)

    # ============================================================
    # MODEL PARAMETERS
    # ============================================================
    # if validation_mode:
    #     # Lighter model for quick validation
    #     model_type = 'resnet18'
    #     encoder_depth = 5
    #     decoder_channels = (128, 64, 32, 16, 8)
    # else:
    # Full model for production
    model_type = 'resnet34'
    encoder_depth = 5
    decoder_channels = (256, 128, 64, 32, 16)

    in_channels = 1
    out_channels = 1
    upsampling_type = 'bilinear'
    skip_type = 'concat'
    center_block = False
    activation = 'sigmoid'

    # ============================================================
    # TRAINING PARAMETERS
    # ============================================================
    if validation_mode:
        # Quick validation settings
        train_batch_size = 32
        val_batch_size = 32
        num_epochs = 20
        learning_rate = 1e-3
        early_stopping_patience = 5
    else:
        # Full training settings, as reported: batch 32, up to 300 epochs,
        # keeping the checkpoint with the best validation R2.
        train_batch_size = 32
        val_batch_size = 32
        num_epochs = 300
        learning_rate = 1e-4
        early_stopping_patience = 20

    # Optimizer
    optimizer_type = 'AdamW'
    weight_decay = 1e-4

    # Learning rate schedule
    use_lr_scheduler = True
    lr_scheduler_type = 'plateau'  # 'plateau' or 'cosine'
    lr_plateau_factor = 0.8
    lr_plateau_patience = 2
    warmup_epochs = 1 if not validation_mode else 0

    # ============================================================
    # HARDWARE & OPTIMIZATION
    # ============================================================
    mixed_precision = "fp16"
    gradient_accumulation_steps = 1
    # In TrainingConfig, change:
    num_workers = 4

    pin_memory = True

    # ============================================================
    # OUTPUT & LOGGING
    # ============================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if validation_mode:
        output_dir = weights_path(f"runs/unet_{model_type}_val_{timestamp}")
    else:
        output_dir = weights_path(f"runs/unet_{model_type}_full_{timestamp}")

    save_every_n_epochs = 1
    visualize_every_n_epochs = 2 if validation_mode else 1

    # ============================================================
    # EXPERIMENT TRACKING
    # ============================================================
    use_wandb = False  # Set to True to enable Weights & Biases logging
    wandb_project = "coverage_prediction"
    wandb_entity = None  # Your wandb username

    # None trains from scratch. To continue a run, point this at its
    # checkpoint_best.pth, e.g. weights_path("runs/<name>/checkpoint_best.pth").
    resume_from_checkpoint = None

config = TrainingConfig()


# --- 2. Helper Functions ---
def save_validation_sample(model, batch, epoch, device, output_dir):
    """Generates and saves a visual comparison of a model's prediction."""
    model.eval()
    with torch.no_grad():
        origin_image = batch["origin_image"].to(device)
        target_heatmap = batch["target_heatmap"].to(device)

        predicted_heatmap = model(origin_image)

        # Take the first item from the batch for visualization
        origin_img_viz = (origin_image[0] * 0.5 + 0.5).cpu().squeeze()
        target_viz = target_heatmap[0].cpu().squeeze()
        predicted_viz = predicted_heatmap[0].cpu().squeeze()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(origin_img_viz, cmap='gray')
        axes[0].set_title("Input: Origin Image")
        axes[0].axis('off')

        axes[1].imshow(target_viz, cmap='viridis')
        axes[1].set_title("Ground Truth Heatmap")
        axes[1].axis('off')

        im = axes[2].imshow(predicted_viz, cmap='viridis')
        axes[2].set_title("Predicted Heatmap")
        axes[2].axis('off')

        # Add colorbar
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        plt.suptitle(f"Epoch {epoch} Validation Sample", fontsize=16)
        save_path = os.path.join(output_dir, f"validation_sample_epoch_{epoch:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    model.train()


def plot_metrics(train_losses, val_losses, train_r2s, val_r2s, output_dir):
    """Plots and saves the training and validation loss and R2 curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot Loss
    ax1.plot(train_losses, label="Train Loss", marker='o', markersize=4)
    ax1.plot(val_losses, label="Validation Loss", marker='s', markersize=4)
    ax1.set_title("Loss Curves", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("MSE Loss", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot R2 Score
    ax2.plot(train_r2s, label="Train R2 Score", marker='o', markersize=4)
    ax2.plot(val_r2s, label="Validation R2 Score", marker='s', markersize=4)
    ax2.set_title("R2 Score Curves", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("R2 Score", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150)
    print(f"Training curves saved to {save_path}")
    plt.close(fig)


def config_dict(config):
    """Every setting of a run. They are class attributes of TrainingConfig, so
    vars() of the instance is empty and a run used to record {}."""
    return {k: v for k, v in vars(type(config)).items()
            if not k.startswith('_') and not callable(v)
            and not isinstance(v, (staticmethod, classmethod, property))}


def save_config(config, output_dir):
    """Save configuration to JSON file"""
    config_dict_ = config_dict(config)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config_dict_, f, indent=2, default=str)
    print(f"Configuration saved to {config_path}")


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


class ScaledMSELoss(nn.Module):
    """
    MSE Loss that scales values before computation for better gradients.
    No need to modify data - just improves loss magnitude for tiny values.
    """

    def __init__(self, scale_factor=10000.0):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, pred, target):
        # Temporarily scale up for loss computation
        # This gives better gradients without changing the actual values
        scaled_pred = pred * self.scale_factor
        scaled_target = target * self.scale_factor
        return F.mse_loss(scaled_pred, scaled_target)


def load_checkpoint_safe(checkpoint_path, model, optimizer, lr_scheduler=None):
    """Load checkpoint and handle _orig_mod. prefix"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model state - strip _orig_mod. prefix if present
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('_orig_mod.', '') if key.startswith('_orig_mod.') else key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state if provided
    if lr_scheduler is not None and 'scheduler_state_dict' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint

def train():
    # import os
    # os.environ['TORCH_COMPILE_DISABLE'] = '1'

    # --- Setup ---
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps
    )

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        save_config(config, config.output_dir)

    # Print configuration
    if accelerator.is_main_process:
        print("=" * 80)
        print("TRAINING CONFIGURATION")
        print("=" * 80)
        print(f"Mode: {'VALIDATION' if config.validation_mode else 'FULL TRAINING'}")
        print(f"Model: {config.model_type}")
        print(f"Data fraction: {config.use_data_fraction * 100:.1f}%")
        print(f"Batch size: {config.train_batch_size}")
        print(f"Epochs: {config.num_epochs}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Output directory: {config.output_dir}")
        if config.resume_from_checkpoint:
            print(f"Resume from: {config.resume_from_checkpoint}")
        print("=" * 80 + "\n")

    # --- FIXED: Smart Data Loading ---
    if accelerator.is_main_process:
        print("Loading dataset (only what we need)...")

    # First, figure out how many samples we need
    metadata_file = Path(config.chunks_dir) / "chunks_metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    total_samples = metadata['total_samples']
    samples_needed = int(total_samples * config.use_data_fraction)

    if accelerator.is_main_process:
        print(f"Total available samples: {total_samples:,}")
        print(f"Samples needed: {samples_needed:,} ({config.use_data_fraction * 100:.1f}%)")

    # Calculate memory needed for subset
    chunk_size = metadata['chunk_size']
    chunks_needed = int(np.ceil(samples_needed / chunk_size))

    # Estimate memory per chunk (total GB / total chunks)
    total_gb = 733.83  # From your error message
    total_chunks = metadata['num_chunks']
    memory_per_chunk = total_gb / total_chunks
    memory_needed = chunks_needed * memory_per_chunk

    # Adjust memory budget for subset
    if config.use_data_fraction < 1.0:
        adjusted_memory_gb = min(config.max_memory_gb, memory_needed * 1.2)  # 20% buffer
        if accelerator.is_main_process:
            print(f"Adjusted memory budget: {adjusted_memory_gb:.1f} GB (enough for {chunks_needed} chunks)\n")
    else:
        adjusted_memory_gb = config.max_memory_gb

    # Load dataset with adjusted memory
    full_dataset = HybridChunkedDataset(
        chunks_dir=config.chunks_dir,
        max_memory_gb=adjusted_memory_gb,
        verbose=accelerator.is_main_process
    )

    # FIXED: Subset IMMEDIATELY after loading
    if config.use_data_fraction < 1.0:
        # Use first N samples (data is already shuffled in chunks)
        indices = list(range(min(samples_needed, len(full_dataset))))
        full_dataset = Subset(full_dataset, indices)
        if accelerator.is_main_process:
            print(f"\n✓ Using subset: {len(full_dataset):,} samples\n")

    # Split into train/val
    val_size = int(len(full_dataset) * config.validation_split_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    print(f"Dataset split: {len(train_dataset):,} training, {len(val_dataset):,} validation samples")
    print(f"Batches per epoch: {len(train_dataloader):,} train, {len(val_dataloader):,} val\n")

    # --- Model Definition ---
    print("Creating model...")
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
        # Custom model
        model = FlexibleUNet(
            encoder=config.model_type,
            encoder_depth=config.encoder_depth,
            decoder_channels=config.decoder_channels,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            upsampling_type=config.upsampling_type,
            skip_type=config.skip_type,
            center_block=config.center_block,
            activation=config.activation
        )

    total_params, trainable_params = count_parameters(model)
    print(f"Model: {config.model_type}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # --- Optimizer, Loss, Scheduler ---
    if config.optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer_type}")

    loss_fn = ScaledMSELoss(scale_factor=10000.0)

    # FIXED: Learning rate scheduler without verbose parameter
    if config.use_lr_scheduler:
        if config.lr_scheduler_type == 'plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=config.lr_plateau_factor,
                patience=config.lr_plateau_patience
            )
        elif config.lr_scheduler_type == 'cosine':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.num_epochs
            )
        else:
            raise ValueError(f"Unknown scheduler: {config.lr_scheduler_type}")
    else:
        lr_scheduler = None

    # ============================================
    # RESUME FROM CHECKPOINT (BEFORE ACCELERATOR)
    # ============================================
    start_epoch = 0
    best_val_r2 = -float('inf')
    best_val_loss = float('inf')
    train_loss_history, val_loss_history = [], []
    train_r2_history, val_r2_history = [], []
    epochs_without_improvement = 0

    if config.resume_from_checkpoint and os.path.exists(config.resume_from_checkpoint):
        print(f"Resuming from checkpoint: {config.resume_from_checkpoint}")
        # checkpoint = torch.load(config.resume_from_checkpoint, map_location='cpu')
        #
        # # Load model state
        # model.load_state_dict(checkpoint['model_state_dict'])
        # print("  ✓ Model state loaded")
        #
        # # Load optimizer state
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # print("  ✓ Optimizer state loaded")
        checkpoint = load_checkpoint_safe(config.resume_from_checkpoint, model, optimizer, lr_scheduler)

        # Load training progress
        start_epoch = checkpoint['epoch'] + 1
        best_val_r2 = checkpoint.get('val_r2', -float('inf'))
        best_val_loss = checkpoint.get('val_loss', float('inf'))

        # Load training history
        history_file = os.path.join(os.path.dirname(config.resume_from_checkpoint), 'training_history.json')
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
            train_loss_history = history['train_loss']
            val_loss_history = history['val_loss']
            train_r2_history = history['train_r2']
            val_r2_history = history['val_r2']
            print(f"  ✓ Training history loaded ({len(train_loss_history)} epochs)")

        # Load scheduler state if using cosine
        if config.use_lr_scheduler and config.lr_scheduler_type == 'cosine':
            if 'scheduler_state_dict' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("  ✓ Scheduler state loaded")

        print(f"\n✓ Resumed from epoch {checkpoint['epoch']}")
        print(f"  Best Val R²: {best_val_r2:.4f}")
        print(f"  Best Val Loss: {best_val_loss:.6f}")
        print(f"  Continuing from epoch {start_epoch}\n")

    # Metrics
    train_r2_metric = R2Score().to(accelerator.device)
    val_r2_metric = R2Score().to(accelerator.device)

    # --- Prepare with Accelerator ---
    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    if lr_scheduler is not None and config.lr_scheduler_type != 'plateau':
        lr_scheduler = accelerator.prepare(lr_scheduler)

    # --- Initialize Weights & Biases (optional) ---
    if config.use_wandb and accelerator.is_main_process:
        import wandb

        # Resume wandb run if resuming training
        if config.resume_from_checkpoint:
            # Try to load wandb run id
            wandb_id_file = os.path.join(os.path.dirname(config.resume_from_checkpoint), 'wandb_run_id.txt')
            if os.path.exists(wandb_id_file):
                with open(wandb_id_file, 'r') as f:
                    wandb_run_id = f.read().strip()
                wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    config=config_dict(config),
                    id=wandb_run_id,
                    resume="must"
                )
            else:
                wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity,
                    config=config_dict(config),
                    name=f"{config.model_type}_{'val' if config.validation_mode else 'full'}_{config.timestamp}"
                )
        else:
            run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=config_dict(config),
                name=f"{config.model_type}_{'val' if config.validation_mode else 'full'}_{config.timestamp}"
            )
            # Save wandb run id for future resume
            with open(os.path.join(config.output_dir, 'wandb_run_id.txt'), 'w') as f:
                f.write(run.id)

        wandb.watch(model, log='all', log_freq=100)

    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80 + "\n")

    # ========== TRAINING LOOP (MODIFIED TO START FROM start_epoch) ==========
    for epoch in range(start_epoch, config.num_epochs):
        # ========== TRAINING ==========
        model.train()
        train_loss_accum = 0.0
        train_r2_metric.reset()

        progress_bar = tqdm(
            total=len(train_dataloader),
            desc=f"Epoch {epoch + 1}/{config.num_epochs} [Train]",
            disable=not accelerator.is_local_main_process
        )

        for batch_idx, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                origin_images = batch["origin_image"]
                target_heatmaps = batch["target_heatmap"]

                predicted_heatmaps = model(origin_images)
                loss = loss_fn(predicted_heatmaps, target_heatmaps)

                accelerator.backward(loss)
                optimizer.step()

                # Update metrics
                train_loss_accum += loss.detach().item()
                train_r2_metric.update(
                    predicted_heatmaps.detach().view(-1),
                    target_heatmaps.view(-1)
                )

                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{loss.item():.6f}")

        progress_bar.close()

        # ========== VALIDATION ==========
        model.eval()
        val_loss_accum = 0.0
        val_r2_metric.reset()

        with torch.no_grad():
            for batch in tqdm(
                    val_dataloader,
                    desc=f"Epoch {epoch + 1}/{config.num_epochs} [Val]  ",
                    disable=not accelerator.is_local_main_process
            ):
                origin_images = batch["origin_image"]
                target_heatmaps = batch["target_heatmap"]

                predicted_heatmaps = model(origin_images)
                loss = loss_fn(predicted_heatmaps, target_heatmaps)

                val_loss_accum += loss.detach().item()
                val_r2_metric.update(
                    predicted_heatmaps.view(-1),
                    target_heatmaps.view(-1)
                )

        # ========== METRICS ==========
        avg_train_loss = train_loss_accum / len(train_dataloader)
        avg_val_loss = val_loss_accum / len(val_dataloader) if len(val_dataloader) > 0 else 0
        epoch_train_r2 = train_r2_metric.compute().item()
        epoch_val_r2 = val_r2_metric.compute().item()

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)
        train_r2_history.append(epoch_train_r2)
        val_r2_history.append(epoch_val_r2)

        # Current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Print metrics
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch + 1}/{config.num_epochs}:")
            print(f"  Train Loss: {avg_train_loss:.6f} | Train R²: {epoch_train_r2:.4f}")
            print(f"  Val Loss:   {avg_val_loss:.6f} | Val R²:   {epoch_val_r2:.4f}")
            print(f"  LR: {current_lr:.2e}")

        # Log to wandb
        if config.use_wandb and accelerator.is_main_process:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_r2': epoch_train_r2,
                'val_r2': epoch_val_r2,
                'learning_rate': current_lr
            })

        # ========== LR SCHEDULER ==========
        if lr_scheduler is not None:
            if config.lr_scheduler_type == 'plateau':
                # Manual verbose output for plateau scheduler
                old_lr = optimizer.param_groups[0]['lr']
                lr_scheduler.step(epoch_val_r2)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr < old_lr and accelerator.is_main_process:
                    print(f"  ⚠️  Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
            else:
                lr_scheduler.step()

        # ========== CHECKPOINTING ==========
        if accelerator.is_main_process:
            # Save best model
            if epoch_val_r2 > best_val_r2:
                best_val_r2 = epoch_val_r2
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0

                print(f"  🏆 New best Val R²: {best_val_r2:.4f}! Saving model...")

                accelerator.save(
                    accelerator.unwrap_model(model).state_dict(),
                    os.path.join(config.output_dir, "model_best.pth")
                )

                # Save full checkpoint (including scheduler state for cosine)
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_r2': best_val_r2,
                    'val_loss': best_val_loss,
                }

                # Add scheduler state if using cosine
                if config.use_lr_scheduler and config.lr_scheduler_type == 'cosine':
                    checkpoint_dict['scheduler_state_dict'] = lr_scheduler.state_dict()

                torch.save(checkpoint_dict, os.path.join(config.output_dir, "checkpoint_best.pth"))
            else:
                epochs_without_improvement += 1

            # Save periodic checkpoint
            if (epoch + 1) % config.save_every_n_epochs == 0:
                accelerator.save(
                    accelerator.unwrap_model(model).state_dict(),
                    os.path.join(config.output_dir, f"model_epoch_{epoch + 1:03d}.pth")
                )

            # Save validation visualization
            if (epoch + 1) % config.visualize_every_n_epochs == 0 and len(val_dataset) > 0:
                save_validation_sample(
                    accelerator.unwrap_model(model),
                    batch,
                    epoch + 1,
                    accelerator.device,
                    config.output_dir
                )

            # Save training history periodically (every 5 epochs)
            if (epoch + 1) % 5 == 0:
                history = {
                    'train_loss': train_loss_history,
                    'val_loss': val_loss_history,
                    'train_r2': train_r2_history,
                    'val_r2': val_r2_history,
                    'best_val_r2': float(best_val_r2),
                    'best_val_loss': float(best_val_loss)
                }
                with open(os.path.join(config.output_dir, 'training_history.json'), 'w') as f:
                    json.dump(history, f, indent=2)

        # ========== EARLY STOPPING ==========
        if epochs_without_improvement >= config.early_stopping_patience:
            if accelerator.is_main_process:
                print(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
                print(f"   No improvement for {config.early_stopping_patience} epochs")
            break

    # ========== FINAL STEPS ==========
    if accelerator.is_main_process:
        # Plot training curves
        plot_metrics(
            train_loss_history,
            val_loss_history,
            train_r2_history,
            val_r2_history,
            config.output_dir
        )

        # Save final model
        accelerator.save(
            accelerator.unwrap_model(model).state_dict(),
            os.path.join(config.output_dir, "model_final.pth")
        )

        # Save training history
        history = {
            'train_loss': train_loss_history,
            'val_loss': val_loss_history,
            'train_r2': train_r2_history,
            'val_r2': val_r2_history,
            'best_val_r2': float(best_val_r2),
            'best_val_loss': float(best_val_loss)
        }
        with open(os.path.join(config.output_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        print(f"Best Validation R²: {best_val_r2:.4f}")
        print(f"Best Validation Loss: {best_val_loss:.6f}")
        print(f"Models saved to: {config.output_dir}")
        print("=" * 80)

    # Finish wandb
    if config.use_wandb and accelerator.is_main_process:
        wandb.finish()


# --- 3. Main Training Function ---
# def train():
#     # import os
#     # os.environ['TORCH_COMPILE_DISABLE'] = '1'
#
#     # --- Setup ---
#     accelerator = Accelerator(
#         mixed_precision=config.mixed_precision,
#         gradient_accumulation_steps=config.gradient_accumulation_steps
#     )
#
#     # Create output directory
#     if accelerator.is_main_process:
#         os.makedirs(config.output_dir, exist_ok=True)
#         save_config(config, config.output_dir)
#
#     # Print configuration
#     if accelerator.is_main_process:
#         print("=" * 80)
#         print("TRAINING CONFIGURATION")
#         print("=" * 80)
#         print(f"Mode: {'VALIDATION' if config.validation_mode else 'FULL TRAINING'}")
#         print(f"Model: {config.model_type}")
#         print(f"Data fraction: {config.use_data_fraction * 100:.1f}%")
#         print(f"Batch size: {config.train_batch_size}")
#         print(f"Epochs: {config.num_epochs}")
#         print(f"Learning rate: {config.learning_rate}")
#         print(f"Output directory: {config.output_dir}")
#         print("=" * 80 + "\n")
#
#     # --- FIXED: Smart Data Loading ---
#     if accelerator.is_main_process:
#         print("Loading dataset (only what we need)...")
#
#     # First, figure out how many samples we need
#     metadata_file = Path(config.chunks_dir) / "chunks_metadata.json"
#     with open(metadata_file, 'r') as f:
#         metadata = json.load(f)
#
#     total_samples = metadata['total_samples']
#     samples_needed = int(total_samples * config.use_data_fraction)
#
#     if accelerator.is_main_process:
#         print(f"Total available samples: {total_samples:,}")
#         print(f"Samples needed: {samples_needed:,} ({config.use_data_fraction * 100:.1f}%)")
#
#     # Calculate memory needed for subset
#     chunk_size = metadata['chunk_size']
#     chunks_needed = int(np.ceil(samples_needed / chunk_size))
#
#     # Estimate memory per chunk (total GB / total chunks)
#     total_gb = 733.83  # From your error message
#     total_chunks = metadata['num_chunks']
#     memory_per_chunk = total_gb / total_chunks
#     memory_needed = chunks_needed * memory_per_chunk
#
#     # Adjust memory budget for subset
#     if config.use_data_fraction < 1.0:
#         adjusted_memory_gb = min(config.max_memory_gb, memory_needed * 1.2)  # 20% buffer
#         if accelerator.is_main_process:
#             print(f"Adjusted memory budget: {adjusted_memory_gb:.1f} GB (enough for {chunks_needed} chunks)\n")
#     else:
#         adjusted_memory_gb = config.max_memory_gb
#
#     # Load dataset with adjusted memory
#     full_dataset = HybridChunkedDataset(
#         chunks_dir=config.chunks_dir,
#         max_memory_gb=adjusted_memory_gb,
#         verbose=accelerator.is_main_process
#     )
#
#     # FIXED: Subset IMMEDIATELY after loading
#     if config.use_data_fraction < 1.0:
#         # Use first N samples (data is already shuffled in chunks)
#         indices = list(range(min(samples_needed, len(full_dataset))))
#         full_dataset = Subset(full_dataset, indices)
#         if accelerator.is_main_process:
#             print(f"\n✓ Using subset: {len(full_dataset):,} samples\n")
#
#     # Split into train/val
#     val_size = int(len(full_dataset) * config.validation_split_ratio)
#     train_size = len(full_dataset) - val_size
#     train_dataset, val_dataset = random_split(
#         full_dataset,
#         [train_size, val_size],
#         generator=torch.Generator().manual_seed(42)
#     )
#
#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=config.train_batch_size,
#         shuffle=True,
#         num_workers=config.num_workers,
#         pin_memory=config.pin_memory
#     )
#     val_dataloader = DataLoader(
#         val_dataset,
#         batch_size=config.val_batch_size,
#         shuffle=False,
#         num_workers=config.num_workers,
#         pin_memory=config.pin_memory
#     )
#
#     print(f"Dataset split: {len(train_dataset):,} training, {len(val_dataset):,} validation samples")
#     print(f"Batches per epoch: {len(train_dataloader):,} train, {len(val_dataloader):,} val\n")
#
#     # --- Model Definition ---
#     print("Creating model...")
#     if config.model_type == 'resnet18':
#         model = create_unet_resnet18(
#             in_channels=config.in_channels,
#             out_channels=config.out_channels
#         )
#     elif config.model_type == 'resnet34':
#         model = create_unet_resnet34(
#             in_channels=config.in_channels,
#             out_channels=config.out_channels
#         )
#     else:
#         # Custom model
#         model = FlexibleUNet(
#             encoder=config.model_type,
#             encoder_depth=config.encoder_depth,
#             decoder_channels=config.decoder_channels,
#             in_channels=config.in_channels,
#             out_channels=config.out_channels,
#             upsampling_type=config.upsampling_type,
#             skip_type=config.skip_type,
#             center_block=config.center_block,
#             activation=config.activation
#         )
#
#     total_params, trainable_params = count_parameters(model)
#     print(f"Model: {config.model_type}")
#     print(f"Total parameters: {total_params:,}")
#     print(f"Trainable parameters: {trainable_params:,}\n")
#
#     # --- Optimizer, Loss, Scheduler ---
#     if config.optimizer_type == 'AdamW':
#         optimizer = torch.optim.AdamW(
#             model.parameters(),
#             lr=config.learning_rate,
#             weight_decay=config.weight_decay
#         )
#     elif config.optimizer_type == 'Adam':
#         optimizer = torch.optim.Adam(
#             model.parameters(),
#             lr=config.learning_rate,
#             weight_decay=config.weight_decay
#         )
#     else:
#         raise ValueError(f"Unknown optimizer: {config.optimizer_type}")
#
#     # loss_fn = nn.MSELoss()
#     loss_fn = ScaledMSELoss(scale_factor=10000.0)
#
#     # FIXED: Learning rate scheduler without verbose parameter
#     if config.use_lr_scheduler:
#         if config.lr_scheduler_type == 'plateau':
#             lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#                 optimizer,
#                 mode='max',
#                 factor=config.lr_plateau_factor,
#                 patience=config.lr_plateau_patience
#                 # REMOVED: verbose=True (not supported in older PyTorch)
#             )
#         elif config.lr_scheduler_type == 'cosine':
#             lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#                 optimizer,
#                 T_max=config.num_epochs
#             )
#         else:
#             raise ValueError(f"Unknown scheduler: {config.lr_scheduler_type}")
#     else:
#         lr_scheduler = None
#
#     # Metrics
#     train_r2_metric = R2Score().to(accelerator.device)
#     val_r2_metric = R2Score().to(accelerator.device)
#
#     # --- Prepare with Accelerator ---
#     model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
#         model, optimizer, train_dataloader, val_dataloader
#     )
#
#     if lr_scheduler is not None and config.lr_scheduler_type != 'plateau':
#         lr_scheduler = accelerator.prepare(lr_scheduler)
#
#     # --- Initialize Weights & Biases (optional) ---
#     if config.use_wandb and accelerator.is_main_process:
#         import wandb
#         wandb.init(
#             project=config.wandb_project,
#             entity=config.wandb_entity,
#             config=vars(config),
#             name=f"{config.model_type}_{'val' if config.validation_mode else 'full'}_{config.timestamp}"
#         )
#         wandb.watch(model, log='all', log_freq=100)
#
#     # --- Training Loop ---
#     best_val_r2 = -float('inf')
#     best_val_loss = float('inf')
#     train_loss_history, val_loss_history = [], []
#     train_r2_history, val_r2_history = [], []
#     epochs_without_improvement = 0
#
#     print("\n" + "=" * 80)
#     print("STARTING TRAINING")
#     print("=" * 80 + "\n")
#
#     for epoch in range(config.num_epochs):
#         # ========== TRAINING ==========
#         model.train()
#         train_loss_accum = 0.0
#         train_r2_metric.reset()
#
#         progress_bar = tqdm(
#             total=len(train_dataloader),
#             desc=f"Epoch {epoch + 1}/{config.num_epochs} [Train]",
#             disable=not accelerator.is_local_main_process
#         )
#
#         for batch_idx, batch in enumerate(train_dataloader):
#             with accelerator.accumulate(model):
#                 optimizer.zero_grad()
#
#                 origin_images = batch["origin_image"]
#                 target_heatmaps = batch["target_heatmap"]
#
#                 predicted_heatmaps = model(origin_images)
#                 loss = loss_fn(predicted_heatmaps, target_heatmaps)
#
#                 accelerator.backward(loss)
#                 optimizer.step()
#
#                 # Update metrics
#                 train_loss_accum += loss.detach().item()
#                 train_r2_metric.update(
#                     predicted_heatmaps.detach().view(-1),
#                     target_heatmaps.view(-1)
#                 )
#
#                 progress_bar.update(1)
#                 progress_bar.set_postfix(loss=f"{loss.item():.6f}")
#
#         progress_bar.close()
#
#         # ========== VALIDATION ==========
#         model.eval()
#         val_loss_accum = 0.0
#         val_r2_metric.reset()
#
#         with torch.no_grad():
#             for batch in tqdm(
#                     val_dataloader,
#                     desc=f"Epoch {epoch + 1}/{config.num_epochs} [Val]  ",
#                     disable=not accelerator.is_local_main_process
#             ):
#                 origin_images = batch["origin_image"]
#                 target_heatmaps = batch["target_heatmap"]
#
#                 predicted_heatmaps = model(origin_images)
#                 loss = loss_fn(predicted_heatmaps, target_heatmaps)
#
#                 val_loss_accum += loss.detach().item()
#                 val_r2_metric.update(
#                     predicted_heatmaps.view(-1),
#                     target_heatmaps.view(-1)
#                 )
#
#         # ========== METRICS ==========
#         avg_train_loss = train_loss_accum / len(train_dataloader)
#         avg_val_loss = val_loss_accum / len(val_dataloader) if len(val_dataloader) > 0 else 0
#         epoch_train_r2 = train_r2_metric.compute().item()
#         epoch_val_r2 = val_r2_metric.compute().item()
#
#         train_loss_history.append(avg_train_loss)
#         val_loss_history.append(avg_val_loss)
#         train_r2_history.append(epoch_train_r2)
#         val_r2_history.append(epoch_val_r2)
#
#         # Current learning rate
#         current_lr = optimizer.param_groups[0]['lr']
#
#         # Print metrics
#         if accelerator.is_main_process:
#             print(f"\nEpoch {epoch + 1}/{config.num_epochs}:")
#             print(f"  Train Loss: {avg_train_loss:.6f} | Train R²: {epoch_train_r2:.4f}")
#             print(f"  Val Loss:   {avg_val_loss:.6f} | Val R²:   {epoch_val_r2:.4f}")
#             print(f"  LR: {current_lr:.2e}")
#
#         # Log to wandb
#         if config.use_wandb and accelerator.is_main_process:
#             wandb.log({
#                 'epoch': epoch + 1,
#                 'train_loss': avg_train_loss,
#                 'val_loss': avg_val_loss,
#                 'train_r2': epoch_train_r2,
#                 'val_r2': epoch_val_r2,
#                 'learning_rate': current_lr
#             })
#
#         # ========== LR SCHEDULER ==========
#         if lr_scheduler is not None:
#             if config.lr_scheduler_type == 'plateau':
#                 # Manual verbose output for plateau scheduler
#                 old_lr = optimizer.param_groups[0]['lr']
#                 lr_scheduler.step(epoch_val_r2)
#                 new_lr = optimizer.param_groups[0]['lr']
#                 if new_lr < old_lr and accelerator.is_main_process:
#                     print(f"  ⚠️  Learning rate reduced: {old_lr:.2e} → {new_lr:.2e}")
#             else:
#                 lr_scheduler.step()
#
#         # ========== CHECKPOINTING ==========
#         if accelerator.is_main_process:
#             # Save best model
#             if epoch_val_r2 > best_val_r2:
#                 best_val_r2 = epoch_val_r2
#                 best_val_loss = avg_val_loss
#                 epochs_without_improvement = 0
#
#                 print(f"  🏆 New best Val R²: {best_val_r2:.4f}! Saving model...")
#
#                 accelerator.save(
#                     accelerator.unwrap_model(model).state_dict(),
#                     os.path.join(config.output_dir, "model_best.pth")
#                 )
#
#                 # Save optimizer state
#                 torch.save({
#                     'epoch': epoch,
#                     'model_state_dict': accelerator.unwrap_model(model).state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'val_r2': best_val_r2,
#                     'val_loss': best_val_loss,
#                 }, os.path.join(config.output_dir, "checkpoint_best.pth"))
#             else:
#                 epochs_without_improvement += 1
#
#             # Save periodic checkpoint
#             if (epoch + 1) % config.save_every_n_epochs == 0:
#                 accelerator.save(
#                     accelerator.unwrap_model(model).state_dict(),
#                     os.path.join(config.output_dir, f"model_epoch_{epoch + 1:03d}.pth")
#                 )
#
#             # Save validation visualization
#             if (epoch + 1) % config.visualize_every_n_epochs == 0 and len(val_dataset) > 0:
#                 save_validation_sample(
#                     accelerator.unwrap_model(model),
#                     batch,
#                     epoch + 1,
#                     accelerator.device,
#                     config.output_dir
#                 )
#
#         # ========== EARLY STOPPING ==========
#         if epochs_without_improvement >= config.early_stopping_patience:
#             if accelerator.is_main_process:
#                 print(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
#                 print(f"   No improvement for {config.early_stopping_patience} epochs")
#             break
#
#     # ========== FINAL STEPS ==========
#     if accelerator.is_main_process:
#         # Plot training curves
#         plot_metrics(
#             train_loss_history,
#             val_loss_history,
#             train_r2_history,
#             val_r2_history,
#             config.output_dir
#         )
#
#         # Save final model
#         accelerator.save(
#             accelerator.unwrap_model(model).state_dict(),
#             os.path.join(config.output_dir, "model_final.pth")
#         )
#
#         # Save training history
#         history = {
#             'train_loss': train_loss_history,
#             'val_loss': val_loss_history,
#             'train_r2': train_r2_history,
#             'val_r2': val_r2_history,
#             'best_val_r2': float(best_val_r2),
#             'best_val_loss': float(best_val_loss)
#         }
#         with open(os.path.join(config.output_dir, 'training_history.json'), 'w') as f:
#             json.dump(history, f, indent=2)
#
#         print("\n" + "=" * 80)
#         print("TRAINING COMPLETE!")
#         print("=" * 80)
#         print(f"Best Validation R²: {best_val_r2:.4f}")
#         print(f"Best Validation Loss: {best_val_loss:.6f}")
#         print(f"Models saved to: {config.output_dir}")
#         print("=" * 80)
#
#     # Finish wandb
#     if config.use_wandb and accelerator.is_main_process:
#         wandb.finish()


if __name__ == '__main__':
    train()
