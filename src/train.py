"""
Entraînement principal (à implémenter par l'étudiant·e).

Doit exposer un main() exécutable via :
    python -m src.train --config configs/config.yaml [--seed 42]

Exigences minimales :
- lire la config YAML
- respecter les chemins 'runs/' et 'artifacts/' définis dans la config
- journaliser les scalars 'train/loss' et 'val/loss' (et au moins une métrique de classification si applicable)
- supporter le flag --overfit_small (si True, sur-apprendre sur un très petit échantillon)
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import os
import random
import numpy as np
from pathlib import Path

from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import set_seed


def create_overfit_dataloader(original_loader, num_samples=32):
    """Create a small subset dataloader for overfitting with diverse labels."""
    # Get the original dataset
    dataset = original_loader.dataset

    available_indices = dataset.indices
    base_dataset = dataset.dataset
    
    # Randomly sample indices to ensure label diversity
    import random
    sampled_indices = random.sample(available_indices, min(num_samples, len(available_indices)))
    
    overfit_dataset = Subset(base_dataset, sampled_indices)
    
    # Display dataset content for verification
    print(f"\n=== OVERFIT DATASET VERIFICATION ===")
    print(f"Total samples: {len(overfit_dataset)}")
    
    # Check label distribution
    labels = []
    for i in range(len(overfit_dataset)):
        _, label = overfit_dataset[i]
        labels.append(label.item() if isinstance(label, torch.Tensor) else label)
    
    # Count unique labels
    from collections import Counter
    label_counts = Counter(labels)
    unique_labels = len(label_counts)
    
    print(f"Unique labels: {unique_labels}")
    print(f"Label distribution: {dict(sorted(label_counts.items()))}")
    print(f"Sample indices: {sorted(sampled_indices[:10])}{'...' if len(sampled_indices) > 10 else ''}")
    print(f"================================\n")
    
    return DataLoader(
        overfit_dataset,
        batch_size=original_loader.batch_size,
        shuffle=True,
        num_workers=0,  # Reduce workers for small dataset
        pin_memory=original_loader.pin_memory
    )


def save_checkpoint(model, optimizer, epoch, val_loss, filepath):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def run_training(model, train_loader, val_loader, optimizer, criterion, 
                max_epochs, device, writer, artifacts_dir, overfit_small=False):
    """
    Generic training function with all parameters specified.
    """
    model.to(device)
    best_val_loss = float('inf')
    best_checkpoint_path = os.path.join(artifacts_dir, 'best.ckpt')
    
    for epoch in range(max_epochs):
        print(f"Epoch {epoch+1}/{max_epochs}")
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            # Log training loss per iteration for overfit mode
            if overfit_small:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('train/loss', loss.item(), global_step)
        
        # Calculate epoch training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Log training metrics with exact tag names
        writer.add_scalar('train/loss', avg_train_loss, epoch)
        writer.add_scalar('train/accuracy', train_acc, epoch)
        
        # Validation phase (skip if overfitting on small set)
        if not overfit_small:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Log validation metrics with exact tag names
            writer.add_scalar('val/loss', avg_val_loss, epoch)
            writer.add_scalar('val/accuracy', val_acc, epoch)
            
            # Save best checkpoint based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(model, optimizer, epoch, avg_val_loss, best_checkpoint_path)
                print(f"New best validation loss: {avg_val_loss:.4f}")
        else:
            avg_val_loss = 0.0
            val_acc = 0.0
        
        # Print progress
        if overfit_small:
            print(f'Epoch {epoch+1}/{max_epochs}: Train Loss: {avg_train_loss:.6f}, Train Acc: {train_acc:.2f}%')
        else:
            print(f'Epoch {epoch+1}/{max_epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overfit_small", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # Get training parameters from config
    train_config = config['train']


    # Set seed
    seed = args.seed if args.seed is not None else train_config.get('seed', 42)
    set_seed(seed)
    
    if args.overfit_small:
        # Override parameters for overfitting
        learning_rate = 0.01  # High learning rate
        weight_decay = 0.0   # No weight decay
        max_epochs = 50      # Enough epochs to overfit
        num_overfit_samples = 32
        run_name = "overfit_small"
        print("OVERFIT MODE: Using hardcoded parameters for overfitting on small dataset")
    else:
        # Use config parameters
        learning_rate = train_config['optimizer'].get('lr', 0.001)
        weight_decay = train_config['optimizer'].get('weight_decay', 0.0001)
        max_epochs = args.max_epochs if args.max_epochs else train_config.get('epochs', 10)
        run_name = train_config.get('run_name', 'experiment')
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    
    # Create small dataset for overfitting if needed
    if args.overfit_small:
        train_loader = create_overfit_dataloader(train_loader, num_overfit_samples)
        print(f"Created overfit dataset with {len(train_loader.dataset)} samples")
    
    # Build model
    print("Building model...")
    model = build_model(config)
    print(f"Model built")
    
    # Setup optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Setup logging and saving
    runs_dir = config.get('paths', {}).get('runs_dir', 'runs')
    artifacts_dir = config.get('paths', {}).get('artifacts_dir', 'artifacts')
    
    # Create artifacts directory
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Setup TensorBoard writer
    log_dir = os.path.join(runs_dir, run_name)
    writer = SummaryWriter(log_dir)
    
    print(f"Logging to: {log_dir}")
    print(f"Artifacts will be saved to: {artifacts_dir}")
    print(f"Training parameters: lr={learning_rate}, weight_decay={weight_decay}, max_epochs={max_epochs}")
    print(f"Setup : Seed={seed}, Overfit_small={args.overfit_small}, run_name={run_name}, runs_dir={runs_dir}, artifacts_dir={artifacts_dir}")
    
    # Run training
    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        max_epochs=max_epochs,
        device=device,
        writer=writer,
        artifacts_dir=artifacts_dir,
        overfit_small=args.overfit_small
    )
    
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()
    