"""
Recherche de taux d'apprentissage (LR finder) — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.lr_finder --config configs/config.yaml

Exigences minimales :
- produire un log/trace permettant de visualiser (lr, loss) dans TensorBoard ou équivalent.
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np

from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import set_seed


class LRFinder:
    """Learning Rate Finder implementation."""
    
    def __init__(self, model, train_loader, criterion, device, 
                 start_lr=1e-7, end_lr=10, num_iter=100, smooth_factor=0.05):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.device = device
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_iter = num_iter
        self.smooth_factor = smooth_factor
        
        # Generate logarithmic scale of learning rates
        self.lrs = np.logspace(np.log10(start_lr), np.log10(end_lr), num_iter)
        
    def find_lr(self, optimizer, writer):
        """
        Perform the learning rate range test.
        
        Args:
            optimizer: PyTorch optimizer
            writer: TensorBoard SummaryWriter
        """
        print(f"Starting LR finder: {self.start_lr:.2e} -> {self.end_lr:.2e} over {self.num_iter} iterations (logarithmic scale)")
        
        # Store initial model state
        model_state = self.model.state_dict()
        optimizer_state = optimizer.state_dict()
        
        self.model.train()
        
        # Initialize tracking variables
        lrs = []
        losses = []
        best_loss = float('inf')
        
        # Create iterator for training data
        data_iter = iter(self.train_loader)
        
        for iteration in range(self.num_iter):
            # Set learning rate from logarithmic scale
            current_lr = self.lrs[iteration]
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # Get next batch (cycle through dataset if needed)
            try:
                data, target = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                data, target = next(data_iter)
            
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Store current values
            lrs.append(current_lr)
            loss_value = loss.item()
            
            # Smooth the loss (exponential moving average)
            if iteration == 0:
                smoothed_loss = loss_value
            else:
                smoothed_loss = self.smooth_factor * loss_value + (1 - self.smooth_factor) * smoothed_loss
            
            losses.append(smoothed_loss)
            
            # Log to TensorBoard
            writer.add_scalar('lr_finder/lr', current_lr, iteration)
            writer.add_scalar('lr_finder/loss', smoothed_loss, iteration)
            
            # Check for loss explosion (early stopping)
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss
            elif smoothed_loss > 4 * best_loss:
                print(f"Stopping early at iteration {iteration}: loss exploded")
                break
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Print progress
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.num_iter}: LR = {current_lr:.2e}, Loss = {smoothed_loss:.6f}")
        
        # Restore initial model state
        self.model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        
        print(f"LR finder completed. Results logged to TensorBoard.")
        return lrs, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Hardcoded parameters
    seed = config['train'].get('seed', 42)
    set_seed(seed)
    
    start_lr = 1e-6
    end_lr = 1
    num_iter = 200
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    print(f"Training data: {len(train_loader)} batches")
    
    # Build model
    print("Building model...")
    model = build_model(config)
    model.to(device)
    
    # Setup criterion and optimizer (we'll reset the optimizer state anyway)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=start_lr)
    
    # Setup TensorBoard writer
    runs_dir = config.get('paths', {}).get('runs_dir', 'runs')
    log_dir = os.path.join(runs_dir, 'lr_finder')
    writer = SummaryWriter(log_dir)
    print(f"Logging to: {log_dir}")
    
    # Initialize LR finder
    lr_finder = LRFinder(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        device=device,
        start_lr=start_lr,
        end_lr=end_lr,
        num_iter=num_iter
    )
    
    # Run LR finder
    lrs, losses = lr_finder.find_lr(optimizer, writer)
    
    writer.close()
    print("LR finder completed!")
    print(f"Explored learning rates from {start_lr:.2e} to {max(lrs):.2e}")
    print("Check TensorBoard logs for lr_finder/lr and lr_finder/loss curves")


if __name__ == "__main__":
    main()