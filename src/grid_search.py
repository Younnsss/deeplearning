"""
Mini grid search — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.grid_search --config configs/config.yaml

Exigences minimales :
- lire la section 'hparams' de la config
- lancer plusieurs runs en variant les hyperparamètres
- journaliser les hparams et résultats de chaque run (ex: TensorBoard HParams ou équivalent)
"""

import argparse
import yaml
import itertools
import os
from pathlib import Path
import copy
from torch.utils.tensorboard import SummaryWriter
from src.train import main as train_main
from src.utils import set_seed

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_hyperparameter_combinations():
    """Generate all combinations of hyperparameters for grid search."""
    # Mini-grille LR "×0.5 ×1 ×2" around base values
    learning_rates = [2.5e-4, 5e-4, 1e-3]
    weight_decays = [1e-5, 1e-4]
    blocks_configs = [(2, 2, 2), (3, 3, 3)]
    dilation_configs = [(2, 2), (2, 3)]  # D2, D3 (D1 always 1)
    
    combinations = []
    for lr, wd, blocks, dilations in itertools.product(
        learning_rates, weight_decays, blocks_configs, dilation_configs
    ):
        combinations.append({
            'lr': lr,
            'weight_decay': wd,
            'blocks_per_stage': list(blocks),
            'dilations': [1] + list(dilations)  # D1 always 1
        })
    
    return combinations

def create_run_name(lr, weight_decay, blocks, dilations):
    """Create structured run name."""
    blocks_str = '-'.join(map(str, blocks))
    dil_str = '-'.join(map(str, dilations[1:]))  # Skip D1 (always 1)
    return f"proj11_lr={lr:.0e}_wd={weight_decay:.0e}_blocks={blocks_str}_dil={dil_str}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    # Load base configuration
    base_config = load_config(args.config)
    
    # Set seed for reproducibility
    seed = base_config['train'].get('seed', 42)
    set_seed(seed)
    
    # Generate hyperparameter combinations
    combinations = generate_hyperparameter_combinations()
    
    print(f"Starting grid search with {len(combinations)} combinations...")
    
    # Create runs directory
    runs_dir = Path(base_config['paths']['runs_dir'])
    runs_dir.mkdir(exist_ok=True)
    
    # Results storage
    results = []
    
    for i, hparams in enumerate(combinations):
        print(f"\nRun {i+1}/{len(combinations)}")
        print(f"Hyperparameters: {hparams}")
        
        # Create modified config for this run
        config = copy.deepcopy(base_config)
        
        # Update hyperparameters
        config['train']['optimizer']['lr'] = hparams['lr']
        config['train']['optimizer']['weight_decay'] = hparams['weight_decay']
        config['model']['blocks_per_stage'] = hparams['blocks_per_stage']
        config['model']['dilations'] = hparams['dilations']
        
        # Set short training for grid search (3 epochs)
        config['train']['epochs'] = 3
        
        # Create run name and log directory
        run_name = create_run_name(
            hparams['lr'], 
            hparams['weight_decay'], 
            hparams['blocks_per_stage'], 
            hparams['dilations']
        )
        log_dir = runs_dir / run_name
        
        # Save modified config for this run
        config_path = log_dir / "config.yaml"
        log_dir.mkdir(exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Create TensorBoard logger
        writer = SummaryWriter(log_dir=str(log_dir))
        
        # Log hyperparameters
        writer.add_hparams(
            hparam_dict={
                'lr': hparams['lr'],
                'weight_decay': hparams['weight_decay'],
                'blocks_B1': hparams['blocks_per_stage'][0],
                'blocks_B2': hparams['blocks_per_stage'][1],
                'blocks_B3': hparams['blocks_per_stage'][2],
                'dilation_D2': hparams['dilations'][1],
                'dilation_D3': hparams['dilations'][2],
            },
            metric_dict={}  # Will be filled after training
        )
        
        try:
            # Run training with modified config
            # Note: You may need to modify train_main to accept config dict
            # For now, we'll save the config and pass the path
            final_metrics = train_with_config(str(config_path), writer)
            
            # Log final metrics
            writer.add_hparams(
                hparam_dict={
                    'lr': hparams['lr'],
                    'weight_decay': hparams['weight_decay'],
                    'blocks_B1': hparams['blocks_per_stage'][0],
                    'blocks_B2': hparams['blocks_per_stage'][1],
                    'blocks_B3': hparams['blocks_per_stage'][2],
                    'dilation_D2': hparams['dilations'][1],
                    'dilation_D3': hparams['dilations'][2],
                },
                metric_dict=final_metrics
            )
            
            # Store results
            result = {
                'run_name': run_name,
                'hparams': hparams,
                'metrics': final_metrics
            }
            results.append(result)
            
            print(f"Run completed. Final metrics: {final_metrics}")
            
        except Exception as e:
            print(f"Run failed with error: {e}")
            
        finally:
            writer.close()
    
    # Print summary of all runs
    print(f"\n\nGrid Search Summary:")
    print(f"Total runs: {len(results)}")
    
    if results:
        # Sort by validation accuracy (assuming it exists)
        if 'val_accuracy' in results[0]['metrics']:
            results.sort(key=lambda x: x['metrics']['val_accuracy'], reverse=True)
            print("\nTop 5 configurations by validation accuracy:")
            for i, result in enumerate(results[:5]):
                print(f"{i+1}. {result['run_name']}: "
                      f"val_acc={result['metrics']['val_accuracy']:.4f}")

def train_with_config(config_path, writer):
    """
    Run training with given config and return final metrics.
    Integrates with the existing training infrastructure.
    """
    import yaml
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from src.data_loading import get_dataloaders
    from src.model import build_model
    
    # Load the config for this specific run
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    
    # Build model
    model = build_model(config)
    model.to(device)
    
    # Setup optimizer and criterion
    train_config = config['train']
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=train_config['optimizer']['lr'],
        weight_decay=train_config['optimizer']['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()
    
    # Run training (adapted from your run_training function)
    max_epochs = train_config['epochs']
    final_metrics = {}
    
    for epoch in range(max_epochs):
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
        
        # Validation phase
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
        
        # Calculate epoch metrics
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Log metrics to TensorBoard
        writer.add_scalar('train/loss', avg_train_loss, epoch)
        writer.add_scalar('train/accuracy', train_acc, epoch)
        writer.add_scalar('val/loss', avg_val_loss, epoch)
        writer.add_scalar('val/accuracy', val_acc, epoch)
        
        # Update final metrics (will be returned)
        final_metrics = {
            'train_loss': avg_train_loss,
            'train_accuracy': train_acc,
            'val_loss': avg_val_loss,
            'val_accuracy': val_acc
        }
        
        print(f'Epoch {epoch+1}/{max_epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    return final_metrics

if __name__ == "__main__":
    main()