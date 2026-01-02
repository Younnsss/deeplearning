"""
Script pour visualiser les données après prétraitement et augmentation.
Sauvegarde des exemples en PNG dans artifacts/ pour le rapport.

Usage:
    python -m src.visualize_data --config configs/config.yaml
"""

import argparse
import os
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
from data_loading import get_dataloaders

def denormalize_tensor(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Dénormalise un tensor pour l'affichage."""
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean

def tensor_to_image(tensor):
    """Convertit un tensor (C, H, W) en image numpy (H, W, C) pour matplotlib."""
    # Clamp pour s'assurer que les valeurs sont dans [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    # Permuter les dimensions et convertir en numpy
    return tensor.permute(1, 2, 0).numpy()

def visualize_samples(config):
    """Visualise et sauvegarde des échantillons après preprocessing/augmentation."""
    
    # Charger les dataloaders
    print("Chargement des dataloaders...")
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)
    
    print(f"Métadonnées du dataset:")
    print(f"  - Nombre de classes: {meta['num_classes']}")
    print(f"  - Forme d'entrée: {meta['input_shape']}")
    print(f"  - Taille des splits: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    
    # Récupérer quelques échantillons de train 
    train_batch = next(iter(train_loader))
    train_images, train_labels = train_batch
    
    # Récupérer quelques échantillons de val
    val_batch = next(iter(val_loader))
    val_images, val_labels = val_batch

    # Récupérer quelques échantillons de test
    test_batch = next(iter(test_loader))
    test_images, test_labels = test_batch

    print(f"\nFormes des tensors:")
    print(f"  - Train batch: {train_images.shape}, dtype: {train_images.dtype}")
    print(f"  - Val batch: {val_images.shape}, dtype: {val_images.dtype}")
    print(f"  - Test batch: {test_images.shape}, dtype: {test_images.dtype}")
    print(f"  - Plage de valeurs train: [{train_images.min():.3f}, {train_images.max():.3f}]")
    print(f"  - Plage de valeurs val: [{val_images.min():.3f}, {val_images.max():.3f}]")
    print(f"  - Plage de valeurs test: [{test_images.min():.3f}, {test_images.max():.3f}]")
    
    # Afficher la distribution des classes dans chaque splits
    def get_class_distribution(dataloader):
        """Calcule la distribution des classes dans un dataloader."""
        class_counts = {}
        for _, labels in dataloader:
            for label in labels:
                label_item = label.item()
                class_counts[label_item] = class_counts.get(label_item, 0) + 1
        return class_counts
    
    print(f"\nDistribution des classes:")
    train_dist = get_class_distribution(train_loader)
    val_dist = get_class_distribution(val_loader)
    test_dist = get_class_distribution(test_loader)
    
    print(f"  - Train: {dict(sorted(train_dist.items()))}")
    print(f"  - Val: {dict(sorted(val_dist.items()))}")
    print(f"  - Test: {dict(sorted(test_dist.items()))}")
    
    # Créer un graphique de distribution des classes
    
    classes = sorted(set(list(train_dist.keys()) + list(val_dist.keys()) + list(test_dist.keys())))
    train_counts = [train_dist.get(c, 0) for c in classes]
    val_counts = [val_dist.get(c, 0) for c in classes]
    test_counts = [test_dist.get(c, 0) for c in classes]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(classes))
    width = 0.25
    
    ax.bar(x - width, train_counts, width, label='Train', alpha=0.8)
    ax.bar(x, val_counts, width, label='Validation', alpha=0.8)
    ax.bar(x + width, test_counts, width, label='Test', alpha=0.8)
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Nombre d\'échantillons')
    ax.set_title('Distribution des classes par split')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("artifacts/class_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nGraphique de distribution sauvegardé: artifacts/class_distribution.png")

def main():
    parser = argparse.ArgumentParser(description="Visualiser les données après preprocessing/augmentation")
    parser.add_argument("--config", type=str, required=True, help="Chemin vers le fichier de configuration")
    
    args = parser.parse_args()
    
    # Charger la configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Visualiser les échantillons
    visualize_samples(config)

if __name__ == "__main__":
    main()
