"""
Test du modèle avec un premier batch d'entraînement.
Calcule la loss et vérifie les gradients après rétropropagation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import build_model
from utils import count_parameters
from data_loading import get_dataloaders
import yaml

def test_premier_batch(model, train_dataloader, num_classes):
    """
    Teste le modèle avec un premier batch d'entraînement.
    
    Args:
        model: Le modèle à tester
        train_dataloader: DataLoader d'entraînement
        num_classes: Nombre de classes pour le calcul de la loss attendue
    """
    # Mettre le modèle en mode entraînement
    model.train()
    
    # Définir la loss function pour multi-classe
    criterion = nn.CrossEntropyLoss()
    
    # Charger un batch d'entraînement
    batch = next(iter(train_dataloader))
    inputs, targets = batch
    
    print(f"Batch shape: {inputs.shape}")
    print(f"Targets shape: {targets.shape}")
    
    # Forward pass
    model.zero_grad()
    logits = model(inputs)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Logits moyens: {logits.mean().item():.4f}")
    print(f"Logits std: {logits.std().item():.4f}")
    
    # Calculer la loss
    loss = criterion(logits, targets)
    print(f"Loss calculée: {loss.item():.4f}")
    
    # Loss attendue pour modèle non-entraîné (uniforme)
    expected_loss = -math.log(1.0 / num_classes)
    print(f"Loss attendue (modèle uniforme): {expected_loss:.4f}")
    
    # Rétropropagation
    loss.backward()
    
    # Vérifier les normes de gradients
    total_grad_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_grad_norm += param_norm.item() ** 2
    
    total_grad_norm = total_grad_norm ** (1. / 2)
    
    print(f"Somme des normes de gradients: {total_grad_norm:.4f}")
    
    if total_grad_norm > 0:
        print("✓ Les gradients sont non-nuls, la rétropropagation fonctionne.")
    else:
        print("⚠ Les gradients sont nuls, vérifiez votre modèle.")
    
    return {
        'loss': loss.item(),
        'expected_loss': expected_loss,
        'grad_norm': total_grad_norm,
        'logits_mean': logits.mean().item(),
        'logits_std': logits.std().item()
    }

if __name__ == "__main__":
    # Charger la config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Construire le modèle
    print("Construction du modèle")
    model = build_model(config)

    # Récupérer les dataloaders
    print("Récupération des dataloaders")
    train_dataloader, val_dataloader, test_dataloader, meta = get_dataloaders(config)

    num_classes = meta["num_classes"]
    results = test_premier_batch(model, train_dataloader, num_classes)
    print("Résultats du test :", results)
