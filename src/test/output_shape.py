"""
Script de test pour vérifier la forme de sortie du modèle.
Validation : sortie de forme (batch_size, num_classes) pour classification multi-classe.
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import build_model
from utils import count_parameters
import yaml

# Charger la config
with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

# Construire le modèle
model = build_model(config)

# Compter les paramètres
total_params = count_parameters(model)
print(f"Nombre de paramètres entraînables: {total_params:,}")

# Créer un batch factice
batch_size = 4
x = torch.randn(batch_size, 3, 64, 64)

# Forward pass
logits = model(x)

print("Output shape:", logits.shape)
