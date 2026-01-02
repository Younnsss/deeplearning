"""
Utils génériques.

Fonctions attendues (signatures imposées) :
- set_seed(seed: int) -> None
- get_device(prefer: str | None = "auto") -> str
- count_parameters(model) -> int
- save_config_snapshot(config: dict, out_dir: str) -> None
"""

import torch
import numpy as np
import random
import os
import logging

def set_seed(seed: int) -> None:
    """Initialise les seeds (numpy/torch/python) pour la reproductibilité."""
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed
    torch.manual_seed(seed)
    
    # Set CUDA random seed (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior on GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Log the seed
    logging.info(f"Seed initialisé à {seed} pour reproductibilité")
    print(f"🌱 Seed initialisé: {seed}")


def get_device(prefer: str | None = "auto") -> str:
    """Retourne 'cpu' ou 'cuda' (ou choix basé sur 'auto'). À implémenter."""
    raise NotImplementedError("get_device doit être implémentée par l'étudiant·e.")


def count_parameters(model) -> int:
    """Retourne le nombre de paramètres entraînables du modèle."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config_snapshot(config: dict, out_dir: str) -> None:
    """Sauvegarde une copie de la config (ex: YAML) dans out_dir. À implémenter."""
    raise NotImplementedError("save_config_snapshot doit être implémentée par l'étudiant·e.")