"""
Évaluation — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt

Exigences minimales :
- charger le modèle et le checkpoint
- calculer et afficher/consigner les métriques de test
"""

import argparse
import yaml
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import numpy as np

from src.data_loading import get_dataloaders
from src.model import build_model

def evaluate_model(model, test_loader, device):
    """
    Évaluer le modèle sur les données de test et renvoyer les métriques.
    """
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # Obtenir les prédictions
            _, predicted = output.max(1)
            
            # Stocker les prédictions et les cibles pour le calcul des métriques
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Calculer les métriques
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_targets, all_predictions)
    f1_macro = f1_score(all_targets, all_predictions, average='macro')
    precision_macro = precision_score(all_targets, all_predictions, average='macro')
    recall_macro = recall_score(all_targets, all_predictions, average='macro')
    
    return {
        'test_loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'predictions': all_predictions,
        'targets': all_targets
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    
    # Charger la configuration
    print(f"Chargement de la configuration depuis {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Configuration de l'appareil
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation de l'appareil : {device}")
    
    # Charger les données (seulement le chargeur de test est nécessaire)
    print("Chargement des données de test...")
    _, _, test_loader, meta = get_dataloaders(config)
    print(f"Taille du jeu de test : {len(test_loader.dataset)} échantillons")
    print(f"Nombre de classes : {meta['num_classes']}")
    print(f"Forme de l'entrée : {meta['input_shape']}")
    
    # Construire le modèle
    print("Construction du modèle...")
    model = build_model(config)
    model.to(device)
    
    # Charger le point de contrôle
    print(f"Chargement du point de contrôle depuis {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Point de contrôle chargé depuis l'époque {checkpoint['epoch']} avec une perte de validation : {checkpoint['val_loss']:.4f}")
    
    # Évaluer le modèle
    print("Évaluation du modèle sur le jeu de test...")
    results = evaluate_model(model, test_loader, device)
    
    # Afficher les résultats
    print("\n" + "="*50)
    print("RésULTATS DE L'ÉVALUATION DU JEU DE TEST")
    print("="*50)
    print(f"Perte de test :      {results['test_loss']:.4f}")
    print(f"Précision :       {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"F1 Macro :       {results['f1_macro']:.4f}")
    print(f"Précision Macro : {results['precision_macro']:.4f}")
    print(f"Rappel Macro :   {results['recall_macro']:.4f}")
    print("="*50)
    
    # Générer le rapport de classification
    print("\nRapport de classification détaillé :")
    print("-"*50)
    report = classification_report(results['targets'], results['predictions'])
    print(report)

if __name__ == "__main__":
    main()