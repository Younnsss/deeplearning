"""
Chargement des données.

Signature imposée :
get_dataloaders(config: dict) -> (train_loader, val_loader, test_loader, meta: dict)

Le dictionnaire meta doit contenir au minimum :
- "num_classes": int
- "input_shape": tuple (ex: (3, 32, 32) pour des images)
"""

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from datasets import load_dataset
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms

from src.preporcessing import get_preprocess_transforms
from src.augmentation import get_augmentation_transforms


class HuggingFaceDataset(Dataset):
    """Wrapper pour convertir un dataset HuggingFace en Dataset PyTorch."""
    
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        image = item['image']
        label = item['label']
        
        # Appliquer les transformations si définies
        if self.transform:
            image = self.transform(image)
        else:
            # Fallback: convertir PIL Image en tensor (code existant)
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Convertir en tensor PyTorch (HWC -> CHW)
            if len(image.shape) == 3:
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            else:
                image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        
        return image, torch.tensor(label, dtype=torch.long)


def get_dataloaders(config: dict):
    """
    Crée et retourne les DataLoaders d'entraînement/validation/test et des métadonnées.
    """
    dataset_config = config['dataset']
    
    # Charger le dataset depuis HuggingFace
    cache_dir = dataset_config.get('root', './data')
    ds = load_dataset("zh-plus/tiny-imagenet", cache_dir=cache_dir)
    
    # Obtenir les transformations de preprocessing et d'augmentation
    preprocess_transforms = get_preprocess_transforms(config)
    augmentation_transforms = get_augmentation_transforms(config)
    
    # Composer les transformations pour l'entraînement (augmentation + preprocessing)
    train_transform = transforms.Compose([
        augmentation_transforms,
        preprocess_transforms["train"]
    ])
    
    # Créer un dataset sans transformation pour faire le split
    raw_train_dataset = HuggingFaceDataset(ds['train'], transform=None)
    
    # Créer un split test stratifié à partir du train
    # Seed fixe pour la reproductibilité (documentée)
    SPLIT_SEED = 42  # Seed fixe pour garantir la reproductibilité des splits train/test
    
    # Extraire les labels pour la stratification
    train_labels = [item['label'] for item in ds['train']]
    
    # Split stratifié train/test (90%/10%)
    train_indices, test_indices = train_test_split(
        range(len(raw_train_dataset)),
        test_size=0.1,
        random_state=SPLIT_SEED,
        stratify=train_labels
    )
    
    # Créer les datasets avec les transformations appropriées APRÈS le split
    train_dataset_with_transforms = HuggingFaceDataset(ds['train'], transform=train_transform)
    test_dataset_with_transforms = HuggingFaceDataset(ds['train'], transform=preprocess_transforms["test"])
    val_dataset = HuggingFaceDataset(ds['valid'], transform=preprocess_transforms["val"])
    
    # Créer les sous-datasets
    train_dataset = Subset(train_dataset_with_transforms, train_indices)
    test_dataset = Subset(test_dataset_with_transforms, test_indices)
    
    # Paramètres des DataLoaders
    batch_size = 32  # Valeur par défaut, sera surchargée par train.batch_size plus tard
    num_workers = dataset_config.get('num_workers', 4)
    shuffle = dataset_config.get('shuffle', True)
    
    # Créer les DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Extraire les métadonnées
    # Tiny ImageNet a 200 classes et des images 64x64x3
    sample_image, sample_label = val_dataset[0]  # Utiliser val_dataset pour éviter l'augmentation
    input_shape = tuple(sample_image.shape)  # (3, 64, 64)
    
    # Déterminer le nombre de classes
    all_labels = set()
    for item in ds['train']:
        all_labels.add(item['label'])
    for item in ds['valid']:
        all_labels.add(item['label'])
    
    num_classes = len(all_labels)
    
    meta = {
        "num_classes": num_classes,
        "input_shape": input_shape
    }
    
    return train_loader, val_loader, test_loader, meta