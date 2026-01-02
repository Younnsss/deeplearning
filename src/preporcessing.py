"""
Pré-traitements.

Signature imposée :
get_preprocess_transforms(config: dict) -> objet/transform callable
"""

from torchvision import transforms

def get_preprocess_transforms(config: dict):
    """
    Retourne les transformations de pré-traitement.
    Applique : redimensionnement, recadrage central (val/test), conversion en tenseur, normalisation.
    Gère les images 1 canal en les convertissant en 3 canaux.
    """
    # Récupération des paramètres depuis config
    preprocess_config = config.get("preprocess", {})
    
    img_size = preprocess_config.get("img_size", 64)
    resize_dims = preprocess_config.get("resize", [64, 64])
    normalize_params = preprocess_config.get("normalize", {})
    mean = normalize_params.get("mean", [0.485, 0.456, 0.406])
    std = normalize_params.get("std", [0.229, 0.224, 0.225])

    # Conversion en RGB (toujours appliquée)
    to_3channels = transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x)

    # Transformations de base pour train
    base = transforms.Compose([
        to_3channels,
        transforms.Resize(tuple(resize_dims)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Pour validation/test : ajout du CenterCrop
    val_test = transforms.Compose([
        to_3channels,
        transforms.Resize(tuple(resize_dims)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return {
        "train": base,
        "val": val_test,
        "test": val_test,
    }