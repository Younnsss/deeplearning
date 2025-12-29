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
    # Paramètres fixes
    img_size = 64
    mean = [0.485, 0.456, 0.406]  # Valeurs ImageNet, à adapter si besoin
    std = [0.229, 0.224, 0.225]

    # Pour gérer les images 1 canal (grayscale)
    to_3channels = transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x)

    base = transforms.Compose([
        to_3channels,
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_test = transforms.Compose([
        to_3channels,
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return {
        "train": base,
        "val": val_test,
        "test": val_test,
    }