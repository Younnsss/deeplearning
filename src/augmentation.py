"""
Data augmentation

Signature imposée :
get_augmentation_transforms(config: dict) -> objet/transform callable (ou None)
"""

from torchvision import transforms

def get_augmentation_transforms(config: dict):
    """
    Retourne les transformations d'augmentation pour l'entraînement.
    - Flip horizontal aléatoire (p=0.5) : invariance gauche-droite.
    - Rotation aléatoire (-15° à 15°, p=0.5) : robustesse à l'orientation.
    - Recadrage aléatoire puis redimensionnement (RandomResizedCrop).
    - ColorJitter léger : robustesse aux variations d'éclairage.
    - RandomErasing (p=0.2) : robustesse à l'occlusion locale.
    Toutes ces transformations préservent le label.
    """
    img_size = config.get("img_size", 64)

    to_3channels = transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x)

    aug = transforms.Compose([
        to_3channels,
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    ])
    return aug