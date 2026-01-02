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
    # Récupération des paramètres depuis config
    preprocess_config = config.get("preprocess", {})
    augment_config = config.get("augment", {})
    
    img_size = preprocess_config.get("img_size", 64)
    
    # Paramètres d'augmentation
    flip_p = augment_config.get("random_horizontal_flip", {}).get("p", 0.5)
    rotation_degrees = augment_config.get("random_rotation", {}).get("degrees", 15)
    
    crop_params = augment_config.get("random_resized_crop", {})
    crop_scale = crop_params.get("scale", [0.8, 1.0])
    crop_ratio = crop_params.get("ratio", [0.9, 1.1])
    
    jitter_params = augment_config.get("color_jitter", {})
    brightness = jitter_params.get("brightness", 0.1)
    contrast = jitter_params.get("contrast", 0.1)
    saturation = jitter_params.get("saturation", 0.1)
    hue = jitter_params.get("hue", 0.05)

    to_3channels = transforms.Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x)

    aug = transforms.Compose([
        to_3channels,
        transforms.RandomResizedCrop(img_size, scale=tuple(crop_scale), ratio=tuple(crop_ratio)),
        transforms.RandomHorizontalFlip(p=flip_p),
        transforms.RandomRotation(degrees=rotation_degrees),
        transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
    ])
    return aug