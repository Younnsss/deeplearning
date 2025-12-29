from datasets import load_dataset
from collections import Counter, defaultdict
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def main():
    ds = load_dataset("zh-plus/tiny-imagenet", cache_dir="./data")
    splits = ['valid']
    class_counts = Counter()
    image_shapes = defaultdict(int)
    total_images = 0

    for split in splits:
        for item in ds[split]:
            label = item['label']
            class_counts[label] += 1
            img = item['image']
            if isinstance(img, Image.Image):
                arr = np.array(img)
            else:
                arr = img
            image_shapes[arr.shape] += 1
            total_images += 1

    print("Distribution des classes :")
    for label, count in sorted(class_counts.items()):
        print(f"Classe {label}: {count} images")
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    if max_count > 1.5 * min_count:
        print("⚠️  Distribution très déséquilibrée !")

    # Graphe de la diversité des classes
    plt.figure(figsize=(12, 5))
    labels = list(class_counts.keys())
    counts = [class_counts[label] for label in labels]
    plt.bar(labels, counts)
    plt.xlabel("Classe")
    plt.ylabel("Nombre d'images")
    plt.title("Distribution des classes dans Tiny ImageNet")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    print("\nTailles d’images rencontrées :")
    for shape, count in image_shapes.items():
        print(f"{shape}: {count} images")
    if len(image_shapes) > 1:
        print("⚠️  Les tailles d’images ne sont pas uniformes !")

if __name__ == "__main__":
    main()
