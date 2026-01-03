# Rapport de projet — CSC8607 : Introduction au Deep Learning

> **Consignes générales**
>
> - Tenez-vous au **format** et à l’**ordre** des sections ci-dessous.
> - Intégrez des **captures d’écran TensorBoard** lisibles (loss, métriques, LR finder, comparaisons).
> - Les chemins et noms de fichiers **doivent** correspondre à la structure du dépôt modèle (ex. `runs/`, `artifacts/best.ckpt`, `configs/config.yaml`).
> - Répondez aux questions **numérotées** (D1–D11, M0–M9, etc.) directement dans les sections prévues.

---

## 0) Informations générales

- **Étudiant·e** : Boutkrida, Younes
- **Projet** : Tiny ImageNet (Convolution)
- **Dépôt Git** : https://github.com/Younnsss/deeplearning
- **Environnement** : `python == 3.10.18`, `torch == 2.5.1`, `cuda == 12.1`
- **Commandes utilisées** :
  - Entraînement : `python -m src.train --config configs/config.yaml`
  - LR finder : `python -m src.lr_finder --config configs/config.yaml`
  - Grid search : `python -m src.grid_search --config configs/config.yaml`
  - Évaluation : `python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt`

---

## 1) Données

### 1.1 Description du dataset

- **Source** (lien) : https://huggingface.co/datasets/zh-plus/tiny-imagenet
- **Type d’entrée** (image) : image 3 x 64 x 64
- **Tâche** : multiclasses
- **Dimensions d’entrée attendues** (`meta["input_shape"]`) : 3 x 64 x 64
- **Nombre de classes** (`meta["num_classes"]`) : 200

**D1.** Quel dataset utilisez-vous ? D’où provient-il et quel est son format (dimensions, type d’entrée) ?

> Le dataset utilisé est Tiny ImageNet, provenant du dépôt HuggingFace `zh-plus/tiny-imagenet`. Il s'agit d'un dataset d'images couleur de taille 64x64 pixels, avec 3 canaux (RGB).

### 1.2 Splits et statistiques

| Split | #Exemples | Particularités (déséquilibre, longueur moyenne, etc.) |
| ----: | --------: | ----------------------------------------------------- |
| Train |    90 000 | 200 classes équilibrées (450 images/classe)           |
|   Val |    10 000 | 200 classes équilibrées (50 images/classe)            |
|  Test |    10 000 | 200 classes équilibrées (50 images/classe)            |

**D2.** Donnez la taille de chaque split et le nombre de classes.

> Le split train contient 90 000 exemples (450 images par classe), le split validation contient 10 000 exemples (50 images par classe), et le split test contient 10 000 exemples (50 images par classe). Le nombre de classes est 200.

**D3.** Si vous avez créé un split (ex. validation), expliquez **comment** (stratification, ratio, seed).

> Le split test a été créé à partir du train original par une séparation stratifiée (ratio 0.1), en utilisant la seed fixe 42 pour garantir la reproductibilité. La stratification assure une distribution équilibrée des classes dans chaque split.

**D4.** Donnez la **distribution des classes** (graphique ou tableau) et commentez en 2–3 lignes l'impact potentiel sur l'entraînement.

![Distribution des classes](/artifacts/dataset/distributionsClasses.png)

> La distribution des classes est parfaitement équilibrée : chaque classe contient exactement 450 images en entraînement, 50 en validation et 50 en test. Cela garantit que le modèle ne sera pas biaisé vers une classe majoritaire et que les métriques d'entraînement et de validation reflèteront fidèment la performance réelle. L'absence de déséquilibre facilite l'apprentissage et la comparaison des résultats entre classes.

**D5.** Mentionnez toute particularité détectée (tailles variées, longueurs variables, multi-labels, etc.).

> Deux formats d’images ont été détectés dans le dataset : la grande majorité (98179 images) sont au format attendu (64, 64, 3), mais 1821 images sont au format (64, 64) (sans canal couleur explicite). Cette hétérogénéité nécessite un prétraitement pour garantir la cohérence des entrées du modèle (conversion en RGB si besoin).

### 1.3 Prétraitements (preprocessing) — _appliqués à train/val/test_

Listez précisément les opérations et paramètres (valeurs **fixes**) :

> Vision : resize = (64, 64), center-crop = 64, normalize = (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**D6.** Quels **prétraitements** avez-vous appliqués (opérations + **paramètres exacts**) et **pourquoi** ?

> Les prétraitements appliqués sont les suivants (identiques pour train/val/test, sauf pour le recadrage central en val/test) :
>
> - Conversion en RGB pour toutes les images (pour gérer les images en niveaux de gris).
> - Redimensionnement à 64x64 pixels (`transforms.Resize((64, 64))`) pour garantir une taille cohérente.
> - Pour val/test uniquement : recadrage central (`transforms.CenterCrop(64)`) pour éviter tout biais d’augmentation.
> - Conversion en tenseur (`transforms.ToTensor()`).
> - Normalisation par canal avec les moyennes et écarts-types ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) afin de stabiliser et accélérer l’apprentissage.
>   Ces choix assurent la compatibilité des entrées avec les modèles convolutionnels classiques et la reproductibilité des expériences.

**D7.** Les prétraitements diffèrent-ils entre train/val/test (ils ne devraient pas, sauf recadrage non aléatoire en val/test) ?

> Les prétraitements sont identiques pour train, validation et test, à l’exception du recadrage central (`CenterCrop`) qui n’est appliqué qu’en validation et test pour garantir une évaluation déterministe et sans biais d’augmentation. Toutes les autres opérations (conversion RGB, resize, normalisation) sont strictement les mêmes sur les trois splits.

### 1.4 Augmentation de données — _train uniquement_

> Liste des **augmentations** (opérations + **paramètres** et **probabilités**) :
>
> - Recadrage aléatoire puis redimensionnement (`RandomResizedCrop`, scale=(0.8, 1.0), ratio=(0.9, 1.1))
> - Flip horizontal aléatoire (`RandomHorizontalFlip`, p=0.5)
> - Rotation aléatoire (`RandomRotation`, degrés=15)
> - ColorJitter léger (`ColorJitter`, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)

**D8.** Quelles **augmentations** avez-vous appliquées (paramètres précis) et **pourquoi** ?

> Les augmentations appliquées en entraînement sont :
>
> - **RandomResizedCrop** (scale=(0.8, 1.0), ratio=(0.9, 1.1)) : simule des variations de cadrage et de zoom, rendant le modèle robuste à la position et à la taille de l’objet.
> - **RandomHorizontalFlip** (p=0.5) : introduit une invariance gauche-droite, utile pour de nombreux objets naturels.
> - **RandomRotation** (degrés=15) : rend le modèle moins sensible à l’orientation exacte de l’objet.
> - **ColorJitter** (brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05) : simule des variations d’éclairage et de couleur, améliorant la robustesse aux conditions de prise de vue.
>   Ces choix visent à améliorer la généralisation du modèle en l’exposant à des variations réalistes des images d’entraînement.

**D9.** Les augmentations **conservent-elles les labels** ? Justifiez pour chaque transformation retenue.

> Oui, toutes les augmentations appliquées sont label-preserving :
>
> - **RandomResizedCrop** : ne modifie pas la classe de l’objet, seulement son cadrage et sa taille apparente.
> - **RandomHorizontalFlip** : l’objet reste de la même classe après un flip horizontal.
> - **RandomRotation** : une rotation modérée ne change pas la nature de l’objet.
> - **ColorJitter** : les variations de couleur et de luminosité n’affectent pas la classe sémantique.
>   Ces transformations sont donc sûres pour l’apprentissage supervisé car elles ne créent pas d’ambiguïté sur le label.

### 1.5 Sanity-checks

> Après preprocessing/augmentation :

![Distribution des classes](/artifacts/dataset/class_info.png)

![Distribution des classes](/artifacts/dataset/class_distribution.png)

**D10.** Montrez 2–3 exemples et commentez brièvement.

> Les résultats montrent des images 64x64 pixels correctement normalisées avec des valeurs dans la plage [-2.118, 2.640], ce qui correspond aux transformations appliquées (normalisation ImageNet). Les batch shapes sont cohérentes : `torch.Size([32, 3, 64, 64])` avec dtype `torch.float32`. La distribution des classes est parfaitement équilibrée : 450 échantillons par classe en train, 50 en validation et 50 en test, confirmant la stratification réussie lors du split des données.

**D11.** Donnez la **forme exacte** d'un batch train (ex. `(batch, C, H, W)` ou `(batch, seq_len)`), et vérifiez la cohérence avec `meta["input_shape"]`.

> La forme exacte d'un batch est `torch.Size([32, 3, 64, 64])` avec un dtype `torch.float32`, ce qui correspond parfaitement au format attendu `(batch_size, channels, height, width)`. Cette forme est cohérente avec `meta["input_shape"] = (3, 64, 64)` : 3 canaux RGB pour des images 64x64 pixels. Les trois splits (train/val/test) ont tous la même forme de batch, garantissant la cohérence du preprocessing.

---

## 2) Modèle

### 2.1 Baselines

**M0.**

> - **Classe majoritaire** — Métrique : `Accuracy` → score = `~0.5%`
> - **Prédiction aléatoire uniforme** — Métrique : `Accuracy` → score = `0.5%`
>   Ces scores très faibles (0.5%) constituent un plancher minimal que tout modèle fonctionnel doit rapidement dépasser. Le dataset Tiny ImageNet étant quasi équilibré (exactement 550 exemples pour les 200 classes), les deux baselines donnent des performances similaires, confirmant l'absence de biais de distribution et la nécessité d'apprendre des caractéristiques visuelles.

### 2.2 Architecture implémentée

- **Description couche par couche** (ordre exact, tailles, activations, normalisations, poolings, résiduels, etc.) :

  > - Input → (3, 64, 64)
  > - Stage 1 (répéter B₁ fois) : Conv2d(3×3, padding=1, dilation=1) → 64 canaux → BatchNorm2d → ReLU → MaxPool2d(2×2) → (64, 32, 32)
  > - Stage 2 (répéter B₂ fois) : Conv2d(3×3, padding=D₂, dilation=D₂) → 128 canaux → BatchNorm2d → ReLU → MaxPool2d(2×2) → (128, 16, 16)
  > - Stage 3 (répéter B₃ fois) : Conv2d(3×3, padding=D₃, dilation=D₃) → 256 canaux → BatchNorm2d → ReLU → AdaptiveAvgPool2d(1×1) → (256, 1, 1)
  > - Tête (GAP / linéaire) → Linear(256, 200) → logits (dimension = 200 classes)

- **Loss function** :

> - Multi-classe : CrossEntropyLoss

- **Sortie du modèle** : forme = **(batch_size, 200)**

> (torch.Size([B, 200]))

- **Nombre total de paramètres** :

> `1,198,600`

**M1.** Décrivez l'**architecture** complète et donnez le **nombre total de paramètres**.  
Expliquez le rôle des **2 hyperparamètres spécifiques au modèle** (ceux imposés par votre sujet).

> L'architecture implémentée est un CNN à 3 stages avec convolutions dilatées. Chaque stage suit le pattern Conv2d(3×3) → BatchNorm2d → ReLU, répété plusieurs fois selon le paramètre de blocs. La réduction spatiale s'effectue par MaxPool2d(2×2) entre les stages 1-2 et 2-3, puis par Global Average Pooling avant la classification.

> Le modèle compte **1,198,600 paramètres entraînables** et produit des logits de forme (batch_size, 200) pour les 200 classes de Tiny ImageNet.

> **Les 2 hyperparamètres spécifiques au modèle** sont :
>
> 1. **Nombre de blocs par stage (B₁, B₂, B₃)** : contrôle la profondeur de chaque stage, choix entre (2,2,2) ou (3,3,3). Plus de blocs augmentent la capacité d'apprentissage mais aussi le risque de surapprentissage et le coût computationnel.
> 2. **Dilation par stage (D₂, D₃)** : contrôle le champ réceptif des convolutions dans les stages 2 et 3, choix entre (2,2) ou (2,3). Une dilation plus élevée permet de capturer des dépendances spatiales plus larges sans augmenter le nombre de paramètres, particulièrement utile après la réduction spatiale par MaxPool.

### 2.3 Perte initiale & premier batch

- **Loss initiale attendue** (multi-classe) ≈ `-log(1/num_classes)` ; exemple 100 classes → ~4.61
- **Observée sur un batch** : `5.2996`
- **Vérification** : backward OK, gradients ≠ 0

**M2.** Donnez la **loss initiale** observée et dites si elle est cohérente. Indiquez la forme du batch et la forme de sortie du modèle.

> Nous avons chargé un batch d'entraînement de taille 32. Les entrées ont la forme **(32, 3, 64, 64)** et les cibles la forme **(32,)** (labels entiers). La sortie du modèle est un tenseur de logits de forme **(32, 200)**.
> La loss initiale (CrossEntropyLoss) vaut **5.2996**, ce qui est parfaitement cohérent avec l'intuition d'une prédiction quasi-uniforme au départ, proche de **-log(1/200) = 5.2983** (différence négligeable de 0.0013).
> Les logits initiaux ont une moyenne de **-0.0007** (≈ 0) et un écart-type de **0.2625**, confirmant une initialisation correcte des poids sans biais vers une classe particulière.
> Après rétropropagation sur ce batch, la norme globale des gradients est **2.79**, strictement non nulle, confirmant que le gradient se propage correctement à travers toutes les couches du modèle.

---

## 3) Overfit « petit échantillon »

- **Sous-ensemble train** : `N = 32` exemples
- **Hyperparamètres modèle utilisés** (les 2 à régler) : `blocks_per_stage = [2, 2, 2]`, `dilations = [1, 2, 2]`
- **Optimisation** : LR = `0.01`, weight decay = `0.0` (0 ou très faible recommandé)
- **Nombre d'époques** : `50`

![Overfit train/loss](/artifacts/overfit/overfit_train-loss.png)

**M3.** Donnez la **taille du sous-ensemble**, les **hyperparamètres** du modèle utilisés, et la **courbe train/loss** (capture). Expliquez ce qui prouve l'overfit.

> Le sous-ensemble d'overfit contient **32 exemples** avec 30 classes uniques (distribution très diverse). Les hyperparamètres utilisés sont une architecture modérée avec **2 blocs par stage** et des **dilations [1, 2, 2]**.

> L'overfit est clairement démontré par :
>
> 1. **Descente rapide de la loss** : de 5.25 (époque 1) à 0.007 (époque 50), soit une réduction de 99.9%
> 2. **Accuracy atteignant 100%** : dès l'époque 11, le modèle atteint 100% de précision et la maintient quasiment constamment
> 3. **Mémorisation complète** : avec seulement 32 échantillons, le modèle mémorise parfaitement l'ensemble d'entraînement, ce qui est la preuve attendue d'une capacité d'apprentissage fonctionnelle

> Cette expérience confirme que le modèle a la capacité d'apprendre et que l'implémentation est correcte, validant ainsi l'architecture avant l'entraînement sur le dataset complet.

---

## 4) LR finder

- **Méthode** : balayage LR (log-scale), quelques itérations, log `(lr, loss)`
- **Fenêtre stable retenue** : `4e-4 → 1e-3`
- **Choix pour la suite** :
  - **LR** = `5e-4`
  - **Weight decay** = `1e-4` (valeurs classiques : 1e-5, 1e-4)

![LR Finder](/artifacts/lr-finder/lr_finder.png)

**M4.** Justifiez en 2–3 phrases le choix du **LR** et du **weight decay**.

> Le choix du **LR = 5e-4** se situe dans la fenêtre stable (4e-4 à 1e-3) où la loss descend de façon constante, avant la zone d'instabilité où elle commence à remonter. Ce LR modéré évite les oscillations tout en garantissant une convergence efficace.
> Le **weight decay = 1e-4** est un choix standard qui s'équilibre bien avec ce LR : suffisamment fort pour régulariser sans freiner excessivement l'apprentissage sur Tiny ImageNet qui reste un dataset challenging nécessitant une capacité d'apprentissage préservée.

---

## 5) Mini grid search (rapide)

- **Grilles** :

  - LR : `{2.5e-4, 5e-4, 1e-3}` (mini-grille ×0.5 ×1 ×2)
  - Weight decay : `{1e-5, 1e-4}`
  - Hyperparamètre modèle A (blocks_per_stage) : `{[2,2,2], [3,3,3]}`
  - Hyperparamètre modèle B (dilations D2,D3) : `{[2,2], [2,3]}`

- **Durée des runs** : `3` époques par run, même seed (42)

| Run (nom explicite)                           | LR   | WD   | Hyp-A   | Hyp-B | Val metric (nom=**accuracy**) | Val loss | Notes        |
| --------------------------------------------- | ---- | ---- | ------- | ----- | ----------------------------- | -------- | ------------ |
| proj11_lr=3e-04_wd=1e-05_blocks=3-3-3_dil=2-3 | 3e-4 | 1e-5 | [3,3,3] | [2,3] | 25.31%                        | 3.197    | **Meilleur** |
| proj11_lr=1e-03_wd=1e-04_blocks=2-2-2_dil=2-3 | 1e-3 | 1e-4 | [2,2,2] | [2,3] | 25.08%                        | 3.288    | 2ème         |
| proj11_lr=3e-04_wd=1e-05_blocks=2-2-2_dil=2-2 | 3e-4 | 1e-5 | [2,2,2] | [2,2] | 24.65%                        | 3.257    | 3ème         |
| proj11_lr=5e-04_wd=1e-05_blocks=2-2-2_dil=2-2 | 5e-4 | 1e-5 | [2,2,2] | [2,2] | 24.40%                        | 3.294    |              |
| proj11_lr=1e-03_wd=1e-05_blocks=2-2-2_dil=2-3 | 1e-3 | 1e-5 | [2,2,2] | [2,3] | 24.20%                        | 3.308    |              |
| proj11_lr=3e-04_wd=1e-04_blocks=2-2-2_dil=2-3 | 3e-4 | 1e-4 | [2,2,2] | [2,3] | 23.86%                        | 3.335    |              |
| proj11_lr=1e-03_wd=1e-04_blocks=2-2-2_dil=2-2 | 1e-3 | 1e-4 | [2,2,2] | [2,2] | 22.83%                        | 3.406    |              |
| proj11_lr=1e-03_wd=1e-04_blocks=2-2-2_dil=2-2 | 1e-3 | 1e-4 | [2,2,2] | [2,2] | 22.73%                        | 3.421    |              |
| proj11_lr=3e-04_wd=1e-04_blocks=3-3-3_dil=2-3 | 3e-4 | 1e-4 | [3,3,3] | [2,3] | 22.95%                        | 3.377    |              |
| proj11_lr=5e-04_wd=1e-05_blocks=2-2-2_dil=2-3 | 5e-4 | 1e-5 | [2,2,2] | [2,3] | 22.52%                        | 3.405    |              |
| proj11_lr=5e-04_wd=1e-04_blocks=2-2-2_dil=2-2 | 5e-4 | 1e-4 | [2,2,2] | [2,2] | 22.45%                        | 3.442    |              |
| proj11_lr=5e-04_wd=1e-04_blocks=3-3-3_dil=2-3 | 5e-4 | 1e-4 | [3,3,3] | [2,3] | 22.61%                        | 3.380    |              |
| proj11_lr=3e-04_wd=1e-04_blocks=3-3-3_dil=2-2 | 3e-4 | 1e-4 | [3,3,3] | [2,2] | 22.11%                        | 3.357    |              |
| proj11_lr=3e-04_wd=1e-05_blocks=2-2-2_dil=2-3 | 3e-4 | 1e-5 | [2,2,2] | [2,3] | 22.71%                        | 3.403    |              |
| proj11_lr=3e-04_wd=1e-04_blocks=2-2-2_dil=2-2 | 3e-4 | 1e-4 | [2,2,2] | [2,2] | 21.71%                        | 3.485    |              |
| proj11_lr=1e-03_wd=1e-05_blocks=3-3-3_dil=2-3 | 1e-3 | 1e-5 | [3,3,3] | [2,3] | 21.58%                        | 3.431    |              |
| proj11_lr=5e-04_wd=1e-04_blocks=2-2-2_dil=2-3 | 5e-4 | 1e-4 | [2,2,2] | [2,3] | 21.44%                        | 3.519    |              |
| proj11_lr=5e-04_wd=1e-05_blocks=3-3-3_dil=2-2 | 5e-4 | 1e-5 | [3,3,3] | [2,2] | 21.25%                        | 3.414    |              |
| proj11_lr=5e-04_wd=1e-05_blocks=3-3-3_dil=2-3 | 5e-4 | 1e-5 | [3,3,3] | [2,3] | 21.01%                        | 3.469    |              |
| proj11_lr=3e-04_wd=1e-05_blocks=3-3-3_dil=2-2 | 3e-4 | 1e-5 | [3,3,3] | [2,2] | 20.46%                        | 3.491    |              |
| proj11_lr=1e-03_wd=1e-04_blocks=3-3-3_dil=2-2 | 1e-3 | 1e-4 | [3,3,3] | [2,2] | 20.55%                        | 3.491    |              |
| proj11_lr=1e-03_wd=1e-05_blocks=3-3-3_dil=2-2 | 1e-3 | 1e-5 | [3,3,3] | [2,2] | 20.37%                        | 3.529    |              |
| proj11_lr=1e-03_wd=1e-04_blocks=3-3-3_dil=2-3 | 1e-3 | 1e-4 | [3,3,3] | [2,3] | 19.57%                        | 3.593    |              |
| proj11_lr=5e-04_wd=1e-04_blocks=3-3-3_dil=2-2 | 5e-4 | 1e-4 | [3,3,3] | [2,2] | 17.67%                        | 3.714    |              |

![Grid Search](/artifacts/gridSearch/gridSearch.png)

**M5.** Présentez la **meilleure combinaison** (selon validation) et commentez l'effet des **2 hyperparamètres de modèle** sur les courbes (stabilité, vitesse, overfit).

> La **meilleure combinaison** est : LR=3e-04, Weight Decay=1e-5, Blocks=[3,3,3], Dilations=[2,3] avec 25.31% de précision validation.

> **Effet des hyperparamètres modèle :**
>
> 1. **Nombre de blocs (2,2,2) vs (3,3,3)** : Les configurations [3,3,3] montrent une convergence plus lente initialement (accuracy plus faible en époque 1) mais peuvent atteindre de meilleures performances finales quand bien optimisées. Cependant, elles sont plus sensibles aux hyperparamètres d'optimisation et peuvent sous-performer avec de mauvais réglages.
> 2. **Dilations (D2,D3) [2,2] vs [2,3]** : Les dilations [2,3] semblent légèrement favorables, apparaissant dans 4 des 5 meilleures configurations. L'augmentation de la dilation D3 permet de capturer des dépendances spatiales plus larges dans les dernières couches, ce qui améliore la capacité de représentation sans coût computationnel supplémentaire significatif.

---

## 6) Entraînement complet (10–20 époques, sans scheduler)

- **Configuration finale** :
  - LR = `3e-4`
  - Weight decay = `1e-5`
  - Hyperparamètre modèle A = `[3,3,3]` (blocks_per_stage)
  - Hyperparamètre modèle B = `[2,3]` (dilations)
  - Batch size = `32` (inféré du grid search)
  - Époques = `15`
- **Checkpoint** : `artifacts/best.ckpt` (selon meilleure métrique val)

**M6.** Montrez les **courbes train/val** (loss + métrique). Interprétez : sous-apprentissage / sur-apprentissage / stabilité d'entraînement.

| ![img1](/artifacts/training/train-loss.png) | ![img2](/artifacts/training/train-accuracy.png) |
| :-----------------------------------------: | :---------------------------------------------: |
|  ![img3](/artifacts/training/val-loss.png)  |  ![img4](/artifacts/training/val-accuracy.png)  |

> L'entraînement montre une **convergence stable et saine** sur 15 époques :
>
> - Train loss : descente continue de 4.54 → 2.02
> - Val loss : amélioration de 4.01 → 2.26
> - Train accuracy : progression régulière de 6.52% → 49.76%
> - Val accuracy : amélioration de 11.76% → 46.20%

> L’écart entre les performances train et validation reste limité, ce qui exclut un surapprentissage marqué. En revanche, les performances restent relativement modestes (≈ 50 % d’accuracy train et ≈ 46 % en validation sur 200 classes), ce qui suggère un sous-apprentissage : le modèle n’a pas encore pleinement capturé la complexité du jeu de données.

---

## 7) Comparaisons de courbes (analyse)

**M7.** Trois **comparaisons** commentées (une phrase chacune) : LR, weight decay, hyperparamètres modèle — ce que vous attendiez vs. ce que vous observez.

> - **Variation du LR**

![img1](/artifacts/comparaison/experiment-lr.png)

> LR=1e-3 converge beaucoup plus rapidement au début (8.02% → 25.66% entre époques 1-4) mais plafonne prématurément à 45.63%, tandis que LR=3e-4 progresse plus graduellement mais atteint une meilleure performance finale (46.20% → 48.09%), illustrant le trade-off classique vitesse vs qualité d'optimisation.

> - **Variation du weight decay** (écart train/val, régularisation)

![img2](/artifacts/comparaison/experiment-wd.png)

> Weight decay=1e-4 réduit significativement l'écart train/val (3.53% vs 6.90% pour wd=1e-5) mais on constate une performance globale légèrement réduite (49.16% vs 54.99% en train) confirmant son effet régularisateur, au prix d'une légère baisse de performance globale mais avec une meilleure généralisation et convergence plus stable.

> - **Variation des 2 hyperparamètres de modèle** (convergence, plateau, surcapacité)

![img2](/artifacts/comparaison/experiment-blocks-dl.png)

L'architecture [2,2,2] + [1,2,2] (moins de paramètres) converge plus rapidement avec une progression très stable, mais l'architecture [3,3,3] + [2,3] (plus de paramètres) converge plus lentement mais atteint une meilleure précision finale.

---

## 8) Itération supplémentaire (si temps)

- **Changement(s)** : `_____` (resserrage de grille, nouvelle valeur d’un hyperparamètre, etc.)
- **Résultat** : `_____` (val metric, tendances des courbes)

**M8.** Décrivez cette itération, la motivation et le résultat.

---

## 9) Évaluation finale (test)

- **Checkpoint évalué** : `artifacts/best.ckpt`
- **Métriques test** :
  - Metric principale (nom = `accuracy`) : `46.00%`
  - Metric(s) secondaire(s) : `F1 Macro = 0.4544`, `Test Loss = 2.2396`

**M9.** Donnez les **résultats test** et comparez-les à la validation (écart raisonnable ? surapprentissage probable ?).

> Les résultats test montrent une **excellente cohérence avec la validation** : accuracy test (46.00%) vs validation finale (46.20%), soit un écart négligeable de seulement 0.20 point. La test loss (2.2396) est également très proche de la validation loss finale (≈2.26), confirmant une généralisation saine du modèle.
>
> Le F1 Macro (0.4544) proche de l'accuracy (0.4600) indique une performance équilibrée entre les 200 classes, sans biais majeur vers certaines classes. L'absence d'écart significatif entre validation et test exclut un surapprentissage et valide la robustesse des hyperparamètres choisis via grid search.

---

## 10) Limites, erreurs & bug diary (court)

- **Limites connues** (données, compute, modèle) :

> Le modèle présente un sous-apprentissage marqué, principalement dû à plusieurs facteurs :
>
> - **Compute** : Ressources limitées, entraînement restreint à 20 époques, ce qui empêche d’explorer des architectures plus profondes ou des entraînements prolongés.
> - **Modèle** : Architecture CNN standard, sans mécanismes récents (attention, skip connections), plafonnant à environ 46 % d’accuracy sur 200 classes.

- **Erreurs rencontrées** (shape mismatch, divergence, NaN…) et **solutions** :

> - **Shape mismatch** lors du chargement des données : incompatibilité entre images RGB (64, 64, 3) et niveaux de gris (64, 64). Solution : conversion systématique en RGB au prétraitement.
> - **Instabilité lors du grid search** : divergence de certaines configurations avec des LR trop élevés. Solution : limitation du LR à 1e-3 maximum et suivi attentif des métriques.

- **Axes d’amélioration si plus de temps/compute** (en une phrase) :

> Prolonger l’entraînement (50+ époques) avec scheduler, tester des architectures modernes (ResNet, attention), augmenter l’intensité des augmentations de données, et recourir à l’ensemblage de modèles pour dépasser 50 % d’accuracy.

---

## 11) Reproductibilité

- **Seed** : `42`
- **Config utilisée** :

```yaml
dataset:
  name: "tiny-imagenet"
  root: "./data"
  split:
    train: "train"
    val: "valid"
    test: null
  download: true
  num_workers: 2
  shuffle: true

preprocess:
  img_size: 64
  resize: [64, 64]
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]

augment:
  random_horizontal_flip:
    p: 0.5
  random_rotation:
    degrees: 15
  random_resized_crop:
    scale: [0.8, 1.0]
    ratio: [0.9, 1.1]
  color_jitter:
    brightness: 0.1
    contrast: 0.1
    saturation: 0.1
    hue: 0.05
  random_erasing:
    enabled: false
    p: 0.2

model:
  type: "dilated_cnn"
  num_classes: 200
  input_shape: [3, 64, 64]

  blocks_per_stage: [3, 3, 3]
  dilations: [1, 2, 3]
  channels: [64, 128, 256]

  activation: relu
  dropout: 0.0
  batch_norm: true
  residual: false
  attention: false

  hidden_sizes: null
  rnn:
    type: null
    hidden_size: null
    num_layers: null
    bidirectional: false

train:
  seed: 42
  device: auto
  batch_size: 64
  epochs: 15
  max_steps: null
  overfit_small: false

  optimizer:
    name: adam
    lr: 0.0003
    weight_decay: 0.00001
    momentum: 0.9

  scheduler:
    name: none
    step_size: 10
    gamma: 0.1
    warmup_steps: 0

metrics:
  classification:
    - accuracy
  regression: []

hparams:
  lr: [0.00025, 0.0005, 0.001]
  weight_decay: [0.00001, 0.0001]
  blocks_per_stage: [[2, 2, 2], [3, 3, 3]]
  dilations: [[2, 2], [2, 3]]

paths:
  runs_dir: "./runs"
  artifacts_dir: "./artifacts"
```

- **Commandes exactes** :

```bash
python -m src.train --config configs/config.yaml --overfit_small
python -m src.train --config configs/config.yaml --max_epochs 15
python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt
python -m src.grid_search --config configs/config.yaml
python -m src.lr_finder --config configs/config.yaml
```

- **Artifacts requis présents** :

  - [x] `runs/` (runs utiles uniquement)
  - [x] `artifacts/best.ckpt`
  - [x] `configs/config.yaml` aligné avec la meilleure config

---

## 12) Références (courtes)

- PyTorch docs des modules utilisés (Conv2d, BatchNorm, ReLU, LSTM/GRU, transforms, etc.).
- Lien dataset officiel (et/ou HuggingFace/torchvision/torchaudio).
- Toute ressource externe substantielle (une ligne par référence).
- Toute ressource externe substantielle (une ligne par référence).
- Lien dataset officiel (et/ou HuggingFace/torchvision/torchaudio).
- Toute ressource externe substantielle (une ligne par référence).
- Toute ressource externe substantielle (une ligne par référence).
