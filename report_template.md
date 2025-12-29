# Rapport de projet — CSC8607 : Introduction au Deep Learning

> **Consignes générales**
>
> - Tenez-vous au **format** et à l’**ordre** des sections ci-dessous.
> - Intégrez des **captures d’écran TensorBoard** lisibles (loss, métriques, LR finder, comparaisons).
> - Les chemins et noms de fichiers **doivent** correspondre à la structure du dépôt modèle (ex. `runs/`, `artifacts/best.ckpt`, `configs/config.yaml`).
> - Répondez aux questions **numérotées** (D1–D11, M0–M9, etc.) directement dans les sections prévues.

---

## 0) Informations générales

- **Étudiant·e** : \_Boutkrida, Younes
- **Projet** : Tiny ImageNet (Convolution)
- **Dépôt Git** : https://github.com/Younnsss/deeplearning
- **Environnement** : `python == ...`, `torch == ...`, `cuda == ...`
- **Commandes utilisées** :
  - Entraînement : `python -m src.train --config configs/config.yaml`
  - LR finder : `python -m src.lr_finder --config configs/config.yaml`
  - Grid search : `python -m src.grid_search --config configs/config.yaml`
  - Évaluation : `python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt`

---

## 1) Données

### 1.1 Description du dataset

- **Source** (lien) :
- **Type d’entrée** (image) : image 3 x 64 x 64
- **Tâche** : multiclasses
- **Dimensions d’entrée attendues** (`meta["input_shape"]`) : 3 x 64 x 64
- **Nombre de classes** (`meta["num_classes"]`) : 200

**D1.** Quel dataset utilisez-vous ? D’où provient-il et quel est son format (dimensions, type d’entrée) ?

> Le dataset utilisé est Tiny ImageNet, provenant du dépôt HuggingFace `zh-plus/tiny-imagenet`. Il s'agit d'un dataset d'images couleur de taille 64x64 pixels, avec 3 canaux (RGB).

### 1.2 Splits et statistiques

| Split | #Exemples | Particularités (déséquilibre, longueur moyenne, etc.) |
| ----: | --------: | ----------------------------------------------------- |
| Train |           |                                                       |
|   Val |           |                                                       |
|  Test |           |                                                       |

**D2.** Donnez la taille de chaque split et le nombre de classes.

> Le split train contient environ 90% des exemples du train original (90% de 100000 -> 90000), le split test 10% stratifié (10% de 100000 -> 10000), et le split validation correspond au split `valid` du dataset HuggingFace (10000). Le nombre de classes est 200.

**D3.** Si vous avez créé un split (ex. validation), expliquez **comment** (stratification, ratio, seed).

> Le split test a été créé à partir du train original par une séparation stratifiée (ratio 0.1), en utilisant la seed fixe 42 pour garantir la reproductibilité. La stratification assure une distribution équilibrée des classes dans chaque split.

**D4.** Donnez la **distribution des classes** (graphique ou tableau) et commentez en 2–3 lignes l’impact potentiel sur l’entraînement.

![Distribution des classes](/img/distributionsClasses.png)

> La distribution des classes est parfaitement équilibrée : chaque classe contient exactement 50 images. Cela garantit que le modèle ne sera pas biaisé vers une classe majoritaire et que les métriques d’entraînement et de validation reflèteront fidèment la performance réelle. L’absence de déséquilibre facilite l’apprentissage et la comparaison des résultats entre classes.

**D5.** Mentionnez toute particularité détectée (tailles variées, longueurs variables, multi-labels, etc.).

> Deux formats d’images ont été détectés dans le dataset : la grande majorité (98179 images) sont au format attendu (64, 64, 3), mais 1821 images sont au format (64, 64) (sans canal couleur explicite). Cette hétérogénéité nécessite un prétraitement pour garantir la cohérence des entrées du modèle (conversion en RGB si besoin).

### 1.3 Prétraitements (preprocessing) — _appliqués à train/val/test_

Listez précisément les opérations et paramètres (valeurs **fixes**) :

- Vision : resize = **, center-crop = **, normalize = (mean=**, std=**)…
- Audio : resample = ** Hz, mel-spectrogram (n_mels=**, n_fft=**, hop_length=**), AmplitudeToDB…
- NLP : tokenizer = **, vocab = **, max_length = **, padding/truncation = **…
- Séries : normalisation par canal, fenêtrage = \_\_…

**D6.** Quels **prétraitements** avez-vous appliqués (opérations + **paramètres exacts**) et **pourquoi** ?

Les prétraitements appliqués sont les suivants (identiques pour train/val/test, sauf pour le recadrage central en val/test) :

- Conversion en RGB pour toutes les images (pour gérer les images en niveaux de gris).
- Redimensionnement à 64x64 pixels (`transforms.Resize((64, 64))`) pour garantir une taille cohérente.
- Pour val/test uniquement : recadrage central (`transforms.CenterCrop(64)`) pour éviter tout biais d’augmentation.
- Conversion en tenseur (`transforms.ToTensor()`).
- Normalisation par canal avec les moyennes et écarts-types ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) afin de stabiliser et accélérer l’apprentissage.

Ces choix assurent la compatibilité des entrées avec les modèles convolutionnels classiques et la reproductibilité des expériences.

**D7.** Les prétraitements diffèrent-ils entre train/val/test (ils ne devraient pas, sauf recadrage non aléatoire en val/test) ?

Les prétraitements sont identiques pour train, validation et test, à l’exception du recadrage central (`CenterCrop`) qui n’est appliqué qu’en validation et test pour garantir une évaluation déterministe et sans biais d’augmentation. Toutes les autres opérations (conversion RGB, resize, normalisation) sont strictement les mêmes sur les trois splits.

### 1.4 Augmentation de données — _train uniquement_

- Liste des **augmentations** (opérations + **paramètres** et **probabilités**) :
  - Recadrage aléatoire puis redimensionnement (`RandomResizedCrop`, scale=(0.8, 1.0), ratio=(0.9, 1.1))
  - Flip horizontal aléatoire (`RandomHorizontalFlip`, p=0.5)
  - Rotation aléatoire (`RandomRotation`, degrés=15)
  - ColorJitter léger (`ColorJitter`, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)

**D8.** Quelles **augmentations** avez-vous appliquées (paramètres précis) et **pourquoi** ?

Les augmentations appliquées en entraînement sont :

- **RandomResizedCrop** (scale=(0.8, 1.0), ratio=(0.9, 1.1)) : simule des variations de cadrage et de zoom, rendant le modèle robuste à la position et à la taille de l’objet.
- **RandomHorizontalFlip** (p=0.5) : introduit une invariance gauche-droite, utile pour de nombreux objets naturels.
- **RandomRotation** (degrés=15) : rend le modèle moins sensible à l’orientation exacte de l’objet.
- **ColorJitter** (brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05) : simule des variations d’éclairage et de couleur, améliorant la robustesse aux conditions de prise de vue.
  Ces choix visent à améliorer la généralisation du modèle en l’exposant à des variations réalistes des images d’entraînement.

**D9.** Les augmentations **conservent-elles les labels** ? Justifiez pour chaque transformation retenue.

Oui, toutes les augmentations appliquées sont label-preserving :

- **RandomResizedCrop** : ne modifie pas la classe de l’objet, seulement son cadrage et sa taille apparente.
- **RandomHorizontalFlip** : l’objet reste de la même classe après un flip horizontal.
- **RandomRotation** : une rotation modérée ne change pas la nature de l’objet.
- **ColorJitter** : les variations de couleur et de luminosité n’affectent pas la classe sémantique.
  Ces transformations sont donc sûres pour l’apprentissage supervisé car elles ne créent pas d’ambiguïté sur le label.

### 1.5 Sanity-checks

- **Exemples** après preprocessing/augmentation (insérer 2–3 images/spectrogrammes) :

> _Insérer ici 2–3 captures illustrant les données après transformation._

**D10.** Montrez 2–3 exemples et commentez brièvement.  
**D11.** Donnez la **forme exacte** d’un batch train (ex. `(batch, C, H, W)` ou `(batch, seq_len)`), et vérifiez la cohérence avec `meta["input_shape"]`.

---

## 2) Modèle

### 2.1 Baselines

**M0.**

- **Classe majoritaire** — Métrique : `_____` → score = `_____`
- **Prédiction aléatoire uniforme** — Métrique : `_____` → score = `_____`  
  _Commentez en 2 lignes ce que ces chiffres impliquent._

### 2.2 Architecture implémentée

- **Description couche par couche** (ordre exact, tailles, activations, normalisations, poolings, résiduels, etc.) :

  - Input → …
  - Stage 1 (répéter N₁ fois) : …
  - Stage 2 (répéter N₂ fois) : …
  - Stage 3 (répéter N₃ fois) : …
  - Tête (GAP / linéaire) → logits (dimension = nb classes)

- **Loss function** :

  - Multi-classe : CrossEntropyLoss
  - Multi-label : BCEWithLogitsLoss
  - (autre, si votre tâche l’impose)

- **Sortie du modèle** : forme = **(batch_size, num_classes)** (ou **(batch_size, num_attributes)**)

- **Nombre total de paramètres** : `_____`

**M1.** Décrivez l’**architecture** complète et donnez le **nombre total de paramètres**.  
Expliquez le rôle des **2 hyperparamètres spécifiques au modèle** (ceux imposés par votre sujet).

### 2.3 Perte initiale & premier batch

- **Loss initiale attendue** (multi-classe) ≈ `-log(1/num_classes)` ; exemple 100 classes → ~4.61
- **Observée sur un batch** : `_____`
- **Vérification** : backward OK, gradients ≠ 0

**M2.** Donnez la **loss initiale** observée et dites si elle est cohérente. Indiquez la forme du batch et la forme de sortie du modèle.

---

## 3) Overfit « petit échantillon »

- **Sous-ensemble train** : `N = ____` exemples
- **Hyperparamètres modèle utilisés** (les 2 à régler) : `_____`, `_____`
- **Optimisation** : LR = `_____`, weight decay = `_____` (0 ou très faible recommandé)
- **Nombre d’époques** : `_____`

> _Insérer capture TensorBoard : `train/loss` montrant la descente vers ~0._

**M3.** Donnez la **taille du sous-ensemble**, les **hyperparamètres** du modèle utilisés, et la **courbe train/loss** (capture). Expliquez ce qui prouve l’overfit.

---

## 4) LR finder

- **Méthode** : balayage LR (log-scale), quelques itérations, log `(lr, loss)`
- **Fenêtre stable retenue** : `_____ → _____`
- **Choix pour la suite** :
  - **LR** = `_____`
  - **Weight decay** = `_____` (valeurs classiques : 1e-5, 1e-4)

> _Insérer capture TensorBoard : courbe LR → loss._

**M4.** Justifiez en 2–3 phrases le choix du **LR** et du **weight decay**.

---

## 5) Mini grid search (rapide)

- **Grilles** :

  - LR : `{_____ , _____ , _____}`
  - Weight decay : `{1e-5, 1e-4}`
  - Hyperparamètre modèle A : `{_____, _____}`
  - Hyperparamètre modèle B : `{_____, _____}`

- **Durée des runs** : `_____` époques par run (1–5 selon dataset), même seed

| Run (nom explicite) | LR  | WD  | Hyp-A | Hyp-B | Val metric (nom=**\_**) | Val loss | Notes |
| ------------------- | --- | --- | ----- | ----- | ----------------------- | -------- | ----- |
|                     |     |     |       |       |                         |          |       |
|                     |     |     |       |       |                         |          |       |

> _Insérer capture TensorBoard (onglet HParams/Scalars) ou tableau récapitulatif._

**M5.** Présentez la **meilleure combinaison** (selon validation) et commentez l’effet des **2 hyperparamètres de modèle** sur les courbes (stabilité, vitesse, overfit).

---

## 6) Entraînement complet (10–20 époques, sans scheduler)

- **Configuration finale** :
  - LR = `_____`
  - Weight decay = `_____`
  - Hyperparamètre modèle A = `_____`
  - Hyperparamètre modèle B = `_____`
  - Batch size = `_____`
  - Époques = `_____` (10–20)
- **Checkpoint** : `artifacts/best.ckpt` (selon meilleure métrique val)

> _Insérer captures TensorBoard :_
>
> - `train/loss`, `val/loss`
> - `val/accuracy` **ou** `val/f1` (classification)

**M6.** Montrez les **courbes train/val** (loss + métrique). Interprétez : sous-apprentissage / sur-apprentissage / stabilité d’entraînement.

---

## 7) Comparaisons de courbes (analyse)

> _Superposez plusieurs runs dans TensorBoard et insérez 2–3 captures :_
>
> - **Variation du LR** (impact au début d’entraînement)
> - **Variation du weight decay** (écart train/val, régularisation)
> - **Variation des 2 hyperparamètres de modèle** (convergence, plateau, surcapacité)

**M7.** Trois **comparaisons** commentées (une phrase chacune) : LR, weight decay, hyperparamètres modèle — ce que vous attendiez vs. ce que vous observez.

---

## 8) Itération supplémentaire (si temps)

- **Changement(s)** : `_____` (resserrage de grille, nouvelle valeur d’un hyperparamètre, etc.)
- **Résultat** : `_____` (val metric, tendances des courbes)

**M8.** Décrivez cette itération, la motivation et le résultat.

---

## 9) Évaluation finale (test)

- **Checkpoint évalué** : `artifacts/best.ckpt`
- **Métriques test** :
  - Metric principale (nom = `_____`) : `_____`
  - Metric(s) secondaire(s) : `_____`

**M9.** Donnez les **résultats test** et comparez-les à la validation (écart raisonnable ? surapprentissage probable ?).

---

## 10) Limites, erreurs & bug diary (court)

- **Limites connues** (données, compute, modèle) :
- **Erreurs rencontrées** (shape mismatch, divergence, NaN…) et **solutions** :
- **Idées « si plus de temps/compute »** (une phrase) :

---

## 11) Reproductibilité

- **Seed** : `_____`
- **Config utilisée** : joindre un extrait de `configs/config.yaml` (sections pertinentes)
- **Commandes exactes** :

```bash
# Exemple (remplacer par vos commandes effectives)
python -m src.train --config configs/config.yaml --max_epochs 15
python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt
```

- **Artifacts requis présents** :

  - [ ] `runs/` (runs utiles uniquement)
  - [ ] `artifacts/best.ckpt`
  - [ ] `configs/config.yaml` aligné avec la meilleure config

---

## 12) Références (courtes)

- PyTorch docs des modules utilisés (Conv2d, BatchNorm, ReLU, LSTM/GRU, transforms, etc.).
- Lien dataset officiel (et/ou HuggingFace/torchvision/torchaudio).
- Toute ressource externe substantielle (une ligne par référence).
- Toute ressource externe substantielle (une ligne par référence).
