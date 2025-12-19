# MicroLens 1M — Pipeline Task1 & Task2 (CLIP + DIN)

Ce projet implémente une pipeline complète pour la compétition MicroLens_1M_MMCTR (Task1 & Task2) :

- Génération d'embeddings multimodaux (texte + image) avec CLIP
- Compression des embeddings en 128 dimensions (PCA)
- Entraînement d'un modèle CTR de type DIN (Deep Interest Network)
- Génération d'une archive `prediction.zip` compatible Codabench

---

## Structure attendue des fichiers

Le dossier `BASE_PATH` (par exemple sur Google Drive) doit contenir :

- `item_feature.parquet` — contient `item_id` et `item_title` (utilisé pour l'encodage texte CLIP)
- `item_images_2.rar` — archive d'images nommées `<item_id>.jpg`
- `MicroLens_1M_x1/` — dossier dataset contenant :
	- `train.parquet`
	- `valid.parquet`
	- `test.parquet`

Le pipeline produit ensuite :

- `item_emb_task1_clip.parquet` — embeddings 128D (CLIP + PCA)
- `models_task1and2/task1and2_best.pt` — meilleur modèle DIN sauvegardé
- `predictions_task1and2/prediction.csv` et `predictions_task1and2/prediction.zip`

---

## Étapes de la pipeline

### Étape 0 — Installation & environnement

Objectif : installer les dépendances requises et monter le Drive (si utilisé dans Colab).

Actions typiques : installation de `clip` (OpenAI), `polars` (lecture Parquet rapide), montage de `/content/drive`.

### Étape 1 — Extraction des images

Objectif : rendre les images disponibles localement.

Action : extraire `item_images_2.rar` dans un dossier (p.ex. `/content/item_images/`). Les images doivent être nommées `<item_id>.jpg`. Si le dossier existe déjà, on ignore cette étape.

### Étape 2 — Génération des embeddings (CLIP)

Objectif : créer un embedding multimodal par item (texte + image).

Processus :

- Texte : tokenisation CLIP de `item_title` (tronqué à 77 tokens)
- Image : prétraitement CLIP (224×224)
- Encodage : `model.encode_text()` et `model.encode_image()`
- Normalisation L2 de chaque embedding
- Fusion multimodale : moyenne simple (texte + image) / 2

Résultat : matrice d'embeddings CLIP (dimension native, p.ex. 512)

### Étape 3 — Réduction dimensionnelle (PCA 512→128)

Objectif : produire des embeddings finaux de dimension 128 (exigence de la compétition).

Actions : appliquer une PCA (`n_components=128`), afficher la variance expliquée et sauvegarder dans `item_emb_task1_clip.parquet` au format :

- `item_id`
- `item_emb_d128` (liste de 128 float32)

Cette étape est ignorée si le fichier existe déjà.

### Étape 4 — Construction de la matrice d'embeddings pour DIN

Objectif : créer des poids d'embedding PyTorch avec un index de padding.

Actions :

- Lire `item_emb_task1_clip.parquet`
- Construire `id_to_idx` : `id_to_idx[item_id] = index + 1` (0 réservé au padding)
- Construire `PRETRAINED_WEIGHTS` : tensor de taille `(num_items + 1, 128)` (ligne 0 = vecteur zéro)

Résultats : `PRETRAINED_WEIGHTS`, `ID_MAP`

### Étape 5 — Préparation du dataset CTR

Objectif : préparer les entrées pour le modèle DIN à partir de `train.parquet`, `valid.parquet`, `test.parquet`.

Champs utilisés :

- `item_id` — item cible
- `item_seq` — séquence historique d'items consultés
- `likes_level` (0..19)
- `views_level` (0..19)
- `label` (train/valid uniquement)
- `ID` (test uniquement)

Tous les `item_id` (cible et séquence) sont mappés via `ID_MAP`. Les ids inconnus deviennent `0` (padding).

### Étape 6 — Modèle DIN (Deep Interest Network)

Entrées : `history` (séquence d'items), `target` (item candidat), `likes_level`, `views_level`.

Architecture principale :

- Embedding item (128D) initialisé avec les poids CLIP+PCA
- Embeddings pour features auxiliaires (`likes_emb` 16D, `views_emb` 16D)
- Mécanisme d'attention (DIN) pour calculer un vecteur d'intérêt utilisateur
- MLP final combinant l'embedding du target, l'embedding d'intérêt, `likes_emb`, `views_emb`
- Activation DICE utilisée dans le MLP

Sortie : logit (avant sigmoid)

### Étape 7 — Entraînement

Objectif : optimiser la classification CTR avec `BCEWithLogitsLoss`.

Paramètres usuels :

- Loss : `BCEWithLogitsLoss`
- Optimizer : `AdamW` (weight_decay=1e-4)
- Scheduler : `ReduceLROnPlateau` surveillant l'AUC de validation
- Label smoothing léger : `lbl = lbl * 0.98 + 0.01`
- Gradient clipping : `clip_grad_norm_` à 5.0
- Early stopping : arrêt si aucune amélioration pendant 3 epochs

Le meilleur modèle est sauvegardé dans : `models_task1and2/task1and2_best.pt`

### Étape 8 — Prédiction et soumission (Task1 & Task2)

Objectif : générer une archive `prediction.zip` compatible Codabench.

Processus :

- Prédictions sur `test.parquet`
- Application de la sigmoid aux logits pour obtenir des probabilités
- Création du CSV contenant les colonnes : `ID`, `Task1`, `Task2`, `Task1&2` (où `Task1` et `Task2` peuvent être 0 si non utilisés, et `Task1&2` contient la prédiction)
- Compression en `prediction.zip` (contenant `prediction.csv`)

---

## Résultat final

À la fin du pipeline, vous obtenez :

- `predictions_task1and2/prediction.zip`, fichier à uploader sur Codabench pour l'évaluation Task1&2

---

## Remarques & bonnes pratiques

- Valider l'existence des fichiers d'entrée avant d'exécuter chaque étape pour éviter les opérations redondantes.
- Sauvegarder les artefacts intermédiaires (`item_emb_task1_clip.parquet`, `ID_MAP`) pour réutilisation.
- Vérifier la compatibilité des versions de `clip`, `torch` et `scikit-learn` (PCA).

