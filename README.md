# AI Model For Lettering

## Description
Ce projet vise à développer un modèle d'intelligence artificielle pour le lettrage automatique des écritures comptables. Il comprend à la fois la génération de données comptables synthétiques et plusieurs modèles d'apprentissage automatique pour effectuer le lettrage automatique.

## Structure du Projet

### 1. DataGeneration/
Contient les scripts pour générer et prétraiter des données comptables synthétiques :
- `accounting_dataset_generator.py` : Générateur principal de données comptables
- `dataset-final-lettrage-preprocessing.ipynb` : Notebook de prétraitement des données
- `expences_transaction_generate.ipynb` : Génération de transactions de dépenses
- `invoices_generate.py` : Génération de factures
- Dossiers de sortie pour les données générées

### 2. Models/
Contient différentes implémentations de modèles d'IA pour le lettrage automatique :
- `GaussianNaiveBayesModel/` : Modèle de classification Naive Bayes gaussien
- `KNNModel/` : Modèle K-plus proches voisins
- `RandomForestModel/` : Modèle de forêt aléatoire
- `OtherModels/` : Autres modèles expérimentaux

## Installation

1. **Cloner le dépôt** :
   ```bash
   git clone [URL_DU_DEPOT]
   cd AI-Model-For-Lettering
   ```

2. **Créer un environnement virtuel** (recommandé) :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows : .\venv\Scripts\activate
   ```

3. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```
   Pour des dépendances spécifiques à un modèle, consultez le fichier `requirements.txt` dans chaque sous-dossier de `Models/`.

## Utilisation

### Génération de données
Pour générer un jeu de données comptable synthétique :
```bash
python DataGeneration/accounting_dataset_generator.py
```

### Entraînement des modèles
Chaque modèle possède son propre dossier avec des instructions spécifiques. Par exemple, pour le modèle de forêt aléatoire :
```bash
cd Models/RandomForestModel
python train.py
```

## Modèles Implémentés

1. **Forêt Aléatoire (Random Forest)**
   - Implémentation complète avec optimisation des hyperparamètres
   - Fonctionnalités de prétraitement des données
   - Évaluation des performances

2. **K-plus proches voisins (KNN)**
   - Classification basée sur la similarité
   - Optimisation du nombre de voisins

3. **Naive Bayes Gaussien**
   - Modèle probabiliste simple et efficace
   - Entraînement rapide sur de grands jeux de données

## 📝 Notes

- Les données générées sont synthétiques et conçues pour simuler des écritures comptables réelles
- Les modèles sont entraînés pour prédire les correspondances entre les transactions bancaires et les factures/dépenses
- Les performances peuvent varier en fonction de la qualité et de la quantité des données d'entraînement

