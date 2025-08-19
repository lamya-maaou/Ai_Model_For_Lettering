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

### 1. Forêt Aléatoire (Random Forest)
- **Type** : Modèle d'ensemble basé sur les arbres de décision
- **Caractéristiques** :
  - Optimisation des hyperparamètres
  - Prétraitement avancé des données
  - Évaluation complète des performances
  - Gestion du sur-apprentissage
- **Fichiers clés** : `RandomForestModel/notebooks/02_Modele_RandomForest.ipynb`

### 2. K-plus proches voisins (KNN)
- **Type** : Algorithme d'apprentissage supervisé basé sur la similarité
- **Caractéristiques** :
  - Optimisation du nombre de voisins (k)
  - Mise à l'échelle des caractéristiques
  - Matrice de confusion et métriques d'évaluation
- **Fichiers clés** : `KNNModel/notebooks/01_modele_knn_lettrage.ipynb`

### 3. Naive Bayes Gaussien
- **Type** : Classificateur probabiliste
- **Caractéristiques** :
  - Hypothèse d'indépendance des caractéristiques
  - Entraînement rapide
  - Efficace sur grands jeux de données

### 4. Modèles Additionnels (OtherModels/)
- **Gradient Boosting** : `GradientGboost_model.pkl`
- **LightGBM** : `LightGBM_model.pkl`
- **Régression Logistique** : `Logestic_Regression_model.pkl`
- **SVM (Machines à Vecteurs de Support)** : `SVM_model.pkl`
- **XGBoost** : `xgb_model.pkl`

### 5. Modèles d'Ensemble
- Combinaison de plusieurs modèles pour améliorer les prédictions
- Approches de vote et de pondération

## 📝 Notes

- Les données générées sont synthétiques et conçues pour simuler des écritures comptables réelles
- Les modèles sont entraînés pour prédire les correspondances entre les transactions bancaires et les factures/dépenses
- Les performances peuvent varier en fonction de la qualité et de la quantité des données d'entraînement

