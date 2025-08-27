from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Literal
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chemins des modèles
MODELS_DIR = Path("../Models/OtherModels/")
MODEL_FILES = {
    "lightgbm": "LightGBM_model.pkl",
    "random_forest": "random_forest_model.pkl",
    "gradient_boosting": "GradientGboost_model.pkl",
    "svm": "SVM_model.pkl",
    "xgboost": "xgb_model.pkl"
}

# Chargement des modèles au démarrage
models = {}
for name, filename in MODEL_FILES.items():
    try:
        model_path = MODELS_DIR / filename
        if model_path.exists():
            models[name] = joblib.load(model_path)
            logger.info(f"Modèle {name} chargé avec succès")
        else:
            logger.warning(f"Fichier du modèle {name} non trouvé: {model_path}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle {name}: {str(e)}")

if not models:
    raise RuntimeError("Aucun modèle n'a pu être chargé. Vérifiez les chemins des fichiers de modèle.")

# Définition des modèles Pydantic pour la validation
class Operation(BaseModel):
    montant: float
    date: str
    reference: Optional[str] = None
    type: Optional[str] = None
    categorie: Optional[str] = None
    libelle: Optional[str] = None

class PredictionRequest(BaseModel):
    debit: Dict
    facture: Dict
    credit: Dict
    depense: Dict
    model_name: Optional[str] = "lightgbm"

class HealthCheck(BaseModel):
    status: str
    models_loaded: Dict[str, bool]
    timestamp: str

# Liste des caractéristiques attendues
FEATURES = [
    "montant_operation", "montant_banque", "presence_num_ref",
    "similarite_cos", "delai_jours", "delai_absolu", "dans_fenetre_valide",
    "montant_ratio", "ecart_montant", "montant_exact_match",
    "type_operation_depense", "type_operation_facture",
    "categorie_abonnements", "categorie_assurances", "categorie_conseil",
    "categorie_divers", "categorie_divertissement", "categorie_déplacements",
    "categorie_formations", "categorie_fournitures_bureau",
    "categorie_frais_bancaires", "categorie_juridique_conformité",
    "categorie_licences_logicielles", "categorie_maintenance",
    "categorie_marketing", "categorie_outils_rh", "categorie_services_cloud",
    "categorie_services_publiques", "categorie_équipement"
]

# Configuration de l'application FastAPI
app = FastAPI(
    title="API Lettrage Comptable",
    description="API pour le lettrage automatique des écritures comptables",
    version="2.0",
    contact={
        "name": "Support",
        "email": "support@example.com"
    },
    license_info={
        "name": "MIT",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calcule la similarité cosinus entre deux textes"""
    if not text1 or not text2:
        return 0.0
    
    # Tokenization simple
    words1 = set(str(text1).lower().split())
    words2 = set(str(text2).lower().split())
    
    # Calcul de la similarité de Jaccard (simplifié)
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

def build_features(mouvement: Dict, operation: Dict) -> Dict[str, float]:
    """
    Transforme une paire (mouvement bancaire, opération comptable) en features pour le modèle.
    
    Args:
        mouvement: Dictionnaire contenant les informations du mouvement bancaire
        operation: Dictionnaire contenant les informations de l'opération comptable
        
    Returns:
        Dictionnaire des caractéristiques pour le modèle
    """
    try:
        montant_mouv = float(mouvement.get("montant", 0))
        montant_op = float(operation.get("montant", 0))
        
        # Extraction des dates
        date_mouv = mouvement.get("date")
        date_op = operation.get("date")
        
        # Calcul du délai entre les dates
        try:
            d1 = datetime.strptime(str(date_mouv), "%Y-%m-%d")
            d2 = datetime.strptime(str(date_op), "%Y-%m-%d")
            delai = (d1 - d2).days
        except (ValueError, TypeError):
            delai = 0
        
        # Calcul de la similarité textuelle
        similarite = calculate_text_similarity(
            str(mouvement.get("libelle", "")),
            str(operation.get("libelle", ""))
        )
        
        # Construction des caractéristiques
        features = {
            "montant_operation": montant_op,
            "montant_banque": montant_mouv,
            "presence_num_ref": 1 if mouvement.get("reference") else 0,
            "similarite_cos": similarite,
            "delai_jours": delai,
            "delai_absolu": abs(delai),
            "dans_fenetre_valide": 1 if abs(delai) <= 30 else 0,
            "montant_ratio": montant_op / montant_mouv if montant_mouv != 0 else 0,
            "ecart_montant": abs(montant_op - montant_mouv),
            "montant_exact_match": 1 if montant_op == montant_mouv else 0,
            "type_operation_depense": 1 if operation.get("type") == "depense" else 0,
            "type_operation_facture": 1 if operation.get("type") == "facture" else 0,
        }
        
        # Initialisation des catégories à 0
        for cat in FEATURES:
            if cat.startswith("categorie_"):
                features[cat] = 0.0
        
        # Activation de la catégorie correspondante
        if operation.get("categorie"):
            cat = f"categorie_{operation['categorie']}"
            if cat in features:
                features[cat] = 1.0
        
        return features
        
    except Exception as e:
        logger.error(f"Erreur dans build_features: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Erreur lors de la construction des caractéristiques: {str(e)}"
        )

@app.get("/", include_in_schema=False)
async def root():
    """Redirige vers la documentation de l'API"""
    return {"message": "Bienvenue sur l'API de lettrage comptable. Accédez à /docs pour la documentation."}

@app.get("/health", response_model=HealthCheck, tags=["Système"])
async def health_check():
    """Vérifie l'état de santé de l'API et des modèles"""
    return {
        "status": "healthy" if models else "degraded",
        "models_loaded": {name: name in models for name in MODEL_FILES},
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/models", tags=["Modèles"])
async def list_models():
    """Liste les modèles disponibles"""
    return {
        "available_models": list(models.keys()),
        "default_model": "lightgbm"
    }


@app.post("/predict", tags=["Prédiction"])
async def predict(request: PredictionRequest):
    """
    Effectue une prédiction de lettrage comptable.
    
    Args:
        request: Objet contenant les données des opérations à analyser
        
    Returns:
        Dictionnaire contenant les prédictions et les caractéristiques utilisées
    """
    try:
        # Vérifier que le modèle demandé est disponible
        model_name = request.model_name.lower()
        if model_name not in models:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Modèle non disponible. Modèles disponibles: {', '.join(models.keys())}"
            )
        
        model = models[model_name]
        
        # Valider les données d'entrée
        debit = request.debit
        facture = request.facture
        credit = request.credit
        depense = request.depense
        
        # Construire les caractéristiques pour chaque paire
        row1 = build_features(debit, facture)   # Débit vs Facture
        row2 = build_features(credit, depense)  # Crédit vs Dépense
        
        # Créer le DataFrame d'entrée
        df = pd.DataFrame([row1, row2])[FEATURES]
        
        # Effectuer la prédiction
        try:
            preds = model.predict(df)
            probas = model.predict_proba(df) if hasattr(model, 'predict_proba') else [[0, 0], [0, 0]]
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Erreur lors de l'exécution du modèle: {str(e)}"
            )
        
        # Formater la réponse
        return {
            "model_used": model_name,
            "predictions": [
                {
                    "pair": "debit-facture",
                    "features": row1,
                    "prediction": int(preds[0]),
                    "confidence": float(probas[0][1]) if hasattr(model, 'predict_proba') else 0.0
                },
                {
                    "pair": "credit-depense",
                    "features": row2,
                    "prediction": int(preds[1]),
                    "confidence": float(probas[1][1]) if hasattr(model, 'predict_proba') else 0.0
                }
            ]
        }
        
    except HTTPException:
        # On laisse passer les exceptions HTTP déjà formatées
        raise
        
    except Exception as e:
        logger.error(f"Erreur inattendue: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Une erreur inattendue s'est produite: {str(e)}"
        )

# Gestion des erreurs personnalisée
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Erreur non gérée: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Une erreur interne du serveur est survenue"}
    )

# Point d'entrée pour l'exécution directe
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
