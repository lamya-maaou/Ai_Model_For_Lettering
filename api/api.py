from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import joblib

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chemin du modèle LightGBM
MODEL_PATH = "C:/Ai_Model_For_Lettering/api/LightGBM_model.pkl"
BASE_DIR = Path(__file__).parent  # répertoire contenant api.py
# Chargement du modèle LightGBM
try:
    # if MODEL_PATH.exists():
        lightgbm_model = joblib.load(MODEL_PATH)
        logger.info("Modèle LightGBM chargé avec succès")
    # else:
    #     raise FileNotFoundError(f"Fichier du modèle non trouvé: {MODEL_PATH}")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
    raise RuntimeError("Le modèle LightGBM n'a pas pu être chargé")

# Initialisation du modèle SentenceTransformer pour la similarité textuelle
try:
    sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    logger.info("Modèle SentenceTransformer chargé avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement de SentenceTransformer: {str(e)}")
    raise RuntimeError("Le modèle SentenceTransformer n'a pas pu être chargé")

# Mappings des colonnes
invoice_columns_mapping = {
    "invoice_id" : "id_operation",
    'INVOICE_DATE': 'date_operation',
    'AMOUNT_TO_PAY': 'montant_operation',
    'TITRE': 'titre_operation',
    'INVOICE_NUMBER': 'numero_reference',
    'LABEL': 'description'
}

expense_columns_mapping = {
    'expense_id': 'id_operation',
    'expense_date': 'date_operation',
    'amount': 'montant_operation',
    'title': 'titre_operation',
    'expense_number': 'numero_reference',
    'category': 'categorie',
    'label': 'description'
}

bank_columns_mapping = {
    'statement_id': 'id_releve',
    'statement_date': 'date_releve',
    'operation_label': 'libelle_operation',
    'additional_label': 'libelle_additionnel',
    'debit': 'debit',
    'credit': 'credit',
    'comments': 'commentaires'
}

# Définition des modèles Pydantic
class BankDebitData(BaseModel):
    statement_id: int
    statement_date: datetime
    operation_label: Optional[str] = None
    additional_label: Optional[str] = None
    debit: Optional[float] = None
    comments: Optional[str] = None

# Classe pour les crédits
class BankCreditData(BaseModel):
    statement_id: int
    statement_date: datetime
    operation_label: Optional[str] = None
    additional_label: Optional[str] = None
    credit: Optional[float] = None
    comments: Optional[str] = None

class InvoiceData(BaseModel):
    invoice_id : int
    INVOICE_DATE: Optional[str] = None
    TOTAL_HT: Optional[float] = None
    MONTANT_TVA: Optional[float] = None
    AMOUNT_TTC: Optional[float] = None
    RAS_5P: Optional[float] = None
    RAS_TVA: Optional[float] = None
    AMOUNT_TO_PAY: float
    PU: Optional[float] = None
    QUANTITY: Optional[float] = None
    EXPECTED_PAYMENT_DATE: Optional[str] = None
    LABEL: Optional[str] = None
    TITRE: Optional[str] = None
    PO: Optional[str] = None
    INVOICE_NUMBER : str
    INVOICE_YEAR: Optional[int] = None

class ExpenseData(BaseModel):
    expense_id: int = None
    title: str = None
    amount: float
    label: Optional[str] = None
    comments: Optional[str] = None
    expense_date: Optional[str] = None
    type: Optional[str] = None
    category: Optional[str] = None
    expense_number: Optional[str] = None
    status: Optional[str] = None

class PredictionRequest(BaseModel):
    debit: List[BankDebitData]
    credit: List[BankCreditData]
    facture: List[InvoiceData]
    depense: List[ExpenseData]
    confidence_threshold: Optional[float] = 0.5

class MatchResult(BaseModel):
    bank_id: int
    operation_id: int
    match_type: str  # "debit-facture" ou "credit-depense"
    confidence: float
    prediction: int

class PredictionSummary(BaseModel):
    total_debit_facture: int
    total_credit_depense: int
    total_matches: int

class PredictionResponse(BaseModel):
    debit_facture: List[MatchResult]
    credit_depense: List[MatchResult]
    summary: PredictionSummary

# Liste des caractéristiques attendues par le modèle
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
    title="API Lettrage Comptable - LightGBM + SentenceTransformer",
    description="API pour le lettrage automatique des écritures comptables utilisant LightGBM et SentenceTransformer",
    version="2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_text(text) :
    # 1. Mettre en minuscules
    text = text.lower()
    # 2. Supprimer les caractères non imprimables
    text = re.sub(r"[^\x20-\x7E\u00C0-\u017F]", "", text)
    # 3. Normaliser les espaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def calculate_cosine_similarity(text1, text2):
    """Calcule la similarité cosinus entre deux textes en utilisant SentenceTransformer"""
    try:
        if not text1 or not text2:
            return 0.0
        
        # Préprocessing des textes
        processed_text1 = preprocess_text(text1)
        processed_text2 = preprocess_text(text2)
        
        if not processed_text1 or not processed_text2:
            return 0.0
        
        # Génération des embeddings avec SentenceTransformer
        embeddings = sentence_model.encode([processed_text1, processed_text2])
        
        # Calcul de la similarité cosinus
        similarity_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
        return float(similarity_matrix[0][0])
        
    except Exception as e:
        logger.warning(f"Erreur dans le calcul de similarité: {str(e)}")
        return 0.0

def apply_column_mapping(data_list, mapping_dict, operation_type):
    """Applique le mapping des colonnes et ajoute le type d'opération"""
    mapped_data = []
    for item in data_list:
        item_dict = item.dict() if hasattr(item, 'dict') else item
        mapped_item = {}
        
        # Application du mapping
        for original_col, mapped_col in mapping_dict.items():
            if original_col in item_dict:
                mapped_item[mapped_col] = item_dict[original_col]
        
        # Ajout du type d'opération
        if operation_type == "facture":
            mapped_item['type_operation'] = 'facture'
            mapped_item['categorie'] = "divers"
        elif operation_type == "depense":
            mapped_item['type_operation'] = 'depense'
            # La catégorie est déjà mappée si elle existe
        
        mapped_data.append(mapped_item)
    
    return mapped_data

import joblib

def build_features(bank_op, accounting_op, ohe_path_dict,sentence_model=None):
    """
    Construit les features pour une paire banque-comptabilité
    en utilisant les encoders sauvegardés et le pipeline complet.

    Args:
        bank_op: dict avec les colonnes de la banque
        accounting_op: dict avec les colonnes de l'opération comptable
        ohe_path_dict: dict contenant le chemin des encoders sauvegardés {'categorie': path, 'type_operation': path}
        sentence_model: modèle SentenceTransformer pour similarité texte

    Returns:
        dict: features prêtes à passer au modèle LightGBM
    """
    # 1Texte concaténé
    cols_tx = ['libelle_operation', 'libelle_additionnel', 'commentaires']
    cols_op = ['titre_operation', 'description', 'numero_reference']

    bank_text = ' '.join([str(bank_op.get(c, '')) for c in cols_tx])
    operation_text = ' '.join([str(accounting_op.get(c, '')) for c in cols_op])

    # 2Similarité cosinus
    sim_cos = 0.0
    if sentence_model:
        emb_bank = sentence_model.encode([bank_text])[0]
        emb_op = sentence_model.encode([operation_text])[0]
        sim_cos = float(cosine_similarity([emb_bank], [emb_op])[0][0])

    # 3️Montants
    montant_banque = abs(float(bank_op.get('debit', 0) or bank_op.get('credit', 0) or 0))
    montant_operation = abs(float(accounting_op.get('montant_operation', 0)))
    ecart_montant = abs(montant_operation - montant_banque)

    # Dates
    date_banque = bank_op.get('date_releve')
    date_op = accounting_op.get('date_operation')
    delai = 0
    if date_banque and date_op:
        try:
            d1 = pd.to_datetime(date_banque)
            d2 = pd.to_datetime(date_op)
            delai = (d1 - d2).days
        except:
            delai = 0

    # 5️⃣ Features de base
    features = {
        "montant_operation": montant_operation,
        "montant_banque": montant_banque,
        "presence_num_ref":  accounting_op.get("numero_reference").lower() in bank_text.lower(),
        "similarite_cos": sim_cos,
        "delai_jours": delai,
        "delai_absolu": abs(delai),
        "dans_fenetre_valide": 1 if (accounting_op.get('type_operation') == 'facture' and abs(delai) <= 60) or accounting_op.get('type_operation') == 'depense' else 0,
        "montant_ratio": montant_operation / montant_banque if montant_banque != 0 else 0,
        "ecart_montant": ecart_montant,
        "montant_exact_match": 1 if ecart_montant == 0  else 0
    }

    # 6️⃣ Features catégorielles avec encoders déjà sauvegardés
    all_categories = [
        "categorie_abonnements","categorie_assurances","categorie_conseil","categorie_divers",
        "categorie_divertissement","categorie_déplacements","categorie_formations","categorie_fournitures_bureau",
        "categorie_frais_bancaires","categorie_juridique_conformité","categorie_licences_logicielles","categorie_maintenance",
        "categorie_marketing","categorie_outils_rh","categorie_services_cloud","categorie_services_publiques",
        "categorie_équipement"
    ]
    # Initialisation à 0
    for cat in all_categories:
        features[cat] = 0.0

    # Categorie
    if ohe_path_dict and 'categorie' in ohe_path_dict and accounting_op.get('categorie'):
        ohe_categorie = joblib.load(ohe_path_dict["categorie"])
        cat_val = str(accounting_op['categorie'])
        encoded = ohe_categorie.transform([[cat_val]])[0]
        for i, cat_name in enumerate(ohe_categorie.categories_[0]):
            feature_name = f"categorie_{cat_name}"
            features[feature_name] = float(encoded[i])

    # type_operation si besoin
    if ohe_path_dict and 'type_operation' in ohe_path_dict and accounting_op.get('type_operation'):
        ohe_type = joblib.load(ohe_path_dict["type_operation"])
        type_val = str(accounting_op['type_operation'])
        encoded = ohe_type.transform([[type_val]])[0]
        for i, t_name in enumerate(ohe_type.categories_[0]):
            feature_name = f"type_operation_{t_name}"
            features[feature_name] = float(encoded[i])
    print(features)
    return features



@app.get("/", include_in_schema=False)
async def root():
    return {"message": "API de lettrage comptable avec LightGBM. Accédez à /docs pour la documentation."}

@app.get("/health", tags=["Système"])
async def health_check():
    return {
        "status": "healthy",
        "lightgbm_model_loaded": True,
        "sentence_transformer_loaded": True,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
async def predict(request: PredictionRequest):
    """
    Effectue le lettrage automatique en séparant clairement :
      - Débit <-> Factures
      - Crédit <-> Dépenses
    """
    try:
        debit_facture_matches = []
        credit_depense_matches = []

        # Mapping
        debit_mapped = apply_column_mapping(request.debit, bank_columns_mapping, "bank")
        credit_mapped = apply_column_mapping(request.credit, bank_columns_mapping, "bank")
        facture_mapped = apply_column_mapping(request.facture, invoice_columns_mapping, "facture")
        depense_mapped = apply_column_mapping(request.depense, expense_columns_mapping, "depense")

        # ---------- Débit <-> Factures ----------
        used_factures = set()
        # ohe_path_dict = {"categorie" :"C:/Ai_Model_For_Lettering/api/categorie_encoder.pkl","type_operation" :"C:/Ai_Model_For_Lettering/api/type_operation_encoder.pkl"}
        ohe_path_dict = {
    "categorie": BASE_DIR / "categorie_encoder.pkl",
    "type_operation": BASE_DIR / "type_operation_encoder.pkl"
}
        scaler = joblib.load(BASE_DIR /"scaler.pkl")
        for debit_op in debit_mapped:
            debit_id = debit_op.get('id_releve', 'unknown_debit')
            candidate_pairs = []
            candidate_features = []

            for facture_op in facture_mapped:
                facture_id = facture_op.get('id_operation', 'unknown_facture')
                if facture_id in used_factures:
                    continue
                features = build_features(debit_op, facture_op,ohe_path_dict,sentence_model)
                candidate_features.append(features)
                candidate_pairs.append({
                    'bank_id': debit_id,
                    'id_operation': facture_id,
                    'match_type': 'debit-facture'
                })

            if candidate_pairs:
                df_features = pd.DataFrame(candidate_features)[FEATURES]
                print(df_features)
                df_features = scaler.transform(df_features)
                predictions = lightgbm_model.predict(df_features)
                print(predictions)
                probabilities = lightgbm_model.predict_proba(df_features)[:, 1]

                best_idx, best_prob = None, -1
                for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                    if pred == 1 and  prob > best_prob:
                        best_prob = prob
                        best_idx = i

                if best_idx is not None:
                    pair = candidate_pairs[best_idx]
                    debit_facture_matches.append(MatchResult(
                        bank_id=pair['bank_id'],
                        operation_id=pair['id_operation'],
                        match_type=pair['match_type'],
                        confidence=float(best_prob),
                        prediction=1
                    ))
                    used_factures.add(pair['id_operation'])

        # ---------- Crédit <-> Dépenses ----------
        used_depenses = set()
        for credit_op in credit_mapped:
            credit_id = credit_op.get('id_releve', 'unknown_credit')
            candidate_pairs = []
            candidate_features = []

            for depense_op in depense_mapped:
                depense_id = depense_op.get('id_operation', 'unknown_depense')
                if depense_id in used_depenses:
                    continue

                features = build_features(credit_op, depense_op,ohe_path_dict,sentence_model)
                candidate_features.append(features)
                candidate_pairs.append({
                    'bank_id': credit_id,
                    'id_operation': depense_id,
                    'match_type': 'credit-depense'
                })

            if candidate_pairs:
                df_features = pd.DataFrame(candidate_features)[FEATURES]
                df_features = scaler.transform(df_features)
                predictions = lightgbm_model.predict(df_features)
                print(predictions)
                probabilities = lightgbm_model.predict_proba(df_features)[:, 1]

                best_idx, best_prob = None, -1
                for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                    if pred == 1  and prob > best_prob:
                        best_prob = prob
                        best_idx = i

                if best_idx is not None:
                    pair = candidate_pairs[best_idx]
                    credit_depense_matches.append(MatchResult(
                        bank_id=pair['bank_id'],
                        operation_id=pair['id_operation'],
                        match_type=pair['match_type'],
                        confidence=float(best_prob),
                        prediction=1
                    ))
                    used_depenses.add(pair['id_operation'])

        # Tri par confiance décroissante dans chaque liste
        debit_facture_matches.sort(key=lambda x: x.confidence, reverse=True)
        credit_depense_matches.sort(key=lambda x: x.confidence, reverse=True)

        return PredictionResponse(
    debit_facture=debit_facture_matches,
    credit_depense=credit_depense_matches,
    summary=PredictionSummary(
        total_debit_facture=len(debit_facture_matches),
        total_credit_depense=len(credit_depense_matches),
        total_matches=len(debit_facture_matches) + len(credit_depense_matches)
    )
)

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )

# Gestion des erreurs
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)