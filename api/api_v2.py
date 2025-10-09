from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- Chemins ----------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "LightGBM_model.pkl"

# ---------------- Chargement des modèles ----------------
try:
    lightgbm_model = joblib.load(MODEL_PATH)
    logger.info("Modèle LightGBM chargé avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle LightGBM: {str(e)}")
    raise RuntimeError("Impossible de charger LightGBM")

# Load scaler if available
try:
    SCALER_PATH = BASE_DIR / "scaler.pkl"
    scaler = joblib.load(SCALER_PATH)
    logger.info("Scaler chargé avec succès")
except Exception as e:
    logger.warning(f"Scaler non trouvé, utilisation sans normalisation: {str(e)}")
    scaler = None

try:
    sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    logger.info("Modèle SentenceTransformer chargé avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement de SentenceTransformer: {str(e)}")
    raise RuntimeError("Impossible de charger SentenceTransformer")

# ---------------- Mappings ----------------
invoice_columns_mapping = {
    "invoice_id": "id_operation",
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

# ---------------- Modèles Pydantic ----------------
class BankDebitData(BaseModel):
    statement_id: int
    statement_date: datetime
    operation_label: Optional[str] = None
    additional_label: Optional[str] = None
    debit: Optional[float] = None
    comments: Optional[str] = None

class BankCreditData(BaseModel):
    statement_id: int
    statement_date: datetime
    operation_label: Optional[str] = None
    additional_label: Optional[str] = None
    credit: Optional[float] = None
    comments: Optional[str] = None

class InvoiceData(BaseModel):
    invoice_id: int
    INVOICE_DATE: Optional[str] = None
    AMOUNT_TO_PAY: float
    LABEL: Optional[str] = None
    TITRE: Optional[str] = None
    INVOICE_NUMBER: str

class ExpenseData(BaseModel):
    expense_id: int
    title: Optional[str] = None
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
    match_type: str
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

# ---------------- FastAPI ----------------
app = FastAPI(title="API Lettrage Comptable - LightGBM + SentenceTransformer",
              version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Utilitaires ----------------
def preprocess_text(text):
    text = text.lower() if text else ""
    text = re.sub(r"[^\x20-\x7E\u00C0-\u017F]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def safe_get(d, key, default=None):
    """Récupère la valeur en évitant AttributeError si ce n'est pas dict"""
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)

def calculate_cosine_similarity(text1, text2):
    try:
        if not text1 or not text2:
            return 0.0
        t1 = preprocess_text(str(text1))
        t2 = preprocess_text(str(text2))
        if not t1 or not t2:
            return 0.0
        emb = sentence_model.encode([t1, t2])
        return float(cosine_similarity([emb[0]], [emb[1]])[0][0])
    except Exception as e:
        logger.warning(f"Erreur calcul similarité cosinus: {str(e)}")
        return 0.0

def apply_column_mapping(data_list, mapping_dict, operation_type):
    mapped_data = []
    for item in data_list:
        item_dict = item.dict() if hasattr(item, "dict") else item
        mapped_item = {}
        for original_col, mapped_col in mapping_dict.items():
            mapped_item[mapped_col] = item_dict.get(original_col)
        if operation_type == "facture":
            mapped_item["type_operation"] = "facture"
            mapped_item["categorie"] = "divers"
        elif operation_type == "depense":
            mapped_item["type_operation"] = "depense"
        mapped_data.append(mapped_item)
    return mapped_data

# ---------------- Features Builder ----------------
FEATURES = [
    "montant_operation", "montant_banque", "presence_num_ref",
    "similarite_cos", "delai_jours", "delai_absolu", "dans_fenetre_valide",
    "montant_ratio", "ecart_montant", "montant_exact_match"
]

def build_features(bank_op, accounting_op):
    bank_text = " ".join([str(safe_get(bank_op, c, "")) for c in ['libelle_operation','libelle_additionnel','commentaires']])
    op_text = " ".join([str(safe_get(accounting_op, c, "")) for c in ['titre_operation','description','numero_reference']])
    sim_cos = calculate_cosine_similarity(bank_text, op_text)

    # Safe amount extraction with better error handling
    try:
        debit_val = safe_get(bank_op, 'debit', 0)
        credit_val = safe_get(bank_op, 'credit', 0)
        montant_banque = abs(float(debit_val or credit_val or 0))
    except (ValueError, TypeError):
        montant_banque = 0.0
    
    try:
        montant_operation = abs(float(safe_get(accounting_op, 'montant_operation', 0)))
    except (ValueError, TypeError):
        montant_operation = 0.0
    ecart_montant = abs(montant_operation - montant_banque)

    # Délais
    delai = 0
    date_banque = safe_get(bank_op,'date_releve')
    date_op = safe_get(accounting_op,'date_operation')
    if date_banque and date_op:
        try:
            d1 = pd.to_datetime(date_banque)
            d2 = pd.to_datetime(date_op)
            delai = (d1 - d2).days
        except:
            delai = 0

    features = {
        "montant_operation": montant_operation,
        "montant_banque": montant_banque,
        "presence_num_ref": 1 if str(safe_get(accounting_op,"numero_reference","")) in bank_text else 0,
        "similarite_cos": sim_cos,
        "delai_jours": delai,
        "delai_absolu": abs(delai),
        "dans_fenetre_valide": 1 if (safe_get(accounting_op,"type_operation")=="facture" and abs(delai)<=60) or (safe_get(accounting_op,"type_operation")=="depense" and abs(delai)<=30) else 0,
        "montant_ratio": montant_operation / montant_banque if montant_banque > 0 else 0,
        "ecart_montant": ecart_montant,
        "montant_exact_match": 1 if ecart_montant==0 else 0
    }
    return features

# ---------------- Endpoints ----------------
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "API Lettrage Comptable prête. Accéder à /docs pour la documentation."}

@app.get("/health", tags=["Système"])
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
async def predict(request: PredictionRequest):
    try:
        # Input validation
        if not request.debit and not request.credit:
            raise HTTPException(status_code=400, detail="Au moins une opération bancaire (debit ou credit) est requise")
        
        if not request.facture and not request.depense:
            raise HTTPException(status_code=400, detail="Au moins une opération comptable (facture ou depense) est requise")
        
        # Validate confidence threshold
        if request.confidence_threshold and (request.confidence_threshold < 0 or request.confidence_threshold > 1):
            raise HTTPException(status_code=400, detail="Le seuil de confiance doit être entre 0 et 1")
        debit_facture_matches = []
        credit_depense_matches = []

        debit_mapped = apply_column_mapping(request.debit, bank_columns_mapping, "bank")
        credit_mapped = apply_column_mapping(request.credit, bank_columns_mapping, "bank")
        facture_mapped = apply_column_mapping(request.facture, invoice_columns_mapping, "facture")
        depense_mapped = apply_column_mapping(request.depense, expense_columns_mapping, "depense")

        used_factures = set()
        used_depenses = set()

        confidence_threshold = request.confidence_threshold or 0.5  # you can put this line once, near top of function

        for debit_op in debit_mapped:
            debit_id = debit_op.get('id_releve', 'unknown_debit')
            candidate_pairs = []
            candidate_features = []

            for facture_op in facture_mapped:
                facture_id = facture_op.get('id_operation', 'unknown_facture')
                if facture_id in used_factures:
                    continue

                # Extract ML features
                features = build_features(debit_op, facture_op)
                candidate_features.append(features)
                candidate_pairs.append({
                    'bank_id': debit_id,
                    'id_operation': facture_id,
                    'match_type': 'debit-facture'
                })

            if not candidate_pairs:
                continue

            try:
                df_features = pd.DataFrame(candidate_features)[FEATURES]
                
                # Validate features
                if df_features.isnull().any().any():
                    logger.warning(f"Valeurs manquantes détectées pour debit {debit_id}, remplacement par 0")
                    df_features = df_features.fillna(0)
                
                # Apply scaling if scaler is available
                if scaler is not None:
                    df_features = scaler.transform(df_features)
                
                predictions = lightgbm_model.predict(df_features)
                probabilities = lightgbm_model.predict_proba(df_features)[:, 1]
            except Exception as e:
                logger.error(f"Erreur lors de la prédiction pour debit {debit_id}: {str(e)}")
                continue

            best_idx = np.argmax(probabilities)
            best_prob = probabilities[best_idx]
            best_pred = predictions[best_idx]

            # ✅ only match if model is confident enough
            if best_pred == 1 and best_prob >= confidence_threshold:
                pair = candidate_pairs[best_idx]
                debit_facture_matches.append(MatchResult(
                    bank_id=pair['bank_id'],
                    operation_id=pair['id_operation'],
                    match_type=pair['match_type'],
                    confidence=float(best_prob),
                    prediction=1
                ))
                used_factures.add(pair['id_operation'])

        # Credit-Depense matching logic
        for credit_op in credit_mapped:
            credit_id = credit_op.get('id_releve', 'unknown_credit')
            candidate_pairs = []
            candidate_features = []

            for depense_op in depense_mapped:
                depense_id = depense_op.get('id_operation', 'unknown_depense')
                if depense_id in used_depenses:
                    continue

                # Extract ML features
                features = build_features(credit_op, depense_op)
                candidate_features.append(features)
                candidate_pairs.append({
                    'bank_id': credit_id,
                    'id_operation': depense_id,
                    'match_type': 'credit-depense'
                })

            if not candidate_pairs:
                continue

            try:
                df_features = pd.DataFrame(candidate_features)[FEATURES]
                
                # Validate features
                if df_features.isnull().any().any():
                    logger.warning(f"Valeurs manquantes détectées pour credit {credit_id}, remplacement par 0")
                    df_features = df_features.fillna(0)
                
                # Apply scaling if scaler is available
                if scaler is not None:
                    df_features = scaler.transform(df_features)
                
                predictions = lightgbm_model.predict(df_features)
                probabilities = lightgbm_model.predict_proba(df_features)[:, 1]
            except Exception as e:
                logger.error(f"Erreur lors de la prédiction pour credit {credit_id}: {str(e)}")
                continue

            best_idx = np.argmax(probabilities)
            best_prob = probabilities[best_idx]
            best_pred = predictions[best_idx]

            # Only match if model is confident enough
            if best_pred == 1 and best_prob >= confidence_threshold:
                pair = candidate_pairs[best_idx]
                credit_depense_matches.append(MatchResult(
                    bank_id=pair['bank_id'],
                    operation_id=pair['id_operation'],
                    match_type=pair['match_type'],
                    confidence=float(best_prob),
                    prediction=1
                ))
                used_depenses.add(pair['id_operation'])

        '''# Simple matching (pour test)
        for d in debit_mapped:
            for f in facture_mapped:
                debit_facture_matches.append(MatchResult(
                    bank_id=safe_get(d,'id_releve',0),
                    operation_id=safe_get(f,'id_operation',0),
                    match_type="debit-facture",
                    confidence=1,
                    prediction=1
                ))
                used_factures.add(safe_get(f,'id_operation',0))
                break

        for c in credit_mapped:
            for dep in depense_mapped:
                credit_depense_matches.append(MatchResult(
                    bank_id=safe_get(c,'id_releve',0),
                    operation_id=safe_get(dep,'id_operation',0),
                    match_type="credit-depense",
                    confidence=1,
                    prediction=1
                ))
                used_depenses.add(safe_get(dep,'id_operation',0))
                break '''

        return PredictionResponse(
            debit_facture=debit_facture_matches,
            credit_depense=credit_depense_matches,
            summary=PredictionSummary(
                total_debit_facture=len(debit_facture_matches),
                total_credit_depense=len(credit_depense_matches),
                total_matches=len(debit_facture_matches)+len(credit_depense_matches)
            )
        )
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")
