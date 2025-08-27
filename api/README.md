# API de Lettrage Comptable

API REST pour le lettrage automatique d'écritures comptables.

## Fonctionnalités

- Gestion des écritures comptables (débits et crédits)
- Proposition automatique de lettrages
- Exécution des lettrages
- Historique des opérations

## Installation

1. Cloner le dépôt
2. Installer les dépendances :
   ```
   pip install -r requirements.txt
   ```
3. Lancer le serveur :
   ```
   uvicorn api:app --reload
   ```

## Endpoints

### Écritures comptables

- `POST /ecritures/ajouter` : Ajouter des écritures
- `GET /ecritures` : Lister les écritures
  - Paramètres :
    - `lettree` (bool) : Filtrer par statut de lettrage
    - `type_operation` (string) : 'debit' ou 'credit'
- `GET /ecritures/{id}` : Obtenir une écriture

### Lettrages

- `POST /lettrages/proposer` : Proposer des paires à lettrer
  - Paramètre : `seuil` (float, 0-1) : Seuil de confiance
- `POST /lettrages/effectuer` : Effectuer un lettrage
- `GET /lettrages` : Lister tous les lettrages
- `GET /lettrages/{id}` : Obtenir un lettrage

## Exemple d'utilisation

### Ajout d'écritures

```bash
curl -X POST "http://localhost:8000/ecritures/ajouter" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "id": "e1",
      "date_operation": "2023-01-15",
      "libelle": "Achat fournitures",
      "montant": 120.50,
      "type_operation": "debit",
      "reference": "FAC-2023-001"
    },
    {
      "id": "e2",
      "date_operation": "2023-01-15",
      "libelle": "Paiement fournitures",
      "montant": 120.50,
      "type_operation": "credit",
      "reference": "FAC-2023-001"
    }
  ]'
```

### Proposition de lettrages

```bash
curl -X POST "http://localhost:8000/lettrages/proposer?seuil=0.7"
```

### Exécution d'un lettrage

```bash
curl -X POST "http://localhost:8000/lettrages/effectuer" \
  -H "Content-Type: application/json" \
  -d '{
    "id_ecriture_debit": "e1",
    "id_ecriture_credit": "e2",
    "montant_lettre": 120.50,
    "commentaire": "Lettrage automatique"
  }'
```

## Structure des données

### Écriture comptable

```typescript
{
  id: string
  date_operation: string  // Format ISO 8601
  libelle: string
  montant: number
  type_operation: 'debit' | 'credit'
  reference?: string
  compte_comptable?: string
  tiers?: string
  piece_jointe?: string
  lettree: boolean
  id_lettrage?: string
}
```

### Proposition de lettrage

```typescript
{
  id_ecriture_debit: string
  id_ecriture_credit: string
  score_confiance: number  // 0-1
  motif: string
  details: {
    montant_debit: number
    montant_credit: number
    reference_debit?: string
    reference_credit?: string
    libelle_debit: string
    libelle_credit: string
  }
}
```

### Opération de lettrage

```typescript
{
  id_ecriture_debit: string
  id_ecriture_credit: string
  montant_lettre: number
  commentaire?: string
}
```

## Développement

### Tests

```bash
pytest
```

### Formatage du code

```bash
black .
```

## Licence

MIT
