import os
import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta
from faker.providers import BaseProvider

# Création d'un provider custom pour les numéros de facture français
class InvoiceProvider(BaseProvider):
    def invoice_number(self, year):
        return f"FAC-{year}-{self.bothify(text='#####')}"

# Configuration
fake = Faker('fr_FR')
fake.add_provider(InvoiceProvider)
os.makedirs('invoices_output', exist_ok=True)

# Paramètres
NUM_INVOICES = 80000
CLIENT_IDS = list(range(1, 101))
CLIENT_TYPES = {cid: random.choice(['PUBLIC', 'PRIVE']) for cid in CLIENT_IDS}
STATUS_DISTRIBUTION = {
    'DRAFT': 0.1,
    'SENT': 0.3,
    'PAID': 0.4,
    'CANCELLED': 0.05,
    'OVERDUE': 0.15
}

# Fonction principale de génération d'une facture
def generate_invoice_base_data():
    invoice_date = fake.date_between(start_date='-2y', end_date='today')
    client_id = random.choice(CLIENT_IDS)
    client_type = CLIENT_TYPES[client_id]
    
    quantity = random.randint(1, 20)
    pu = round(random.uniform(50, 2000), 2)
    total_ht = round(pu * quantity, 2)
    
    if client_type == "PUBLIC":
        montant_tva = round(total_ht * 0.2, 2)
        amount_ttc = round(total_ht * 1.2, 2)
        ras_5p = round(total_ht * 0.05, 2)
        ras_tva = round(total_ht * 0.2 * 0.75, 2)
        amount_to_pay = round(amount_ttc - ras_5p - ras_tva, 2)
    else:
        montant_tva = round(total_ht * 0.2, 2)
        amount_ttc = round(total_ht * 1.2, 2)
        ras_5p = 0
        ras_tva = 0
        amount_to_pay = amount_ttc

    label = random.choice([
        "Développement logiciel", "Consulting IT", "Maintenance SaaS",
        "Formation technique", "Licence logicielle", "Hébergement cloud"
    ])
    titre = f"Facture - {label.split()[0]}"

    return {
        'INVOICE_DATE': invoice_date,
        'CLIENT_ID': client_id,
        'CLIENT_TYPE': client_type,
        'TOTAL_HT': total_ht,
        'MONTANT_TVA': montant_tva,
        'AMOUNT_TTC': amount_ttc,
        'RAS_5P': ras_5p,
        'RAS_TVA': ras_tva,
        'AMOUNT_TO_PAY': amount_to_pay,
        'PU': pu,
        'QUANTITY': quantity,
        'ELECTRONIC_DATE': invoice_date + timedelta(days=random.randint(0, 2)),
        'PHYSICAL_DATE': invoice_date + timedelta(days=random.randint(1, 5)) if random.random() > 0.3 else None,
        'EXPECTED_PAYMENT_DATE': invoice_date + timedelta(days=random.randint(30, 90)),
        'LABEL': label,
        'TITRE': titre,
        'PO': fake.bothify(text="PO-#####-??"),
        'INVOICE_YEAR': invoice_date.year
    }

def generate_all_invoices(num_invoices):
    invoices = []
    for i in range(num_invoices):
        base_data = generate_invoice_base_data()
        status = random.choices(
            list(STATUS_DISTRIBUTION.keys()),
            weights=list(STATUS_DISTRIBUTION.values())
        )[0]
        payment_date = None
        if status == 'PAID':
            payment_date = base_data['EXPECTED_PAYMENT_DATE'] + timedelta(days=random.randint(-15, 15))
        invoice_data = {
            'INVOICE_ID': i + 1,
            **base_data,
            'STATUS': status,
            'PAYMENT_DATE': payment_date,
            'INVOICE_NUMBER': fake.invoice_number(base_data['INVOICE_YEAR'])
        }
        invoices.append(invoice_data)
    return pd.DataFrame(invoices)

def split_invoices(df_invoices):
    paid_mask = df_invoices['STATUS'] == 'PAID'
    df_paid = df_invoices[paid_mask].copy()
    matched_count = int(len(df_paid) * 0.4)
    unmatched_count = int(len(df_paid) * 0.4)
    grouped_count = int(len(df_paid) * 0.1)
    matched = df_paid.iloc[:matched_count].copy()
    unmatched = df_paid.iloc[matched_count:matched_count+unmatched_count].copy()
    grouped = df_paid.iloc[matched_count+unmatched_count:matched_count+unmatched_count+grouped_count].copy()
    partial = df_paid.iloc[matched_count+unmatched_count+grouped_count:].copy()
    return {
        'matched': matched,
        'partial': partial,
        'grouped': grouped,
        'unmatched': unmatched,
        'non_paid': df_invoices[~paid_mask]
    }

def generate_bank_statements(invoice_splits):
    statements = []
    statement_id = 1
    for _, row in invoice_splits['matched'].iterrows():
        statements.append({
            'STATEMENT_ID': statement_id,
            'STATEMENT_DATE': row['PAYMENT_DATE'],
            'OPERATION_LABEL': "VIREMENT RECU",
            'ADDITIONAL_LABEL': f"REF: {row['INVOICE_NUMBER']} - {row['LABEL']}",
            'DEBIT': None,
            'CREDIT': row['AMOUNT_TO_PAY'],
            'COMMENTS': f"Paiement facture {row['INVOICE_NUMBER']}",
            'RELATED_INVOICE_ID': row['INVOICE_ID'],
            'CREATED_AT': datetime.now(),
            'SOURCE_FILENAME': f"releve_{row['PAYMENT_DATE'].strftime('%Y%m')}.csv",
            'FILE_BLOB': None,
            'MIME_TYPE': 'text/csv',
            'RELATED_EXPENSE_ID': None,
            'VALUE_DATE': row['PAYMENT_DATE'],
            'MATCH_TYPE': 'MATCHED'
        })
        statement_id += 1

    for _, row in invoice_splits['partial'].iterrows():
        partial_payments = random.randint(2, 4)
        remaining = row['AMOUNT_TO_PAY']
        for i in range(partial_payments):
            amount = round(remaining / (partial_payments - i), 2) if i != partial_payments - 1 else remaining
            remaining -= amount
            payment_date = row['PAYMENT_DATE'] + timedelta(days=random.randint(0, 30))
            statements.append({
                'STATEMENT_ID': statement_id,
                'STATEMENT_DATE': payment_date,
                'OPERATION_LABEL': "VIREMENT RECU",
                'ADDITIONAL_LABEL': f"PAIEMENT PARTIEL {i+1}/{partial_payments} - REF: {row['INVOICE_NUMBER']}",
                'DEBIT': None,
                'CREDIT': amount,
                'COMMENTS': f"Paiement partiel {i+1} facture {row['INVOICE_NUMBER']}",
                'RELATED_INVOICE_ID': row['INVOICE_ID'],
                'CREATED_AT': datetime.now(),
                'SOURCE_FILENAME': f"releve_{payment_date.strftime('%Y%m')}.csv",
                'FILE_BLOB': None,
                'MIME_TYPE': 'text/csv',
                'RELATED_EXPENSE_ID': None,
                'VALUE_DATE': payment_date,
                'MATCH_TYPE': 'PARTIAL'
            })
            statement_id += 1

    grouped_invoices = invoice_splits['grouped']
    chunks = [grouped_invoices[i:i+3] for i in range(0, len(grouped_invoices), 3)]
    for chunk in chunks:
        total = chunk['AMOUNT_TO_PAY'].sum()
        refs = ", ".join(chunk['INVOICE_NUMBER'])
        invoice_ids = ",".join([str(x) for x in chunk['INVOICE_ID']])
        payment_date = chunk.iloc[0]['PAYMENT_DATE'] + timedelta(days=random.randint(0, 5))
        statements.append({
            'STATEMENT_ID': statement_id,
            'STATEMENT_DATE': payment_date,
            'OPERATION_LABEL': "VIREMENT RECU",
            'ADDITIONAL_LABEL': f"PAIEMENT GROUPE {len(chunk)} FACTURES",
            'DEBIT': None,
            'CREDIT': total,
            'COMMENTS': f"Paiement groupé factures: {refs}",
            'RELATED_INVOICE_ID': chunk.iloc[0]['INVOICE_ID'],
            'CREATED_AT': datetime.now(),
            'SOURCE_FILENAME': f"releve_{payment_date.strftime('%Y%m')}.csv",
            'FILE_BLOB': None,
            'MIME_TYPE': 'text/csv',
            'RELATED_EXPENSE_ID': None,
            'VALUE_DATE': payment_date,
            'MATCH_TYPE': 'GROUPED',
            'GROUPED_INVOICE_IDS': invoice_ids
        })
        statement_id += 1

    for _, row in invoice_splits['unmatched'].iterrows():
        statements.append({
            'STATEMENT_ID': statement_id,
            'STATEMENT_DATE': row['PAYMENT_DATE'],
            'OPERATION_LABEL': "VIREMENT RECU",
            'ADDITIONAL_LABEL': fake.company().upper(),
            'DEBIT': None,
            'CREDIT': row['AMOUNT_TO_PAY'],
            'COMMENTS': f"Virement sans référence claire - {fake.bothify(text='????#####')}",
            'RELATED_INVOICE_ID': row['INVOICE_ID'],
            'CREATED_AT': datetime.now(),
            'SOURCE_FILENAME': f"releve_{row['PAYMENT_DATE'].strftime('%Y%m')}.csv",
            'FILE_BLOB': None,
            'MIME_TYPE': 'text/csv',
            'RELATED_EXPENSE_ID': None,
            'VALUE_DATE': row['PAYMENT_DATE'],
            'MATCH_TYPE': 'UNMATCHED',
            'ACTUAL_INVOICE_ID': row['INVOICE_ID']
        })
        statement_id += 1

    for _ in range(200):
        expense_date = fake.date_between(start_date='-2y', end_date='today')
        statements.append({
            'STATEMENT_ID': statement_id,
            'STATEMENT_DATE': expense_date,
            'OPERATION_LABEL': random.choice(['PRELEVEMENT', 'VIREMENT EMIS', 'CHEQUE']),
            'ADDITIONAL_LABEL': random.choice([
                'ELECTRICITE', 'TELEPHONIE', 'FOURNITURES BUREAU', 
                'SALAIRES', 'CHARGES SOCIALES', 'LOYER'
            ]),
            'DEBIT': round(random.uniform(100, 5000), 2),
            'CREDIT': None,
            'COMMENTS': fake.sentence(nb_words=6),
            'RELATED_INVOICE_ID': None,
            'CREATED_AT': datetime.now(),
            'SOURCE_FILENAME': f"releve_{expense_date.strftime('%Y%m')}.csv",
            'FILE_BLOB': None,
            'MIME_TYPE': 'text/csv',
            'RELATED_EXPENSE_ID': random.randint(1, 50),
            'VALUE_DATE': expense_date,
            'MATCH_TYPE': 'EXPENSE'
        })
        statement_id += 1

    return pd.DataFrame(statements)

def save_datasets(invoice_splits, bank_statements):
    all_invoices = pd.concat([
        invoice_splits['matched'],
        invoice_splits['partial'],
        invoice_splits['grouped'],
        invoice_splits['unmatched'],
        invoice_splits['non_paid']
    ], ignore_index=True)
    all_invoices.to_csv('invoices_output/all_invoices.csv', index=False)
    invoice_splits['matched'].to_csv('invoices_output/invoices_matched.csv', index=False)
    invoice_splits['partial'].to_csv('invoices_output/invoices_partial_payments.csv', index=False)
    invoice_splits['grouped'].to_csv('invoices_output/invoices_grouped_payments.csv', index=False)
    invoice_splits['unmatched'].to_csv('invoices_output/invoices_unmatched.csv', index=False)
    invoice_splits['non_paid'].to_csv('invoices_output/invoices_non_paid.csv', index=False)
    bank_statements.to_csv('invoices_output/bank_statements_all.csv', index=False)
    for match_type in bank_statements['MATCH_TYPE'].unique():
        subset = bank_statements[bank_statements['MATCH_TYPE'] == match_type]
        subset.to_csv(f'invoices_output/bank_statements_{match_type.lower()}.csv', index=False)

def main():
    print("Génération des factures de base...")
    df_invoices = generate_all_invoices(NUM_INVOICES)
    
    print("Découpage des factures en catégories...")
    invoice_splits = split_invoices(df_invoices)
    
    print("Génération des relevés bancaires...")
    bank_statements = generate_bank_statements(invoice_splits)
    
    print("Sauvegarde des fichiers...")
    save_datasets(invoice_splits, bank_statements)
    
    print(f"""Génération terminée. Fichiers créés dans invoices_output/ :
    FACTURES :
    - all_invoices.csv
    - invoices_matched.csv
    - invoices_partial_payments.csv
    - invoices_grouped_payments.csv
    - invoices_unmatched.csv
    - invoices_non_paid.csv
    RELEVÉS BANCAIRES :
    - bank_statements_all.csv
    - par type : matched / partial / grouped / unmatched / expense
    """)

if __name__ == "__main__":
    main()
