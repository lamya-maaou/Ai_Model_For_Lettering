"""
Générateur de Dataset Comptable Synthétique - Compatible Oracle DB
================================================================

Génère un dataset réaliste pour tester des systèmes de lettrage automatique
ou entraîner des modèles IA de reconnaissance de paiement.

Adapté au schéma Oracle DB avec tables BANK_STATEMENT, INVOICES et EXPENSES.

"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import csv
import os
from typing import List, Dict, Tuple
import uuid

print("Script démarré !") 
fake = Faker('fr_FR')  # Locale français
random.seed(42)
np.random.seed(42)
Faker.seed(42)

class AccountingDatasetGenerator:
    """Générateur de dataset comptable synthétique compatible Oracle DB."""
    
    def __init__(self):
        self.clients = []
        self.invoices = []
        self.bank_statements = []
        self.expenses = []
        self.invoice_statuses = []
        
        # Paramètres de génération MAJ pour les nouveaux volumes
        self.nb_invoices = 5000
        self.nb_bank_statements = 8000  # Augmenté à 8000
        self.nb_expenses = 5000         # Nouveau paramètre spécifique
        self.nb_clients = 800
        
        # Templates de libellés bancaires réalistes
        self.operation_labels = {
            'payment': [
                "VIR {company}",
                "PRLV {company}",
                "CB {company}",
                "CHEQUE {company}",
                "VIR SEPA {company}",
                "PAIEMENT {company}",
                "VIREMENT {company}",
                "REGLEMENT {company}"
            ],
            'expense': [
                "CB CARREFOUR",
                "PRLV EDF PARIS",
                "CB STATION SERVICE",
                "PRLV SFR MOBILE",
                "CB AMAZON EU",
                "VIREMENT SALAIRE",
                "CB LECLERC",
                "PRLV ORANGE FRANCE",
                "CB FNAC",
                "LOYER BUREAUX"
            ],
            'orphan': [
                "FRAIS BANCAIRES",
                "AGIOS",
                "COMMISSION VIREMENT",
                "COTISATION CARTE",
                "FRAIS TENUE COMPTE",
                "VIR DIVERS",
                "REMBOURSEMENT",
                "INTERETS CREDITEURS",
                "PENALITES RETARD",
                "FRAIS CHANGE"
            ]
        }
        
        self.additional_labels = {
            'payment': [
                "FACT {invoice_number}",
                "REF {invoice_number}",
                "FACTURE {invoice_number}",
                "PAIEMENT FACTURE {invoice_number}",
                "REGLEMENT {invoice_number}",
                "N° {invoice_number}",
                "FACT N°{invoice_number}",
                "REF FACTURE {invoice_number}"
            ],
            'expense': [
                "ACHAT {date}",
                "FRAIS {date}",
                "DEPENSE {date}",
                "FOURNITURES",
                "SERVICES",
                "UTILITIES",
                "MAINTENANCE",
                "APPROVISIONNEMENT"
            ],
            'orphan': [
                "FRAIS MENSUELS",
                "COMMISSION",
                "PENALITE",
                "AJUSTEMENT",
                "CORRECTION",
                "REGULARISATION",
                "DIVERS",
                "AUTRE OPERATION"
            ]
        }
    
    def generate_invoice_statuses(self) -> List[Dict]:
        """Génère les statuts de facture."""
        statuses = [
            {'STATUS_CODE': 'DRAFT', 'DESCRIPTION': 'Brouillon'},
            {'STATUS_CODE': 'SENT', 'DESCRIPTION': 'Envoyée'},
            {'STATUS_CODE': 'PAID', 'DESCRIPTION': 'Payée'},
            {'STATUS_CODE': 'UNPAID', 'DESCRIPTION': 'Impayée'},
            {'STATUS_CODE': 'OVERDUE', 'DESCRIPTION': 'En retard'},
            {'STATUS_CODE': 'CANCELLED', 'DESCRIPTION': 'Annulée'},
            {'STATUS_CODE': 'PARTIAL', 'DESCRIPTION': 'Partiellement payée'}
        ]
        
        self.invoice_statuses = statuses
        return statuses
    
    def generate_clients(self) -> List[Dict]:
        """Génère la liste des clients."""
        print("Génération des clients...")
        
        clients = []
        for i in range(self.nb_clients):
            client_type = random.choice(['PUBLIC', 'PRIVATE'])
            
            if client_type == 'PUBLIC':
                # Organismes publics
                company_suffixes = ['Mairie', 'Conseil Départemental', 'Préfecture', 
                                  'Hôpital', 'Université', 'Lycée', 'Collège']
                company = f"{fake.city()} {random.choice(company_suffixes)}"
            else:
                # Entreprises privées
                company = fake.company()
            
            client = {
                'CLIENT_ID': i + 1,  # Oracle IDENTITY commence à 1
                'COMPANY_NAME': company,
                'CLIENT_TYPE': client_type,
                'CONTACT_NAME': fake.name(),
                'EMAIL': fake.email(),
                'PHONE': fake.phone_number(),
                'ADDRESS': fake.address().replace('\n', ', '),
                'CITY': fake.city(),
                'POSTAL_CODE': fake.postcode(),
                'SIRET': fake.siret() if client_type == 'PRIVATE' else None,
                'CREATED_AT': fake.date_between(start_date='-5y', end_date='today')
            }
            clients.append(client)
        
        self.clients = clients
        return clients
    
    def calculate_invoice_amounts(self, ht_amount: float, client_type: str) -> Dict[str, float]:
        """Calcule les montants d'une facture selon le type de client."""
        if client_type == 'PUBLIC':
            # Calculs spécifiques pour les clients publics
            montant_tva = ht_amount * 0.2  # TVA 20%
            amount_ttc = ht_amount + montant_tva
            ras_5p = ht_amount * 0.05  # Retenue à la source 5%
            ras_tva = montant_tva * 0.75  # RAS TVA = TVA * 75%
            amount_to_pay = amount_ttc - ras_5p - ras_tva
            
            return {
                'TOTAL_HT': round(ht_amount, 2),
                'AMOUNT_TTC': round(amount_ttc, 2),
                'MONTANT_TVA': round(montant_tva, 2),
                'RAS_5P': round(ras_5p, 2),
                'RAS_TVA': round(ras_tva, 2),
                'AMOUNT_TO_PAY': round(amount_to_pay, 2)
            }
        else:
            # Calculs classiques pour les clients privés
            tva_rate = random.choice([0.055, 0.10, 0.20])  # 5,5%, 10% ou 20%
            montant_tva = ht_amount * tva_rate
            amount_ttc = ht_amount + montant_tva
            
            return {
                'TOTAL_HT': round(ht_amount, 2),
                'AMOUNT_TTC': round(amount_ttc, 2),
                'MONTANT_TVA': round(montant_tva, 2),
                'RAS_5P': 0.0,
                'RAS_TVA': 0.0,
                'AMOUNT_TO_PAY': round(amount_ttc, 2)
            }
    
    def generate_invoices(self) -> List[Dict]:
        """Génère les factures selon le schéma Oracle INVOICES."""
        print("Génération des factures...")
        
        invoices = []
        current_year = datetime.now().year
        
        for i in range(self.nb_invoices):
            client = random.choice(self.clients)
            
            # Génération des dates
            invoice_date = fake.date_between(start_date='-18m', end_date='today')
            invoice_year = invoice_date.year
            
            # Dates spécifiques au schéma
            electronic_date = invoice_date + timedelta(days=random.randint(0, 2))
            physical_date = electronic_date + timedelta(days=random.randint(1, 5))
            expected_payment_date = invoice_date + timedelta(days=random.randint(15, 90))
            
            # Statut de paiement (80% payées)
            status_weights = [
                ('PAID', 0.75),
                ('UNPAID', 0.15),
                ('OVERDUE', 0.05),
                ('PARTIAL', 0.03),
                ('SENT', 0.02)
            ]
            status = random.choices([s[0] for s in status_weights], 
                                  weights=[s[1] for s in status_weights])[0]
            
            payment_date = None
            if status in ['PAID', 'PARTIAL']:
                # Date de paiement entre la date de facture et quelques jours après l'échéance
                max_payment_date = min(expected_payment_date + timedelta(days=30), datetime.now().date())
                if max_payment_date > invoice_date:
                    payment_date = fake.date_between(
                        start_date=invoice_date, 
                        end_date=max_payment_date
                    )
            
            # Génération du numéro de facture
            invoice_number = f"FACT-{invoice_year}-{i+1:06d}"
            
            # Génération d'un montant HT réaliste
            ht_amount = round(random.uniform(100, 10000), 2)
            
            # Génération de la quantité (entre 1 et 100)
            quantity = random.randint(1, 100)
            
            # Calcul du prix unitaire avec arrondi
            pu = round(ht_amount / quantity, 2)
            
            # Vérification que le prix unitaire est réaliste
            if pu < 1 or pu > 1000:
                continue  # Régénérer la facture si le prix unitaire n'est pas réaliste
            
            # Calculs selon le type de client
            amounts = self.calculate_invoice_amounts(ht_amount, client['CLIENT_TYPE'])
            
            invoice = {
                'INVOICE_ID': i + 1,  # Oracle IDENTITY
                'CLIENT_ID': client['CLIENT_ID'],
                'INVOICE_DATE': invoice_date,
                'PAYMENT_DATE': payment_date,
                'STATUS': status,
                'INVOICE_NUMBER': invoice_number,
                'INVOICE_YEAR': invoice_year,
                'PO': f"PO-{random.randint(1000, 9999)}" if random.random() < 0.7 else None,
                'PU': pu,
                'QUANTITY': quantity,
                'ELECTRONIC_DATE': electronic_date,
                'PHYSICAL_DATE': physical_date,
                'EXPECTED_PAYMENT_DATE': expected_payment_date,
                'LABEL': f"Prestation {fake.catch_phrase()}",
                'CLIENT_TYPE': client['CLIENT_TYPE'],
                'CREATED_AT': fake.date_between(start_date=invoice_date, end_date='today'),
                **amounts
            }
            
            invoices.append(invoice)
        
        self.invoices = invoices
        return invoices
    
    def generate_expenses(self) -> List[Dict]:
        """Génère des dépenses conformément au schéma Oracle EXPENSES."""
        print("Génération des dépenses...")
        
        expenses = []
        
        # Catégories et types de dépenses enrichis
        categories = [
            'Fournitures bureau', 'Frais professionnels', 'Déplacements', 
            'Communication', 'Formation', 'Logiciels', 'Matériel informatique',
            'Frais bancaires', 'Assurances', 'Loyer', 'Services publics',
            'Marketing', 'R&D', 'Frais de représentation', 'Abonnements',
            'Maintenance', 'Transport', 'Restauration', 'Hébergement'
        ]
        
        types = [
            'professional', 'travel', 'equipment', 'software', 
            'subscription', 'office', 'other', 'marketing',
            'research', 'maintenance', 'food', 'lodging'
        ]
        
        statuses = ['unpaid', 'paid']
        
        for i in range(self.nb_expenses):
            # Génération des dates avec une plage plus large
            expense_date = fake.date_between(start_date='-24m', end_date='today')
            created_at = expense_date + timedelta(days=random.randint(0, 2))
            updated_at = created_at if random.random() < 0.7 else created_at + timedelta(days=random.randint(1, 30))
            
            # Montant entre 5 et 5000 € avec distribution log-normale pour plus de réalisme
            amount = round(np.random.lognormal(mean=4, sigma=0.8), 2)
            amount = min(max(amount, 5), 5000)  # Bornage entre 5 et 5000
            
            expense = {
                'EXPENSE_ID': i + 1,
                'TITLE': f"{fake.word().capitalize()} {random.choice(['Dépense', 'Frais', 'Achat', 'Facture'])}",
                'AMOUNT': amount,
                'LABEL': random.choice([
                    f"Frais {fake.word()}",
                    f"Note {fake.city()}",
                    f"Facture {fake.company()}",
                    f"Remboursement {fake.last_name()}",
                    f"Achat {fake.word()}",
                    f"Service {fake.word()}"
                ]),
                'COMMENTS': fake.sentence() if random.random() < 0.6 else None,
                'EXPENSE_DATE': expense_date,
                'CREATED_AT': created_at,
                'ATTACHMENT': None,
                'TYPE': random.choice(types),
                'CATEGORY': random.choice(categories),
                'EXPENSE_NUMBER': f"EXP-{expense_date.year}-{i+1:05d}",
                'UPDATED_AT': updated_at,
                'STATUS': random.choices(
                    statuses, 
                    weights=[0.3, 0.7]  # 70% de paid, 30% unpaid
                )[0],
                'EXPECTED_PAYMENT_DATE': expense_date + timedelta(days=random.randint(1, 60))
            }
            
            # Appliquer les triggers si les dates sont nulles
            if expense['EXPENSE_DATE'] is None:
                expense['EXPENSE_DATE'] = datetime.now().replace(day=1).date()
                
            if expense['EXPECTED_PAYMENT_DATE'] is None:
                expense['EXPECTED_PAYMENT_DATE'] = datetime.now().replace(day=1, month=datetime.now().month+1).date()
                
            expenses.append(expense)
        
        self.expenses = expenses
        return expenses
    
    def generate_bank_statements(self) -> List[Dict]:
        """Génère les relevés bancaires selon le schéma Oracle BANK_STATEMENT."""
        print("Génération des relevés bancaires...")
        
        bank_statements = []
        paid_invoices = [inv for inv in self.invoices if inv['STATUS'] in ['PAID', 'PARTIAL']]
        
        # Répartition adaptée pour 8000 relevés
        nb_invoice_payments = int(self.nb_bank_statements * 0.65)  # 65% -> 5200
        nb_expense_payments = int(self.nb_bank_statements * 0.25)  # 25% -> 2000
        nb_orphan_statements = self.nb_bank_statements - nb_invoice_payments - nb_expense_payments  # 10% -> 800
        
        statement_id = 1
        
        # 1. Relevés liés aux factures payées (5200)
        print(f"  Génération de {nb_invoice_payments} paiements de factures...")
        selected_invoices = random.sample(paid_invoices, 
                                        min(nb_invoice_payments, len(paid_invoices)))
        
        for invoice in selected_invoices:
            # Variation de montant (±5%)
            amount_variation = random.uniform(-0.05, 0.05)
            bank_amount = invoice['AMOUNT_TO_PAY'] * (1 + amount_variation)
            
            # Génération d'une date de relevé entre la date de paiement et aujourd'hui
            statement_date = fake.date_between(
                start_date=invoice['PAYMENT_DATE'],
                end_date=datetime.now().date()
            )
            
            # Date de valeur proche de la date de relevé
            value_date = statement_date + timedelta(days=random.randint(0, 2))
            
            # Génération des libellés bancaires
            operation_template = random.choice(self.operation_labels['payment'])
            additional_template = random.choice(self.additional_labels['payment'])
            
            # Récupération du nom du client
            client = next(c for c in self.clients if c['CLIENT_ID'] == invoice['CLIENT_ID'])
            company_short = client['COMPANY_NAME'][:20]  # Limitation Oracle VARCHAR2(255)
            
            operation_label = operation_template.format(company=company_short)
            additional_label = additional_template.format(
                invoice_number=invoice['INVOICE_NUMBER'],
                date=statement_date.strftime('%d/%m')
            )
            
            # Commentaires aléatoires
            comments_options = [
                "Paiement conforme",
                "Règlement client",
                "Virement reçu",
                "Facture soldée",
                None
            ]
            comments = random.choice(comments_options)
            
            statement = {
                'STATEMENT_ID': statement_id,
                'STATEMENT_DATE': statement_date,
                'OPERATION_LABEL': operation_label,
                'ADDITIONAL_LABEL': additional_label,
                'DEBIT': None,
                'CREDIT': round(bank_amount, 2),
                'COMMENTS': comments,
                'RELATED_INVOICE_ID': invoice['INVOICE_ID'],
                'RELATED_EXPENSE_ID': None,
                'VALUE_DATE': value_date,
                'SOURCE_FILENAME': None,
                'MIME_TYPE': None,
                'CREATED_AT': fake.date_between(start_date=statement_date, end_date='today')
            }
            
            bank_statements.append(statement)
            statement_id += 1
        
        # 2. Relevés liés aux dépenses (2000)
        print(f"  Génération de {nb_expense_payments} paiements de dépenses...")
        if self.expenses:
            selected_expenses = random.sample(self.expenses, 
                                            min(nb_expense_payments, len(self.expenses)))
            
            for expense in selected_expenses:
                # Variation de montant (±2%)
                amount_variation = random.uniform(-0.02, 0.02)
                bank_amount = expense['AMOUNT'] * (1 + amount_variation)
                
                # Génération d'une date de relevé entre la date de dépense et aujourd'hui
                statement_date = fake.date_between(
                    start_date=expense['EXPENSE_DATE'],
                    end_date=datetime.now().date()
                )
                
                # Date de valeur proche de la date de relevé
                value_date = statement_date + timedelta(days=random.randint(0, 2))
                
                # Génération des libellés bancaires
                operation_label = random.choice(self.operation_labels['expense'])
                additional_label = random.choice(self.additional_labels['expense']).format(
                    date=statement_date.strftime('%d/%m')
                )
                
                statement = {
                    'STATEMENT_ID': statement_id,
                    'STATEMENT_DATE': statement_date,
                    'OPERATION_LABEL': operation_label,
                    'ADDITIONAL_LABEL': additional_label,
                    'DEBIT': round(bank_amount, 2),
                    'CREDIT': None,
                    'COMMENTS': f"Dépense {expense['CATEGORY']}",
                    'RELATED_INVOICE_ID': None,
                    'RELATED_EXPENSE_ID': expense['EXPENSE_ID'],
                    'VALUE_DATE': value_date,
                    'SOURCE_FILENAME': None,
                    'MIME_TYPE': None,
                    'CREATED_AT': fake.date_between(start_date=statement_date, end_date='today')
                }
                
                bank_statements.append(statement)
                statement_id += 1
        
        # 3. Relevés orphelins (800)
        print(f"  Génération de {nb_orphan_statements} relevés orphelins...")
        for i in range(nb_orphan_statements):
            # Utilisation d'une distribution log-normale pour les montants
            amount = round(np.random.lognormal(mean=3, sigma=1.2), 2)
            amount = random.choice([-1, 1]) * min(abs(amount), 500)  # Bornage et signe aléatoire
            
            statement_date = fake.date_between(start_date='-24m', end_date='today')
            value_date = statement_date + timedelta(days=random.randint(-1, 1))
            
            operation_label = random.choice(self.operation_labels['orphan'])
            additional_label = random.choice(self.additional_labels['orphan'])
            
            statement = {
                'STATEMENT_ID': statement_id,
                'STATEMENT_DATE': statement_date,
                'OPERATION_LABEL': operation_label,
                'ADDITIONAL_LABEL': additional_label,
                'DEBIT': round(abs(amount), 2) if amount < 0 else None,
                'CREDIT': round(amount, 2) if amount > 0 else None,
                'COMMENTS': "Opération automatique",
                'RELATED_INVOICE_ID': None,
                'RELATED_EXPENSE_ID': None,
                'VALUE_DATE': value_date,
                'SOURCE_FILENAME': None,
                'MIME_TYPE': None,
                'CREATED_AT': fake.date_between(start_date=statement_date, end_date='today')
            }
            
            bank_statements.append(statement)
            statement_id += 1
        
        # Tri par date
        bank_statements.sort(key=lambda x: x['STATEMENT_DATE'])
        
        self.bank_statements = bank_statements
        return bank_statements
    
    def export_to_csv(self, output_dir: str = 'output'):
        """Exporte les données en fichiers CSV compatibles Oracle."""
        print(f"Export des données vers le répertoire '{output_dir}'...")
        
        # Création du répertoire de sortie
        os.makedirs(output_dir, exist_ok=True)
        
        # Export des statuts de facture
        if self.invoice_statuses:
            statuses_df = pd.DataFrame(self.invoice_statuses)
            statuses_df.to_csv(f'{output_dir}/invoice_statuses.csv', index=False, encoding='utf-8-sig')
            print(f"  ✓ {len(self.invoice_statuses)} statuts exportés vers invoice_statuses.csv")
        
        # Export des clients
        if self.clients:
            clients_df = pd.DataFrame(self.clients)
            # Formatage des dates pour Oracle
            for col in clients_df.select_dtypes(include=['datetime64']).columns:
                clients_df[col] = clients_df[col].dt.strftime('%Y-%m-%d')
            clients_df.to_csv(f'{output_dir}/clients.csv', index=False, encoding='utf-8-sig')
            print(f"  ✓ {len(self.clients)} clients exportés vers clients.csv")
        
        # Export des factures (table INVOICES)
        if self.invoices:
            invoices_df = pd.DataFrame(self.invoices)
            # Formatage des dates pour Oracle
            date_columns = ['INVOICE_DATE', 'PAYMENT_DATE', 'ELECTRONIC_DATE', 
                          'PHYSICAL_DATE', 'EXPECTED_PAYMENT_DATE', 'CREATED_AT']
            for col in date_columns:
                if col in invoices_df.columns:
                    invoices_df[col] = pd.to_datetime(invoices_df[col]).dt.strftime('%Y-%m-%d')
            
            invoices_df.to_csv(f'{output_dir}/invoices.csv', index=False, encoding='utf-8-sig')
            print(f"  ✓ {len(self.invoices)} factures exportées vers invoices.csv")
        
        # Export des relevés bancaires (table BANK_STATEMENT)
        if self.bank_statements:
            statements_df = pd.DataFrame(self.bank_statements)
            # Formatage des dates pour Oracle
            date_columns = ['STATEMENT_DATE', 'VALUE_DATE', 'CREATED_AT']
            for col in date_columns:
                if col in statements_df.columns:
                    statements_df[col] = pd.to_datetime(statements_df[col]).dt.strftime('%Y-%m-%d')
            
            statements_df.to_csv(f'{output_dir}/bank_statements.csv', index=False, encoding='utf-8-sig')
            print(f"  ✓ {len(self.bank_statements)} relevés bancaires exportés vers bank_statements.csv")
        
        # Export des dépenses (table EXPENSES)
        if self.expenses:
            expenses_df = pd.DataFrame(self.expenses)
            # Formatage des dates pour Oracle
            date_columns = ['EXPENSE_DATE', 'CREATED_AT', 'UPDATED_AT', 'EXPECTED_PAYMENT_DATE']
            for col in date_columns:
                if col in expenses_df.columns:
                    expenses_df[col] = pd.to_datetime(expenses_df[col]).dt.strftime('%Y-%m-%d')
            expenses_df.to_csv(f'{output_dir}/expenses.csv', index=False, encoding='utf-8-sig')
            print(f"  ✓ {len(self.expenses)} dépenses exportées vers expenses.csv")
        
        # Génération d'un script SQL d'insertion
        self.generate_sql_inserts(output_dir)
        
        # Génération d'un rapport de synthèse
        self.generate_summary_report(output_dir)
    
    def generate_sql_inserts(self, output_dir: str):
        """Génère des scripts SQL d'insertion pour Oracle."""
        sql_path = f'{output_dir}/insert_data.sql'
        
        with open(sql_path, 'w', encoding='utf-8') as f:
            f.write("-- Script d'insertion pour Oracle DB\n")
            f.write("-- Dataset Comptable Synthétique\n")
            f.write(f"-- Généré le {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
            
            # Insertion des statuts de facture
            f.write("-- Insertion des statuts de facture\n")
            for status in self.invoice_statuses:
                f.write(f"INSERT INTO INVOICE_STATUSES (STATUS_CODE, DESCRIPTION) VALUES ('{status['STATUS_CODE']}', '{status['DESCRIPTION']}');\n")
            f.write("\n")
            
            # Note sur les insertions des autres tables
            f.write("-- Les autres insertions peuvent être effectuées via:\n")
            f.write("-- 1. SQL*Loader avec les fichiers CSV générés\n")
            f.write("-- 2. Import/Export Oracle\n")
            f.write("-- 3. Scripts d'insertion générés automatiquement\n\n")
            
            f.write("-- Exemple de structure d'insertion pour INVOICES:\n")
            if self.invoices:
                sample_invoice = self.invoices[0]
                f.write("/*\n")
                f.write("INSERT INTO INVOICES (\n")
                f.write("    CLIENT_ID, INVOICE_DATE, PAYMENT_DATE, TOTAL_HT, STATUS,\n")
                f.write("    INVOICE_NUMBER, INVOICE_YEAR, PO, PU, QUANTITY,\n")
                f.write("    AMOUNT_TTC, RAS_5P, RAS_TVA, AMOUNT_TO_PAY,\n")
                f.write("    ELECTRONIC_DATE, PHYSICAL_DATE, EXPECTED_PAYMENT_DATE,\n")
                f.write("    LABEL, CLIENT_TYPE, MONTANT_TVA\n")
                f.write(") VALUES (\n")
                f.write(f"    {sample_invoice['CLIENT_ID']}, DATE '{sample_invoice['INVOICE_DATE']}', ")
                payment_date = f"DATE '{sample_invoice['PAYMENT_DATE']}'" if sample_invoice['PAYMENT_DATE'] else "NULL"
                f.write(f"{payment_date}, {sample_invoice['TOTAL_HT']}, '{sample_invoice['STATUS']}',\n")
                f.write(f"    '{sample_invoice['INVOICE_NUMBER']}', {sample_invoice['INVOICE_YEAR']}, ")
                po = "'{}'".format(sample_invoice['PO']) if sample_invoice['PO'] else "NULL"
                f.write(f"{po}, {sample_invoice['PU']}, {sample_invoice['QUANTITY']},\n")
                f.write(f"    {sample_invoice['AMOUNT_TTC']}, {sample_invoice['RAS_5P']}, {sample_invoice['RAS_TVA']}, {sample_invoice['AMOUNT_TO_PAY']},\n")
                f.write(f"    DATE '{sample_invoice['ELECTRONIC_DATE']}', DATE '{sample_invoice['PHYSICAL_DATE']}', DATE '{sample_invoice['EXPECTED_PAYMENT_DATE']}',\n")
                f.write("    '{}', '{}', {}\n".format(
                    sample_invoice['LABEL'].replace("'", "''"),
                    sample_invoice['CLIENT_TYPE'],
                    sample_invoice['MONTANT_TVA']
                ))
                f.write(");\n")
                f.write("*/\n\n")
            
            f.write("COMMIT;\n")
        
        print(f"  ✓ Script SQL généré: insert_data.sql")
    
    def generate_summary_report(self, output_dir: str):
        """Génère un rapport de synthèse du dataset."""
        report_path = f'{output_dir}/dataset_summary.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("RAPPORT DE SYNTHÈSE - DATASET COMPTABLE SYNTHÉTIQUE ORACLE\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Date de génération: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"Compatible avec le schéma Oracle DB\n\n")
            
            # Statistiques clients
            if self.clients:
                public_clients = len([c for c in self.clients if c['CLIENT_TYPE'] == 'PUBLIC'])
                private_clients = len([c for c in self.clients if c['CLIENT_TYPE'] == 'PRIVATE'])
                f.write(f"CLIENTS ({len(self.clients)} total):\n")
                f.write(f"  - Publics: {public_clients}\n")
                f.write(f"  - Privés: {private_clients}\n\n")
            
            # Statistiques factures
            if self.invoices:
                status_counts = {}
                for invoice in self.invoices:
                    status = invoice['STATUS']
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                total_ht = sum([i['TOTAL_HT'] for i in self.invoices])
                total_ttc = sum([i['AMOUNT_TTC'] for i in self.invoices])
                total_to_pay = sum([i['AMOUNT_TO_PAY'] for i in self.invoices])
                
                f.write(f"FACTURES ({len(self.invoices)} total):\n")
                for status, count in status_counts.items():
                    f.write(f"  - {status}: {count}\n")
                f.write(f"  - Montant total HT: {total_ht:,.2f} €\n")
                f.write(f"  - Montant total TTC: {total_ttc:,.2f} €\n")
                f.write(f"  - Montant total à payer: {total_to_pay:,.2f} €\n\n")
            
            # Statistiques relevés bancaires
            if self.bank_statements:
                linked_to_invoices = len([s for s in self.bank_statements if s['RELATED_INVOICE_ID']])
                linked_to_expenses = len([s for s in self.bank_statements if s['RELATED_EXPENSE_ID']])
                orphan_statements = len(self.bank_statements) - linked_to_invoices - linked_to_expenses
                
                total_credits = sum([s['CREDIT'] for s in self.bank_statements if s['CREDIT']])
                total_debits = sum([s['DEBIT'] for s in self.bank_statements if s['DEBIT']])
                
                f.write(f"RELEVES BANCAIRES ({len(self.bank_statements)} total):\n")
                f.write(f"  - Liés à des factures: {linked_to_invoices}\n")
                f.write(f"  - Liés à des dépenses: {linked_to_expenses}\n")
                f.write(f"  - Orphelins: {orphan_statements}\n")
                f.write(f"  - Total crédits: {total_credits:,.2f} €\n")
                f.write(f"  - Total débits: {total_debits:,.2f} €\n\n")
            
            # Statistiques dépenses
            if self.expenses:
                status_counts = {}
                category_counts = {}
                type_counts = {}
                total_amount = 0
                
                for expense in self.expenses:
                    status = expense['STATUS']
                    category = expense['CATEGORY']
                    expense_type = expense['TYPE']
                    
                    status_counts[status] = status_counts.get(status, 0) + 1
                    category_counts[category] = category_counts.get(category, 0) + 1
                    type_counts[expense_type] = type_counts.get(expense_type, 0) + 1
                    total_amount += expense['AMOUNT']
                
                f.write(f"DEPENSES ({len(self.expenses)} total):\n")
                f.write(f"  - Montant total: {total_amount:,.2f} €\n")
                f.write("  - Par statut:\n")
                for status, count in status_counts.items():
                    f.write(f"    - {status}: {count}\n")
                f.write("  - Par catégorie:\n")
                for category, count in category_counts.items():
                    f.write(f"    - {category}: {count}\n")
                f.write("  - Par type:\n")
                for expense_type, count in type_counts.items():
                    f.write(f"    - {expense_type}: {count}\n")
                f.write("\n")
            
            f.write("FIN DU RAPPORT\n")

if __name__ == "__main__":
    try:
        print("\n=== Démarrage du générateur de dataset ===")
        generator = AccountingDatasetGenerator()
        print("✓ Instance du générateur créée")
        
        # Génération des données
        print("\n=== Génération des données ===")
        print("Génération des clients...")
        generator.generate_clients()
        print(f"✓ {len(generator.clients)} clients générés")
        
        print("Génération des statuts de facture...")
        generator.generate_invoice_statuses()
        print(f"✓ {len(generator.invoice_statuses)} statuts générés")
        
        print("Génération des factures...")
        generator.generate_invoices()
        print(f"✓ {len(generator.invoices)} factures générées")
        
        print("Génération des dépenses...")
        generator.generate_expenses()
        print(f"✓ {len(generator.expenses)} dépenses générées")
        
        print("Génération des relevés bancaires...")
        generator.generate_bank_statements()
        print(f"✓ {len(generator.bank_statements)} relevés bancaires générés")
        
        print("\n=== Export des données ===")
        generator.export_to_csv()
        print("✓ Données exportées vers le dossier 'output/'")
        
        print("\n=== Génération terminée ===")
        print("\nVérifiez le dossier 'output/' pour les fichiers générés.")
        print("Fichiers créés :")
        print("- clients.csv")
        print("- invoices.csv")
        print("- bank_statements.csv")
        print("- expenses.csv")
        print("- invoice_statuses.csv")
        print("- insert_data.sql")
        print("- dataset_summary.txt")
    except Exception as e:
        print(f"\nErreur lors de la génération : {str(e)}")
        raise
            