# from sentence_transformers import SentenceTransformer, util

# Charger le modèle (multilingue, donc parfait pour FR/EN)
# model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Deux textes à comparer
text1 = "VIR AMAZON AWS IRELAND 28/08/2025"
text2 = "VIR AMAZON AWS IRELAND 28/08/2025"

# # Obtenir les embeddings
# embedding1 = model.encode(text1, convert_to_tensor=True)
# embedding2 = model.encode(text2, convert_to_tensor=True)

# # Calculer la similarité cosinus
# similarity = util.cos_sim(embedding1, embedding2)

print("AMAZON" in text1)
