
# ============================================
# PROJET : Analyse des sentiments et émotions
# Fichier : src/prediction_emotion.py
# Version locale + Hugging Face
# ============================================

# 1. IMPORTATION DES BIBLIOTHÈQUES

from pathlib import Path
from functools import lru_cache

import torch

from huggingface_hub import snapshot_download

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# 2. CHEMIN DU PROJET

BASE_DIR = Path(__file__).resolve().parent.parent


# 3. CHEMIN DU MODÈLE LOCAL

MODEL_PATH = (
    BASE_DIR / "models" / "distilbert_emotions_ameliore"
)


# 4. DÉPÔT HUGGING FACE

HF_REPO_ID = "sozame/distilbert-emotions-ameliore"


# 5. APPAREIL UTILISÉ

device = "cuda" if torch.cuda.is_available() else "cpu"


# 6. CHARGEMENT DU MODÈLE

@lru_cache(maxsize=1)
def charger_modele_emotion():
    """
    Charge le modèle DistilBERT amélioré.

    Sur le PC :
        Charge le modèle depuis models/.

    Sur Streamlit Cloud :
        Télécharge le modèle depuis Hugging Face.

    Le modèle n'est chargé qu'une seule fois
    par processus Python.
    """

    # Vérifier que les fichiers nécessaires existent
    modele_local_disponible = (
        (MODEL_PATH / "config.json").exists()
        and (MODEL_PATH / "model.safetensors").exists()
        and (MODEL_PATH / "tokenizer.json").exists()
    )

    if modele_local_disponible:

        print("Modèle d'émotions chargé localement.")

        chemin_modele = MODEL_PATH

    else:

        print(
            "Modèle local absent. "
            "Téléchargement depuis Hugging Face..."
        )

        try:

            chemin_modele = snapshot_download(
                repo_id=HF_REPO_ID,
                repo_type="model",
                allow_patterns=[
                    "*.json",
                    "*.safetensors"
                ]
            )

        except Exception as erreur:

            raise RuntimeError(
                "Impossible de télécharger le modèle "
                "d'émotions depuis Hugging Face. "
                "Vérifiez la connexion Internet et "
                f"les permissions du dépôt {HF_REPO_ID}."
            ) from erreur

    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        chemin_modele
    )

    # Charger le modèle de classification
    modele = AutoModelForSequenceClassification.from_pretrained(
        chemin_modele
    )

    # Déplacer le modèle vers CPU ou GPU
    modele.to(device)

    # Activer le mode évaluation
    modele.eval()

    print("Modèle DistilBERT prêt pour les prédictions.")

    return tokenizer, modele


# 7. FONCTION DE PRÉDICTION

def predire_emotion(texte):
    """
    Prédit l'émotion d'un texte anglais.

    Classes :
    Tristesse, Joie, Amour,
    Colère, Peur, Surprise.
    """

    # Vérifier le texte
    if not isinstance(texte, str) or not texte.strip():
        raise ValueError("Le texte est vide.")

    # Charger le tokenizer et le modèle
    tokenizer, modele = charger_modele_emotion()

    # Préparer le texte pour DistilBERT
    entrees = tokenizer(
        texte,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    # Placer les données sur le même appareil
    entrees = {
        cle: valeur.to(device)
        for cle, valeur in entrees.items()
    }

    # Prédire sans calculer de gradients
    with torch.no_grad():

        sortie = modele(**entrees)

        prediction = sortie.logits.argmax(
            dim=1
        ).item()

    # Retourner le nom de l'émotion
    return modele.config.id2label[prediction]


# 8. TEST DU MODÈLE

if __name__ == "__main__":

    avis_tests = [
        "I am very happy today!",
        "I feel really sad and lonely.",
        "I am angry about this situation.",
        "I am afraid of what will happen.",
        "I love my family so much.",
        "I was surprised by the news!"
    ]

    print("=" * 60)
    print("TEST DU MODÈLE D'ÉMOTIONS")
    print("=" * 60)

    for i, texte in enumerate(avis_tests, 1):

        try:

            emotion = predire_emotion(texte)

            print(f"\nAvis {i} : {texte}")
            print("Émotion prédite :", emotion)
            print("-" * 60)

        except Exception as erreur:

            print("Erreur :", erreur)

    print("\nTest terminé !")
