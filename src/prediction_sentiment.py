
# ============================================
# PROJET : Analyse automatique des sentiments
# Fichier : src/prediction_sentiment.py
# Version locale + Hugging Face
# ============================================

# 1. IMPORTATION DES BIBLIOTHÈQUES

from pathlib import Path
from functools import lru_cache

import joblib
from huggingface_hub import hf_hub_download


# 2. CHEMINS DU MODÈLE

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR / "models" / "meilleur_modele_sentiment.joblib"
)

# Dépôt contenant notre modèle entraîné
HF_REPO_ID = "sozame/sentiment-fr-logistic"

HF_FILENAME = "meilleur_modele_sentiment.joblib"


# 3. FONCTION POUR CHARGER LE MODÈLE

@lru_cache(maxsize=1)
def charger_modele_sentiment():
    """
    Charge le modèle de sentiment.

    - Sur le PC : utilise le modèle local.
    - Sur Streamlit Cloud : télécharge le modèle
      depuis Hugging Face s'il est absent.
    """

    if MODEL_PATH.exists():

        # Utiliser le modèle présent sur le PC
        chemin_modele = MODEL_PATH

        print("Modèle de sentiment chargé localement.")

    else:

        # Télécharger le modèle depuis Hugging Face
        print(
            "Modèle local absent. "
            "Téléchargement depuis Hugging Face..."
        )

        try:

            chemin_modele = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_FILENAME,
                repo_type="model"
            )

        except Exception as erreur:

            raise RuntimeError(
                "Impossible de télécharger le modèle "
                "de sentiment depuis Hugging Face. "
                "Vérifiez la connexion et les permissions "
                f"du dépôt {HF_REPO_ID}."
            ) from erreur

    # Charger le modèle entraîné
    modele = joblib.load(chemin_modele)

    return modele


# 4. FONCTION POUR PRÉDIRE LE SENTIMENT

def predire_sentiment(texte):
    """
    Prédit le sentiment d'un avis français.

    0 = Négatif
    1 = Positif
    """

    if not isinstance(texte, str) or not texte.strip():
        raise ValueError("Le commentaire est vide.")

    modele = charger_modele_sentiment()

    prediction = modele.predict([texte])[0]

    if int(prediction) == 1:
        return "Positif"

    elif int(prediction) == 0:
        return "Négatif"

    else:
        raise ValueError(
            f"Classe inconnue : {prediction}"
        )


# 5. TESTER NOTRE MODÈLE SUR 10 AVIS

if __name__ == "__main__":

    avis_tests = [
        "Je suis très content de ce film !",
        "Ce film est excellent.",
        "J'ai adoré ce film.",
        "Ce film est vraiment magnifique.",
        "Je suis très heureux de ce film.",
        "Ce film est une merveille.",
        "Ce film est mauvais.",
        "Je déteste ce film.",
        "Je suis déçu par ce film.",
        "Ce film est catastrophique."
    ]

    sentiments_attendus = [
        "Positif",
        "Positif",
        "Positif",
        "Positif",
        "Positif",
        "Positif",
        "Négatif",
        "Négatif",
        "Négatif",
        "Négatif"
    ]

    print("=" * 60)
    print("TEST DU MODÈLE DE SENTIMENTS")
    print("=" * 60)

    bonnes_predictions = 0

    for i, avis in enumerate(avis_tests):

        sentiment_predit = predire_sentiment(avis)
        sentiment_attendu = sentiments_attendus[i]

        correct = (
            sentiment_predit == sentiment_attendu
        )

        if correct:
            bonnes_predictions += 1
            resultat = "CORRECT"
        else:
            resultat = "ERREUR"

        print(f"\nAvis {i + 1} : {avis}")
        print("Sentiment attendu :", sentiment_attendu)
        print("Sentiment prédit  :", sentiment_predit)
        print("Résultat          :", resultat)
        print("-" * 60)

    total = len(avis_tests)

    pourcentage = (
        bonnes_predictions / total * 100
    )

    print("\n" + "=" * 60)
    print("RÉSULTATS FINAUX")
    print("=" * 60)

    print("Nombre d'avis testés :", total)
    print("Prédictions correctes :", bonnes_predictions)
    print(
        "Prédictions incorrectes :",
        total - bonnes_predictions
    )

    print(
        f"Pourcentage de réussite : {pourcentage:.2f}%"
    )

    print("\nTest terminé !")
