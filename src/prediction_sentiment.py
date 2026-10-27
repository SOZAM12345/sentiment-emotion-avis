
# ============================================
# PROJET : Analyse automatique des sentiments
# Fichier : src/prediction_sentiment.py
# ============================================

# 1. IMPORTATION DES BIBLIOTHÈQUES

from pathlib import Path
from functools import lru_cache
import joblib


# 2. CHEMIN DU MODÈLE SAUVEGARDÉ

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR / "models" / "meilleur_modele_sentiment.joblib"
)


# 3. FONCTION POUR CHARGER LE MODÈLE

@lru_cache(maxsize=1)
def charger_modele_sentiment():

    # Vérifier si le modèle existe
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    # Charger le modèle entraîné
    modele = joblib.load(MODEL_PATH)

    return modele


# 4. FONCTION POUR PRÉDIRE LE SENTIMENT

def predire_sentiment(texte):
    """
    Prédit le sentiment d'un avis français.

    0 = Négatif
    1 = Positif
    """

    # Vérifier le texte
    if not isinstance(texte, str) or not texte.strip():
        raise ValueError("Le commentaire est vide.")

    # Charger notre modèle
    modele = charger_modele_sentiment()

    # Prédire le sentiment
    prediction = modele.predict([texte])[0]

    # Transformer la prédiction en texte
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
        # 6 avis positifs
        "Je suis très content de ce film !",
        "Ce film est excellent.",
        "J'ai adoré ce film.",
        "Ce film est vraiment magnifique.",
        "Je suis très heureux de ce film.",
        "Ce film est une merveille.",

        # 4 avis négatifs
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

        # Vérifier si la prédiction est correcte
        correct = sentiment_predit == sentiment_attendu

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

    # 6. RÉSUMÉ FINAL

    total = len(avis_tests)

    pourcentage = bonnes_predictions / total * 100

    print("\n" + "=" * 60)
    print("RÉSULTATS FINAUX")
    print("=" * 60)

    print("Nombre d'avis testés :", total)
    print("Prédictions correctes :", bonnes_predictions)
    print("Prédictions incorrectes :", total - bonnes_predictions)
    print(f"Pourcentage de réussite : {pourcentage:.2f}%")

    print("\nTest terminé !")
