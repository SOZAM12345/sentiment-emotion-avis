
from pathlib import Path
from functools import lru_cache

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# 1. Chemin du projet
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Chemin du modèle amélioré
MODEL_PATH = (
    BASE_DIR / "models" / "distilbert_emotions_ameliore"
)

# 3. Appareil utilisé
device = "cuda" if torch.cuda.is_available() else "cpu"

# 4. Charger le modèle une seule fois
@lru_cache(maxsize=1)
def charger_modele_emotion():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    modele = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    modele.to(device)
    modele.eval()

    return tokenizer, modele


# 5. Fonction de prédiction
def predire_emotion(texte):
    """
    Prédit l'émotion d'un texte anglais.

    Classes :
    Tristesse, Joie, Amour,
    Colère, Peur, Surprise.
    """

    if not isinstance(texte, str) or not texte.strip():
        raise ValueError("Le texte est vide.")

    tokenizer, modele = charger_modele_emotion()

    entrees = tokenizer(
        texte,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )

    entrees = {
        cle: valeur.to(device)
        for cle, valeur in entrees.items()
    }

    with torch.no_grad():
        sortie = modele(**entrees)

        prediction = sortie.logits.argmax(
            dim=1
        ).item()

    return modele.config.id2label[prediction]


# 6. Tester notre fonction
if __name__ == "__main__":

    avis = [
        "I am very happy today!",
        "I feel really sad and lonely.",
        "I am angry about this situation.",
        "I am afraid of what will happen.",
        "I love my family so much.",
        "I was surprised by the news!"
    ]

    for texte in avis:

        emotion = predire_emotion(texte)

        print("Avis :", texte)
        print("Émotion :", emotion)
        print("-" * 50)
