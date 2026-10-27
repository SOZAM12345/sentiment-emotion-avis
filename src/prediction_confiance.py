
# ============================================
# SCORES DE CONFIANCE DES MODELES
# ============================================

import torch

from prediction_sentiment import (
    charger_modele_sentiment
)

from prediction_emotion import (
    charger_modele_emotion,
    device
)


# 1. Score de confiance du sentiment
def confiance_sentiment(texte):

    modele = charger_modele_sentiment()

    # Le pipeline LogisticRegression fournit predict_proba
    probabilites = modele.predict_proba([texte])[0]

    prediction = modele.predict([texte])[0]

    # Trouver la probabilité de la classe prédite
    classes = list(modele.classes_)
    indice = classes.index(prediction)

    return float(probabilites[indice])


# 2. Score de confiance de l'émotion
def confiance_emotion(texte):

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

        probabilites = torch.softmax(
            sortie.logits,
            dim=1
        )[0]

        indice = torch.argmax(probabilites).item()

        confiance = probabilites[indice].item()

    return float(confiance)


# 3. Niveau de confiance
def niveau_confiance(score, seuil=0.70):

    if score < seuil:
        return "Incertain"

    return "Confiance élevée (indicative)"


# 4. Test
if __name__ == "__main__":

    texte_fr = "Ce film est magnifique !"
    texte_en = "I am very happy today!"

    print(
        "Confiance sentiment :",
        f"{confiance_sentiment(texte_fr) * 100:.2f}%"
    )

    print(
        "Confiance émotion :",
        f"{confiance_emotion(texte_en) * 100:.2f}%"
    )
