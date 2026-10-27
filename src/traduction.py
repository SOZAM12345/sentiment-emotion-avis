
from functools import lru_cache
import torch

from transformers import MarianMTModel, MarianTokenizer

# Modèles de traduction
MODELES = {
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr"
}


@lru_cache(maxsize=2)
def charger_traducteur(langue_source, langue_cible):

    nom_modele = MODELES[(langue_source, langue_cible)]

    print("Chargement du modèle :", nom_modele)

    tokenizer = MarianTokenizer.from_pretrained(nom_modele)
    modele = MarianMTModel.from_pretrained(nom_modele)

    modele.eval()

    return tokenizer, modele


def traduire_texte(texte, langue_source, langue_cible):

    if not isinstance(texte, str) or not texte.strip():
        raise ValueError("Le texte est vide.")

    if langue_source == langue_cible:
        return texte

    if (langue_source, langue_cible) not in MODELES:
        raise ValueError("Traduction non prise en charge.")

    tokenizer, modele = charger_traducteur(
        langue_source,
        langue_cible
    )

    entrees = tokenizer(
        [texte],
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        sortie = modele.generate(
            **entrees,
            max_new_tokens=512,
            num_beams=4
        )

    traduction = tokenizer.batch_decode(
        sortie,
        skip_special_tokens=True
    )[0]

    return traduction


if __name__ == "__main__":

    avis_francais = "Je suis vraiment déçu par ce produit."

    avis_anglais = "I am very happy with this product."

    print("=== FRANÇAIS VERS ANGLAIS ===")

    print(
        traduire_texte(
            avis_francais,
            "fr",
            "en"
        )
    )

    print("\n=== ANGLAIS VERS FRANÇAIS ===")

    print(
        traduire_texte(
            avis_anglais,
            "en",
            "fr"
        )
    )
