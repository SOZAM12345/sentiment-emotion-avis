
from langdetect import detect, DetectorFactory, LangDetectException

DetectorFactory.seed = 42

def detecter_langue(texte):
    if not isinstance(texte, str) or not texte.strip():
        return "inconnue"

    try:
        langue = detect(texte)

        if langue == "fr":
            return "français"
        elif langue == "en":
            return "anglais"
        else:
            return "non prise en charge"

    except LangDetectException:
        return "inconnue"


if __name__ == "__main__":

    avis = [
        "Ce produit est vraiment magnifique, je suis très content.",
        "I am really disappointed with this product.",
        "La livraison est catastrophique et le service est mauvais.",
        "I absolutely love this movie, it's fantastic."
    ]

    for texte in avis:
        print("Avis :", texte)
        print("Langue :", detecter_langue(texte))
        print("-" * 50)
