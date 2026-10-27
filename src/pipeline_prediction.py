
# ============================================
# PROJET : Analyse automatique des sentiments
# et des émotions dans les avis utilisateurs
#
# Fichier : src/pipeline_prediction.py
# ============================================


# ============================================
# 1. IMPORTER LES FONCTIONS
# ============================================

from detection_langue import detecter_langue

from traduction import traduire_texte

from prediction_sentiment import predire_sentiment

from prediction_emotion import predire_emotion

from prediction_confiance import (
    confiance_sentiment,
    confiance_emotion,
    niveau_confiance
)


# ============================================
# 2. FONCTION PRINCIPALE
# ============================================

def analyser_avis(texte):
    """
    Analyse automatiquement un avis utilisateur.

    Fonctionnalités :
    - Détection de langue
    - Traduction français / anglais
    - Prédiction du sentiment
    - Prédiction de l'émotion
    - Calcul des scores de confiance
    - Détection des prédictions incertaines

    Retourne un dictionnaire contenant
    tous les résultats de l'analyse.
    """

    # ETAPE 1 : Vérifier le texte

    if not isinstance(texte, str) or not texte.strip():

        raise ValueError(
            "Le commentaire est vide."
        )

    texte = texte.strip()


    # ETAPE 2 : Détecter la langue

    langue = detecter_langue(texte)


    # ETAPE 3 : Préparer les traductions

    if langue == "français":

        texte_fr = texte

        texte_en = traduire_texte(
            texte,
            "fr",
            "en"
        )

    elif langue == "anglais":

        texte_en = texte

        texte_fr = traduire_texte(
            texte,
            "en",
            "fr"
        )

    else:

        raise ValueError(
            "Langue non prise en charge. "
            "Veuillez utiliser le français "
            "ou l'anglais."
        )


    # ETAPE 4 : Prédire le sentiment

    sentiment = predire_sentiment(
        texte_fr
    )


    # ETAPE 5 : Prédire l'émotion

    emotion = predire_emotion(
        texte_en
    )


    # ETAPE 6 : Calculer les scores de confiance

    score_sentiment = confiance_sentiment(
        texte_fr
    )

    score_emotion = confiance_emotion(
        texte_en
    )


    # ETAPE 7 : Vérifier les niveaux de confiance

    niveau_sentiment = niveau_confiance(
        score_sentiment
    )

    niveau_emotion = niveau_confiance(
        score_emotion
    )


    # ETAPE 8 : Regrouper tous les résultats

    resultat = {

        "avis": texte,

        "langue": langue,

        "texte_francais": texte_fr,

        "texte_anglais": texte_en,

        "sentiment": sentiment,

        "emotion": emotion,

        "confiance_sentiment": score_sentiment,

        "confiance_emotion": score_emotion,

        "niveau_sentiment": niveau_sentiment,

        "niveau_emotion": niveau_emotion

    }

    return resultat


# ============================================
# 3. TESTS DU PIPELINE
# ============================================

if __name__ == "__main__":

    avis_tests = [

        "Ce restaurant est excellent !",

        "I am angry about this terrible service.",

        "Je suis vraiment déçu par ce produit.",

        "I am very happy with my new phone!"

    ]

    print("=" * 60)
    print("TEST DU PIPELINE DE PREDICTION")
    print("=" * 60)

    for avis in avis_tests:

        print("\n" + "=" * 60)

        try:

            resultat = analyser_avis(avis)

            print("Avis :", resultat["avis"])

            print("Langue :", resultat["langue"])

            print(
                "Texte français :",
                resultat["texte_francais"]
            )

            print(
                "Texte anglais :",
                resultat["texte_anglais"]
            )

            print(
                "\nSentiment :",
                resultat["sentiment"]
            )

            print(
                "Confiance sentiment :",
                f"{resultat['confiance_sentiment'] * 100:.2f}%"
            )

            print(
                "Niveau sentiment :",
                resultat["niveau_sentiment"]
            )

            print(
                "\nEmotion :",
                resultat["emotion"]
            )

            print(
                "Confiance émotion :",
                f"{resultat['confiance_emotion'] * 100:.2f}%"
            )

            print(
                "Niveau émotion :",
                resultat["niveau_emotion"]
            )

        except Exception as erreur:

            print(
                "Erreur pendant l'analyse :",
                erreur
            )

    print("\n" + "=" * 60)
    print("FIN DES TESTS")
    print("=" * 60)
