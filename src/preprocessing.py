
import re
import html


# Fonction pour nettoyer un avis utilisateur
def nettoyer_texte(texte):

    # 1. Convertir les entités HTML
    texte = html.unescape(texte)

    # 2. Supprimer les balises HTML
    texte = re.sub(r"<[^>]+>", " ", texte)

    # 3. Supprimer les liens Internet
    texte = re.sub(r"https?://\S+|www\.\S+", " ", texte)

    # 4. Transformer les majuscules en minuscules
    texte = texte.lower()

    # 5. Supprimer les espaces multiples
    texte = re.sub(r"\s+", " ", texte)

    # 6. Retourner le texte nettoyé
    return texte.strip()


# Tester notre fonction
if __name__ == "__main__":

    avis = "J'ADORE ce film !!! 😍 https://exemple.com"

    print("AVANT NETTOYAGE :")
    print(avis)

    resultat = nettoyer_texte(avis)

    print("\nAPRÈS NETTOYAGE :")
    print(resultat)
