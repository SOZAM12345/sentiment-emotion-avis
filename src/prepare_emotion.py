
import pandas as pd
import re
import html
from pathlib import Path

# 1. Chemins du projet
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw" / "emotion"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "emotion"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# 2. Fonction de nettoyage léger
def nettoyer_texte(texte):
    texte = html.unescape(str(texte))

    # Supprimer les balises HTML
    texte = re.sub(r"<[^>]+>", " ", texte)

    # Supprimer les liens
    texte = re.sub(r"https?://\S+|www\.\S+", " ", texte)

    # Corriger les espaces multiples
    texte = re.sub(r"\s+", " ", texte)

    return texte.strip()


# 3. Préparer chaque ensemble
donnees = {}

for partie in ["train", "validation", "test"]:

    print(f"\n--- Préparation de {partie} ---")

    df = pd.read_csv(RAW_DIR / f"{partie}.csv")

    print("Avant nettoyage :", len(df))

    # Supprimer les valeurs manquantes
    df = df.dropna(subset=["text", "label"]).copy()

    # Nettoyer les textes
    df["text"] = df["text"].apply(nettoyer_texte)

    # Retirer les textes vides
    df = df[df["text"].str.len() > 0].copy()

    # Supprimer les doublons selon le texte
    df = df.drop_duplicates(subset=["text"]).copy()

    donnees[partie] = df

    print("Après nettoyage :", len(df))


# 4. Supprimer les textes communs entre ensembles
# Priorité : train, puis validation, puis test

train = donnees["train"]

validation = donnees["validation"]
test = donnees["test"]

textes_train = set(train["text"])

validation = validation[
    ~validation["text"].isin(textes_train)
].copy()

textes_validation = set(validation["text"])

textes_exclus = textes_train | textes_validation

test = test[
    ~test["text"].isin(textes_exclus)
].copy()


# 5. Sauvegarder les données
donnees_finales = {
    "train": train,
    "validation": validation,
    "test": test
}

for partie, df in donnees_finales.items():

    chemin = PROCESSED_DIR / f"{partie}.csv"

    df.to_csv(chemin, index=False)

    print(f"{partie} : {len(df)} textes sauvegardés")


# 6. Vérifier les fuites de données
print("\n--- Vérification finale ---")

train_textes = set(train["text"])
val_textes = set(validation["text"])
test_textes = set(test["text"])

print("Train / Validation :", len(train_textes & val_textes))
print("Train / Test :", len(train_textes & test_textes))
print("Validation / Test :", len(val_textes & test_textes))

print("\nPrétraitement terminé !")
