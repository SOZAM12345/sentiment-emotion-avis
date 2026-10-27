
import pandas as pd
from pathlib import Path
from preprocessing import nettoyer_texte

# 1. Définir les chemins des dossiers
BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw" / "allocine"
PROCESSED_DIR = BASE_DIR / "data" / "processed" / "allocine"

# 2. Créer le dossier des données nettoyées
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 3. Traiter chaque partie du dataset
for partie in ["train", "validation", "test"]:

    print(f"\n--- Traitement de {partie} ---")

    # 4. Charger les données originales
    df = pd.read_csv(RAW_DIR / f"{partie}.csv")

    print("Nombre d'avis avant nettoyage :", len(df))

    # 5. Supprimer les doublons exacts
    df = df.drop_duplicates().copy()

    # 6. Appliquer la fonction de nettoyage aux avis
    df["review"] = df["review"].apply(nettoyer_texte)

    # 7. Supprimer les textes vides après nettoyage
    df = df[df["review"].str.len() > 0].copy()

    # 8. Supprimer les doublons apparus après nettoyage
    df = df.drop_duplicates().copy()

    # 9. Sauvegarder les données nettoyées
    chemin_sortie = PROCESSED_DIR / f"{partie}.csv"
    df.to_csv(chemin_sortie, index=False)

    print("Nombre d'avis après nettoyage :", len(df))
    print("Fichier sauvegardé :", chemin_sortie)

print("\nPréparation du dataset Allociné terminée !")
