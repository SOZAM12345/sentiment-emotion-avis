
from datasets import load_dataset
from pathlib import Path
import pandas as pd

# 1. Définir le dossier principal du projet
BASE_DIR = Path(__file__).resolve().parent.parent

# 2. Définir le dossier de sauvegarde
RAW_DIR = BASE_DIR / "data" / "raw" / "emotion"

RAW_DIR.mkdir(parents=True, exist_ok=True)

# 3. Télécharger le dataset depuis Hugging Face
print("Téléchargement du dataset Emotion...")

dataset = load_dataset("dair-ai/emotion", "split")

# 4. Récupérer les noms des émotions
noms_emotions = dataset["train"].features["label"].names

print("\nÉmotions disponibles :")
for i, nom in enumerate(noms_emotions):
    print(i, ":", nom)

# 5. Sauvegarder train, validation et test
for partie in ["train", "validation", "test"]:

    df = dataset[partie].to_pandas()

    chemin = RAW_DIR / f"{partie}.csv"

    df.to_csv(chemin, index=False)

    print(f"{partie} : {len(df)} textes sauvegardés")

print("\nTéléchargement et sauvegarde terminés !")
