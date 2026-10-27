
import pandas as pd
from pathlib import Path

DOSSIER = Path(__file__).resolve().parent.parent / "data" / "processed" / "allocine"

train = pd.read_csv(DOSSIER / "train.csv")
val = pd.read_csv(DOSSIER / "validation.csv")
test = pd.read_csv(DOSSIER / "test.csv")

# Récupérer les textes de chaque ensemble
textes_train = set(train["review"].dropna())
textes_val = set(val["review"].dropna())
textes_test = set(test["review"].dropna())

# Chercher les textes présents dans plusieurs ensembles
train_val = textes_train.intersection(textes_val)
train_test = textes_train.intersection(textes_test)
val_test = textes_val.intersection(textes_test)

print("Avis communs entre Train et Validation :", len(train_val))
print("Avis communs entre Train et Test :", len(train_test))
print("Avis communs entre Validation et Test :", len(val_test))
