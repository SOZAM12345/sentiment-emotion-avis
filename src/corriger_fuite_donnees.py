
import pandas as pd
from pathlib import Path

# 1. Chemin des données nettoyées
BASE_DIR = Path(__file__).resolve().parent.parent
DOSSIER = BASE_DIR / "data" / "processed" / "allocine"

# 2. Charger les trois ensembles
train = pd.read_csv(DOSSIER / "train.csv")
val = pd.read_csv(DOSSIER / "validation.csv")
test = pd.read_csv(DOSSIER / "test.csv")

# 3. Vérifier les tailles initiales
print("AVANT CORRECTION")
print("Train :", len(train))
print("Validation :", len(val))
print("Test :", len(test))

# 4. Conserver le train
textes_train = set(train["review"])

# 5. Supprimer de validation les avis présents dans train
val = val[~val["review"].isin(textes_train)].copy()

# 6. Supprimer de test les avis présents dans train ou validation
textes_interdits = textes_train.union(set(val["review"]))

test = test[~test["review"].isin(textes_interdits)].copy()

# 7. Sauvegarder les données corrigées
train.to_csv(DOSSIER / "train.csv", index=False)
val.to_csv(DOSSIER / "validation.csv", index=False)
test.to_csv(DOSSIER / "test.csv", index=False)

# 8. Afficher les résultats
print("\nAPRÈS CORRECTION")
print("Train :", len(train))
print("Validation :", len(val))
print("Test :", len(test))

# 9. Vérifier une dernière fois
print("\nVÉRIFICATION")
print("Train / Validation :", len(set(train["review"]) & set(val["review"])))
print("Train / Test :", len(set(train["review"]) & set(test["review"])))
print("Validation / Test :", len(set(val["review"]) & set(test["review"])))

print("\nCorrection terminée !")
