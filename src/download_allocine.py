from datasets import load_dataset
import os

print("Téléchargement du dataset Allociné...")

dataset = load_dataset("tblard/allocine")

os.makedirs("data/raw/allocine", exist_ok=True)

dataset["train"].to_csv("data/raw/allocine/train.csv", index=False)
dataset["validation"].to_csv("data/raw/allocine/validation.csv", index=False)
dataset["test"].to_csv("data/raw/allocine/test.csv", index=False)

print("Dataset Allociné sauvegardé dans data/raw/allocine/")