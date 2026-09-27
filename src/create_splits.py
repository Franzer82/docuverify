import random
from pathlib import Path

import pandas as pd

MANIFEST_FILE = Path("data/manifest.csv")
RANDOM_SPLIT_DIR = Path("data/splits_random")
COUNTRY_SPLIT_DIR = Path("data/splits_country")

RANDOM_SEED = 42

# Reduziert von 3 auf 2 Test-Länder: das gibt dem Modell mehr Trainings-
# vielfalt (8 statt 7 Länder), während die Generalisierungsprüfung
# (unbekannte Länder im Test) erhalten bleibt.
TEST_COUNTRIES = ["lva", "svk"]


def create_random_split(manifest: pd.DataFrame, seed: int = RANDOM_SEED):
    shuffled = manifest.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_total = len(shuffled)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    train_df = shuffled.iloc[:n_train]
    val_df = shuffled.iloc[n_train:n_train + n_val]
    test_df = shuffled.iloc[n_train + n_val:]

    return train_df, val_df, test_df


def create_country_split(manifest: pd.DataFrame, test_countries: list, seed: int = RANDOM_SEED):
    test_df = manifest[manifest["country"].isin(test_countries)].reset_index(drop=True)
    remaining_df = manifest[~manifest["country"].isin(test_countries)].reset_index(drop=True)

    shuffled_remaining = remaining_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n_val = int(len(shuffled_remaining) * 0.1)

    val_df = shuffled_remaining.iloc[:n_val]
    train_df = shuffled_remaining.iloc[n_val:]

    return train_df, val_df, test_df


def print_split_summary(name: str, train_df, val_df, test_df):
    print(f"\n=== {name} ===")
    print(f"Training:    {len(train_df):4d} Bilder | "
          f"{train_df['label'].sum()} gefälscht, {len(train_df) - train_df['label'].sum()} echt | "
          f"Länder: {sorted(train_df['country'].unique())}")
    print(f"Validierung: {len(val_df):4d} Bilder | "
          f"{val_df['label'].sum()} gefälscht, {len(val_df) - val_df['label'].sum()} echt | "
          f"Länder: {sorted(val_df['country'].unique())}")
    print(f"Test:        {len(test_df):4d} Bilder | "
          f"{test_df['label'].sum()} gefälscht, {len(test_df) - test_df['label'].sum()} echt | "
          f"Länder: {sorted(test_df['country'].unique())}")


if __name__ == "__main__":
    manifest = pd.read_csv(MANIFEST_FILE)

    RANDOM_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train_r, val_r, test_r = create_random_split(manifest)
    train_r.to_csv(RANDOM_SPLIT_DIR / "train.csv", index=False)
    val_r.to_csv(RANDOM_SPLIT_DIR / "val.csv", index=False)
    test_r.to_csv(RANDOM_SPLIT_DIR / "test.csv", index=False)
    print_split_summary("Zufälliger Split", train_r, val_r, test_r)

    COUNTRY_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train_c, val_c, test_c = create_country_split(manifest, TEST_COUNTRIES)
    train_c.to_csv(COUNTRY_SPLIT_DIR / "train.csv", index=False)
    val_c.to_csv(COUNTRY_SPLIT_DIR / "val.csv", index=False)
    test_c.to_csv(COUNTRY_SPLIT_DIR / "test.csv", index=False)
    print_split_summary("Länder-Split (unbekannte Länder im Test)", train_c, val_c, test_c)