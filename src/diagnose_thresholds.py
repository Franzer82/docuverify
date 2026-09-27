from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from pipeline import load_pipeline, predict_and_explain

SPLIT_DIR = Path("data/splits_country")


def main():
    pipeline = load_pipeline()
    test_df = pd.read_csv(SPLIT_DIR / "test.csv")

    results = []
    for _, row in test_df.iterrows():
        image = Image.open(row["filepath"])
        result = predict_and_explain(image, pipeline)
        results.append({
            "country": row["country"],
            "true_label": "gefälscht" if row["label"] == 1 else "echt",
            "fake_probability": result["fake_probability"],
        })

    results_df = pd.DataFrame(results)

    print("=== Durchschnittliche Fälschungs-Wahrscheinlichkeit, pro Land UND echtem/gefälschtem Label ===\n")
    summary = results_df.groupby(["country", "true_label"])["fake_probability"].agg(["mean", "min", "max", "count"])
    print(summary)

    print("\n=== Nur ECHTE Dokumente, wie viele werden ueber 70% als 'gefaelscht' eingestuft? ===")
    real_only = results_df[results_df["true_label"] == "echt"]
    for country in real_only["country"].unique():
        country_reals = real_only[real_only["country"] == country]
        above_threshold = (country_reals["fake_probability"] >= 0.70).sum()
        print(f"{country}: {above_threshold}/{len(country_reals)} echte Dokumente werden als 'gefälscht' (>70%) eingestuft")


if __name__ == "__main__":
    main()