import json
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/SIDTD/SIDTD/SIDTD/templates")
IMAGES_DIR = DATA_DIR / "Images"
ANNOTATIONS_DIR = DATA_DIR / "Annotations"

OUTPUT_FILE = Path("data/manifest.csv")

# Erkennt das Länderkürzel am Dateianfang, z.B. "alb_id_00.jpg" -> "alb"
COUNTRY_PATTERN = re.compile(r"^([a-z]{3})_")


def extract_country(filename: str) -> str:
    """Extrahiert das 3-Buchstaben-Länderkürzel aus dem Dateinamen."""
    match = COUNTRY_PATTERN.match(filename)
    return match.group(1) if match else "unknown"


def build_manifest() -> pd.DataFrame:
    """Baut eine zentrale Übersichtstabelle aller Bilder: Pfad, Label
    (0=echt, 1=gefälscht), Land, sowie bei Fälschungen zusätzlich den
    Fälschungstyp und das manipulierte Feld aus der Annotation-Datei."""
    rows = []

    # Echte Dokumente einsammeln
    real_files = sorted((IMAGES_DIR / "reals").glob("*.jpg"))
    for path in real_files:
        rows.append({
            "filepath": str(path),
            "label": 0,
            "country": extract_country(path.name),
            "fake_type": None,
            "manipulated_field": None,
        })

    # Gefälschte Dokumente + zugehörige Annotationen einsammeln
    fake_files = sorted((IMAGES_DIR / "fakes").glob("*.jpg"))
    for path in fake_files:
        annotation_path = ANNOTATIONS_DIR / "fakes" / f"{path.stem}.json"

        fake_type = None
        manipulated_field = None
        if annotation_path.exists():
            with open(annotation_path, "r", encoding="utf-8") as f:
                annotation = json.load(f)
            fake_type = annotation.get("ctype")
            manipulated_field = annotation.get("field")

        rows.append({
            "filepath": str(path),
            "label": 1,
            "country": extract_country(path.name),
            "fake_type": fake_type,
            "manipulated_field": manipulated_field,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    manifest = build_manifest()
    manifest.to_csv(OUTPUT_FILE, index=False)

    print(f"Manifest gespeichert unter: {OUTPUT_FILE}")
    print(f"Gesamtzahl Bilder: {len(manifest)}\n")

    print("=== Verteilung echt/gefälscht ===")
    print(manifest["label"].value_counts().rename({0: "echt", 1: "gefälscht"}))

    print("\n=== Anzahl Länder ===")
    print(f"{manifest['country'].nunique()} unterschiedliche Länderkürzel gefunden")

    print("\n=== Bilder pro Land ===")
    print(manifest["country"].value_counts())

    print("\n=== Häufigste Fälschungstypen ===")
    print(manifest["fake_type"].value_counts())