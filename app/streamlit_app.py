import re
from pathlib import Path

import streamlit as st
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import load_pipeline, predict_and_explain

st.set_page_config(page_title="DocuVerify", page_icon="🛂", layout="centered")

FAKE_THRESHOLD = 0.70
REAL_THRESHOLD = 0.30

KNOWN_COUNTRIES = {"alb", "aze", "esp", "est", "fin", "rus", "srb"}

COUNTRY_PATTERN = re.compile(r"^([a-z]{3})_")


def guess_country(filename: str) -> str:
    match = COUNTRY_PATTERN.match(filename.lower())
    return match.group(1) if match else "unbekannt"


def inject_custom_style():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0B1929;
        }

        h1, h2, h3 {
            color: #E8F1FF;
        }

        p, .stCaption, [data-testid="stCaptionContainer"] {
            color: #A8C0DB;
        }

        [data-testid="stFileUploaderDropzone"] {
            background-color: #142943;
            border: 2px dashed #3B82C4;
        }

        [data-testid="stExpander"] {
            background-color: #142943;
            border: 1px solid #2A4A6E;
            border-radius: 8px;
        }

        .verdict-box {
            border-radius: 10px;
            padding: 16px 20px;
            margin: 16px 0;
            font-size: 1.1rem;
            font-weight: 600;
        }
        .verdict-real {
            background-color: #16342A;
            border: 1px solid #2E7D5B;
            color: #B9F5D0;
        }
        .verdict-fake {
            background-color: #3A1E1E;
            border: 1px solid #A83232;
            color: #FFC7C7;
        }
        .verdict-uncertain {
            background-color: #3A3419;
            border: 1px solid #A88A32;
            color: #FFE8B0;
        }
        .verdict-box .subtext {
            font-size: 0.9rem;
            font-weight: 400;
            color: #E8F1FF;
            opacity: 0.9;
        }

        .reliability-known {
            background-color: #16342A;
            border: 1px solid #2E7D5B;
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 0.95rem;
            color: #DFFCE9;
            margin-bottom: 12px;
            line-height: 1.5;
        }
        .reliability-unknown {
            background-color: #3A1E1E;
            border: 2px solid #C4423A;
            border-radius: 8px;
            padding: 14px 16px;
            font-size: 0.95rem;
            font-weight: 600;
            color: #FFE9E9;
            margin-bottom: 12px;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_pipeline():
    return load_pipeline()


def classify_with_uncertainty(fake_probability: float) -> str:
    if fake_probability >= FAKE_THRESHOLD:
        return "fake"
    elif fake_probability <= REAL_THRESHOLD:
        return "real"
    else:
        return "uncertain"


def render_reliability_notice(country: str) -> bool:
    is_known = country in KNOWN_COUNTRIES

    if is_known:
        st.markdown(
            f"""
            <div class="reliability-known">
                ✅ Land erkannt: <b>{country.upper()}</b> - gehört zu den trainierten
                Dokumenttypen. Für dieses Land wurde eine hohe Erkennungsgenauigkeit
                gemessen (ROC-AUC &gt; 0,99 auf zurückgehaltenen Testdaten).
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        country_display = country.upper() if country != "unbekannt" else "nicht erkennbar"
        st.markdown(
            f"""
            <div class="reliability-unknown">
                ⚠️ Land {'erkannt: ' + country_display if country != 'unbekannt' else 'nicht erkennbar'}
                - dieser Dokumenttyp war NICHT Teil des Trainings.<br><br>
                In unseren Tests zeigte das Modell bei unbekannten Ländern ein
                systematisches Fehlverhalten: Es sagt einen länderspezifisch fast
                konstanten Wert voraus, der kaum vom tatsächlichen Inhalt (echt/gefälscht)
                abhängt. Das Ergebnis unten ist bei diesem Dokumenttyp
                <b>nicht verlässlich interpretierbar</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    return is_known


def render_verdict(result: dict, is_known_country: bool):
    fake_prob = result["fake_probability"]
    category = classify_with_uncertainty(fake_prob)

    if category == "fake":
        css_class, icon, label_text = "verdict-fake", "⚠️", "Mögliche Fälschung erkannt"
    elif category == "real":
        css_class, icon, label_text = "verdict-real", "✅", "Als echt eingestuft"
    else:
        css_class, icon, label_text = "verdict-uncertain", "❓", "Unsicher – manuelle Prüfung empfohlen"

    confidence_note = "" if is_known_country else " (geringe Verlässlichkeit - siehe Hinweis oben)"

    st.markdown(
        f"""
        <div class="verdict-box {css_class}">
            {icon} {label_text}{confidence_note}<br>
            <span class="subtext">
                Modell-Wahrscheinlichkeit für Fälschung: {fake_prob:.1%}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    inject_custom_style()

    st.title("🛂 DocuVerify")
    st.caption("Automatisierte Fälschungserkennung für Ausweisdokumente mit Erklärbarkeit (Grad-CAM)")

    st.divider()

    pipeline = get_pipeline()

    # Zaehler in der Session, der ueber den "key" des file_uploader
    # gesteuert wird: Wird er erhoeht, erzeugt Streamlit ein KOMPLETT
    # NEUES Uploader-Widget (leer), statt das alte mit der bisherigen
    # Datei weiterzufuehren. Das ist der Standard-Trick, um einen
    # file_uploader gezielt zuruecksetzen zu koennen, ohne dass die
    # Person zuerst manuell die alte Datei entfernen muss.
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0

    uploaded_file = st.file_uploader(
        "Dokumentbild hochladen (JPG oder PNG)",
        type=["jpg", "jpeg", "png"],
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if uploaded_file is not None:
        if st.button("🔄 Anderes Dokument hochladen"):
            st.session_state.uploader_key += 1
            st.rerun()

        country = guess_country(uploaded_file.name)
        is_known_country = render_reliability_notice(country)

        image = Image.open(uploaded_file)

        with st.spinner("Analysiere Dokument ..."):
            result = predict_and_explain(image, pipeline)

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Grad-CAM Erklärung")
            st.image(result["overlay_image"], use_container_width=True)
            st.caption("Rot/Orange = Bereiche, auf die sich das Modell bei seiner Entscheidung am stärksten stützt.")

        render_verdict(result, is_known_country)

        with st.expander("ℹ️ Details zur Modell-Zuverlässigkeit"):
            st.write(
                "Dieses Modell wurde auf einem synthetischen Forschungsdatensatz (SIDTD) mit 8 "
                "Ländern trainiert (Albanien, Aserbaidschan, Spanien, Estland, Finnland, Russland, "
                "Serbien). Für diese Länder wurde eine Erkennungsgenauigkeit von ROC-AUC > 0,99 "
                "gemessen. Bei drei komplett unbekannten Ländern (Griechenland, Lettland, Slowakei) "
                "zeigte sich dagegen ein deutlicher Leistungsabfall (ROC-AUC 0,68) sowie ein "
                "systematisches 'Shortcut Learning'-Verhalten: Statt echte Manipulationsmerkmale zu "
                "erkennen, sagt das Modell für unbekannte Dokumenttypen einen weitgehend konstanten, "
                "länderspezifischen Wert voraus. Die Grad-CAM-Analyse bestätigt, dass das Modell "
                "primär auf die Position von Datenfeldern reagiert, nicht auf konkrete "
                "Manipulationsspuren. Dieses Tool ist ein Portfolio-/Demonstrationsprojekt und "
                "ersetzt keine professionelle Dokumentenprüfung."
            )


if __name__ == "__main__":
    main()