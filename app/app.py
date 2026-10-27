# ============================================================
# PROJET : ANALYSE AUTOMATIQUE DES SENTIMENTS ET DES EMOTIONS
# FICHIER : app/app.py
# APPLICATION STREAMLIT - VERSION PROFESSIONNELLE
# ============================================================

import html
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except ImportError:  # l'application reste utilisable sans plotly
    PLOTLY_OK = False


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from pipeline_prediction import analyser_avis  # noqa: E402

st.set_page_config(
    page_title="Analyse des avis | Sentiments et émotions",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEUIL_CONFIANCE = 0.70
HISTORIQUE_PATH = BASE_DIR / "data" / "custom" / "historique_analyses.csv"

# Palette
INK = "#0F1B2D"
ACCENT = "#0E7C86"
MUTED = "#64748B"
BORDER = "#E2E8F0"

COULEURS_SENTIMENT = {
    "Positif": "#2F855A",
    "Négatif": "#C53030",
    "Neutre": "#718096",
}

COULEURS_EMOTION = {
    "Joie": "#D69E2E",
    "Amour": "#D53F8C",
    "Colère": "#C53030",
    "Tristesse": "#3182CE",
    "Peur": "#6B46C1",
    "Surprise": "#DD6B20",
}

COULEUR_DEFAUT = "#94A3B8"
PALETTE_LANGUES = ["#0E7C86", "#3182CE", "#D69E2E", "#6B46C1", "#94A3B8"]


# ============================================================
# 2. STYLE CSS
# ============================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, .stApp, [class*="css"] {
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
}
.stApp { background: #F5F7FA; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1180px;
}

h1, h2, h3 { letter-spacing: -0.3px; color: #0F1B2D; }
h3 { font-size: 1.1rem; font-weight: 600; }

/* ---------- Barre latérale ---------- */
section[data-testid="stSidebar"] {
    background: #0F1B2D;
    border-right: none;
}
section[data-testid="stSidebar"] * { color: #D9E1EC; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12); }
.sb-titre { font-size: 1.15rem; font-weight: 600; color: #FFFFFF !important; margin-bottom: .2rem; }
.sb-sous  { font-size: .85rem; color: #93A4BC !important; line-height: 1.4; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: .45rem .6rem;
    border-radius: 6px;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.07);
}

/* ---------- En-tête de page ---------- */
.page-header {
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 1rem;
    margin-bottom: 1.6rem;
}
.page-header h1 { font-size: 1.85rem; font-weight: 600; margin: 0 0 .25rem 0; padding: 0; }
.page-header p  { color: #64748B; margin: 0; font-size: 1rem; max-width: 70ch; }

/* ---------- Cartes ---------- */
.card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    height: 100%;
}
.card-label { color: #64748B; font-size: .85rem; font-weight: 500; margin-bottom: .6rem; }
.card-note  { color: #64748B; font-size: .85rem; margin-top: .7rem; }

.badge {
    display: inline-block;
    padding: .35rem .95rem;
    border-radius: 999px;
    color: #FFFFFF;
    font-weight: 600;
    font-size: 1.15rem;
}
.chip {
    display: inline-block;
    padding: .2rem .65rem;
    border: 1px solid #E2E8F0;
    border-radius: 999px;
    background: #FFFFFF;
    color: #334155;
    font-size: .85rem;
    margin: 0 .35rem .35rem 0;
}

.conf-row   { display: flex; justify-content: space-between; font-size: .85rem; color: #64748B; margin: 1rem 0 .35rem 0; }
.conf-track { height: 8px; background: #E8EDF3; border-radius: 99px; overflow: hidden; }
.conf-fill  { height: 100%; border-radius: 99px; }

.alerte {
    margin-top: .9rem;
    padding: .55rem .8rem;
    border-left: 3px solid #D69E2E;
    background: #FEF9E7;
    color: #7A5B0B;
    font-size: .88rem;
    border-radius: 0 6px 6px 0;
}

/* ---------- Pipeline ---------- */
.pipeline { display: flex; flex-wrap: wrap; gap: .6rem; margin: .4rem 0 .5rem 0; }
.etape {
    flex: 1 1 150px;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-top: 3px solid #0E7C86;
    border-radius: 6px;
    padding: .8rem .9rem;
}
.etape b { display: block; font-size: .95rem; color: #0F1B2D; margin-bottom: .15rem; }
.etape span { font-size: .82rem; color: #64748B; }

/* ---------- Fonctionnalités ---------- */
.fonction { border-left: 3px solid #0E7C86; padding: .1rem 0 .1rem .9rem; margin-bottom: 1.1rem; }
.fonction b { display: block; color: #0F1B2D; margin-bottom: .1rem; }
.fonction span { color: #64748B; font-size: .92rem; }

/* ---------- Composants Streamlit ---------- */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: .9rem 1.1rem;
}
div[data-testid="stMetricLabel"] { color: #64748B; }
div[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 1.2rem;
}
div[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
}
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    border-radius: 6px;
    font-weight: 500;
}
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
    background: #0E7C86;
    border-color: #0E7C86;
    color: #FFFFFF;
}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {
    background: #0B656D;
    border-color: #0B656D;
}
.pied { color: #94A3B8; font-size: .82rem; text-align: center; padding-top: .5rem; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# 3. OUTILS D'AFFICHAGE
# ============================================================

def e(texte):
    """Échapper un texte avant de l'insérer dans du HTML."""
    return html.escape(str(texte))


def en_tete_page(titre, sous_titre):
    st.markdown(
        f'<div class="page-header"><h1>{e(titre)}</h1>'
        f"<p>{e(sous_titre)}</p></div>",
        unsafe_allow_html=True,
    )


def afficher_tableau(df, **kwargs):
    """Afficher un DataFrame pleine largeur (compatible anciennes et récentes versions)."""
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except Exception:
        st.dataframe(df, use_container_width=True, **kwargs)


def afficher_graphique(fig):
    try:
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    except Exception:
        st.plotly_chart(
            fig, use_container_width=True, config={"displayModeBar": False}
        )


def pourcentage(score):
    """Convertir un score entre 0 et 1 en pourcentage."""
    if score is None or pd.isna(score):
        return None
    return round(float(score) * 100, 2)


def cle(valeur):
    """Normaliser un libellé : minuscules, sans accents ni espaces autour."""
    texte = unicodedata.normalize("NFD", str(valeur).strip().lower())
    return "".join(c for c in texte if unicodedata.category(c) != "Mn")


def couleur_de(valeur, table):
    for nom, couleur in table.items():
        if cle(nom) == cle(valeur):
            return couleur
    return COULEUR_DEFAUT


def carte_prediction(titre, prediction, score, table_couleurs):
    """Carte HTML : étiquette colorée + barre de confiance."""
    couleur = couleur_de(prediction, table_couleurs)

    if score is None:
        confiance = (
            '<div class="card-note">Score de confiance indisponible.</div>'
        )
    else:
        valeur = max(0.0, min(1.0, float(score)))
        confiance = (
            f'<div class="conf-row"><span>Confiance du modèle</span>'
            f"<b>{valeur * 100:.1f} %</b></div>"
            f'<div class="conf-track"><div class="conf-fill" '
            f'style="width:{valeur * 100:.1f}%; background:{couleur};"></div></div>'
        )
        if valeur < SEUIL_CONFIANCE:
            confiance += (
                '<div class="alerte">Prédiction incertaine : '
                "le score est inférieur à 70 %.</div>"
            )

    st.markdown(
        f'<div class="card"><div class="card-label">{e(titre)}</div>'
        f'<span class="badge" style="background:{couleur};">{e(prediction)}</span>'
        f"{confiance}</div>",
        unsafe_allow_html=True,
    )


def afficher_resultat(resultat):
    """Afficher le résultat détaillé d'un avis."""

    st.markdown(
        f'<span class="chip">Langue détectée : {e(resultat["langue"])}</span>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        carte_prediction(
            "Sentiment",
            resultat["sentiment"],
            resultat.get("confiance_sentiment"),
            COULEURS_SENTIMENT,
        )
    with col2:
        carte_prediction(
            "Émotion",
            resultat["emotion"],
            resultat.get("confiance_emotion"),
            COULEURS_EMOTION,
        )

    st.caption(
        "Les scores sont des probabilités estimées par les modèles. "
        "Ils ne garantissent pas que la prédiction est correcte."
    )

    with st.expander("Voir les traductions"):
        st.write("**Texte français :**", resultat["texte_francais"])
        st.write("**Texte anglais :**", resultat["texte_anglais"])


def convertir_resultat(resultat):
    """Transformer un résultat en ligne de tableau."""
    return {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Avis": resultat["avis"],
        "Langue": resultat["langue"],
        "Sentiment": resultat["sentiment"],
        "Confiance sentiment (%)": pourcentage(resultat.get("confiance_sentiment")),
        "Émotion": resultat["emotion"],
        "Confiance émotion (%)": pourcentage(resultat.get("confiance_emotion")),
    }


# ============================================================
# 4. GRAPHIQUES
# ============================================================

def graphique_repartition(serie, titre, table_couleurs=None, donut=False):
    """Répartition d'une colonne catégorielle (donut ou barres horizontales)."""

    comptes = serie.dropna().value_counts()
    if comptes.empty:
        st.info("Aucune donnée à afficher.")
        return

    st.markdown(f"**{titre}**")

    if not PLOTLY_OK:
        st.bar_chart(comptes)
        return

    if table_couleurs is not None:
        couleurs = [couleur_de(k, table_couleurs) for k in comptes.index]
    else:
        couleurs = [PALETTE_LANGUES[i % len(PALETTE_LANGUES)] for i in range(len(comptes))]

    if donut:
        fig = go.Figure(
            go.Pie(
                labels=list(comptes.index),
                values=list(comptes.values),
                hole=0.62,
                marker=dict(colors=couleurs, line=dict(color="#FFFFFF", width=2)),
                textinfo="percent",
                sort=False,
            )
        )
    else:
        comptes = comptes.sort_values()
        couleurs = [couleur_de(k, table_couleurs) if table_couleurs else ACCENT
                    for k in comptes.index]
        fig = go.Figure(
            go.Bar(
                x=list(comptes.values),
                y=list(comptes.index),
                orientation="h",
                marker_color=couleurs,
                text=list(comptes.values),
                textposition="outside",
            )
        )
        fig.update_xaxes(showgrid=True, gridcolor="#EDF1F5", zeroline=False)
        fig.update_yaxes(showgrid=False)

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, sans-serif", color="#334155", size=13),
        legend=dict(orientation="h", y=-0.05),
        showlegend=donut,
    )
    afficher_graphique(fig)


# ============================================================
# 5. HISTORIQUE
# ============================================================

def charger_historique():
    """Lire les analyses sauvegardées."""
    if not HISTORIQUE_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(HISTORIQUE_PATH)
    except (OSError, pd.errors.ParserError) as erreur:
        st.error(f"Impossible de lire l'historique : {erreur}")
        return pd.DataFrame()


def enregistrer_historique(liste_resultats):
    """Enregistrer des analyses dans un CSV local."""
    if not liste_resultats:
        return

    HISTORIQUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    nouvelles_lignes = pd.DataFrame(liste_resultats)
    ancien = charger_historique()

    if not ancien.empty:
        nouvelles_lignes = pd.concat([ancien, nouvelles_lignes], ignore_index=True)

    nouvelles_lignes.to_csv(HISTORIQUE_PATH, index=False, encoding="utf-8-sig")


# ============================================================
# 6. STATISTIQUES
# ============================================================

def afficher_statistiques(df):
    """Afficher les statistiques d'un ensemble d'avis."""

    if df.empty or "Sentiment" not in df.columns:
        st.info("Aucune prédiction disponible.")
        return

    sentiments = df["Sentiment"].dropna()
    total = len(sentiments)

    if total == 0:
        st.info("Aucune prédiction disponible.")
        return

    positifs = int((sentiments.map(cle) == "positif").sum())
    negatifs = int((sentiments.map(cle) == "negatif").sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avis analysés", total)
    col2.metric("Positifs", f"{positifs}", f"{positifs / total * 100:.1f} %", delta_color="off")
    col3.metric("Négatifs", f"{negatifs}", f"{negatifs / total * 100:.1f} %", delta_color="off")

    if "Confiance sentiment (%)" in df.columns:
        moyenne = pd.to_numeric(df["Confiance sentiment (%)"], errors="coerce").mean()
        col4.metric("Confiance moyenne", "-" if pd.isna(moyenne) else f"{moyenne:.1f} %")

    st.write("")
    g1, g2 = st.columns(2, gap="large")

    with g1:
        graphique_repartition(
            sentiments, "Répartition des sentiments", COULEURS_SENTIMENT, donut=True
        )
    with g2:
        if "Émotion" in df.columns:
            graphique_repartition(
                df["Émotion"], "Répartition des émotions", COULEURS_EMOTION
            )

    if "Langue" in df.columns and df["Langue"].dropna().nunique() > 1:
        graphique_repartition(df["Langue"], "Langues détectées", None)


# ============================================================
# 7. RECOMMANDATIONS
# ============================================================

RECO_EMOTIONS = {
    "colère": "Vérifier si l'avis signale un problème de service, de qualité ou de délai.",
    "tristesse": "Étudier les attentes non satisfaites et envisager une réponse attentive.",
    "joie": "Identifier les expériences positives à préserver.",
    "amour": "Repérer les éléments qui suscitent un fort attachement.",
    "peur": "Examiner les inquiétudes éventuelles concernant la fiabilité du service.",
    "surprise": "Vérifier si la surprise exprimée correspond à une expérience positive ou négative.",
}


def generer_recommandations(sentiment, emotion):
    """Générer des conseils à partir des prédictions."""

    conseils = []

    if cle(sentiment) == "positif":
        conseils.append(
            "Identifier les éléments appréciés dans le commentaire "
            "pour conserver les points forts du produit ou du service."
        )
    elif cle(sentiment) == "negatif":
        conseils.append(
            "Examiner les causes de l'insatisfaction afin de proposer des améliorations."
        )

    # Recherche insensible à la casse ("Colère", "colère", "COLÈRE"...)
    conseil_emotion = {cle(k): v for k, v in RECO_EMOTIONS.items()}.get(cle(emotion))
    if conseil_emotion:
        conseils.append(conseil_emotion)

    return conseils


def afficher_conseils(conseils):
    if not conseils:
        st.info("Aucune recommandation disponible pour ce résultat.")
        return
    for conseil in conseils:
        st.markdown(
            f'<div class="fonction"><span style="color:#1F2937;">{e(conseil)}</span></div>',
            unsafe_allow_html=True,
        )


# ============================================================
# 8. NAVIGATION
# ============================================================

st.sidebar.markdown(
    '<div class="sb-titre">Analyse des avis</div>'
    '<div class="sb-sous">Sentiments et émotions dans les avis utilisateurs</div>',
    unsafe_allow_html=True,
)
st.sidebar.write("")

page = st.sidebar.radio(
    "Navigation",
    ["Tableau de bord", "Détection", "Performance", "Historique", "Recommandations"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown(
    '<div class="sb-sous">Projet Data Science<br>NLP · Machine Learning · Deep Learning</div>',
    unsafe_allow_html=True,
)


# ============================================================
# PAGE 1 : TABLEAU DE BORD
# ============================================================

if page == "Tableau de bord":

    en_tete_page(
        "Tableau de bord",
        "Mesurer la satisfaction des clients en analysant automatiquement "
        "le sentiment et l'émotion de leurs avis, en français ou en anglais.",
    )

    historique = charger_historique()

    st.subheader("Vue d'ensemble")

    if historique.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Analyses enregistrées", 0)
        c2.metric("Avis positifs", 0)
        c3.metric("Avis négatifs", 0)
        st.caption(
            "Aucune analyse enregistrée. Rendez-vous dans Détection "
            "pour analyser un premier avis."
        )
    else:
        afficher_statistiques(historique)

    st.write("")
    st.subheader("Fonctionnement")

    etapes = [
        ("Avis", "Texte saisi ou fichier CSV"),
        ("Langue", "Détection automatique"),
        ("Traduction", "Vers le français et l'anglais"),
        ("Modèles", "Sentiment et émotion"),
        ("Résultats", "Scores et recommandations"),
    ]
    st.markdown(
        '<div class="pipeline">'
        + "".join(f'<div class="etape"><b>{e(t)}</b><span>{e(d)}</span></div>' for t, d in etapes)
        + "</div>",
        unsafe_allow_html=True,
    )

    st.write("")
    st.subheader("Fonctionnalités")

    fonctions = [
        ("Détection de la langue", "Reconnaissance automatique du français et de l'anglais."),
        ("Analyse des sentiments", "Classification des avis en positifs ou négatifs."),
        ("Détection des émotions", "Six émotions : joie, tristesse, amour, colère, peur et surprise."),
        ("Scores de confiance", "Probabilité estimée par les modèles pour chaque prédiction."),
        ("Alerte d'incertitude", "Avertissement lorsque la confiance est inférieure à 70 %."),
        ("Analyse par fichier CSV", "Traitement de plusieurs commentaires en une seule fois."),
        ("Comparaison de formulations", "Effet de la formulation sur les prédictions."),
        ("Recommandations", "Conseils fondés sur le sentiment et l'émotion détectés."),
    ]
    c1, c2 = st.columns(2, gap="large")
    for i, (titre, desc) in enumerate(fonctions):
        with (c1 if i % 2 == 0 else c2):
            st.markdown(
                f'<div class="fonction"><b>{e(titre)}</b><span>{e(desc)}</span></div>',
                unsafe_allow_html=True,
            )

    st.subheader("Technologies")
    technos = [
        "Python", "Pandas", "Scikit-learn", "TF-IDF", "Régression logistique",
        "DistilBERT", "Transformers", "Streamlit",
    ]
    st.markdown("".join(f'<span class="chip">{e(t)}</span>' for t in technos), unsafe_allow_html=True)


# ============================================================
# PAGE 2 : DETECTION
# ============================================================

elif page == "Détection":

    en_tete_page(
        "Détection",
        "Analysez un commentaire ou un fichier complet pour obtenir "
        "le sentiment et l'émotion dominante.",
    )

    mode = st.radio(
        "Mode d'analyse",
        ["Un commentaire", "Un fichier CSV"],
        horizontal=True,
    )
    st.write("")

    # --------------------------------------------------------
    # MODE 1 : UN COMMENTAIRE
    # --------------------------------------------------------
    if mode == "Un commentaire":

        with st.form("formulaire_avis"):
            avis = st.text_area(
                "Commentaire à analyser",
                placeholder="Exemple : Ce restaurant est excellent !",
                height=140,
            )
            bouton_analyser = st.form_submit_button("Analyser le commentaire", type="primary")

        if bouton_analyser:
            if not avis.strip():
                st.warning("Saisissez un commentaire avant de lancer l'analyse.")
            else:
                try:
                    with st.spinner("Analyse en cours..."):
                        resultat = analyser_avis(avis)
                    st.session_state["dernier_resultat"] = resultat
                except Exception as erreur:
                    st.session_state.pop("dernier_resultat", None)
                    st.error(f"L'analyse a échoué : {erreur}")

        if "dernier_resultat" in st.session_state:
            resultat = st.session_state["dernier_resultat"]

            st.write("")
            st.subheader("Résultat")
            afficher_resultat(resultat)

            st.write("")
            st.subheader("Recommandations rapides")
            afficher_conseils(
                generer_recommandations(resultat["sentiment"], resultat["emotion"])
            )

            if st.button("Enregistrer dans l'historique"):
                enregistrer_historique([convertir_resultat(resultat)])
                st.success("Analyse enregistrée.")

        # ---------------- Comparaison de formulations ----------------
        st.write("")
        with st.expander("Comparer plusieurs formulations"):
            st.write("Écrivez entre 2 et 10 phrases, une par ligne.")

            with st.form("formulaire_comparaison"):
                phrases_texte = st.text_area(
                    "Phrases à comparer",
                    value=(
                        "Ce produit est excellent !\n"
                        "Je suis très heureux de ce produit.\n"
                        "Ce produit est mauvais."
                    ),
                    height=150,
                )
                lancer_comparaison = st.form_submit_button("Comparer les phrases")

            if lancer_comparaison:
                phrases = [p.strip() for p in phrases_texte.splitlines() if p.strip()]

                if not 2 <= len(phrases) <= 10:
                    st.warning("Saisissez entre 2 et 10 phrases.")
                else:
                    lignes = []
                    barre = st.progress(0)

                    for i, phrase in enumerate(phrases):
                        try:
                            ligne = convertir_resultat(analyser_avis(phrase))
                            ligne["Erreur"] = ""
                        except Exception as erreur:
                            ligne = {"Avis": phrase, "Erreur": str(erreur)}
                        lignes.append(ligne)
                        barre.progress((i + 1) / len(phrases))

                    barre.empty()
                    st.session_state["resultats_comparaison"] = pd.DataFrame(lignes)

            if "resultats_comparaison" in st.session_state:
                st.markdown("**Résultats de la comparaison**")
                afficher_tableau(st.session_state["resultats_comparaison"])

    # --------------------------------------------------------
    # MODE 2 : FICHIER CSV
    # --------------------------------------------------------
    else:

        st.write("Importez un fichier CSV contenant une colonne de commentaires.")

        fichier = st.file_uploader("Choisir un fichier CSV", type=["csv"])

        if fichier is not None:
            try:
                df = pd.read_csv(fichier)
            except Exception as erreur:
                st.error(f"Impossible de lire le CSV : {erreur}")
                df = pd.DataFrame()

            if df.empty:
                st.warning("Le fichier est vide ou illisible.")
            else:
                st.success(f"Fichier chargé : {len(df)} lignes.")

                st.markdown("**Aperçu des données**")
                afficher_tableau(df.head(10))

                col_a, col_b = st.columns(2)
                with col_a:
                    colonne = st.selectbox(
                        "Colonne des avis", df.columns.tolist()
                    )
                with col_b:
                    nombre = st.number_input(
                        "Nombre de commentaires à analyser",
                        min_value=1,
                        max_value=len(df),
                        value=min(5, len(df)),
                        step=1,
                    )

                if st.button("Lancer l'analyse", type="primary"):
                    lignes = []
                    barre = st.progress(0)

                    for i, texte in enumerate(df[colonne].head(int(nombre))):
                        if pd.isna(texte) or not str(texte).strip():
                            ligne = {"Avis": texte, "Erreur": "Avis vide"}
                        else:
                            try:
                                ligne = convertir_resultat(analyser_avis(str(texte)))
                                ligne["Erreur"] = ""
                            except Exception as erreur:
                                ligne = {"Avis": texte, "Erreur": str(erreur)}
                        lignes.append(ligne)
                        barre.progress((i + 1) / int(nombre))

                    barre.empty()
                    st.session_state["resultats_csv"] = pd.DataFrame(lignes)
                    st.session_state["csv_fichier_source"] = fichier.name

        if "resultats_csv" in st.session_state:
            resultats = st.session_state["resultats_csv"]

            st.divider()
            st.subheader("Résultats")
            afficher_tableau(resultats)

            st.write("")
            st.subheader("Statistiques")
            afficher_statistiques(resultats)

            st.write("")
            col_dl, col_sv = st.columns([1, 1])

            with col_dl:
                st.download_button(
                    "Télécharger les résultats (CSV)",
                    data=resultats.to_csv(index=False).encode("utf-8-sig"),
                    file_name="resultats_analyse.csv",
                    mime="text/csv",
                )

            with col_sv:
                if st.button("Enregistrer dans l'historique"):
                    if "Date" in resultats.columns:
                        lignes_valides = resultats[resultats["Date"].notna()].to_dict("records")
                        if lignes_valides:
                            enregistrer_historique(lignes_valides)
                            st.success("Analyses enregistrées.")
                        else:
                            st.info("Aucune analyse valide à enregistrer.")


# ============================================================
# PAGE 3 : PERFORMANCE DES MODELES
# ============================================================

elif page == "Performance":

    en_tete_page(
        "Performance des modèles",
        "Résultats obtenus lors des expérimentations du projet.",
    )

    # Valeurs issues des expériences ; à confirmer avec les notebooks.
    performances = pd.DataFrame([
        {"Modèle": "Régression logistique", "Tâche": "Sentiment",
         "Dataset": "Allociné", "Accuracy (%)": 93.45, "F1 macro": None},
        {"Modèle": "Régression logistique", "Tâche": "Émotion",
         "Dataset": "Emotion", "Accuracy (%)": 87.91, "F1 macro": 0.8510},
        {"Modèle": "DistilBERT amélioré", "Tâche": "Émotion",
         "Dataset": "Emotion", "Accuracy (%)": 91.82, "F1 macro": 0.8903},
    ])

    st.subheader("Tableau des performances")
    afficher_tableau(performances, hide_index=True)

    st.write("")
    st.subheader("Accuracy par expérience")

    performances["Expérience"] = performances["Modèle"] + " (" + performances["Tâche"] + ")"
    tri = performances.sort_values("Accuracy (%)")

    if PLOTLY_OK:
        couleurs_tache = {"Sentiment": ACCENT, "Émotion": "#3182CE"}
        fig = go.Figure(
            go.Bar(
                x=tri["Accuracy (%)"],
                y=tri["Expérience"],
                orientation="h",
                marker_color=[couleurs_tache.get(t, COULEUR_DEFAUT) for t in tri["Tâche"]],
                text=[f"{v:.2f} %" for v in tri["Accuracy (%)"]],
                textposition="outside",
            )
        )
        fig.update_xaxes(range=[0, 100], showgrid=True, gridcolor="#EDF1F5", zeroline=False)
        fig.update_yaxes(showgrid=False)
        fig.update_layout(
            height=280,
            margin=dict(l=10, r=40, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="IBM Plex Sans, sans-serif", color="#334155", size=13),
        )
        afficher_graphique(fig)
    else:
        st.bar_chart(performances.set_index("Expérience")["Accuracy (%)"])

    st.info(
        "Ces valeurs proviennent des résultats rapportés lors des expériences. "
        "Vérifiez-les dans les notebooks avant le rapport final. "
        "Les tâches sentiment et émotion ne se comparent pas directement."
    )

    st.write("")
    st.subheader("Modèles utilisés dans l'application")
    m1, m2 = st.columns(2, gap="medium")
    with m1:
        st.markdown(
            '<div class="card"><div class="card-label">Sentiment</div>'
            "<b>TF-IDF + Régression logistique</b></div>",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            '<div class="card"><div class="card-label">Émotion</div>'
            "<b>DistilBERT amélioré</b></div>",
            unsafe_allow_html=True,
        )

    st.write("")
    st.subheader("Comprendre les métriques")

    with st.expander("Accuracy"):
        st.write("Pourcentage de prédictions correctes sur les données d'évaluation.")
    with st.expander("Précision"):
        st.write(
            "Pour une classe, proportion des prédictions de cette classe qui sont correctes."
        )
    with st.expander("Rappel"):
        st.write(
            "Capacité du modèle à retrouver les véritables exemples d'une classe."
        )
    with st.expander("F1-score"):
        st.write(
            "Combine la précision et le rappel. Le F1 macro donne le même poids à chaque classe."
        )

    st.caption(
        "Les scores de confiance affichés pendant la détection sont différents "
        "de l'accuracy mesurée sur un jeu d'évaluation."
    )


# ============================================================
# PAGE 4 : HISTORIQUE
# ============================================================

elif page == "Historique":

    en_tete_page(
        "Historique",
        "Toutes les analyses enregistrées sur cet ordinateur.",
    )

    historique = charger_historique()

    if historique.empty:
        st.info("Aucune analyse enregistrée. Commencez dans la page Détection.")

    else:
        historique_filtre = historique

        if "Sentiment" in historique.columns:
            options = ["Tous"] + sorted(historique["Sentiment"].dropna().unique().tolist())
            choix = st.selectbox("Filtrer par sentiment", options)
            if choix != "Tous":
                historique_filtre = historique[historique["Sentiment"] == choix]

        st.subheader("Analyses enregistrées")
        afficher_tableau(historique_filtre)

        st.write("")
        st.subheader("Statistiques")
        afficher_statistiques(historique_filtre)

        st.divider()

        st.download_button(
            "Télécharger l'historique (CSV)",
            data=historique.to_csv(index=False).encode("utf-8-sig"),
            file_name="historique_analyses.csv",
            mime="text/csv",
        )

        st.warning(
            "L'historique est stocké localement dans data/custom/historique_analyses.csv. "
            "N'enregistrez pas de commentaires contenant des informations "
            "personnelles ou confidentielles."
        )

        confirmer = st.checkbox("Je confirme vouloir supprimer tout l'historique.")

        if st.button("Supprimer l'historique", disabled=not confirmer):
            HISTORIQUE_PATH.unlink(missing_ok=True)
            st.success("Historique supprimé.")
            st.rerun()


# ============================================================
# PAGE 5 : RECOMMANDATIONS
# ============================================================

elif page == "Recommandations":

    en_tete_page(
        "Recommandations",
        "Conseils fondés sur le sentiment et l'émotion du dernier commentaire analysé.",
    )

    if "dernier_resultat" not in st.session_state:
        st.info("Analysez d'abord un commentaire dans la page Détection.")

    else:
        resultat = st.session_state["dernier_resultat"]

        st.subheader("Commentaire analysé")
        st.markdown(
            f'<div class="card" style="font-size:1.02rem;">{e(resultat["avis"])}</div>',
            unsafe_allow_html=True,
        )

        st.write("")
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            carte_prediction(
                "Sentiment", resultat["sentiment"],
                resultat.get("confiance_sentiment"), COULEURS_SENTIMENT,
            )
        with col2:
            carte_prediction(
                "Émotion", resultat["emotion"],
                resultat.get("confiance_emotion"), COULEURS_EMOTION,
            )

        st.write("")
        st.subheader("Recommandations proposées")
        afficher_conseils(
            generer_recommandations(resultat["sentiment"], resultat["emotion"])
        )

        score_s = resultat.get("confiance_sentiment")
        score_e = resultat.get("confiance_emotion")

        if (
            score_s is None
            or score_e is None
            or score_s < SEUIL_CONFIANCE
            or score_e < SEUIL_CONFIANCE
        ):
            st.warning(
                "Une prédiction est incertaine ou sans score disponible. "
                "Vérifiez ces recommandations avec prudence."
            )

        st.caption(
            "Les recommandations reposent sur des règles simples. "
            "Elles ne déterminent pas les causes réelles d'un problème."
        )


# ============================================================
# PIED DE PAGE
# ============================================================

st.divider()
st.markdown(
    '<div class="pied">Analyse des sentiments et des émotions · Projet Data Science</div>',
    unsafe_allow_html=True,
)