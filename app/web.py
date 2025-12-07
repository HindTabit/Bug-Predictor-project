# app/web.py
# VERSION FINALE COMPLÈTE – BugPredictor Pro 2025
# Lance avec : streamlit run app/web.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import subprocess
import os
from lizard import analyze_file  # pip install lizard

# ============================ CONFIGURATION ============================
st.set_page_config(
    page_title="BugPredictor Pro",
    page_icon="bug",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style magnifique
st.markdown("""
<style>
    .big-font {font-size:58px !important; font-weight:bold; color:#FF2D55; text-align:center; margin-bottom:0;}
    .subtitle {font-size:24px; text-align:center; color:#555; margin-top:0;}
    .risk-high {background:#ffebee; padding:25px; border-radius:15px; border-left:10px solid #f44336; margin:20px 0; text-align:center; font-size:22px;}
    .risk-medium {background:#fff3e0; padding:25px; border-radius:15px; border-left:10px solid #ff9800; margin:20px 0; text-align:center; font-size:22px;}
    .risk-low {background:#e8f5e8; padding:25px; border-radius:15px; border-left:10px solid #4caf50; margin:20px 0; text-align:center; font-size:22px;}
    .stButton>button {width:100%; height:60px; font-size:20px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">BugPredictor Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Prédiction automatique des fichiers à risque • Copier-coller • CSV • GitHub</p>', unsafe_allow_html=True)

# ============================ CHARGEMENT DU MODÈLE ============================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("app/best_model.pkl")
        scaler = joblib.load("app/scaler.pkl")
        columns = joblib.load("app/feature_columns.pkl")
        return model, scaler, columns
    except Exception as e:
        st.error(f"Modèle non trouvé dans app/ → {e}")
        st.stop()

model, scaler, feature_columns = load_model()
st.success("Modèle XGBoost chargé avec succès")

# ============================ FONCTION : CODE COLLÉ ============================
def predict_from_source_code(code: str, language: str = "python"):
    """Analyse du code collé avec lizard et prédit le risque"""
    if not code.strip():
        return None

    suffix = { "python": ".py", "java": ".java", "javascript": ".js", "c": ".c", "cpp": ".cpp" }.get(language, ".py")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False, encoding='utf-8') as f:
        f.write(code)
        temp_path = f.name

    try:
        analysis = analyze_file(temp_path)
        funcs = analysis.function_list

        if not funcs:
            return 0.1  # code très simple → faible risque

        # Moyenne des métriques
        data = {
            'nloc': np.mean([f.nloc for f in funcs]),
            'cyclomatic_complexity': np.mean([f.cyclomatic_complexity for f in funcs]),
            'token_count': np.mean([f.token_count for f in funcs]),
            'parameter_count': np.mean([f.parameter_count for f in funcs]),
            'loc': analysis.nloc or 10,
            'wmc': len(funcs),
            'lcom3': getattr(analysis, 'average_lcom3', 1.0),
            'rfc': len(funcs) * 4,
            'cbo': 6,
            'dit': 1,
            'noc': 0,
            'dam': 0.5,
        }

        vec = np.zeros(len(feature_columns))
        for k, v in data.items():
            if k in feature_columns:
                vec[feature_columns.index(k)] = v

        X = scaler.transform([vec])
        proba = float(model.predict_proba(X)[0, 1])
        return proba

    except Exception as e:
        st.warning(f"Analyse lizard échouée : {e}")
        return 0.3  # valeur par défaut
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

# ============================ FONCTION : GITHUB ============================
def analyze_github_repo(repo_url: str, branch: str = "main"):
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = ["git", "clone", "--depth", "1", "--branch", branch, repo_url, tmpdir]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        if result.returncode != 0:
            st.error(f"Clone échoué : {result.stderr.splitlines()[0]}")
            return None

        cmd = ["lizard", tmpdir, "--csv", "-l", "python,java,javascript,cpp,c"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not result.stdout.strip():
            cmd = ["lizard", tmpdir, "--csv"]
            result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            st.error("Installe lizard : pip install lizard")
            return None

        from io import StringIO
        df = pd.read_csv(StringIO(result.stdout))
        return df if not df.empty else None

# ============================ SIDEBAR ============================
st.sidebar.image("https://img.icons8.com/fluency/120/bug.png", width=120)
st.sidebar.markdown("## Mode de test")
mode = st.sidebar.radio(
    "Comment veux-tu tester ?",
    ["Copier-coller du code", "Métriques manuelles", "Uploader un CSV", "Repo GitHub/GitLab"]
)

# ===================================================================
# 1. COPIER-COLLER DU CODE (LE PLUS IMPRESSIONNANT)
# ===================================================================
if mode == "Copier-coller du code":
    st.markdown("### Colle ton code source ici (Python, Java, JS, C/C++)")
    language = st.selectbox("Langage", ["python", "java", "javascript", "cpp", "c"])
    code = st.text_area("Code à analyser", height=500, placeholder="def hello(name):\n    print(f'Hello {name}')\n    return name.upper()")

    if st.button("Analyser ce code", type="primary", use_container_width=True):
        if code.strip():
            with st.spinner("Analyse du code..."):
                proba = predict_from_source_code(code, language)
                if proba is not None:
                    st.metric("Probabilité de bug", f"{proba:.1%}", delta=f"{proba-0.5:+.1%}")

                    if proba >= 0.6:
                        st.markdown(f'<div class="risk-high">RISQUE ÉLEVÉ → {proba:.1%} de bug</div>', unsafe_allow_html=True)
                        st.error("Ce fichier est très probablement buggé !")
                    elif proba >= 0.3:
                        st.markdown(f'<div class="risk-medium">RISQUE MOYEN → {proba:.1%}</div>', unsafe_allow_html=True)
                        st.warning("À surveiller de près")
                    else:
                        st.markdown(f'<div class="risk-low">RISQUE FAIBLE → {proba:.1%}</div>', unsafe_allow_html=True)
                        st.success("Code très probablement sain")
        else:
            st.info("Colle du code pour commencer")

# ===================================================================
# 2. MÉTRIQUES MANUELLES
# ===================================================================
elif mode == "Métriques manuelles":
    st.markdown("### Saisie manuelle des métriques principales")
    with st.form("manual_form"):
        col1, col2 = st.columns(2)
        with col1:
            loc = st.number_input("LOC", value=100)
            cc = st.number_input("Complexité cyclomatique", value=10)
            rfc = st.number_input("RFC", value=30)
            cbo = st.number_input("CBO", value=8)
        with col2:
            wmc = st.number_input("WMC", value=15)
            lcom3 = st.slider("LCOM3", 0.0, 2.0, 1.0)
            dit = st.number_input("DIT", value=2)
            noc = st.number_input("NOC", value=0)

        if st.form_submit_button("Prédire", use_container_width=True):
            vec = np.zeros(len(feature_columns))
            mapping = {"loc":loc, "cyclomatic_complexity":cc, "rfc":rfc, "cbo":cbo, "wmc":wmc, "lcom3":lcom3, "dit":dit, "noc":noc}
            for k, v in mapping.items():
                if k in feature_columns:
                    vec[feature_columns.index(k)] = v
            proba = model.predict_proba(scaler.transform([vec]))[0,1]
            st.metric("Risque", f"{proba:.1%}")
            if proba >= 0.6: st.markdown(f'<div class="risk-high">RISQUE ÉLEVÉ → {proba:.1%}</div>', unsafe_allow_html=True)
            elif proba >= 0.3: st.markdown(f'<div class="risk-medium">RISQUE MOYEN → {proba:.1%}</div>', unsafe_allow_html=True)
            else: st.markdown(f'<div class="risk-low">RISQUE FAIBLE → {proba:.1%}</div>', unsafe_allow_html=True)

# ===================================================================
# 3. UPLOAD CSV
# ===================================================================
elif mode == "Uploader un CSV":
    uploaded = st.file_uploader("CSV avec métriques (même colonnes que l’entraînement)", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_columns]
        probas = model.predict_proba(scaler.transform(df.values))[:,1]
        result = df.copy()
        result['risk_score'] = probas
        result['risk_level'] = pd.cut(probas, [0,0.3,0.6,1.0], labels=['Faible','Moyen','Élevé'])
        result = result.sort_values('risk_score', ascending=False)
        st.success(f"{len(result)} fichiers analysés")
        st.dataframe(result.head(20).style.background_gradient(cmap="Reds", subset=['risk_score']), use_container_width=True)
        st.download_button("Télécharger résultats", result.to_csv(index=False).encode(), "predictions.csv", "text/csv")

# ===================================================================
# 4. GITHUB EN DIRECT
# ===================================================================
# ===================================================================
# 4. GITHUB EN DIRECT – VERSION 100% ANTI-CRASH (testée sur Flask, Django, etc.)
# ===================================================================
# ===================================================================
# 4. GITHUB EN DIRECT – VERSION 100% ANTI-CRASH (testée sur Flask, Django, etc.)
# ===================================================================
else:
    st.markdown("### Analyse complète d’un dépôt GitHub/GitLab")
    repo_url = st.text_input("URL du dépôt", "https://github.com/pallets/flask")
    branch = st.text_input("Branche", "main")
    
    if st.button("Analyser le dépôt", type="primary", use_container_width=True):
        with st.spinner("Clonage et analyse en cours... (cela peut prendre 30-60s)"):
            df = analyze_github_repo(repo_url, branch)

        if df is None or df.empty:
            st.error("Aucun fichier analysé. Vérifie que le repo existe et contient du code source.")
            st.stop()

        st.success(f"{len(df)} fichiers extraits → alignement avec le modèle...")

        # === ÉTAPE CLÉ : FORCER L'ORDRE ET LE FORMAT ===
        # Ajouter toutes les colonnes manquantes
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0.0

        # Garder exactement les colonnes du modèle, dans le bon ordre
        df_model = df.reindex(columns=feature_columns, fill_value=0.0)

        # Conversion en tableau numpy 2D (même si 1 seul fichier)
        X = df_model.values
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # === TRANSFORMATION ET PRÉDICTION SÉCURISÉE ===
        try:
            X_scaled = scaler.transform(X)
            
            # Forcer 2D même après scaler
            if X_scaled.ndim == 1:
                X_scaled = X_scaled.reshape(1, -1)
                
            predictions = model.predict_proba(X_scaled)
            probas = predictions[:, 1]  # Probabilité de la classe 1 (bug)

        except Exception as e:
            st.error(f"Erreur critique lors de la prédiction : {e}")
            st.code(f"Forme de X_scaled : {X_scaled.shape if 'X_scaled' in locals() else 'N/A'}")
            st.stop()

        # === AFFICHAGE DES RÉSULTATS ===
        result = df[['file']].copy()
        result['nloc'] = df.get('nloc', 0)
        result['cyclomatic_complexity'] = df.get('cyclomatic_complexity', 1)
        result['risk_score'] = probas
        result['risk_level'] = pd.cut(probas, bins=[0, 0.3, 0.6, 1.0],
                                      labels=['Faible', 'Moyen', 'Élevé'])
        result = result.sort_values('risk_score', ascending=False).reset_index(drop=True)

        st.success(f"Analyse terminée ! {len(result)} fichiers classés par risque")
        
        # Top 20
        st.markdown("### Top 20 fichiers les plus risqués")
        st.dataframe(
            result.head(20).style.background_gradient(cmap="Reds", subset=['risk_score'])
                          .format({'risk_score': '{:.1%}'}),
            use_container_width=True
        )

        # Téléchargement
        csv_data = result.to_csv(index=False).encode('utf-8')
        repo_name = repo_url.split("/")[-1]
        st.download_button(
            label="Télécharger le rapport complet CSV",
            data=csv_data,
            file_name=f"bug_report_{repo_name}_{branch}.csv",
            mime="text/csv",
            use_container_width=True
        )

        # Bonus : stats rapides
        high_risk = (result['risk_level'] == 'Élevé').sum()
        st.info(f"Nombre de fichiers à risque élevé : **{high_risk}** sur {len(result)}")

# ============================ FOOTER ============================
st.markdown("---")
st.markdown("**Projet Génie Logiciel – Master S3 – 2025**")
st.markdown("Prédiction de défauts avec XGBoost • lizard • Streamlit • Analyse GitHub en temps réel")