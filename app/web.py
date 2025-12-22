# app/web.py
# VERSION CORRIGÉE POUR STREAMLIT CLOUD – BugPredictor Pro 2025
# Compatible Python 3.9+ et 3.13

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import subprocess
import os

# ============================ GESTION DES IMPORTS CONDITIONNELS ============================
# Import sécurisé pour éviter les erreurs sur Streamlit Cloud

IMPORT_ERRORS = []

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError as e:
    JOBLIB_AVAILABLE = False
    IMPORT_ERRORS.append(f"joblib: {str(e)}")

try:
    from lizard import analyze_file
    LIZARD_AVAILABLE = True
except ImportError as e:
    LIZARD_AVAILABLE = False
    IMPORT_ERRORS.append(f"lizard: {str(e)}")

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
    .warning-box {background:#fff3cd; border-left:6px solid #ffc107; padding:15px; margin:15px 0; border-radius:5px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">BugPredictor Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Prédiction automatique des fichiers à risque • Copier-coller • CSV • GitHub</p>', unsafe_allow_html=True)

# Avertissement si imports manquants
if IMPORT_ERRORS:
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.warning("⚠️ **Certaines dépendances sont manquantes**")
    st.write("Modules non disponibles :")
    for error in IMPORT_ERRORS:
        st.write(f"- {error}")
    st.write("**Solution** : Vérifiez que `requirements.txt` contient : `joblib`, `lizard`")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================ CHARGEMENT DU MODÈLE (VERSION ROBUSTE) ============================
@st.cache_resource
def load_model():
    """Charge le modèle avec gestion d'erreurs robuste"""
    if not JOBLIB_AVAILABLE:
        st.error("❌ **joblib n'est pas disponible**")
        st.error("Installez-le avec : `pip install joblib`")
        return None, None, None
    
    try:
        # Chemins robustes
        base_dir = Path(__file__).parent
        model_path = base_dir / "best_model.pkl"
        scaler_path = base_dir / "scaler.pkl"
        columns_path = base_dir / "feature_columns.pkl"
        
        # Vérifier l'existence des fichiers
        missing_files = []
        for path, name in [(model_path, "best_model.pkl"), 
                          (scaler_path, "scaler.pkl"), 
                          (columns_path, "feature_columns.pkl")]:
            if not path.exists():
                missing_files.append(name)
        
        if missing_files:
            st.error(f"❌ Fichiers manquants dans 'app/' : {', '.join(missing_files)}")
            st.error("Assurez-vous que les fichiers .pkl sont dans le dossier 'app/'")
            return None, None, None
        
        # Charger les fichiers
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        columns = joblib.load(columns_path)
        
        st.success("✅ Modèle chargé avec succès")
        return model, scaler, columns
        
    except Exception as e:
        st.error(f"❌ Erreur de chargement du modèle : {str(e)}")
        
        # Mode démonstration en cas d'échec
        st.warning("""
        **Mode démonstration activé**
        
        L'application fonctionne en mode démo avec un modèle simplifié.
        Pour utiliser le modèle complet :
        1. Vérifiez que `requirements.txt` contient toutes les dépendances
        2. Les fichiers .pkl doivent être dans le dossier `app/`
        3. Sur Streamlit Cloud, sélectionnez Python 3.9 dans les paramètres
        """)
        
        # Créer un modèle de démonstration
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Features par défaut (exemple)
        demo_columns = ['nloc', 'cyclomatic_complexity', 'cbo', 'rfc', 'wmc', 
                       'lcom3', 'dit', 'noc', 'dam', 'loc']
        
        # Créer un modèle simple
        demo_model = RandomForestClassifier(n_estimators=10, random_state=42)
        demo_scaler = StandardScaler()
        
        # Données d'entraînement factices
        np.random.seed(42)
        X_demo = np.random.randn(100, len(demo_columns))
        y_demo = np.random.randint(0, 2, 100)
        
        # Entraîner
        demo_scaler.fit(X_demo)
        X_scaled = demo_scaler.transform(X_demo)
        demo_model.fit(X_scaled, y_demo)
        
        return demo_model, demo_scaler, demo_columns

# Charger le modèle
model, scaler, feature_columns = load_model()

if model is None:
    st.stop()

# ============================ FONCTION : CODE COLLÉ ============================
def predict_from_source_code(code: str, language: str = "python"):
    """Analyse du code collé avec lizard"""
    if not LIZARD_AVAILABLE:
        st.error("❌ lizard n'est pas installé")
        st.error("Installez-le avec : `pip install lizard`")
        return 0.3  # Valeur par défaut
    
    if not code.strip():
        return None

    suffix = {"python": ".py", "java": ".java", "javascript": ".js", "c": ".c", "cpp": ".cpp"}.get(language, ".py")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False, encoding='utf-8') as f:
        f.write(code)
        temp_path = f.name

    try:
        analysis = analyze_file(temp_path)
        funcs = analysis.function_list

        if not funcs:
            return 0.1  # Faible risque par défaut

        # Calcul des métriques
        data = {
            'nloc': np.mean([f.nloc for f in funcs]) if funcs else 10,
            'cyclomatic_complexity': np.mean([f.cyclomatic_complexity for f in funcs]) if funcs else 1,
            'token_count': np.mean([f.token_count for f in funcs]) if funcs else 50,
            'parameter_count': np.mean([f.parameter_count for f in funcs]) if funcs else 1,
            'loc': analysis.nloc or 10,
            'wmc': len(funcs),
            'lcom3': getattr(analysis, 'average_lcom3', 1.0),
            'rfc': len(funcs) * 4,
            'cbo': 6,
            'dit': 1,
            'noc': 0,
            'dam': 0.5,
        }

        # Construction du vecteur
        vec = np.zeros(len(feature_columns))
        for k, v in data.items():
            if k in feature_columns:
                idx = feature_columns.index(k)
                vec[idx] = v

        # Prédiction sécurisée
        vec_2d = vec.reshape(1, -1)
        X_scaled = scaler.transform(vec_2d)
        
        if X_scaled.ndim == 1:
            X_scaled = X_scaled.reshape(1, -1)
            
        proba_raw = model.predict_proba(X_scaled)
        
        if proba_raw.ndim == 1:
            proba = float(proba_raw[1])
        else:
            proba = float(proba_raw[0, 1])

        return proba

    except Exception as e:
        st.warning(f"Analyse lizard échouée : {e}")
        return 0.2
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

# ============================ FONCTION : GITHUB ============================
def analyze_github_repo(repo_url: str, branch: str = "main"):
    if not LIZARD_AVAILABLE:
        st.error("❌ lizard n'est pas disponible")
        return None
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Clone
        cmd = ["git", "clone", "--depth", "1", "--branch", branch, repo_url, tmpdir]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        
        if result.returncode != 0:
            cmd = ["git", "clone", "--depth", "1", repo_url, tmpdir]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            
            if result.returncode != 0:
                st.error(f"❌ Clone échoué")
                return None
        
        # Analyse avec lizard
        try:
            cmd = ["lizard", tmpdir, "-l", "python", "--csv"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            output = result.stdout.strip()
            if not output:
                return None
            
            # Parsing simple
            lines = output.split('\n')
            data = []
            
            for line in lines[1:]:  # Skip header
                if not line.strip():
                    continue
                
                parts = line.split(',')
                if len(parts) >= 6:
                    try:
                        row = {
                            'nloc': float(parts[0]),
                            'cyclomatic_complexity': float(parts[1]),
                            'token_count': float(parts[2]),
                            'parameter_count': float(parts[3]),
                            'file_path': parts[5],
                            'filename': Path(parts[5]).name
                        }
                        data.append(row)
                    except:
                        continue
            
            if not data:
                return None
            
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            st.error(f"❌ Erreur lizard: {str(e)}")
            return None

# ============================ SIDEBAR ============================
st.sidebar.image("https://img.icons8.com/fluency/120/bug.png", width=120)
st.sidebar.markdown("## Mode de test")

# Désactiver certaines options si lizard n'est pas disponible
if not LIZARD_AVAILABLE:
    st.sidebar.warning("⚠️ lizard manquant")
    mode_options = ["Métriques manuelles", "Uploader un CSV"]
else:
    mode_options = ["Copier-coller du code", "Métriques manuelles", "Uploader un CSV", "Repo GitHub/GitLab"]

mode = st.sidebar.radio("Comment veux-tu tester ?", mode_options)

# ============================ SECTIONS PRINCIPALES ============================

if mode == "Copier-coller du code" and LIZARD_AVAILABLE:
    st.markdown("### Colle ton code source ici")
    language = st.selectbox("Langage", ["python", "java", "javascript", "cpp", "c"])
    code = st.text_area("Code à analyser", height=500, placeholder="def hello(name):\n    print(f'Hello {name}')\n    return name.upper()")

    if st.button("Analyser ce code", type="primary", use_container_width=True):
        if code.strip():
            with st.spinner("Analyse du code..."):
                proba = predict_from_source_code(code, language)
                if proba is not None:
                    st.metric("Probabilité de bug", f"{proba:.1%}")
                    
                    if proba >= 0.6:
                        st.markdown(f'<div class="risk-high">RISQUE ÉLEVÉ → {proba:.1%}</div>', unsafe_allow_html=True)
                    elif proba >= 0.3:
                        st.markdown(f'<div class="risk-medium">RISQUE MOYEN → {proba:.1%}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="risk-low">RISQUE FAIBLE → {proba:.1%}</div>', unsafe_allow_html=True)
        else:
            st.info("Colle du code pour commencer")

elif mode == "Métriques manuelles":
    st.markdown("### Saisie manuelle des métriques")
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
            mapping = {"loc": loc, "cyclomatic_complexity": cc, "rfc": rfc, "cbo": cbo,
                      "wmc": wmc, "lcom3": lcom3, "dit": dit, "noc": noc}
            
            for k, v in mapping.items():
                if k in feature_columns:
                    idx = feature_columns.index(k)
                    vec[idx] = v
            
            vec_2d = vec.reshape(1, -1)
            
            try:
                proba_raw = model.predict_proba(scaler.transform(vec_2d))
                if proba_raw.ndim == 2:
                    proba = float(proba_raw[0, 1])
                else:
                    proba = float(proba_raw[1])
                
                st.metric("Risque", f"{proba:.1%}")
                
                if proba >= 0.6:
                    st.markdown(f'<div class="risk-high">RISQUE ÉLEVÉ → {proba:.1%}</div>', unsafe_allow_html=True)
                elif proba >= 0.3:
                    st.markdown(f'<div class="risk-medium">RISQUE MOYEN → {proba:.1%}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-low">RISQUE FAIBLE → {proba:.1%}</div>', unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Erreur de prédiction : {e}")

elif mode == "Uploader un CSV":
    st.markdown("### Uploader un CSV avec métriques")
    uploaded = st.file_uploader("CSV avec métriques", type=["csv"])
    
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"✅ CSV chargé : {len(df)} lignes")
            
            # Ajouter colonnes manquantes
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0.0
            
            df = df[feature_columns]
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
            
            # Prédiction
            X_scaled = scaler.transform(df.values)
            predictions = model.predict_proba(X_scaled)
            
            # Extraction sécurisée
            if predictions.ndim == 2 and predictions.shape[1] >= 2:
                probas = predictions[:, 1]
            elif predictions.ndim == 1:
                probas = predictions
            else:
                st.error("Format de prédiction inattendu")
                probas = np.zeros(len(df))
            
            # Résultats
            result = df.copy()
            result['risk_score'] = probas
            result['risk_level'] = pd.cut(probas, [0, 0.3, 0.6, 1.0], 
                                         labels=['🟢 Faible', '🟡 Moyen', '🔴 Élevé'])
            result = result.sort_values('risk_score', ascending=False)
            
            # Affichage
            st.success(f"🎯 {len(result)} fichiers évalués")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🔴 Haut risque", (result['risk_level'] == '🔴 Élevé').sum())
            with col2:
                st.metric("🟡 Moyen risque", (result['risk_level'] == '🟡 Moyen').sum())
            with col3:
                st.metric("🟢 Faible risque", (result['risk_level'] == '🟢 Faible').sum())
            
            # Top 20
            st.markdown("#### Top 20 fichiers les plus risqués")
            
            display_df = result[['risk_score', 'risk_level']].head(20).copy()
            display_df.insert(0, 'Fichier', [f"Ligne {i+1}" for i in range(len(display_df))])
            display_df['risk_score'] = display_df['risk_score'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # Téléchargement
            csv_data = result.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Télécharger résultats", csv_data, 
                             "predictions.csv", "text/csv", use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")

elif mode == "Repo GitHub/GitLab" and LIZARD_AVAILABLE:
    st.markdown("### Analyse d'un dépôt GitHub")
    repo_url = st.text_input("URL du dépôt", "https://github.com/pallets/flask")
    branch = st.text_input("Branche", "main")
    
    if st.button("🚀 Analyser", type="primary", use_container_width=True):
        with st.spinner("Clonage et analyse en cours..."):
            df_raw = analyze_github_repo(repo_url, branch)
        
        if df_raw is None or df_raw.empty:
            st.error("Échec de l'analyse")
            st.stop()
        
        # Agrégation par fichier
        df_files = df_raw.groupby('filename').agg({
            'nloc': 'sum',
            'cyclomatic_complexity': 'mean'
        }).reset_index()
        
        # Préparation pour modèle
        X_data = pd.DataFrame(0.0, index=range(len(df_files)), columns=feature_columns)
        
        # Mapping simple
        if 'nloc' in feature_columns:
            X_data['nloc'] = df_files.get('nloc', 0)
        if 'cyclomatic_complexity' in feature_columns:
            X_data['cyclomatic_complexity'] = df_files.get('cyclomatic_complexity', 0)
        
        # Valeurs par défaut
        X_data['loc'] = df_files.get('nloc', 0)
        X_data['cbo'] = 5.0
        X_data['rfc'] = 15.0
        
        # Prédiction
        try:
            X_scaled = scaler.transform(X_data.values)
            predictions = model.predict_proba(X_scaled)
            
            if predictions.ndim == 2:
                probas = predictions[:, 1]
            else:
                probas = predictions
            
            # Résultats
            result = pd.DataFrame({
                'Fichier': df_files['filename'],
                'Score': probas,
                'LOC': df_files.get('nloc', 0).astype(int),
                'Complexité': df_files.get('cyclomatic_complexity', 0).round(2)
            })
            
            result['Risque'] = pd.cut(probas, [0, 0.3, 0.6, 1.0], 
                                     labels=['🟢 Faible', '🟡 Moyen', '🔴 Élevé'])
            result = result.sort_values('Score', ascending=False)
            
            # Affichage
            st.success(f"✅ {len(result)} fichiers analysés")
            
            # Statistiques
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Score max", f"{result['Score'].max():.1%}")
            with col2:
                st.metric("Score moyen", f"{result['Score'].mean():.1%}")
            with col3:
                st.metric("Fichiers risqués", (result['Score'] >= 0.6).sum())
            
            # Top 10
            st.markdown("#### Top 10 fichiers")
            display_df = result.head(10)[['Fichier', 'Risque', 'Score', 'LOC', 'Complexité']].copy()
            display_df['Score'] = display_df['Score'].apply(lambda x: f"{x:.1%}")
            st.dataframe(display_df, use_container_width=True)
            
            # Téléchargement
            csv_string = result.to_csv(index=False).encode('utf-8')
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            st.download_button("📥 Télécharger rapport", csv_string,
                             f"bug_report_{repo_name}.csv", "text/csv", use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Erreur : {str(e)}")

# ============================ FOOTER ============================
st.markdown("---")
st.markdown("**Projet Génie Logiciel – Master S3 – 2025**")
st.markdown("Prédiction de défauts • Streamlit • Analyse de code")

# Debug info (optionnel)
with st.expander("🔍 Informations de debug"):
    st.write(f"Python version: {sys.version}")
    st.write(f"Joblib disponible: {JOBLIB_AVAILABLE}")
    st.write(f"Lizard disponible: {LIZARD_AVAILABLE}")
    st.write(f"Nombre de features: {len(feature_columns) if feature_columns else 'N/A'}")
    st.write(f"Modules chargés: {', '.join([m for m in ['pandas', 'numpy', 'streamlit']])}")