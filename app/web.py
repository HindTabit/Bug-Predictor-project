# app/web.py
# VERSION FINALE COMPLÈTE ET CORRIGÉE – BugPredictor Pro 2025
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
        # Chemins robustes : fichiers dans le même dossier que web.py
        base_dir = Path(__file__).parent
        model_path = base_dir / "best_model.pkl"
        scaler_path = base_dir / "scaler.pkl"
        columns_path = base_dir / "feature_columns.pkl"
        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        columns = joblib.load(columns_path)
        
        return model, scaler, columns
    except Exception as e:
        st.error(f"Modèle non trouvé → {e}")
        st.error("Vérifie que best_model.pkl, scaler.pkl et feature_columns.pkl sont dans le dossier 'app/' avec web.py")
        st.stop()

model, scaler, feature_columns = load_model()
st.success("Modèle XGBoost chargé avec succès")

# ============================ FONCTION : CODE COLLÉ (CORRIGÉE & ROBUSTE) ============================
def predict_from_source_code(code: str, language: str = "python"):
    """Analyse du code collé avec lizard et prédit le risque de bug"""
    if not code.strip():
        return None

    suffix = {"python": ".py", "java": ".java", "javascript": ".js", "c": ".c", "cpp": ".cpp"}.get(language, ".py")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False, encoding='utf-8') as f:
        f.write(code)
        temp_path = f.name

    try:
        analysis = analyze_file(temp_path)
        funcs = analysis.function_list

        # Cas où lizard ne détecte aucune fonction (script simple, imports, etc.)
        if not funcs:
            return 0.1  # Faible risque par défaut

        # Calcul des métriques moyennes
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

        # Construction du vecteur d'entrée
        vec = np.zeros(len(feature_columns))
        for k, v in data.items():
            if k in feature_columns:
                idx = feature_columns.index(k)
                vec[idx] = v

        # === PRÉDICTION SÉCURISÉE (plus jamais d'erreur d'indice) ===
        vec_2d = vec.reshape(1, -1)                    # Force 2D
        X_scaled = scaler.transform(vec_2d)
        
        if X_scaled.ndim == 1:                         # Sécurité supplémentaire
            X_scaled = X_scaled.reshape(1, -1)
            
        proba_raw = model.predict_proba(X_scaled)
        
        if proba_raw.ndim == 1:                        # Cas rare mais possible
            proba = float(proba_raw[1])
        else:
            proba = float(proba_raw[0, 1])

        return proba

    except Exception as e:
        st.warning(f"Analyse lizard échouée : {e}")
        return 0.2  # Valeur par défaut raisonnable (risque faible-moyen)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

# ============================ FONCTION : GITHUB (VERSION AVEC DIAGNOSTIQUE) ============================
def analyze_github_repo(repo_url: str, branch: str = "main"):
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Clone le dépôt
        cmd = ["git", "clone", "--depth", "1", "--branch", branch, repo_url, tmpdir]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        
        if result.returncode != 0:
            cmd = ["git", "clone", "--depth", "1", repo_url, tmpdir]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            
            if result.returncode != 0:
                st.error(f"❌ Clone échoué: {result.stderr[:200] if result.stderr else 'Erreur'}")
                return None
        
        # Exécuter lizard avec une sortie CSV standard
        try:
            cmd = ["lizard", tmpdir, "-l", "python", "--csv"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                st.warning(f"⚠️ Lizard warnings: {result.stderr[:200] if result.stderr else ''}")
            
            output = result.stdout.strip()
            
            if not output:
                st.error("La sortie de lizard est vide")
                return None
            
            # 🔥 PARSING CRITIQUE : Lizard produit un format CSV spécifique
            # Format: NLOC,CCN,token,param,function,file,long_name,start,end
            # Mais parfois sans guillemets, avec des virgules dans les champs
            
            lines = output.split('\n')
            if not lines:
                return None
            
            data = []
            for line in lines:
                if not line.strip():
                    continue
                    
                # Split intelligent qui gère les virgules dans les champs
                parts = []
                current = ""
                in_quotes = False
                
                for char in line:
                    if char == '"':
                        in_quotes = not in_quotes
                    elif char == ',' and not in_quotes:
                        parts.append(current.strip('"').strip())
                        current = ""
                        continue
                    current += char
                
                parts.append(current.strip('"').strip())
                
                # Nous attendons au moins 6 colonnes
                if len(parts) >= 6:
                    row = {
                        'nloc': float(parts[0]) if parts[0].replace('.', '', 1).isdigit() else 0,
                        'cyclomatic_complexity': float(parts[1]) if parts[1].replace('.', '', 1).isdigit() else 0,
                        'token_count': float(parts[2]) if parts[2].replace('.', '', 1).isdigit() else 0,
                        'parameter_count': float(parts[3]) if parts[3].replace('.', '', 1).isdigit() else 0,
                        'function_name': parts[4] if len(parts) > 4 else '',
                        'file_path': parts[5] if len(parts) > 5 else '',
                        'filename': Path(parts[5] if len(parts) > 5 else '').name  # Extraire juste le nom du fichier
                    }
                    data.append(row)
                else:
                    # Format alternatif ou ligne de somme
                    continue
            
            if not data:
                # Essayer un parsing plus simple
                st.write("Tentative de parsing alternatif...")
                data = []
                for line in lines:
                    parts = line.split(',')
                    if len(parts) >= 6:
                        try:
                            row = {
                                'nloc': float(parts[0]),
                                'cyclomatic_complexity': float(parts[1]),
                                'token_count': float(parts[2]),
                                'parameter_count': float(parts[3]),
                                'function_name': parts[4],
                                'file_path': parts[5],
                                'filename': Path(parts[5]).name
                            }
                            data.append(row)
                        except:
                            continue
            
            if not data:
                st.error("Impossible de parser la sortie de lizard")
                return None
            
            df = pd.DataFrame(data)
            st.write(f"✅ {len(df)} fonctions analysées dans {df['filename'].nunique()} fichiers")
            return df
            
        except Exception as e:
            st.error(f"❌ Erreur lizard: {str(e)}")
            return None

# ============================ SIDEBAR ============================
st.sidebar.image("https://img.icons8.com/fluency/120/bug.png", width=120)
st.sidebar.markdown("## Mode de test")
mode = st.sidebar.radio(
    "Comment veux-tu tester ?",
    ["Copier-coller du code", "Métriques manuelles", "Uploader un CSV", "Repo GitHub/GitLab"]
)

# ===================================================================
# 1. COPIER-COLLER DU CODE
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
            mapping = {"loc": loc, "cyclomatic_complexity": cc, "rfc": rfc, "cbo": cbo,
                       "wmc": wmc, "lcom3": lcom3, "dit": dit, "noc": noc}
            for k, v in mapping.items():
                if k in feature_columns:
                    vec[feature_columns.index(k)] = v
            
            vec_2d = vec.reshape(1, -1)
            proba = float(model.predict_proba(scaler.transform(vec_2d))[0, 1])
            
            st.metric("Risque", f"{proba:.1%}")
            if proba >= 0.6:
                st.markdown(f'<div class="risk-high">RISQUE ÉLEVÉ → {proba:.1%}</div>', unsafe_allow_html=True)
            elif proba >= 0.3:
                st.markdown(f'<div class="risk-medium">RISQUE MOYEN → {proba:.1%}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-low">RISQUE FAIBLE → {proba:.1%}</div>', unsafe_allow_html=True)

# ===================================================================
# 3. UPLOAD CSV
# ===================================================================
# ===================================================================
# 3. UPLOAD CSV (VERSION CORRIGÉE)
# ===================================================================
elif mode == "Uploader un CSV":
    st.markdown("### 📊 Uploader un fichier CSV avec métriques")
    uploaded = st.file_uploader("CSV avec métriques (même colonnes que l'entraînement)", type=["csv"])
    
    if uploaded:
        try:
            # Charger le CSV
            df = pd.read_csv(uploaded)
            st.success(f"✅ CSV chargé : {len(df)} lignes, {len(df.columns)} colonnes")
            
            # Vérifier les colonnes
            st.write(f"📋 Colonnes du CSV : {list(df.columns)}")
            st.write(f"🎯 Colonnes attendues par le modèle : {len(feature_columns)} features")
            
            # Ajouter les colonnes manquantes
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0.0
                    st.warning(f"⚠️ Colonne '{col}' manquante, remplie avec 0.0")
            
            # Sélectionner uniquement les colonnes requises
            df = df[feature_columns]
            
            # S'assurer que toutes les valeurs sont numériques
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
            
            # Aperçu des données
            st.markdown("#### 📝 Aperçu des données préparées")
            st.dataframe(df.head(), use_container_width=True)
            
            # Prédiction SÉCURISÉE
            X_scaled = scaler.transform(df.values)
            
            st.write(f"🔍 Forme des données : {X_scaled.shape}")
            
            # Prédiction avec vérification
            predictions = model.predict_proba(X_scaled)
            st.write(f"📊 Forme des prédictions : {predictions.shape}")
            
            # Extraction SÉCURISÉE des probabilités
            if predictions.ndim == 2 and predictions.shape[1] >= 2:
                probas = predictions[:, 1]  # Probabilité de la classe 1 (bug)
            elif predictions.ndim == 1:
                probas = predictions  # Déjà les probabilités de la classe 1
            else:
                st.error(f"Format de prédiction inattendu : {predictions.shape}")
                probas = np.zeros(len(df))
            
            # Résultats
            result = df.copy()
            result['risk_score'] = probas
            result['risk_level'] = pd.cut(
                probas, 
                [0, 0.3, 0.6, 1.0], 
                labels=['🟢 Faible', '🟡 Moyen', '🔴 Élevé']
            )
            
            # Trier par risque
            result = result.sort_values('risk_score', ascending=False)
            
            # Statistiques
            st.success(f"🎯 **Analyse terminée !** {len(result)} fichiers évalués")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                high_risk = (result['risk_level'] == '🔴 Élevé').sum()
                st.metric("🔴 Risque Élevé", high_risk)
            with col2:
                medium_risk = (result['risk_level'] == '🟡 Moyen').sum()
                st.metric("🟡 Risque Moyen", medium_risk)
            with col3:
                low_risk = (result['risk_level'] == '🟢 Faible').sum()
                st.metric("🟢 Risque Faible", low_risk)
            
            # Top 20 fichiers risqués
            st.markdown("#### 🏆 Top 20 fichiers les plus risqués")
            
            display_cols = []
            if 'filename' in df.columns or 'file' in df.columns or 'File' in df.columns:
                # Chercher une colonne de nom de fichier
                file_col = None
                for col in ['filename', 'file', 'File', 'file_name', 'path']:
                    if col in result.columns:
                        file_col = col
                        break
                
                if file_col:
                    display_df = result[[file_col, 'risk_score', 'risk_level']].head(20).copy()
                    display_df = display_df.rename(columns={file_col: 'Fichier'})
                else:
                    display_df = result[['risk_score', 'risk_level']].head(20).copy()
                    display_df.insert(0, 'Fichier', [f"Ligne {i+1}" for i in range(20)])
            else:
                display_df = result[['risk_score', 'risk_level']].head(20).copy()
                display_df.insert(0, 'Fichier', [f"Ligne {i+1}" for i in range(20)])
            
            display_df['Rang'] = range(1, len(display_df) + 1)
            display_df = display_df[['Rang', 'Fichier', 'risk_score', 'risk_level']]
            display_df['risk_score'] = display_df['risk_score'].apply(lambda x: f"{x:.1%}")
            
            # Colorisation
            def color_risk(val):
                if '🔴' in str(val):
                    return 'color: #d32f2f; font-weight: bold'
                elif '🟡' in str(val):
                    return 'color: #f57c00'
                elif '🟢' in str(val):
                    return 'color: #388e3c'
                return ''
            
            st.dataframe(
                display_df.style.applymap(color_risk, subset=['risk_level']),
                use_container_width=True,
                height=500
            )
            
            # Téléchargement
            st.markdown("#### 💾 Téléchargement des résultats")
            
            # Préparer CSV pour téléchargement
            csv_data = result.copy()
            csv_data['risk_score'] = csv_data['risk_score'].apply(lambda x: f"{x:.4f}")
            
            # Bouton de téléchargement
            download_button = st.download_button(
                label="📥 Télécharger toutes les prédictions (CSV)",
                data=csv_data.to_csv(index=False).encode('utf-8'),
                file_name="predictions_complete.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Téléchargement du top 20 uniquement
            top20_csv = display_df.copy()
            top20_csv['risk_score'] = top20_csv['risk_score'].str.replace('%', '')
            top20_button = st.download_button(
                label="📥 Télécharger le Top 20 (CSV)",
                data=top20_csv.to_csv(index=False).encode('utf-8'),
                file_name="predictions_top20.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement du CSV : {str(e)}")
            import traceback
            st.code(traceback.format_exc())


# ===================================================================
# 4. REPO GITHUB / GITLAB (VERSION SIMPLIFIÉE ET CORRIGÉE)
# ===================================================================
else:
    st.markdown("### 🔍 Analyse complète d'un dépôt GitHub")
    repo_url = st.text_input("URL du dépôt", "https://github.com/pallets/flask")
    branch = st.text_input("Branche", "main")
    
    if st.button("🚀 Analyser le dépôt", type="primary", use_container_width=True):
        with st.spinner("Clonage et analyse en cours (2-3 min)..."):
            df_raw = analyze_github_repo(repo_url, branch)
        
        if df_raw is None or df_raw.empty:
            st.error("Échec de l'analyse")
            st.stop()
        
        st.success(f"✅ {len(df_raw)} fonctions analysées dans {df_raw['filename'].nunique()} fichiers")
        
        # ===== ÉTAPE 1: AGRÉGATION PAR FICHIER =====
        aggregation = {
            'nloc': 'sum',
            'cyclomatic_complexity': 'mean',
            'token_count': 'sum',
            'parameter_count': 'mean'
        }
        
        valid_agg = {k: v for k, v in aggregation.items() if k in df_raw.columns}
        
        if not valid_agg:
            st.error("Aucune métrique valide pour l'agrégation")
            st.stop()
        
        df_files = df_raw.groupby('filename').agg(valid_agg).reset_index()
        
        # ===== ÉTAPE 2: AJOUTER LES FEATURES MANQUANTES =====
        st.write(f"📊 Le modèle attend {len(feature_columns)} features")
        
        X_data = pd.DataFrame(0.0, index=range(len(df_files)), columns=feature_columns)
        
        mapping = {
            'nloc': 'nloc',
            'cyclomatic_complexity': 'cyclomatic_complexity',
            'loc': 'nloc',
            'wmc': 'cyclomatic_complexity',
        }
        
        for model_feature, source_feature in mapping.items():
            if model_feature in X_data.columns and source_feature in df_files.columns:
                X_data[model_feature] = df_files[source_feature]
        
        X_data['cbo'] = 5.0
        X_data['rfc'] = 15.0
        X_data['lcom3'] = 1.0
        X_data['dam'] = 0.5
        X_data['dit'] = 1.0
        X_data['noc'] = 0.0
        
        # ===== ÉTAPE 3: PRÉDICTION SÉCURISÉE =====
        try:
            st.write(f"✅ X_data shape: {X_data.shape}")
            st.write(f"✅ X_data columns: {list(X_data.columns[:10])}...")
            
            X_data = X_data[feature_columns]
            X_scaled = scaler.transform(X_data)
            
            st.write(f"✅ X_scaled shape: {X_scaled.shape}")
            
            probas = model.predict_proba(X_scaled)
            st.write(f"✅ Predictions shape: {probas.shape}")
            
            if probas.ndim == 2 and probas.shape[1] >= 2:
                bug_probas = probas[:, 1]
            elif probas.ndim == 1:
                bug_probas = probas
            else:
                st.error(f"Format de prédiction inattendu: {probas.shape}")
                bug_probas = np.zeros(len(df_files))
        
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()
        
        # ===== ÉTAPE 4: PRÉSENTATION DES RÉSULTATS =====
        result = pd.DataFrame({
            'Fichier': df_files['filename'],
            'Risque (%)': bug_probas * 100,
            'Score': bug_probas,
            'LOC': df_files.get('nloc', 0).astype(int),
            'Complexité': df_files.get('cyclomatic_complexity', 0).round(2)
        })
        
        result['Rang'] = result['Score'].rank(method='first', ascending=False).astype(int)
        result = result.sort_values('Score', ascending=False).reset_index(drop=True)
        
        def categoriser_risque(score):
            if score >= 0.6:
                return '🔴 Élevé'
            elif score >= 0.3:
                return '🟡 Moyen'
            else:
                return '🟢 Faible'
        
        result['Niveau'] = result['Score'].apply(categoriser_risque)
        
        # ===== ÉTAPE 5: AFFICHAGE =====
        st.success(f"🎯 **Analyse terminée!** {len(result)} fichiers évalués")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            risque_max = result['Score'].max() * 100
            st.metric("Risque max", f"{risque_max:.1f}%")
        with col2:
            risque_moyen = result['Score'].mean() * 100
            st.metric("Risque moyen", f"{risque_moyen:.1f}%")
        with col3:
            fichiers_risque = (result['Score'] >= 0.6).sum()
            st.metric("Fichiers à risque", fichiers_risque)
        with col4:
            fichiers_sains = (result['Score'] < 0.3).sum()
            st.metric("Fichiers sains", fichiers_sains)
        
        # Tableau des 20 plus risqués
        st.markdown("### 🏆 Top 20 fichiers les plus risqués")
        
        top20 = result.head(20).copy()
        display_df = top20[['Rang', 'Fichier', 'Niveau', 'Risque (%)', 'LOC', 'Complexité']].copy()
        display_df['Risque (%)'] = display_df['Risque (%)'].apply(lambda x: f"{x:.1f}%")
        
        def color_risk(val):
            if '🔴' in str(val):
                return 'color: #d32f2f; font-weight: bold'
            elif '🟡' in str(val):
                return 'color: #f57c00'
            elif '🟢' in str(val):
                return 'color: #388e3c'
            return ''
        
        st.dataframe(
            display_df.style.applymap(color_risk, subset=['Niveau'])
                     .format({'Complexité': '{:.2f}'}),
            use_container_width=True,
            height=700
        )
        
        # ===== ÉTAPE 6: TÉLÉCHARGEMENT =====
        st.markdown("### 💾 Téléchargement des résultats")
        
        csv_data = result.copy()
        csv_data['Risque (%)'] = csv_data['Risque (%)'].astype(str).str.replace('%', '')
        csv_string = csv_data.to_csv(index=False, encoding='utf-8')
        
        repo_name = repo_url.split('/')[-1].replace('.git', '')
        st.download_button(
            label="📥 Télécharger le rapport complet (CSV)",
            data=csv_string.encode('utf-8'),
            file_name=f"bug_risk_{repo_name}_{branch}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Résumé final
        st.markdown("---")
        st.markdown(f"""
        **📈 Résumé de l'analyse:**
        - **Dépôt analysé:** {repo_url}
        - **Branche:** {branch}
        - **Fichiers analysés:** {len(result)}
        - **Score de risque moyen:** {risque_moyen:.1f}%
        - **Fichiers nécessitant une review (risque > 60%):** {fichiers_risque}
        - **Fichiers considérés sains (risque < 30%):** {fichiers_sains}
        """)
# ============================ FOOTER ============================
st.markdown("---")
st.markdown("**Projet Génie Logiciel – Master S3 – 2025**")
st.markdown("Prédiction de défauts avec XGBoost • lizard • Streamlit • Analyse GitHub en temps réel")