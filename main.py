# pipeline/trainer.py
import sys
from pathlib import Path
# AJOUTE LA RACINE DU PROJET AU PATH (indispensable)
sys.path.append(str(Path(__file__).parent.parent))

# Maintenant les imports marchent parfaitement
from core.dataset import DefectDataset
from utils.preprocessing import clean_label, preprocess_features
from core.xgboost_model import XGBoostDefectModel
from utils.metrics import compute_global_score
from sklearn.model_selection import train_test_split
import joblib




# main.py  ←  À ÉCRASER COMPLÈTEMENT CE FICHIER
import sys
from pathlib import Path

# Ajoute la racine du projet au chemin Python
ROOT_DIR = Path(__file__).parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import de la fonction d'entraînement
from pipeline.trainer import train_and_save_xgboost   # LIGNE INDISPENSABLE
import pandas as pd

def main():
    print("="*65)
    print("        BUG PREDICTOR - Entraînement du modèle XGBoost")
    print("="*65)

    # Chemin vers ton dataset
    csv_path = "data/PROMISE-unified-class.csv"

    if not Path(csv_path).exists():
        print(f"ERREUR : Fichier non trouvé → {csv_path}")
        print("   → Déplace ton CSV dans le dossier 'data/' ou corrige le chemin")
        return

    print(f"Dataset trouvé : {csv_path}")
    print("Démarrage de l'entraînement...\n")

    try:
        train_and_save_xgboost(csv_path=csv_path, label_col="bug")
        print("\nENTRAÎNEMENT TERMINÉ AVEC SUCCÈS !")
        print("Fichiers générés dans le dossier app/ :")
        print("   ├── best_model.pkl")
        print("   ├── scaler.pkl")
        print("   ├── feature_columns.pkl")
        print("   └── processed_defects.parquet")
        print("\nTu peux maintenant lancer l'interface web ou l'API !")

    except Exception as e:
        print(f"Erreur fatale : {e}")
        raise

if __name__ == "__main__":
    main()