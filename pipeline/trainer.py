# pipeline/trainer.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from core.dataset import DefectDataset
from utils.preprocessing import clean_label, preprocess_features
from core.xgboost_model import XGBoostDefectModel
from utils.metrics import compute_global_score
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

def train_and_save_xgboost(csv_path: str = "data/software_defects.csv", label_col: str = "bug"):
    # 1. Chargement
    dataset = DefectDataset(csv_path, label_col)
    df = dataset.load()

    # 2. Pré-traitement
    dataset.y = clean_label(df, label_col)
    X_scaled, scaler, feature_columns = preprocess_features(df.drop(columns=[label_col]))
    dataset.X = pd.DataFrame(X_scaled, columns=feature_columns)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, dataset.y, test_size=0.2, random_state=42, stratify=dataset.y
    )

    # 4. Modèle XGBoost
    print("\nEntraînement du XGBoostDefectModel...")
    model = XGBoostDefectModel()
    model.fit(X_train, y_train, feature_names=feature_columns)

    # 5. Évaluation rapide
    from sklearn.metrics import f1_score, recall_score, roc_auc_score, accuracy_score
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    score = 0.45*f1_score(y_test,y_pred) + 0.3*recall_score(y_test,y_pred) + 0.2*roc_auc_score(y_test,y_proba) + 0.05*accuracy_score(y_test,y_pred)
    print(f"Score global → {score:.4f}")

    # 6. Sauvegarde dans app/
    model.save("app/best_model.pkl")
    joblib.dump(scaler, "app/scaler.pkl")
    joblib.dump(feature_columns, "app/feature_columns.pkl")
    dataset.save_processed()

    # 7. Top features
    print("\nTop 15 features les plus importantes :")
    for name, imp in sorted(model.get_feature_importance().items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"   {name}: {imp:.4f}")

    print("\nTout est prêt dans le dossier app/ !")
    return model