# pipeline/evaluator.py
import sys
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # ← AJOUTE ÇA EN PREMIER

import pandas as pd
from utils.metrics import print_full_evaluation, compute_global_score  # ← Maintenant ça marche !

class DefectEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = None
        self.y_proba = None

    def evaluate(self):
        self.y_pred = self.model.predict(self.X_test)
        self.y_proba = self.model.predict_proba(self.X_test)

        print_full_evaluation(self.y_test, self.y_pred, self.y_proba, "XGBoostDefectModel")
        return compute_global_score(self.y_test, self.y_pred, self.y_proba)

    def get_predictions_df(self):
        df = pd.DataFrame({
            'y_true': self.y_test,
            'y_pred': self.y_pred,
            'proba_bug': self.y_proba
        })
        df['risk_level'] = pd.cut(df['proba_bug'], 
                                  bins=[0, 0.3, 0.6, 1.0], 
                                  labels=['Faible', 'Moyen', 'Élevé'])
        return df