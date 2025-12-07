# core/xgboost_model.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from xgboost import XGBClassifier
from core.model import BaseDefectModel
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

class XGBoostDefectModel(BaseDefectModel):
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist",
            eval_metric="logloss",
            verbosity=0
        )
        self.feature_names = None

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self):
        if self.feature_names is None:
            return dict(zip(self.feature_names, self.model.feature_importances_))
        else:
            return {f"feat_{i}": v for i, v in enumerate(self.model.feature_importances_)}