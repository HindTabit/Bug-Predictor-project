# core/model.py
from abc import ABC, abstractmethod
import joblib
from pathlib import Path

class BaseDefectModel(ABC):
    """Classe abstraite pour tous les modèles de prédiction de bugs"""

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

    def save(self, path: str = "app/best_model.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"Modèle sauvegardé → {path}")

    @classmethod
    def load(cls, path: str):
        model = joblib.load(path)
        print(f"Modèle chargé depuis → {path}")
        return model