# core/dataset.py
import pandas as pd
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

class DefectDataset:
    def __init__(self, csv_path: str, label_column: str = "bug"):
        self.csv_path = csv_path
        self.label_column = label_column
        self.raw_df = None
        self.X = None
        self.y = None
        self.processed_path = None

    def load(self):
        print(f"Chargement depuis {self.csv_path}")
        self.raw_df = pd.read_csv(self.csv_path, low_memory=False)
        print(f"Dataset chargé → {self.raw_df.shape}")
        return self.raw_df

    def save_processed(self, save_dir="app", name="processed_defects"):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        full_df = pd.concat([self.X, self.y], axis=1)
        self.processed_path = Path(save_dir) / f"{name}.parquet"
        full_df.to_parquet(self.processed_path, index=False)
        print(f"Dataset traité sauvegardé → {self.processed_path}")