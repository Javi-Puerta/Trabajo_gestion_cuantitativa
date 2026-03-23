import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, BaseCrossValidator

class ModeloBase(ABC):
    def train(self, df: pd.DataFrame, feature_cols: list[str]) -> None:
        """Entrena el modelo con los datos proporcionados, aplicando un grid search por el método
        de Walk-Forward cross validation."""
        X = df[feature_cols]
        y = df["Target"]
        tscv = WalkForwardCV(n_splits=self.n_splits, val_ratio=0.20)
        grid = GridSearchCV(estimator=self.clf,
                            param_grid=self.param_grid,
                            cv=tscv,
                            scoring="roc_auc")
        grid.fit(X, y, groups=df["Fecha"])
        self.clf = grid.best_estimator_

        fecha_min = df["Fecha"].min().date()
        fecha_max = df["Fecha"].max().date()
        print(f"[Train] {fecha_min} → {fecha_max} | AUC={grid.best_score_:.4f} | {grid.best_params_}")

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Devuelve la probabilidad de clase positiva para cada fila."""
        return pd.Series(self.clf.predict_proba(X)[:, 1], index=X.index)


class RandomForestModel(ModeloBase):
    def __init__(self, random_state: int = 42, n_splits=5):
        self.n_splits = n_splits
        self.clf = RandomForestClassifier(random_state=random_state)
        self.param_grid = {
            "n_estimators": [150, 250, 400], "max_depth": [3, 6],
            "min_samples_leaf": [0.01, 0.05], "max_features": ["sqrt", 0.8],
            "class_weight": [{0:1, 1:3}, {0:1, 1:5}]
        }
    
class XGBoostModel(ModeloBase):
    def __init__(self, random_state: int = 42, n_splits=5):
        self.n_splits = n_splits
        self.clf = XGBClassifier(random_state=random_state)
        self.param_grid = {
            "n_estimators":  [150, 250, 400],
            "max_depth":     [3, 6],
            "learning_rate": [0.05, 0.1],
            "subsample":     [0.7, 1.0],
        }

class WalkForwardCV(BaseCrossValidator):
    '''Implementación de cross validation para series temporales. Divide los datos en n_splits partes,
    y en cada iteración utiliza los datos desde el inicio como entrenamiento y el val_ratio posterior
    como validación.'''
    def __init__(self, n_splits=3, val_ratio=0.20):
        self.n_splits  = n_splits
        self.val_ratio = val_ratio

    def split(self, X, y=None, groups=None):
        fechas   = np.array(groups)
        unicas   = sorted(set(fechas))
        n        = len(unicas)
        val_size = max(1, int(n * self.val_ratio))
        avail    = n - val_size

        for i in range(1, self.n_splits + 1):
            train_end = int(i * avail / self.n_splits)
            val_end   = train_end + val_size
            if train_end == 0 or val_end > n:
                continue
            train_dates = set(unicas[:train_end])
            val_dates   = set(unicas[train_end:val_end])
            yield (np.where(np.isin(fechas, list(train_dates)))[0],
                   np.where(np.isin(fechas, list(val_dates)))[0])

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def _iter_test_indices(self, X=None, y=None, groups=None):
        for _, test in self.split(X, y, groups):
            yield test