from abc import ABC, abstractmethod
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

class ModeloBase(ABC):
    """
    Interfaz mínima que debe cumplir cualquier modelo de stock-picking.
    Permite sustituir RandomForest por XGBoost, red neuronal, etc.
    """

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Entrena el modelo con los datos proporcionados."""
        ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Devuelve la probabilidad de clase positiva para cada fila."""
        ...


class RandomForestModel(ModeloBase):
    def __init__(self, n_estimators: int = 250, max_depth: int = 4,
                 class_weight: dict = None, random_state: int = 42,
                 positive_class_weight: float = 10.0):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight or {0: 1, 1: positive_class_weight},
            random_state=random_state,
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.clf.fit(X, y)

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.clf.predict_proba(X)[:, 1], index=X.index)
    
class XGBoostModel(ModeloBase):
    def __init__(self, n_estimators: int = 250, max_depth: int = 4,
                 class_weight: dict = None, random_state: int = 42,
                 positive_class_weight: float = 10.0):
        weights = class_weight or {0: 1, 1: positive_class_weight}
        scale_pos_weight = weights[1] / weights[0]

        self.clf = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            objective="binary:logistic",
            eval_metric="logloss",
        )

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.clf.fit(X, y)
    
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self.clf.predict_proba(X)[:, 1], index=X.index)