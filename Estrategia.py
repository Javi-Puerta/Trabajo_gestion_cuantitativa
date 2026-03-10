import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from Modelos import ModeloBase
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize


class EstrategiaBase(ABC):
    """
    Interfaz que cualquier estrategia debe implementar.
    Responsabilidad: decidir qué tickers comprar y con qué peso.
    """

    @abstractmethod
    def train(self, df: pd.DataFrame, feature_cols: list[str], tickers_validos: set) -> bool:
        """
        Entrena la estrategia con los datos del periodo.
        Devuelve True si el entrenamiento fue exitoso, False si no hay datos suficientes.
        """
        ...

    @abstractmethod
    def seleccionar(self, df_hoy: pd.DataFrame, feature_cols: list[str],
                    cartera: dict[str, float]) -> dict[str, float]:
        """
        Devuelve un diccionario {ticker: peso} con la cartera para esa semana.
        Los pesos deben sumar 1.
        """
        ...


class EstrategiaMLEquiponderada(EstrategiaBase):
    """
    Usa un modelo ML para puntuar los activos y asigna pesos iguales
    entre los n_activos_obj mejor puntuados, con buffer de permanencia
    para reducir rotación.

    Parámetros
    ----------
    modelo        : ModeloBase — modelo de clasificación
    n_activos_obj : int        — número de activos en cartera
    umbral_salida : int        — top-N para el buffer de permanencia
    """

    def __init__(self, modelo: ModeloBase, n_activos_obj: int, umbral_salida: int):
        self.modelo        = modelo
        self.n_activos_obj = n_activos_obj
        self.umbral_salida = umbral_salida
        self._cartera_actual: set = set()

    def train(self, df: pd.DataFrame, feature_cols: list[str], tickers_validos: set) -> bool:
        train_data = df[df["Ticker"].isin(tickers_validos)].copy()
        if train_data.empty:
            return False
        self.modelo.train(train_data[feature_cols], train_data["Target"])
        return True

    def seleccionar(self, df_hoy: pd.DataFrame, feature_cols: list[str],
                    cartera: dict[str, float]) -> dict[str, float]:
        datos = df_hoy.copy()
        if datos.empty:
            return {}

        proba = self.modelo.predict_proba(datos[feature_cols])
        datos["Score"] = proba[:, 1] if getattr(proba, "ndim", 1) > 1 else proba
        datos = datos.sort_values("Score", ascending=False)

        # Posiciones actuales (ignorando cash)
        cartera_actual = set(cartera.keys()) - {"cash"}

        tickers_hoy = set(datos["Ticker"])
        mantener_base = cartera_actual & tickers_hoy  # elimina delisted automáticamente

        if not mantener_base:
            nueva_cartera = set(datos.head(self.n_activos_obj)["Ticker"])
        else:
            top_mant = set(datos.head(self.umbral_salida)["Ticker"])
            mantener = mantener_base & top_mant
            huecos = self.n_activos_obj - len(mantener)
            candidatos = [t for t in datos["Ticker"] if t not in mantener]
            nueva_cartera = mantener | set(candidatos[:huecos])

        peso = 1 / len(nueva_cartera)
        return {t: peso for t in nueva_cartera}


class EstrategiaMarkI(EstrategiaBase):
    """
    Usa un modelo ML para puntuar los activos. 
    Elige el top-N mejor puntuados y que a la vez tienen una probabilidad de superar al mercado (columna "Target") mayor a 0.75. 
    Asigna pesos iguales entre los seleccionados, con buffer de permanencia para reducir rotación.

    Parámetros
    ----------
    modelo        : ModeloBase — modelo de clasificación
    n_activos_obj : int        — número de activos en cartera
    umbral_salida : int        — top-N para el buffer de permanencia
    umbral_target : float      — probabilidad mínima de superar al mercado para ser considerado en la selección
    """

    def __init__(self, modelo: ModeloBase, n_activos_obj: int, umbral_salida: int, umbral_target: float = 0.75):
        self.modelo        = modelo
        self.n_activos_obj = n_activos_obj
        self.umbral_salida = umbral_salida
        self.umbral_target = umbral_target
        self._cartera_actual: set = set()

    def train(self, df: pd.DataFrame, feature_cols: list[str], tickers_validos: set) -> bool:
        train_data = df[df["Ticker"].isin(tickers_validos)]
        if train_data.empty:
            return False
        self.modelo.train(train_data[feature_cols], train_data["Target"])
        return True

    def seleccionar(self, df_hoy: pd.DataFrame, feature_cols: list[str],
                    cartera: dict[str, float]) -> dict[str, float]:
        datos = df_hoy.copy()
        if datos.empty:
            return {}

        proba = self.modelo.predict_proba(datos[feature_cols])
        datos["Score"] = proba[:, 1] if getattr(proba, "ndim", 1) > 1 else proba
        
        # Filtrar solo activos con Score >= umbral_target
        datos = datos[datos["Score"] >= self.umbral_target]
        if datos.empty:
            return {}
        
        datos = datos.sort_values("Score", ascending=False)

        # Posiciones actuales (ignorando cash)
        cartera_actual = set(cartera.keys()) - {"cash"}

        tickers_hoy = set(datos["Ticker"])
        mantener_base = cartera_actual & tickers_hoy  # elimina delisted automáticamente

        if not mantener_base:
            nueva_cartera = set(datos.head(self.n_activos_obj)["Ticker"])
        else:
            top_mant = set(datos.head(self.umbral_salida)["Ticker"])
            mantener = mantener_base & top_mant
            huecos = self.n_activos_obj - len(mantener)
            candidatos = [t for t in datos["Ticker"] if t not in mantener]
            nueva_cartera = mantener | set(candidatos[:huecos])

        peso = 1 / len(nueva_cartera) if nueva_cartera else 0
        return {t: peso for t in nueva_cartera}
    
class EstrategiaMLMinVarAlphaTilt(EstrategiaBase):
    """
    PRUEBA JAVI
    """

    def __init__(
        self,
        modelo: ModeloBase,
        n_activos_obj: int = 20,
        umbral_salida: int = 60,
        p_neutral: float = 0.50,
        alpha_scale: float = 1.0,
        lambda_risk: float = 8.0,
        lambda_tc: float = 0.002,
        w_max: float = 0.10,
        turnover_max: float = 0.20,
        no_trade_band: float = 0.003,
        coste_transaccion: float = 0.0005,
        utility_buffer: float = 0.0002,
        min_hist_obs: int = 26,
    ):
        self.modelo = modelo
        self.n_activos_obj = n_activos_obj
        self.umbral_salida = umbral_salida

        self.p_neutral = p_neutral
        self.alpha_scale = alpha_scale
        self.lambda_risk = lambda_risk
        self.lambda_tc = lambda_tc
        self.w_max = w_max
        self.turnover_max = turnover_max
        self.no_trade_band = no_trade_band
        self.coste_transaccion = coste_transaccion
        self.utility_buffer = utility_buffer
        self.min_hist_obs = min_hist_obs

        self._cov = pd.DataFrame()
        self._trained_tickers = set()

    def train(self, df: pd.DataFrame, feature_cols: list[str], tickers_validos: set) -> bool:
        train_data = df[df["Ticker"].isin(tickers_validos)].copy()
        if train_data.empty:
            return False

        # 1) Entrenamos alpha model
        self.modelo.train(train_data[feature_cols], train_data["Target"])

        # 2) Estimamos covarianza robusta sobre retornos semanales
        ret = (
            train_data.pivot(index="Fecha", columns="Ticker", values="Retorno_1W")
            .sort_index()
            .dropna(axis=1, thresh=self.min_hist_obs)
        )
        if ret.shape[1] < 2:
            self._cov = pd.DataFrame()
            self._trained_tickers = set()
            return True

        ret = ret.fillna(0.0)
        lw = LedoitWolf()
        lw.fit(ret.values)
        cov = pd.DataFrame(lw.covariance_, index=ret.columns, columns=ret.columns)

        # Jitter para estabilidad numérica
        cov.values[np.diag_indices_from(cov)] += 1e-8

        self._cov = cov
        self._trained_tickers = set(cov.columns)
        return True

    @staticmethod
    def _l1_smooth(x: np.ndarray, eps: float = 1e-8) -> float:
        return float(np.sum(np.sqrt(x * x + eps)))

    def _current_weights_from_cartera(
        self, cartera: dict[str, float], precios_hoy: pd.Series, activos: list[str]
    ) -> np.ndarray:
        # cartera trae cantidades de acciones + cash
        valor_total = float(cartera.get("cash", 0.0))
        for t, qty in cartera.items():
            if t == "cash":
                continue
            px = precios_hoy.get(t, np.nan)
            if pd.notna(px):
                valor_total += float(qty) * float(px)

        if valor_total <= 0:
            return np.zeros(len(activos), dtype=float)

        w_prev = np.zeros(len(activos), dtype=float)
        for i, t in enumerate(activos):
            qty = float(cartera.get(t, 0.0))
            px = precios_hoy.get(t, np.nan)
            if pd.notna(px) and qty > 0:
                w_prev[i] = qty * float(px) / valor_total
        return w_prev

    def _optimize_weights(
        self, alpha: np.ndarray, cov: np.ndarray, w_prev: np.ndarray
    ) -> np.ndarray:
        n = len(alpha)
        if n == 0:
            return np.array([], dtype=float)

        # Inicio: combinación de prev + uniforme
        x0 = 0.5 * w_prev + 0.5 * np.ones(n) / n
        x0 = np.clip(x0, 0.0, self.w_max)
        s0 = x0.sum()
        if s0 <= 0:
            x0 = np.ones(n) / n
        else:
            x0 = x0 / s0

        def obj(w: np.ndarray) -> float:
            risk = self.lambda_risk * float(w @ cov @ w)
            ret = -float(alpha @ w)
            tc = self.lambda_tc * self._l1_smooth(w - w_prev)
            return ret + risk + tc

        cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
        bounds = [(0.0, self.w_max) for _ in range(n)]

        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
        if not res.success:
            return w_prev if w_prev.sum() > 0 else np.ones(n) / n

        w = np.clip(res.x, 0.0, self.w_max)
        s = w.sum()
        if s <= 0:
            w = w_prev if w_prev.sum() > 0 else np.ones(n) / n
            s = w.sum()
        w = w / s
        return w

    def seleccionar(self, df_hoy: pd.DataFrame, feature_cols: list[str], cartera: dict[str, float]) -> dict[str, float]:
        datos = df_hoy.copy()
        if datos.empty:
            return {}

        # Necesitamos cov entrenada para min-var
        if self._cov.empty:
            return {}

        # Score ML -> alpha
        proba = self.modelo.predict_proba(datos[feature_cols])
        p = np.asarray(proba).reshape(-1)
        datos["Score"] = p

        # Candidatos: top umbral_salida + holdings actuales (buffer de permanencia)
        datos = datos.sort_values("Score", ascending=False)
        top = set(datos.head(self.umbral_salida)["Ticker"])
        actuales = set(cartera.keys()) - {"cash"}
        candidatos = list((top | actuales) & set(datos["Ticker"]) & self._trained_tickers)
        if len(candidatos) < 2:
            # fallback: top n_activos_obj
            candidatos = list((set(datos.head(self.n_activos_obj)["Ticker"])) & self._trained_tickers)
            if len(candidatos) == 0:
                return {}

        sub = datos[datos["Ticker"].isin(candidatos)].copy()
        sub = sub.drop_duplicates("Ticker").set_index("Ticker").loc[candidatos]

        alpha = self.alpha_scale * (sub["Score"].values - self.p_neutral)
        alpha = np.clip(alpha, 0.0, None)  # long-only, no invertimos en alpha negativo

        # Si todos quedan en 0, usa score sin centrar para no vaciar cartera
        if np.all(alpha <= 0):
            alpha = np.clip(sub["Score"].values, 0.0, None)

        cov_df = self._cov.reindex(index=candidatos, columns=candidatos).fillna(0.0)
        cov = cov_df.values
        cov[np.diag_indices_from(cov)] += 1e-8

        precios_hoy = sub["Precio_Close"]
        w_prev = self._current_weights_from_cartera(cartera, precios_hoy, candidatos)

        # Optimización principal
        w_new = self._optimize_weights(alpha=alpha, cov=cov, w_prev=w_prev)

        # No-trade band por activo
        delta = w_new - w_prev
        mask_small = np.abs(delta) < self.no_trade_band
        w_new[mask_small] = w_prev[mask_small]

        # Renormalizar
        if w_new.sum() > 0:
            w_new = w_new / w_new.sum()

        # Cap de turnover total (L1)
        turnover = float(np.sum(np.abs(w_new - w_prev)))
        if turnover > self.turnover_max and turnover > 0:
            k = self.turnover_max / turnover
            w_new = w_prev + k * (w_new - w_prev)
            if w_new.sum() > 0:
                w_new = w_new / w_new.sum()

        # Gate de mejora neta vs coste esperado (evita sobretrading)
        util_prev = float(alpha @ w_prev - self.lambda_risk * (w_prev @ cov @ w_prev))
        util_new = float(alpha @ w_new - self.lambda_risk * (w_new @ cov @ w_new))
        coste_estimado = self.coste_transaccion * float(np.sum(np.abs(w_new - w_prev)))
        mejora_neta = util_new - util_prev - coste_estimado

        if mejora_neta <= self.utility_buffer:
            # Mantener cartera actual si la mejora no compensa fricciones
            if w_prev.sum() <= 0:
                return {}
            return {t: float(w_prev[i]) for i, t in enumerate(candidatos) if w_prev[i] > 0}

        # Mantener solo top n_activos_obj por peso final
        order = np.argsort(-w_new)
        keep_idx = order[: self.n_activos_obj]
        w_final = np.zeros_like(w_new)
        w_final[keep_idx] = w_new[keep_idx]
        if w_final.sum() > 0:
            w_final = w_final / w_final.sum()

        return {t: float(w_final[i]) for i, t in enumerate(candidatos) if w_final[i] > 0}
    
    
    
class EstrategiaMLBlackLitterman(EstrategiaBase):
    def __init__(
        self,
        modelo: ModeloBase,
        n_activos_obj: int = 20,
        umbral_salida: int = 60,
        tau: float = 0.05,
        delta: float = 2.5,
        score_neutral: float = 0.50,
        view_scale: float = 0.02,
        w_max: float = 0.10,
        turnover_max: float = 0.20,
        no_trade_band: float = 0.003,
        coste_transaccion: float = 0.0005,
        lambda_tc: float = 0.002,
        lambda_risk: float = 8.0,
        utility_buffer: float = 0.0002,
        min_hist_obs: int = 26,
    ):
        self.modelo = modelo
        self.n_activos_obj = n_activos_obj
        self.umbral_salida = umbral_salida
        self.tau = tau
        self.delta = delta
        self.score_neutral = score_neutral
        self.view_scale = view_scale
        self.w_max = w_max
        self.turnover_max = turnover_max
        self.no_trade_band = no_trade_band
        self.coste_transaccion = coste_transaccion
        self.lambda_tc = lambda_tc
        self.lambda_risk = lambda_risk
        self.utility_buffer = utility_buffer
        self.min_hist_obs = min_hist_obs

        self._cov = pd.DataFrame()
        self._trained_tickers = set()

    def train(self, df: pd.DataFrame, feature_cols: list[str], tickers_validos: set) -> bool:
        train_data = df[df["Ticker"].isin(tickers_validos)].copy()
        if train_data.empty:
            return False

        self.modelo.train(train_data[feature_cols], train_data["Target"])

        ret = (
            train_data.pivot(index="Fecha", columns="Ticker", values="Retorno_1W")
            .sort_index()
            .dropna(axis=1, thresh=self.min_hist_obs)
        )
        if ret.shape[1] < 2:
            self._cov = pd.DataFrame()
            self._trained_tickers = set()
            return True

        ret = ret.fillna(0.0)
        lw = LedoitWolf()
        lw.fit(ret.values)
        cov = pd.DataFrame(lw.covariance_, index=ret.columns, columns=ret.columns)
        cov.values[np.diag_indices_from(cov)] += 1e-8

        self._cov = cov
        self._trained_tickers = set(cov.columns)
        return True

    def _current_weights_from_cartera(self, cartera, precios_hoy, activos):
        valor_total = float(cartera.get("cash", 0.0))
        for t, qty in cartera.items():
            if t == "cash":
                continue
            px = precios_hoy.get(t, np.nan)
            if pd.notna(px):
                valor_total += float(qty) * float(px)

        if valor_total <= 0:
            return np.zeros(len(activos), dtype=float)

        w_prev = np.zeros(len(activos), dtype=float)
        for i, t in enumerate(activos):
            qty = float(cartera.get(t, 0.0))
            px = precios_hoy.get(t, np.nan)
            if pd.notna(px) and qty > 0:
                w_prev[i] = qty * float(px) / valor_total
        return w_prev

    def _build_prior_weights(self, scores: np.ndarray) -> np.ndarray:
        # prior simple: equiponderado
        n = len(scores)
        return np.ones(n) / n

    def _build_views(self, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Views absolutas por activo
        centered = scores - self.score_neutral
        q = self.view_scale * 2.0 * centered
        q = np.clip(q, -1.0, 1.0)  # cap +/-100%
        conf = np.clip(np.abs(centered) / max(self.score_neutral, 1e-8), 0.01, 1.0)
        P = np.eye(len(scores))
        return P, q.reshape(-1, 1), conf

    def _build_omega(self, cov: np.ndarray, P: np.ndarray, conf: np.ndarray) -> np.ndarray:
        # diagonal con incertidumbre inversa a confianza
        base = np.diag(P @ (self.tau * cov) @ P.T)
        omega_diag = base / conf
        omega_diag = np.maximum(omega_diag, 1e-8)
        return np.diag(omega_diag)

    def _black_litterman_posterior(self, pi, cov, P, Q, omega):
        tau_cov = self.tau * cov
        inv_tau_cov = np.linalg.pinv(tau_cov)
        inv_omega = np.linalg.pinv(omega)

        A = inv_tau_cov + P.T @ inv_omega @ P
        b = inv_tau_cov @ pi.reshape(-1, 1) + P.T @ inv_omega @ Q
        mu_bl = np.linalg.pinv(A) @ b
        return mu_bl.ravel()

    def _optimize_weights(self, mu_bl, cov, w_prev):
        n = len(mu_bl)
        if n == 0: 
            return np.array([], dtype=float)

        
        x0 = np.ones(n) / n if w_prev.sum() <= 0 else w_prev.copy()
        x0 = np.clip(x0, 0.0, self.w_max)
        if x0.sum() <= 0: 
            x0 = np.ones(n) / n 
        else: 
            x0 = x0 / x0.sum()

        def obj(w):
            ret = -float(mu_bl @ w)
            risk = self.lambda_risk * float(w @ cov @ w)
            tc = self.lambda_tc * float(np.sum(np.sqrt((w - w_prev) ** 2 + 1e-8)))
            return ret + risk + tc

        cons = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
        bounds = [(0.0, self.w_max) for _ in range(n)]

        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 300, "ftol": 1e-9})
        
        if not res.success:
            s = x0.sum()
            return x0 / s if s > 0 else np.ones(n) / n
        
        w = np.clip(res.x, 0.0, self.w_max)
        s = w.sum()
        if s <= 0: 
            s0 = x0.sum()
            return x0 / s0 if s0 > 0 else np.ones(n) / n
        return w / s

    def seleccionar(self, df_hoy: pd.DataFrame, feature_cols: list[str], cartera: dict[str, float]) -> dict[str, float]:
        datos = df_hoy.copy()
        if datos.empty or self._cov.empty:
            return {}

        proba = self.modelo.predict_proba(datos[feature_cols])
        datos["Score"] = proba[:, 1] if getattr(proba, "ndim", 1) > 1 else np.asarray(proba).reshape(-1)
        datos = datos.sort_values("Score", ascending=False)

        top = set(datos.head(self.umbral_salida)["Ticker"])
        actuales = set(cartera.keys()) - {"cash"}
        candidatos = list((top | actuales) & set(datos["Ticker"]) & self._trained_tickers)
        if len(candidatos) < 2:
            candidatos = list(set(datos.head(self.n_activos_obj)["Ticker"]) & self._trained_tickers)
            if len(candidatos) < 2:
                return {}

        sub = datos[datos["Ticker"].isin(candidatos)].drop_duplicates("Ticker").set_index("Ticker")
        sub = sub.loc[candidatos]

        cov_df = self._cov.reindex(index=candidatos, columns=candidatos).fillna(0.0)
        cov = cov_df.values
        cov[np.diag_indices_from(cov)] += 1e-8

        scores = sub["Score"].values
        w_mkt = self._build_prior_weights(scores)
        pi = self.delta * cov @ w_mkt

        P, Q, conf = self._build_views(scores)
        omega = self._build_omega(cov, P, conf)
        mu_bl = self._black_litterman_posterior(pi, cov, P, Q, omega)

        precios_hoy = sub["Precio_Close"]
        w_prev = self._current_weights_from_cartera(cartera, precios_hoy, candidatos)
        w_new = self._optimize_weights(mu_bl, cov, w_prev)

        delta_w = w_new - w_prev
        w_new[np.abs(delta_w) < self.no_trade_band] = w_prev[np.abs(delta_w) < self.no_trade_band]
        if w_new.sum() > 0:
            w_new = w_new / w_new.sum()

        turnover = float(np.sum(np.abs(w_new - w_prev)))
        if turnover > self.turnover_max and turnover > 0:
            k = self.turnover_max / turnover
            w_new = w_prev + k * (w_new - w_prev)
            if w_new.sum() > 0: 
                w_new = w_new / w_new.sum()
                
        # filtro de utilidad neta 
        util_prev = float(mu_bl @ w_prev - self.lambda_risk * (w_prev @ cov @ w_prev)) 
        util_new = float(mu_bl @ w_new - self.lambda_risk * (w_new @ cov @ w_new)) 
        coste_estimado = self.coste_transaccion * float(np.sum(np.abs(w_new - w_prev))) 
        mejora_neta = util_new - util_prev - coste_estimado 
        if mejora_neta <= self.utility_buffer: 
            if w_prev.sum() <= 0: 
                return {} 
            return {t: float(w_prev[i]) for i, t in enumerate(candidatos) if w_prev[i] > 0}


        order = np.argsort(-w_new)
        keep = order[:self.n_activos_obj]
        w_final = np.zeros_like(w_new)
        w_final[keep] = w_new[keep]
        if w_final.sum() > 0: 
            w_final = w_final / w_final.sum()

        return {t: float(w_final[i]) for i, t in enumerate(candidatos) if w_final[i] > 0}