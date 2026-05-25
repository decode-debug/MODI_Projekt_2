import numpy as np


import numpy as np

class MNKSolver:
    """Solver Metody Najmniejszych Kwadratów."""

    @staticmethod
    def solve(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Rozwiązuje układ Phi * theta = y w sensie MNK (odpowiednik Phi \ y w MATLAB).
        Wykorzystuje rozkład QR (transformacje Householdera) omijając budowę
        układu normalnego (Phi^T Phi), co zapewnia lepszą stabilność numeryczną.

        Phi: macierz regresorów (N x M)
        y:   wektor wartości docelowych (N,)
        """
        R = Phi.astype(float).copy()
        d = y.astype(float).copy()
        N, M = R.shape
        for k in range(M):
            x = R[k:, k]
            norm_x = np.linalg.norm(x)

            if norm_x < 1e-12:
                continue
            s = 1.0 if x[0] >= 0 else -1.0
            u1 = x[0] + s * norm_x
            v = x / u1
            v[0] = 1.0
            beta = 2.0 / np.dot(v, v)
            R[k:, k:] -= beta * np.outer(v, np.dot(v, R[k:, k:]))
            d[k:] -= beta * v * np.dot(v, d[k:])

        theta = np.zeros(M)
        for i in range(M - 1, -1, -1):
            if abs(R[i, i]) < 1e-12:
                return np.full(M, np.nan)

            theta[i] = (d[i] - np.dot(R[i, i+1:M], theta[i+1:])) / R[i, i]

        return theta

class BaseModel:
    """Wspólna logika fit / predict / evaluate dla modeli MNK."""

    def __init__(self):
        self._theta: np.ndarray | None = None

    def _build_regressor(self, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, u_ucz: np.ndarray, y_ucz: np.ndarray) -> 'BaseModel':
        self._theta = MNKSolver.solve(self._build_regressor(u_ucz), y_ucz)
        return self

    def predict(self, u: np.ndarray) -> np.ndarray:
        if self._theta is None:
            raise RuntimeError("Model niewyuczony – wywołaj fit() przed predict().")
        return self._build_regressor(u) @ self._theta

    def evaluate(self, u: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(u)
        mse = float(np.mean((y - y_pred) ** 2))
        return {'y_pred': y_pred, 'mse': mse, 'rmse': float(np.sqrt(mse))}

    @property
    def coefficients(self) -> np.ndarray | None:
        return self._theta


class LinearModel(BaseModel):
    """Model liniowy: y(u) = a0 + a1·u."""

    def __init__(self):
        super().__init__()
        self.a0: float | None = None
        self.a1: float | None = None

    def _build_regressor(self, u: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(u)), u])

    def fit(self, u_ucz: np.ndarray, y_ucz: np.ndarray) -> 'LinearModel':
        super().fit(u_ucz, y_ucz)
        self.a0, self.a1 = float(self._theta[0]), float(self._theta[1])
        return self

    def __str__(self) -> str:
        if self._theta is None:
            return "LinearModel (niewyuczony)"
        return f"Model liniowy:  y(u) = {self.a0:.6f} + {self.a1:.6f} * u"


class PolynomialModel(BaseModel):
    """Model wielomianowy stopnia N: y(u) = a0 + a1·u + … + aN·u^N."""

    def __init__(self, degree: int):
        if degree < 1:
            raise ValueError("Stopień wielomianu musi być >= 1.")
        super().__init__()
        self.degree = degree

    def _build_regressor(self, u: np.ndarray) -> np.ndarray:
        return np.column_stack([u ** i for i in range(self.degree + 1)])

    def __str__(self) -> str:
        if self._theta is None:
            return f"PolynomialModel(N={self.degree}, niewyuczony)"
        terms = [f'{self._theta[0]:.4f}'] + [
            f'{c:+.4f}·u' if i == 1 else f'{c:+.4f}·u^{i}'
            for i, c in enumerate(self._theta[1:], 1)
        ]
        return f"Model wielomianowy N={self.degree}:  y(u) = {'  '.join(terms)}"


