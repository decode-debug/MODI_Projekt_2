import numpy as np


class MNKSolver:
    """Solver Metody Najmniejszych Kwadratów."""

    @staticmethod
    def solve(Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Rozwiązuje układ normalny (Phi^T Phi) θ = Phi^T y
        metodą eliminacji Gaussa z częściowym wyborem pivotu.
        Phi: macierz regresorów (N x M)
        y:   wektor wartości docelowych (N,)
        """
        A = (Phi.T @ Phi).astype(float)
        b = (Phi.T @ y).astype(float)
        n = len(b)
        # Eliminacja w przód z częściowym wyborem pivotu
        for k in range(n):
            pivot = k + np.argmax(np.abs(A[k:, k]))
            A[[k, pivot]], b[[k, pivot]] = A[[pivot, k]].copy(), b[[pivot, k]].copy()
            for i in range(k + 1, n):
                f = A[i, k] / A[k, k]
                A[i, k:] -= f * A[k, k:]
                b[i]     -= f * b[k]
        # Podstawianie wstecz
        theta = np.zeros(n)
        for i in range(n - 1, -1, -1):
            theta[i] = (b[i] - A[i, i+1:] @ theta[i+1:]) / A[i, i]
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


