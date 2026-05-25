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

class ARXModel:
    """
    Dynamiczny model liniowy ARX rzędu (nA, nB):

        y(k) = b1·u(k-1) + … + bNB·u(k-nB) + a1·y(k-1) + … + aNA·y(k-nA)

    Parametry wyznaczane MNK z solverem eliminacji Gaussa.
    """

    def __init__(self, nA: int, nB: int):
        if nA < 1 or nB < 1:
            raise ValueError("nA i nB muszą być >= 1.")
        self.nA = nA
        self.nB = nB
        self._theta: np.ndarray | None = None   # [b1,…,bNB, a1,…,aNA]

    # ── Prywatne ─────────────────────────────────────────────────────────────

    def _build_regressor(self, u: np.ndarray, y: np.ndarray) -> tuple:
        """Buduje macierz regresji Phi i offset n = max(nA, nB)."""
        n = max(self.nA, self.nB)
        rows = len(u) - n
        Phi = np.zeros((rows, self.nB + self.nA))
        for k in range(rows):
            idx = k + n
            Phi[k, :self.nB] = [u[idx - i - 1] for i in range(self.nB)]
            Phi[k, self.nB:] = [y[idx - i - 1] for i in range(self.nA)]
        return Phi, n

    # ── Publiczny interfejs ───────────────────────────────────────────────────

    def fit(self, u_ucz: np.ndarray, y_ucz: np.ndarray) -> 'ARXModel':
        """Trenuje model MNK na zbiorze uczącym; zwraca self (fluent API)."""
        Phi, n = self._build_regressor(u_ucz, y_ucz)
        self._theta = MNKSolver.solve(Phi, y_ucz[n:])
        return self

    def predict(self, u: np.ndarray, y: np.ndarray,
                recursive: bool = False) -> tuple:
        """
        Predykcja modelu. Zwraca (y_hat, n).

          recursive=False – bez rekurencji: regressor używa mierzonych y
          recursive=True  – z rekurencją (symulacja): używa własnych ŷ
        """
        if self._theta is None:
            raise RuntimeError("Model niewyuczony – wywołaj fit() przed predict().")
        n = max(self.nA, self.nB)
        b, a = self._theta[:self.nB], self._theta[self.nB:]

        if not recursive:
            Phi, n = self._build_regressor(u, y)
            return Phi @ self._theta, n

        N = len(u)
        y_hat = np.zeros(N)
        y_hat[:n] = y[:n]           # warunki początkowe z pomiarów
        for k in range(n, N):
            y_hat[k] = (b @ np.array([u[k - i - 1] for i in range(self.nB)]) +
                        a @ np.array([y_hat[k - i - 1] for i in range(self.nA)]))
        return y_hat[n:], n

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        mse = float(np.mean((y_true - y_pred) ** 2))
        return {'mse': mse, 'rmse': float(np.sqrt(mse))}

    @property
    def coefficients(self) -> np.ndarray | None:
        return self._theta

    def __str__(self) -> str:
        if self._theta is None:
            return f"ARXModel(nA={self.nA}, nB={self.nB}, niewyuczony)"
        b, a = self._theta[:self.nB], self._theta[self.nB:]
        b_str = '  '.join(f'{b[i]:+.4f}·u(k-{i+1})' for i in range(self.nB))
        a_str = '  '.join(f'{a[i]:+.4f}·y(k-{i+1})' for i in range(self.nA))
        return f"ARX(nA={self.nA}, nB={self.nB}):  y(k) = {b_str}  {a_str}"


class NARXModel:
    """
    Dynamiczny model wielomianowy NARX rzędu (nA, nB) stopnia deg:

        y(k) = Σ_{i=1}^{nB} Σ_{p=1}^{deg} w_{i,p}·u(k-i)^p
             + Σ_{i=1}^{nA} Σ_{p=1}^{deg} w'_{i,p}·y(k-i)^p

    Parametry wyznaczane MNK z solverem eliminacji Gaussa.
    Układ kolumn theta: [u(k-1)^1, u(k-1)^2, …, u(k-nB)^deg,
                         y(k-1)^1, y(k-1)^2, …, y(k-nA)^deg]
    """

    _POW_SYM = {1: '', 2: '²', 3: '³', 4: '⁴'}

    def __init__(self, nA: int, nB: int, deg: int):
        if nA < 1 or nB < 1 or deg < 1:
            raise ValueError("nA, nB >= 1 i deg >= 1.")
        self.nA  = nA
        self.nB  = nB
        self.deg = deg
        self._theta: np.ndarray | None = None   # długość = (nB + nA) * deg

    # ── Prywatne ─────────────────────────────────────────────────────────────

    def _build_regressor(self, u: np.ndarray, y: np.ndarray) -> tuple:
        """Buduje macierz regresji Phi i offset n = max(nA, nB)."""
        n    = max(self.nA, self.nB)
        rows = len(u) - n
        pows = np.arange(1, self.deg + 1)  # [1, 2, ..., deg]

        # u_lags[k, i] = u[k+n-i-1], shape (rows, nB)
        u_lags = np.column_stack([u[n-i-1:n-i-1+rows] for i in range(self.nB)])
        # y_lags[k, i] = y[k+n-i-1], shape (rows, nA)
        y_lags = np.column_stack([y[n-i-1:n-i-1+rows] for i in range(self.nA)])

        # Broadcasting: (rows, nX, 1)^(1, 1, deg) → (rows, nX, deg) → (rows, nX*deg)
        u_pows = (u_lags[:, :, None] ** pows).reshape(rows, self.nB * self.deg)
        y_pows = (y_lags[:, :, None] ** pows).reshape(rows, self.nA * self.deg)

        return np.hstack([u_pows, y_pows]), n

    # ── Publiczny interfejs ───────────────────────────────────────────────────

    def fit(self, u_ucz: np.ndarray, y_ucz: np.ndarray) -> 'NARXModel':
        """Trenuje model MNK; zwraca self (fluent API)."""
        Phi, n = self._build_regressor(u_ucz, y_ucz)
        self._theta = MNKSolver.solve(Phi, y_ucz[n:])
        return self

    def predict(self, u: np.ndarray, y: np.ndarray,
                recursive: bool = False) -> tuple:
        """
        Predykcja modelu. Zwraca (y_hat, n).

          recursive=False – bez rekurencji: regressor używa mierzonych y
          recursive=True  – z rekurencją (symulacja): używa własnych ŷ
        """
        if self._theta is None:
            raise RuntimeError("Model niewyuczony – wywołaj fit() przed predict().")
        n = max(self.nA, self.nB)

        if not recursive:
            Phi, n = self._build_regressor(u, y)
            return Phi @ self._theta, n

        N    = len(u)
        n    = max(self.nA, self.nB)
        pows = np.arange(1, self.deg + 1)  # precomputed once
        y_hat = np.zeros(N)
        y_hat[:n] = y[:n]
        for k in range(n, N):
            u_lags_k = np.array([u[k-i-1] for i in range(self.nB)])
            y_lags_k = np.array([y_hat[k-i-1] for i in range(self.nA)])
            # (nX, deg) broadcasting, then flatten → regressor vector
            u_pows = (u_lags_k[:, None] ** pows).ravel()  # (nB*deg,)
            y_pows = (y_lags_k[:, None] ** pows).ravel()  # (nA*deg,)
            val = float(np.dot(np.concatenate([u_pows, y_pows]), self._theta))
            y_hat[k] = val if np.isfinite(val) else 0.0
        return y_hat[n:], n

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        if not np.all(np.isfinite(y_pred)):
            return {'mse': np.inf, 'rmse': np.inf}
        mse = float(np.mean((y_true - y_pred) ** 2))
        return {'mse': mse, 'rmse': float(np.sqrt(mse))}

    @property
    def n_params(self) -> int:
        return (self.nB + self.nA) * self.deg

    def __str__(self) -> str:
        if self._theta is None:
            return f"NARX(nA={self.nA}, nB={self.nB}, deg={self.deg}, niewyuczony)"
        return f"NARX(nA={self.nA}, nB={self.nB}, deg={self.deg})"
