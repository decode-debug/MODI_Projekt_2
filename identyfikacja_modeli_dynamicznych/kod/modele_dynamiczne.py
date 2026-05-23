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
