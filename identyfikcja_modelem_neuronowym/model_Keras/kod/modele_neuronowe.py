"""Statyczny model neuronowy z jedną warstwą ukrytą, budowany przez Keras 3.

Wymaga backendu torch (TensorFlow nie wspiera Pythona 3.14).
Zmienna środowiskowa KERAS_BACKEND jest ustawiana przed importem Keras.
"""
import os

# Ustaw backend PRZED importem keras – zmiana po imporcie jest ignorowana.
os.environ.setdefault('KERAS_BACKEND', 'torch')

import numpy as np

_KNOWN_OPTIMIZERS = ('adam', 'sgd', 'rmsprop')


def _get_optimizers():
    """Leniwy import Keras – wykonywany dopiero przy pierwszym uczeniu."""
    import keras
    return {
        'adam':    keras.optimizers.Adam,
        'sgd':     keras.optimizers.SGD,
        'rmsprop': keras.optimizers.RMSprop,
    }


class KerasNNModel:
    """Statyczny model neuronowy: Input(1) → Dense(n_hidden, tanh) → Dense(1, linear).

    Parametry
    ----------
    n_hidden       : liczba neuronów w warstwie ukrytej
    optimizer_name : 'adam' | 'sgd' | 'rmsprop'
    lr             : współczynnik uczenia
    seed           : ziarno losowości (punkt startowy wag)
    """

    def __init__(self, n_hidden: int, optimizer_name: str = 'adam',
                 lr: float = 0.001, seed: int = 42):
        if optimizer_name not in _KNOWN_OPTIMIZERS:
            raise ValueError(f"Nieznany optymalizator: {optimizer_name!r}. "
                             f"Dostępne: {list(_KNOWN_OPTIMIZERS)}")
        self.n_hidden       = n_hidden
        self.optimizer_name = optimizer_name
        self.lr             = lr
        self.seed           = seed
        self._model   = None
        self._history = None

    # ── Budowanie modelu ───────────────────────────────────────────────────────

    def _build(self):
        import keras
        from keras import layers
        keras.utils.set_random_seed(self.seed)
        _OPTIMIZERS = _get_optimizers()
        model = keras.Sequential([
            layers.Input(shape=(1,)),
            layers.Dense(self.n_hidden, activation='tanh'),
            layers.Dense(1),
        ])
        opt = _OPTIMIZERS[self.optimizer_name](learning_rate=self.lr)
        model.compile(optimizer=opt, loss='mse')
        return model

    # ── Uczenie ────────────────────────────────────────────────────────────────

    def fit(self, u_ucz: np.ndarray, y_ucz: np.ndarray,
            epochs: int = 3000, batch_size: int = 32,
            verbose: int = 0) -> 'KerasNNModel':
        self._model = self._build()
        self._history = self._model.fit(
            u_ucz.reshape(-1, 1), y_ucz,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        return self

    # ── Predykcja i ocena ──────────────────────────────────────────────────────

    def predict(self, u: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model niewyuczony – wywołaj fit() przed predict().")
        return self._model.predict(u.reshape(-1, 1), verbose=0).ravel()

    def evaluate(self, u: np.ndarray, y: np.ndarray) -> dict:
        """Zwraca słownik z y_pred, mse, rmse."""
        y_pred = self.predict(u)
        mse = float(np.mean((y - y_pred) ** 2))
        return {'y_pred': y_pred, 'mse': mse, 'rmse': float(np.sqrt(mse))}

    # ── Właściwości ────────────────────────────────────────────────────────────

    @property
    def n_params(self) -> int | None:
        return int(self._model.count_params()) if self._model else None

    @property
    def history_loss(self) -> list[float] | None:
        """Lista wartości MSE (loss) w kolejnych epokach uczenia."""
        if self._history is None:
            return None
        return [float(v) for v in self._history.history['loss']]

    def __str__(self) -> str:
        return (f"KerasNN(hidden={self.n_hidden}, "
                f"opt={self.optimizer_name}, lr={self.lr}, seed={self.seed})")

    def __repr__(self) -> str:
        return self.__str__()
