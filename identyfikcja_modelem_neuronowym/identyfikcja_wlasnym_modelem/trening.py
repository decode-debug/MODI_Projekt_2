"""
Trening modelu MLP do regresji (MSE loss, wyjście liniowe).
Skopiowany i zaadaptowany z LAB_5_WSI.
"""

import numpy as np
from .model import Model


class TrenujModel:
    """Mini-batch SGD z propagacją wsteczną dla regresji (MSE + liniowe wyjście)."""

    def __init__(self, model: Model, learning_rate: float = 0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.loss_history    = []
        self.val_mse_history = []

    # ── Loss ───────────────────────────────────────────────────────────

    def _mse_loss(self, y_pred, y_true):
        return float(np.mean((y_pred - y_true) ** 2))

    # ── Backward ───────────────────────────────────────────────────────

    def _backward(self, y_pred, y_true):
        """
        Wsteczna propagacja błędu.
        y_pred: (N, 1)  y_true: (N, 1)
        Zakłada liniowe wyjście + MSE: dZ_out = 2*(y_pred - y_true)/N
        """
        n = len(y_true)
        gradients = []

        # Gradient dla warstwy wyjściowej (liniowa aktywacja + MSE)
        error = 2.0 * (y_pred - y_true) / n   # (N, output_size)

        for i in reversed(range(len(self.model.layers))):
            a_in, z, a_out = self.model.cache[i]

            dW = error.T @ a_in          # (nodes_out, nodes_in)
            db = np.sum(error, axis=0)   # (nodes_out,)
            gradients.append((i, dW, db))

            if i > 0:
                W, _ = self.model.layers[i]._get_weights()
                _, z_prev, _ = self.model.cache[i - 1]
                act = self.model.layers[i - 1].activation_name
                if act == 'sigmoid':
                    sig   = 1.0 / (1.0 + np.exp(-np.clip(z_prev, -500, 500)))
                    deriv = sig * (1.0 - sig)
                elif act == 'linear':
                    deriv = np.ones_like(z_prev)
                else:  # relu
                    deriv = (z_prev > 0).astype(float)
                error = (error @ W) * deriv

        return gradients

    # ── Gradient descent step ──────────────────────────────────────────

    def gradient_descent(self, X, y):
        y_pred = self.model._forward(X)
        loss   = self._mse_loss(y_pred, y)
        grads  = self._backward(y_pred, y)
        for layer_idx, dW, db in grads:
            layer = self.model.layers[layer_idx]
            W, b  = layer._get_weights()
            W    -= self.learning_rate * dW
            b    -= self.learning_rate * db
            layer._set_weights(W, b)
        return loss

    # ── Full training loop ─────────────────────────────────────────────

    def train(self, X_train, y_train, X_val, y_val,
              epochs: int, batch_size: int = 32, verbose: bool = True):
        self.loss_history    = []
        self.val_mse_history = []
        n = X_train.shape[0]

        for epoch in range(1, epochs + 1):
            idx       = np.random.permutation(n)
            X_shuf    = X_train[idx]
            y_shuf    = y_train[idx]
            bs        = batch_size if batch_size is not None else n

            epoch_loss = []
            for start in range(0, n, bs):
                Xb = X_shuf[start:start + bs]
                yb = y_shuf[start:start + bs]
                epoch_loss.append(self.gradient_descent(Xb, yb))

            if epoch % 10 == 0 or epoch == 1:
                train_mse = float(np.mean(epoch_loss))
                val_mse   = self.model.mse(X_val, y_val)
                self.loss_history.append(round(train_mse, 6))
                self.val_mse_history.append(round(val_mse, 6))
                if verbose:
                    print(f"Epoka {epoch:>4}/{epochs}  train MSE={train_mse:.5f}  val MSE={val_mse:.5f}")
