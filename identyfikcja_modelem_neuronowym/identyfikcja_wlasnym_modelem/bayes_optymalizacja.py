"""
Optymalizacja Bayesowska hiperparametrów sieci do regresji statycznej.
Skopiowana i zaadaptowana z LAB_5_WSI (zmiana: MSE zamiast accuracy, wyjście liniowe).
"""

import numpy as np
from .model import Model
from .trening import TrenujModel


# ── Gaussian Process (prosty model zastępczy) ──────────────────────────────────

class GaussianProcess:
    def __init__(self, noise: float = 1e-3):
        self.noise   = noise
        self.X_seen  = []
        self.y_seen  = []

    def _rbf_kernel(self, a, b, length_scale: float = 1.0):
        diff = a - b
        return np.exp(-0.5 * np.dot(diff, diff) / (length_scale ** 2))

    def _kernel_matrix(self, X1, X2):
        return np.array([[self._rbf_kernel(x1, x2) for x2 in X2] for x1 in X1])

    def fit(self, X, y):
        self.X_seen = np.array(X, dtype=float)
        self.y_seen = np.array(y, dtype=float)

    def predict(self, X_new):
        X_new = np.array(X_new, dtype=float)
        n     = len(self.X_seen)
        K     = self._kernel_matrix(self.X_seen, self.X_seen)
        K_inv = np.linalg.inv(K + self.noise * np.eye(n))
        means, stds = [], []
        for x in X_new:
            k    = np.array([self._rbf_kernel(x, xs) for xs in self.X_seen])
            mean = float(k @ K_inv @ self.y_seen)
            var  = max(0.0, 1.0 - float(k @ K_inv @ k))
            means.append(mean)
            stds.append(np.sqrt(var))
        return np.array(means), np.array(stds)


def acquisition_ucb(means, stds, kappa: float = 2.0):
    """Upper Confidence Bound: wysoka wartość = obiecujący kandydat."""
    return means + kappa * stds


# ── Główna klasa optymalizatora ────────────────────────────────────────────────

class BayesOptymalizator:
    """
    Poszukuje najlepszej architektury i hiperparametrów sieci regresyjnej.

    Metryka: val MSE (minimalizacja → negujemy aby GP maksymalizował score).
    """

    def __init__(
        self,
        X_train, y_train,
        X_val,   y_val,
        layers_range      = (1, 4),
        nodes_range       = (8, 128),
        lr_range          = (1e-3, 0.1),
        epochs_range      = (100, 500),
        batch_size_range  = (16, 128),
        n_random_starts   = 5,
        n_iterations      = 20,
        n_candidates      = 200,
    ):
        self.X_train, self.y_train = X_train, y_train
        self.X_val,   self.y_val   = X_val,   y_val

        self.bounds = {
            'num_layers': layers_range,
            'nodes'     : nodes_range,
            'lr'        : lr_range,
            'epochs'    : epochs_range,
            'batch_size': batch_size_range,
        }
        self.max_layers      = layers_range[1]
        self.n_random_starts = n_random_starts
        self.n_iterations    = n_iterations
        self.n_candidates    = n_candidates
        self.gp      = GaussianProcess()
        self.history = []

    def _encode(self, config):
        lo_l, hi_l  = self.bounds['num_layers']
        lo_lr,hi_lr = self.bounds['lr']
        lo_e, hi_e  = self.bounds['epochs']
        lo_b, hi_b  = self.bounds['batch_size']
        return [
            (config['num_layers'] - lo_l)  / max(hi_l  - lo_l,  1),
            (config['lr']         - lo_lr) / (hi_lr - lo_lr),
            (config['epochs']     - lo_e)  / max(hi_e  - lo_e,  1),
            (config['batch_size'] - lo_b)  / max(hi_b  - lo_b,  1),
        ]

    def _random_config(self):
        cfg = {
            'num_layers': np.random.randint(*self.bounds['num_layers']),
            'lr'        : np.random.uniform(*self.bounds['lr']),
            'epochs'    : np.random.randint(*self.bounds['epochs']),
            'batch_size': np.random.randint(
                self.bounds['batch_size'][0],
                self.bounds['batch_size'][1] + 1,
            ),
        }
        for i in range(1, self.max_layers + 1):
            cfg[f'nodes_{i}'] = np.random.randint(*self.bounds['nodes'])
        return cfg

    def _evaluate(self, config):
        input_size  = self.X_train.shape[1]
        output_size = 1

        hidden_sizes = [config[f'nodes_{i+1}'] for i in range(config['num_layers'])]
        layer_sizes  = [input_size] + hidden_sizes + [output_size]
        activations  = ['sigmoid'] * config['num_layers'] + ['linear']

        model   = Model(layer_sizes, activations)
        trainer = TrenujModel(model, learning_rate=config['lr'])
        trainer.train(
            self.X_train, self.y_train,
            self.X_val,   self.y_val,
            epochs     = config['epochs'],
            batch_size = config['batch_size'],
            verbose    = False,
        )
        val_mse = model.mse(self.X_val, self.y_val)
        # Negujemy MSE – GP maksymalizuje score (wyższy = lepszy)
        score = -val_mse
        return score, trainer

    def optymalizuj(self):
        for iteration in range(1, self.n_iterations + 1):
            if iteration <= self.n_random_starts:
                config = self._random_config()
            else:
                candidates = [self._random_config() for _ in range(self.n_candidates)]
                encoded    = np.array([self._encode(c) for c in candidates])
                X_hist     = [self._encode(h['config']) for h in self.history]
                y_hist     = [h['score']                for h in self.history]
                self.gp.fit(X_hist, y_hist)
                means, stds = self.gp.predict(encoded)
                ucb_scores  = acquisition_ucb(means, stds)
                config      = candidates[int(np.argmax(ucb_scores))]

            score, trainer = self._evaluate(config)
            self.history.append({'config': config, 'score': score, 'trainer': trainer})

        best = max(self.history, key=lambda h: h['score'])
        return best['config'], -best['score'], best['trainer']
