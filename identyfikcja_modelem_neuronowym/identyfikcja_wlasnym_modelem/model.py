"""
Prosty model MLP (Multilayer Perceptron) do regresji – implementacja na czystym NumPy.
Skopiowany i zaadaptowany z LAB_5_WSI (pierwotnie klasyfikacja); zmiana: wyjście liniowe + MSE.
"""

import numpy as np


class Node:
    """Jeden neuron: wagi + bias."""
    def __init__(self, num_inputs, init_scale=None):
        self.num_inputs = num_inputs
        if init_scale is None:
            init_scale = np.sqrt(2.0 / num_inputs)  # He (domyślnie dla relu)
        self.weights = np.random.randn(num_inputs) * init_scale
        self.bias = 0.0


class Layer:
    """Warstwa neuronów."""
    def __init__(self, num_nodes, num_inputs, activation_name):
        self.num_nodes = num_nodes
        self.num_inputs = num_inputs
        self.activation_name = activation_name
        # Xavier dla sigmoid, He dla relu/linear
        if activation_name == 'sigmoid':
            init_scale = np.sqrt(1.0 / num_inputs)
        else:
            init_scale = np.sqrt(2.0 / num_inputs)
        self.nodes = [Node(num_inputs, init_scale=init_scale) for _ in range(num_nodes)]

    def _get_weights(self):
        W = np.array([node.weights for node in self.nodes])
        b = np.array([node.bias    for node in self.nodes])
        return W, b

    def _set_weights(self, W, b):
        for i, node in enumerate(self.nodes):
            node.weights = W[i]
            node.bias    = b[i]


class Model:
    """Sieć MLP do regresji z konfigurowalnymi warstwami i funkcjami aktywacji."""

    def __init__(self, layer_sizes: list, activations: list):
        self.num_layers = len(layer_sizes) - 1
        self.layers = []
        self.cache  = None

        for i in range(self.num_layers):
            layer = Layer(layer_sizes[i + 1], layer_sizes[i], activations[i])
            self.layers.append(layer)

    # ── Activations ────────────────────────────────────────────────────

    def _relu(self, z):
        return np.maximum(0, z)

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _linear(self, z):
        return z

    # ── Forward pass ───────────────────────────────────────────────────

    def _forward(self, X):
        A = X
        self.cache = []
        for layer in self.layers:
            W, b   = layer._get_weights()
            A_prev = A
            Z      = A_prev @ W.T + b
            act    = layer.activation_name
            if act == 'sigmoid':
                A = self._sigmoid(Z)
            elif act == 'linear':
                A = self._linear(Z)
            else:  # relu (default)
                A = self._relu(Z)
            self.cache.append((A_prev, Z, A))
        return A

    # ── Metrics ────────────────────────────────────────────────────────

    def mse(self, X, y):
        """Mean squared error na zbiorze (X, y). y może mieć kształt (N,) lub (N,1)."""
        y_pred = self._forward(X).flatten()
        y_true = np.asarray(y).flatten()
        return float(np.mean((y_pred - y_true) ** 2))

    # ── Serialisation ──────────────────────────────────────────────────

    def save(self, path: str):
        arrays = {}
        layer_sizes = [self.layers[0].num_inputs]
        activations = []
        for i, layer in enumerate(self.layers):
            W, b = layer._get_weights()
            arrays[f'W_{i}'] = W
            arrays[f'b_{i}'] = b
            layer_sizes.append(layer.num_nodes)
            activations.append(layer.activation_name)
        arrays['layer_sizes'] = np.array(layer_sizes, dtype=np.int32)
        arrays['activations'] = np.array(activations, dtype=object)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> 'Model':
        if not path.endswith('.npz'):
            path = path + '.npz'
        data        = np.load(path, allow_pickle=True)
        layer_sizes = data['layer_sizes'].tolist()
        activations = data['activations'].tolist()
        model = cls(layer_sizes, activations)
        for i, layer in enumerate(model.layers):
            layer._set_weights(data[f'W_{i}'], data[f'b_{i}'])
        return model
