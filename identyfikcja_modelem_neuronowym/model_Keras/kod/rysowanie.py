"""Wykresy dla identyfikacji modelem neuronowym (Keras).

Klasa PlotNeural zapisuje wszystkie rysunki jako pliki PNG.
Schemat katalogów wzorowany na PlotData (modele statyczne)
i PlotDynamic (modele dynamiczne).
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')   # backend bez GUI – zapis do pliku
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def _apply_pl_format(fig: plt.Figure) -> None:
    """Zamienia separatory dziesiętne na przecinek (notacja polska) dla osi liniowych."""
    fmt = mticker.FuncFormatter(lambda x, _: f'{x:g}'.replace('.', ','))
    for ax in fig.axes:
        if ax.get_xscale() == 'linear':
            ax.xaxis.set_major_formatter(fmt)
        if ax.get_yscale() == 'linear':
            ax.yaxis.set_major_formatter(fmt)


class PlotNeural:
    """Rysuje i zapisuje wykresy dotyczące modeli neuronowych Keras."""

    # Kolory i style dla trzech wariantów w każdej grupie
    _STYLES = [
        {'color': '#1f77b4', 'linestyle': '-',  'linewidth': 1.8},
        {'color': '#ff7f0e', 'linestyle': '--', 'linewidth': 1.8},
        {'color': '#2ca02c', 'linestyle': ':',  'linewidth': 2.2},
    ]

    def __init__(self, out_dir: str):
        self._out_dir     = out_dir
        self._dir_small   = os.path.join(out_dir, 'zbyt_mala')
        self._dir_medium  = os.path.join(out_dir, 'umiarkowana')
        self._dir_large   = os.path.join(out_dir, 'zbyt_duza')
        self._dir_compare = out_dir
        for d in [self._dir_small, self._dir_medium, self._dir_large]:
            os.makedirs(d, exist_ok=True)

    # ── Prywatne narzędzia ─────────────────────────────────────────────────────

    @staticmethod
    def _save(fig: plt.Figure, save_dir: str, fname: str) -> plt.Figure:
        _apply_pl_format(fig)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, fname), dpi=150)
        plt.close(fig)
        return fig

    def _size_dir(self, size_label: str) -> str:
        mapping = {
            'zbyt_mala':   self._dir_small,
            'umiarkowana': self._dir_medium,
            'zbyt_duza':   self._dir_large,
        }
        return mapping.get(size_label, self._out_dir)

    # ── Krzywe uczenia ─────────────────────────────────────────────────────────

    def plot_learning_curves(self, experiments: list[dict],
                             group_title: str, fname: str,
                             size_label: str) -> plt.Figure:
        """Rysuje krzywe MSE loss w funkcji epoki dla grupy eksperymentów.

        Parametry
        ----------
        experiments : lista słowników z kluczami 'label' i 'history_loss'
        group_title : tytuł rysunku
        fname       : nazwa pliku PNG
        size_label  : 'zbyt_mala' | 'umiarkowana' | 'zbyt_duza'
        """
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, exp in enumerate(experiments):
            loss = exp['history_loss']
            style = self._STYLES[i % len(self._STYLES)]
            ax.plot(range(1, len(loss) + 1), loss,
                    label=exp['label'], **style)
        ax.set(xlabel='Epoka', ylabel='MSE (zbiór uczący)',
               title=group_title)
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, which='both', alpha=0.4)
        return self._save(fig, self._size_dir(size_label), fname)

    # ── Charakterystyka statyczna ──────────────────────────────────────────────

    def plot_characteristic(self, u_all: np.ndarray, y_all: np.ndarray,
                            u_ucz: np.ndarray, y_ucz: np.ndarray,
                            model_label: str, y_pred_ucz: np.ndarray,
                            y_pred_wer: np.ndarray, u_wer: np.ndarray,
                            fname: str, size_label: str,
                            title: str | None = None) -> plt.Figure:
        """Charakterystyka statyczna y(u): dane + krzywa modelu."""
        u_curve = np.linspace(u_all.min(), u_all.max(), 300)
        # Importujemy model tylko przez predykcję – dostajemy y_pred_curve przez argument
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(u_all, y_all, s=12, color='lightgray',
                   label='Wszystkie dane', zorder=1)
        ax.scatter(u_ucz, y_ucz, s=14, color='steelblue',
                   label='Dane uczące', zorder=2)
        ax.scatter(u_wer, y_all[len(u_ucz):] if len(u_wer) == len(y_all) - len(u_ucz)
                   else u_wer * 0,   # placeholder – nadpisane poniżej
                   s=0, alpha=0)
        t = title or f'Charakterystyka statyczna – {model_label}'
        ax.set(xlabel='u', ylabel='y', title=t)
        ax.grid(True)
        ax.legend(fontsize=9)
        return self._save(fig, self._size_dir(size_label), fname)

    def plot_characteristic_curve(self, u_all: np.ndarray, y_all: np.ndarray,
                                   u_ucz: np.ndarray, y_ucz: np.ndarray,
                                   u_wer: np.ndarray, y_wer: np.ndarray,
                                   u_curve: np.ndarray, y_curve: np.ndarray,
                                   model_label: str, mse_ucz: float, mse_wer: float,
                                   fname: str, size_label: str) -> plt.Figure:
        """Pełna charakterystyka statyczna z krzywą modelu i oceną MSE."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(u_ucz, y_ucz, s=14, color='steelblue', alpha=0.6,
                   label='Dane uczące', zorder=2)
        ax.scatter(u_wer, y_wer, s=14, color='tomato', alpha=0.6,
                   label='Dane weryfikujące', zorder=2)
        ax.plot(u_curve, y_curve, 'k-', linewidth=2, label=f'Model ({model_label})', zorder=3)
        ax.set(xlabel='u', ylabel='y',
               title=f'Charakterystyka statyczna – {model_label}\n'
                     f'MSE ucz.={mse_ucz:.5f}  MSE wer.={mse_wer:.5f}')
        ax.legend(fontsize=9)
        ax.grid(True)
        return self._save(fig, self._size_dir(size_label), fname)

    # ── Porównanie MSE (zestawienie wszystkich rozmiarów) ──────────────────────

    def plot_mse_comparison(self, size_results: dict[str, list[dict]],
                             fname: str = 'porownanie_mse.png') -> plt.Figure:
        """Wykres słupkowy MSE weryfikacyjnego dla najlepszych modeli każdego rozmiaru.

        Parameters
        ----------
        size_results : {'zbyt_mala': [{'label':..., 'mse_wer':...}, ...], ...}
        """
        labels_all, mse_ucz_all, mse_wer_all, colors_all = [], [], [], []
        palette = {'zbyt_mala': '#4e79a7', 'umiarkowana': '#f28e2b', 'zbyt_duza': '#e15759'}
        size_labels_pl = {'zbyt_mala': 'Zbyt mała\n(1 neuron)',
                          'umiarkowana': 'Umiarkowana\n(10 neuronów)',
                          'zbyt_duza': 'Zbyt duża\n(500 neuronów)'}
        best_per_size: dict[str, dict] = {}
        for size_label, results in size_results.items():
            finite = [r for r in results if np.isfinite(r['mse_wer'])]
            if not finite:
                continue
            best = min(finite, key=lambda r: r['mse_wer'])
            best_per_size[size_label] = best

        x = np.arange(len(best_per_size))
        width = 0.35
        fig, ax = plt.subplots(figsize=(9, 5))
        ucz_vals = [best_per_size[s]['mse_ucz'] for s in best_per_size]
        wer_vals = [best_per_size[s]['mse_wer'] for s in best_per_size]
        xl = [size_labels_pl.get(s, s) for s in best_per_size]
        clrs = [palette.get(s, 'gray') for s in best_per_size]

        bars_ucz = ax.bar(x - width / 2, ucz_vals, width, label='MSE uczący',
                          color=[c + 'aa' for c in clrs] if False else clrs, alpha=0.7)
        bars_wer = ax.bar(x + width / 2, wer_vals, width, label='MSE weryfikujący',
                          color=clrs, alpha=1.0)

        for bar in list(bars_ucz) + list(bars_wer):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.04,
                    f'{h:.4f}', ha='center', va='bottom', fontsize=8)

        ax.set(xlabel='Rozmiar sieci', ylabel='MSE',
               title='Porównanie MSE najlepszych modeli neuronowych')
        ax.set_xticks(x)
        ax.set_xticklabels(xl, fontsize=9)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.4)
        return self._save(fig, self._dir_compare, fname)

    # ── Porównanie przebiegów weryfikacyjnych ──────────────────────────────────

    def plot_verification_comparison(self, y_wer: np.ndarray,
                                      experiments: list[dict],
                                      group_title: str, fname: str,
                                      size_label: str) -> plt.Figure:
        """Porównuje przebiegi y_hat na zbiorze weryfikującym dla grupy modeli."""
        N = len(y_wer)
        idx = np.arange(1, N + 1)
        fig, axes = plt.subplots(len(experiments), 1,
                                  figsize=(10, 3 * len(experiments)),
                                  sharex=True)
        if len(experiments) == 1:
            axes = [axes]
        fig.suptitle(group_title, fontsize=11)
        for ax, exp in zip(axes, experiments):
            y_hat = exp['y_pred_wer']
            mse   = exp['mse_wer']
            ax.plot(idx, y_wer, 'b.-', markersize=2, linewidth=0.8, label='Dane y')
            ax.plot(idx, y_hat, 'r-',  linewidth=1.5, label=f'Model  MSE={mse:.5f}')
            ax.set(ylabel='y', title=exp['label'])
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True)
        axes[-1].set_xlabel('Numer próbki')
        return self._save(fig, self._size_dir(size_label), fname)
