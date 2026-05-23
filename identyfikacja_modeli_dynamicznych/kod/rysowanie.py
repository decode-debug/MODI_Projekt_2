import os
import numpy as np
import matplotlib.pyplot as plt


class PlotDynamic:
    """Rysuje i zapisuje wykresy danych dynamicznych i wyników modeli ARX."""

    # ══ Inicjalizacja ══════════════════════════════════════════════════════════

    def __init__(self, out_dir: str):
        self._dir_surowe = os.path.join(str(out_dir), 'dane_surowe')
        self._dir_arx    = os.path.join(str(out_dir), 'modele_arx')
        for d in [self._dir_surowe, self._dir_arx]:
            os.makedirs(d, exist_ok=True)

    def _order_dir(self, order: int) -> str:
        path = os.path.join(self._dir_arx, f'order{order}')
        os.makedirs(path, exist_ok=True)
        return path

    # ══ Prywatne metody generyczne ════════════════════════════════════════════

    @staticmethod
    def _save(fig: plt.Figure, save_dir: str, fname: str) -> plt.Figure:
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, fname), dpi=150)
        return fig

    @staticmethod
    def _plot_signals(u: np.ndarray, y: np.ndarray,
                      suptitle: str, u_title: str, y_title: str) -> plt.Figure:
        N = len(u)
        fig, (ax_u, ax_y) = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle(suptitle)
        ax_u.plot(range(1, N + 1), u, 'b.-', markersize=3)
        ax_u.set(xlabel='Numer próbki', ylabel='u', title=u_title)
        ax_u.grid(True)
        ax_y.plot(range(1, N + 1), y, 'r.-', markersize=3)
        ax_y.set(xlabel='Numer próbki', ylabel='y', title=y_title)
        ax_y.grid(True)
        return fig

    @staticmethod
    def _plot_model_vs_verification(y_wer: np.ndarray, y_hat: np.ndarray,
                                    idx: np.ndarray, suptitle: str,
                                    mse: float, rmse: float) -> plt.Figure:
        fig, (ax_out, ax_err) = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle(suptitle)
        ax_out.plot(idx, y_wer, 'b.-', markersize=2, label='Dane y')
        ax_out.plot(idx, y_hat, 'r-',  linewidth=1.5, label='Model $\\hat{y}$')
        ax_out.set(xlabel='Numer próbki', ylabel='y',
                   title=f'Wyjście modelu vs dane weryfikujące  '
                         f'(MSE={mse:.4f}, RMSE={rmse:.4f})')
        ax_out.legend()
        ax_out.grid(True)
        ax_err.stem(idx, y_wer - y_hat, linefmt='grey', markerfmt='ko', basefmt='k-')
        ax_err.axhline(0, color='k', linewidth=0.8)
        ax_err.set(xlabel='Numer próbki', ylabel='e = y − ŷ', title='Błąd predykcji')
        ax_err.grid(True)
        return fig

    @staticmethod
    def _plot_scatter_comparison(y_true: np.ndarray, y_pred: np.ndarray,
                                 title: str) -> plt.Figure:
        lims = [min(y_true.min(), y_pred.min()) - 0.1,
                max(y_true.max(), y_pred.max()) + 0.1]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(lims, lims, 'r--', linewidth=1.2, label='Idealny model (y = ŷ)')
        ax.scatter(y_true, y_pred, s=10, color='steelblue', alpha=0.5, label='Dane weryf.')
        ax.set(xlabel='y  (dane rzeczywiste)', ylabel='ŷ  (wyjście modelu)', title=title)
        ax.set_aspect('equal', 'box')
        ax.legend()
        ax.grid(True)
        return fig

    # ══ Dane surowe ════════════════════════════════════════════════════════════

    def plot_training_data(self, u_ucz: np.ndarray, y_ucz: np.ndarray) -> plt.Figure:
        fig = self._plot_signals(u_ucz, y_ucz,
            f'Dane dynamiczne – zbiór uczący (N = {len(u_ucz)})',
            'Sygnał wejściowy u – zbiór uczący',
            'Sygnał wyjściowy y – zbiór uczący')
        return self._save(fig, self._dir_surowe, 'dane_dynamiczne_uczace.png')

    def plot_verification_data(self, u_wer: np.ndarray, y_wer: np.ndarray) -> plt.Figure:
        fig = self._plot_signals(u_wer, y_wer,
            f'Dane dynamiczne – zbiór weryfikujący (N = {len(u_wer)})',
            'Sygnał wejściowy u – zbiór weryfikujący',
            'Sygnał wyjściowy y – zbiór weryfikujący')
        return self._save(fig, self._dir_surowe, 'dane_dynamiczne_weryfikujace.png')

    # ══ Modele ARX ═════════════════════════════════════════════════════════════

    def plot_verification_results(self, y_wer: np.ndarray, y_hat: np.ndarray,
                                  n: int, mse: float, rmse: float,
                                  order: int, recursive: bool) -> plt.Figure:
        mode   = 'z rekurencją' if recursive else 'bez rekurencji'
        suffix = 'r' if recursive else 'nr'
        idx = np.arange(n + 1, n + 1 + len(y_wer))
        fig = self._plot_model_vs_verification(y_wer, y_hat, idx,
            f'ARX rzędu {order} – zbiór weryfikujący ({mode})', mse, rmse)
        return self._save(fig, self._order_dir(order),
                          f'order{order}_{suffix}_weryfikacja.png')

    def plot_scatter(self, y_wer: np.ndarray, y_hat: np.ndarray,
                     order: int, recursive: bool) -> plt.Figure:
        mode   = 'z rekurencją' if recursive else 'bez rekurencji'
        suffix = 'r' if recursive else 'nr'
        fig = self._plot_scatter_comparison(y_wer, y_hat,
            f'Relacja danych weryf. vs model ARX({order}) – {mode}')
        return self._save(fig, self._order_dir(order),
                          f'order{order}_{suffix}_scatter.png')

    # ══ Podsumowanie ═══════════════════════════════════════════════════════════

    def plot_metrics_table(self, results: list) -> plt.Figure:
        """Tabela MSE/RMSE dla wszystkich rzędów i trybów predykcji."""
        col_labels = ['Rząd',
                      'MSE ucz NR', 'RMSE ucz NR', 'MSE wer NR', 'RMSE wer NR',
                      'MSE ucz R',  'RMSE ucz R',  'MSE wer R',  'RMSE wer R']
        table_data = [
            [f'n={r["order"]}',
             f'{r["mse_ucz_nr"]:.4f}', f'{r["rmse_ucz_nr"]:.4f}',
             f'{r["mse_wer_nr"]:.4f}', f'{r["rmse_wer_nr"]:.4f}',
             f'{r["mse_ucz_r"]:.4f}',  f'{r["rmse_ucz_r"]:.4f}',
             f'{r["mse_wer_r"]:.4f}',  f'{r["rmse_wer_r"]:.4f}']
            for r in results
        ]
        n_rows = len(results)
        fig, ax = plt.subplots(figsize=(16, n_rows * 0.55 + 1.8))
        ax.axis('off')
        tbl = ax.table(cellText=table_data, colLabels=col_labels,
                       loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.2, 1.6)
        for j in range(len(col_labels)):
            tbl[(0, j)].set_facecolor('#4472C4')
            tbl[(0, j)].set_text_props(color='white', fontweight='bold')
        best = min(range(n_rows), key=lambda i: results[i]['mse_wer_r'])
        for j in range(len(col_labels)):
            tbl[(best + 1, j)].set_facecolor('#E2EFDA')
        ax.set_title('Metryki błędów – modele ARX\n'
                     '(NR = bez rekurencji, R = z rekurencją, zielony = najniższy MSE wer R)',
                     pad=14, fontsize=10, fontweight='bold')
        return self._save(fig, self._dir_arx, 'tabela_metryk_arx.png')

    @staticmethod
    def show() -> None:
        plt.show()
