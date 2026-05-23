import os
import numpy as np
import matplotlib.pyplot as plt


class PlotData:
    """Rysuje i zapisuje wykresy danych statycznych i wyników modeli."""


    def __init__(self, out_dir: str):
        self._out_dir          = out_dir
        self._dir_surowe       = os.path.join(out_dir, 'dane_surowe')
        self._dir_liniowy      = os.path.join(out_dir, 'model_liniowy')
        self._dir_wielomianowy = os.path.join(out_dir, 'model_wielomianowy')
        for d in [self._dir_surowe, self._dir_liniowy, self._dir_wielomianowy]:
            os.makedirs(d, exist_ok=True)

    def _poly_dir(self, degree: int) -> str:
        """Tworzy (jeśli brak) i zwraca ścieżkę podfolderu poly{N}."""
        path = os.path.join(self._dir_wielomianowy, f'poly{degree}')
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _save(fig: plt.Figure, save_dir: str, fname: str) -> plt.Figure:
        """Dopasowuje układ i zapisuje rysunek do pliku PNG."""
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, fname), dpi=150)
        plt.close(fig)
        return fig

    @staticmethod
    def _plot_signals(u: np.ndarray, y: np.ndarray,
                      suptitle: str, u_title: str, y_title: str,
                      idx_start: int = 1) -> plt.Figure:
        """Dwa przebiegi czasowe u(k) i y(k) na osobnych subplotach."""
        idx = range(idx_start, idx_start + len(u))
        fig, (ax_u, ax_y) = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle(suptitle)
        ax_u.plot(idx, u, 'b.-', markersize=4)
        ax_u.set(xlabel='Numer próbki', ylabel='u', title=u_title)
        ax_u.grid(True)
        ax_y.plot(idx, y, 'r.-', markersize=4)
        ax_y.set(xlabel='Numer próbki', ylabel='y', title=y_title)
        ax_y.grid(True)
        return fig

    @staticmethod
    def _plot_characteristic(u: np.ndarray, y: np.ndarray,
                             title: str, fmt: str = 'ko',
                             label: str | None = None) -> plt.Figure:
        """Charakterystyka statyczna y = f(u) jako wykres punktowy."""
        fig, ax = plt.subplots(figsize=(7, 5))
        kw = dict(markersize=4) if label is None else dict(markersize=4, label=label)
        ax.plot(u, y, fmt, **kw)
        ax.set(xlabel='u', ylabel='y', title=title)
        ax.grid(True)
        if label:
            ax.legend()
        return fig

    @staticmethod
    def _plot_characteristic_with_model(u: np.ndarray, y: np.ndarray,
                                        u_curve: np.ndarray, y_curve: np.ndarray,
                                        curve_label: str, title: str) -> plt.Figure:
        """Dane punktowe z nałożoną krzywą modelu."""
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(u, y, 'k.', markersize=5, alpha=0.45, label='Dane')
        ax.plot(u_curve, y_curve, 'r-', linewidth=2, label=curve_label)
        ax.set(xlabel='u', ylabel='y', title=title)
        ax.legend()
        ax.grid(True)
        return fig

    @staticmethod
    def _plot_model_vs_verification(y_wer: np.ndarray, y_hat: np.ndarray,
                                    idx: np.ndarray, suptitle: str,
                                    mse: float, rmse: float) -> plt.Figure:
        """Dwa subploty: wyjście modelu vs dane + błąd predykcji."""
        fig, (ax_out, ax_err) = plt.subplots(2, 1, figsize=(10, 6))
        fig.suptitle(suptitle)
        ax_out.plot(idx, y_wer, 'b.-', markersize=4, label='Dane y')
        ax_out.plot(idx, y_hat, 'r-',  linewidth=1.5, label='Model $\\hat{y}$')
        ax_out.set(xlabel='Numer próbki', ylabel='y',
                   title=f'Wyjście modelu na tle danych weryfikujących  '
                         f'(MSE={mse:.4f}, RMSE={rmse:.4f})')
        ax_out.legend()
        ax_out.grid(True)
        ax_err.stem(idx, y_wer - y_hat, linefmt='grey', markerfmt='ko', basefmt='k-')
        ax_err.axhline(0, color='k', linewidth=0.8)
        ax_err.set(xlabel='Numer próbki', ylabel='e = y - ŷ', title='Błąd predykcji')
        ax_err.grid(True)
        return fig

    @staticmethod
    def _plot_scatter_comparison(y_true: np.ndarray, y_pred: np.ndarray,
                                 title: str) -> plt.Figure:
        """Wykres rozrzutu: y rzeczywiste vs predykcja modelu."""
        lims = [min(y_true.min(), y_pred.min()) - 0.1,
                max(y_true.max(), y_pred.max()) + 0.1]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(lims, lims, 'r--', linewidth=1.2, label='Idealny model (y = ŷ)')
        ax.scatter(y_true, y_pred, s=20, color='steelblue', alpha=0.7, label='Dane weryf.')
        ax.set(xlabel='y  (dane rzeczywiste)', ylabel='ŷ  (wyjście modelu)', title=title)
        ax.set_aspect('equal', 'box')
        ax.legend()
        ax.grid(True)
        return fig


    def plot_all_data(self, u: np.ndarray, y: np.ndarray) -> plt.Figure:
        fig = self._plot_signals(u, y,
            suptitle=f'Dane statyczne – wszystkie próbki (N = {len(u)})',
            u_title='Sygnał wejściowy u', y_title='Sygnał wyjściowy y')
        return self._save(fig, self._dir_surowe, 'dane_statyczne_wszystkie.png')

    def plot_static_characteristic(self, u: np.ndarray, y: np.ndarray) -> plt.Figure:
        fig = self._plot_characteristic(u, y,
            'Charakterystyka statyczna y = f(u) – wszystkie dane')
        return self._save(fig, self._dir_surowe, 'charakterystyka_statyczna_wszystkie.png')

    def plot_training_data(self, u_ucz: np.ndarray, y_ucz: np.ndarray) -> plt.Figure:
        fig = self._plot_signals(u_ucz, y_ucz,
            suptitle=f'Zbiór uczący – 60 % próbek (N_ucz = {len(u_ucz)})',
            u_title='Sygnał wejściowy u – zbiór uczący',
            y_title='Sygnał wyjściowy y – zbiór uczący')
        return self._save(fig, self._dir_surowe, 'dane_statyczne_uczace.png')

    def plot_training_characteristic(self, u_ucz: np.ndarray, y_ucz: np.ndarray) -> plt.Figure:
        fig = self._plot_characteristic(u_ucz, y_ucz,
            'Charakterystyka statyczna – zbiór uczący (60 %)', fmt='bo', label='zbiór uczący')
        return self._save(fig, self._dir_surowe, 'charakterystyka_statyczna_uczace.png')

    def plot_verification_data(self, u_wer: np.ndarray, y_wer: np.ndarray,
                               N_uczacy: int) -> plt.Figure:
        fig = self._plot_signals(u_wer, y_wer,
            suptitle=f'Zbiór weryfikujący – 40 % próbek (N_wer = {len(u_wer)})',
            u_title='Sygnał wejściowy u – zbiór weryfikujący',
            y_title='Sygnał wyjściowy y – zbiór weryfikujący',
            idx_start=N_uczacy + 1)
        return self._save(fig, self._dir_surowe, 'dane_statyczne_weryfikujace.png')

    def plot_verification_characteristic(self, u_wer: np.ndarray, y_wer: np.ndarray) -> plt.Figure:
        fig = self._plot_characteristic(u_wer, y_wer,
            'Charakterystyka statyczna – zbiór weryfikujący (40 %)',
            fmt='rs', label='zbiór weryfikujący')
        return self._save(fig, self._dir_surowe, 'charakterystyka_statyczna_weryfikujace.png')


    def plot_model_characteristic(self, u: np.ndarray, y: np.ndarray,
                                  a0: float, a1: float) -> plt.Figure:
        u_lin = np.linspace(u.min(), u.max(), 300)
        fig = self._plot_characteristic_with_model(u, y, u_lin, a0 + a1 * u_lin,
            curve_label=f'Model: $y = {a0:.3f} + {a1:.3f}\\,u$',
            title='Charakterystyka statyczna – model liniowy')
        return self._save(fig, self._dir_liniowy, 'model_liniowy_charakterystyka.png')

    def plot_verification_results(self, y_wer: np.ndarray, y_hat_wer: np.ndarray,
                                  N_uczacy: int, mse: float, rmse: float) -> plt.Figure:
        idx = np.arange(N_uczacy + 1, N_uczacy + len(y_wer) + 1)
        fig = self._plot_model_vs_verification(y_wer, y_hat_wer, idx,
            suptitle='Model liniowy – zbiór weryfikujący', mse=mse, rmse=rmse)
        return self._save(fig, self._dir_liniowy, 'model_liniowy_weryfikacja_przebieg.png')

    def plot_scatter(self, y_wer: np.ndarray, y_hat_wer: np.ndarray) -> plt.Figure:
        fig = self._plot_scatter_comparison(y_wer, y_hat_wer,
            'Relacja danych weryfikujących i wyjścia modelu')
        return self._save(fig, self._dir_liniowy, 'model_liniowy_weryfikacja_scatter.png')

    def plot_poly_characteristic(self, u: np.ndarray, y: np.ndarray,
                                 model, degree: int) -> plt.Figure:
        u_lin = np.linspace(u.min(), u.max(), 400)
        fig = self._plot_characteristic_with_model(u, y, u_lin, model.predict(u_lin),
            curve_label=f'Model N={degree}',
            title=f'Charakterystyka statyczna – model wielomianowy N={degree}')
        return self._save(fig, self._poly_dir(degree), f'poly{degree}_charakterystyka.png')

    def plot_poly_verification_results(self, y_wer: np.ndarray, y_hat_wer: np.ndarray,
                                       N_uczacy: int, mse: float, rmse: float,
                                       degree: int) -> plt.Figure:
        idx = np.arange(N_uczacy + 1, N_uczacy + len(y_wer) + 1)
        fig = self._plot_model_vs_verification(y_wer, y_hat_wer, idx,
            suptitle=f'Model wielomianowy N={degree} – zbiór weryfikujący',
            mse=mse, rmse=rmse)
        return self._save(fig, self._poly_dir(degree), f'poly{degree}_weryfikacja_przebieg.png')

    def plot_poly_scatter(self, y_wer: np.ndarray, y_hat_wer: np.ndarray,
                          degree: int) -> plt.Figure:
        fig = self._plot_scatter_comparison(y_wer, y_hat_wer,
            f'Relacja danych weryfikujących i wyjścia modelu (N={degree})')
        return self._save(fig, self._poly_dir(degree), f'poly{degree}_weryfikacja_scatter.png')

    # Podsumowanie

    def plot_metrics_table(self, results: list) -> plt.Figure:
        """Tabela MSE/RMSE dla wszystkich testowanych stopni wielomianu."""
        col_labels = ['Stopień N', 'MSE uczący', 'RMSE uczący', 'MSE weryf.', 'RMSE weryf.']
        table_data = [
            [f'N={r["degree"]}',
             f'{r["mse_ucz"]:.6f}', f'{r["rmse_ucz"]:.6f}',
             f'{r["mse_wer"]:.6f}', f'{r["rmse_wer"]:.6f}']
            for r in results
        ]
        n_rows = len(results)
        fig, ax = plt.subplots(figsize=(11, n_rows * 0.55 + 1.5))
        ax.axis('off')
        tbl = ax.table(cellText=table_data, colLabels=col_labels,
                       loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.6)
        for j in range(len(col_labels)):
            tbl[(0, j)].set_facecolor('#4472C4')
            tbl[(0, j)].set_text_props(color='white', fontweight='bold')
        best = min(range(n_rows), key=lambda i: results[i]['mse_wer'])
        for j in range(len(col_labels)):
            tbl[(best + 1, j)].set_facecolor('#E2EFDA')
        ax.set_title('Metryki błędów – modele wielomianowe\n(zielony = najniższy MSE weryf.)',
                     pad=14, fontsize=11, fontweight='bold')
        return self._save(fig, self._dir_wielomianowy, 'tabela_metryk_wielomianowe.png')

    def plot_all_poly_characteristics(self, u: np.ndarray, y: np.ndarray,
                                      models: list, degrees: list) -> plt.Figure:
        """Wszystkie krzywe wielomianowe nałożone na jeden wykres."""
        u_lin  = np.linspace(u.min(), u.max(), 400)
        colors = plt.cm.tab10(np.linspace(0, 1, len(degrees)))
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(u, y, 'k.', markersize=4, alpha=0.35, label='Dane')
        for model, deg, col in zip(models, degrees, colors):
            ax.plot(u_lin, model.predict(u_lin), '-', linewidth=1.8,
                    color=col, label=f'N={deg}')
        ax.set(xlabel='u', ylabel='y',
               title='Charakterystyki statyczne – wszystkie modele wielomianowe')
        ax.legend(fontsize=8)
        ax.grid(True)
        return self._save(fig, self._dir_wielomianowy, 'poly_wszystkie_charakterystyki.png')

    @staticmethod
    def show() -> None:
        plt.show()

