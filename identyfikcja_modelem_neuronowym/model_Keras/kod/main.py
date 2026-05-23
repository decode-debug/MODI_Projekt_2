"""Identyfikacja modelem neuronowym (Keras) – statyczna charakterystyka y(u).

Eksperyment porównuje trzy rozmiary sieci jednowarstwowej:
  - „zbyt mała"   : 1 neuron ukryty
  - „umiarkowana" : 10 neuronów ukrytych
  - „zbyt duża"   : 500 neuronów ukrytych

Dla każdego rozmiaru uczenie powtarzane jest przy zmianach:
  a) algorytmu optymalizacji : Adam, SGD, RMSprop  (lr=FIX_LR, seed=FIX_SEED)
  b) współczynnika uczenia   : 0.001, 0.01, 0.1    (opt=FIX_OPT, seed=FIX_SEED)
  c) punktu startowego wag   : seed=42, 123, 999   (opt=FIX_OPT, lr=FIX_LR)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os

# Ustaw backend Keras PRZED importem modeli
os.environ.setdefault('KERAS_BACKEND', 'torch')

import numpy as np

from import_danych    import ImportStaticData
from modele_neuronowe import KerasNNModel
from rysowanie        import PlotNeural

# ── Konfiguracja eksperymentów ─────────────────────────────────────────────────

HIDDEN_SIZES: dict[str, int] = {
    'zbyt_mala':   1,
    'umiarkowana': 10,
    'zbyt_duza':   500,
}

OPTIMIZERS   = ['adam', 'sgd', 'rmsprop']
LR_LIST      = [0.001, 0.01, 0.1]
SEEDS        = [42, 123, 999]

FIX_OPT  = 'adam'   # ustalony optymalizator przy zmianach lr i seed
FIX_LR   = 0.01     # ustalony lr przy zmianach optymalizatora i seed
FIX_SEED = 42       # ustalony seed przy zmianach opt i lr

# Liczba epok – dobrana tak, żeby sieć zdążyła się schodzić / wyraźnie się przetrenować
EPOCHS: dict[str, int] = {
    'zbyt_mala':   8000,
    'umiarkowana': 4000,
    'zbyt_duza':   1500,
}

BATCH_SIZE = 32


# ══════════════════════════════════════════════════════════════════════════════
# Funkcje pomocnicze
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(dane: ImportStaticData,
                   n_hidden: int, optimizer_name: str,
                   lr: float, seed: int,
                   epochs: int) -> dict:
    """Trenuje jeden model i zwraca pełny słownik wyników."""
    model = KerasNNModel(n_hidden=n_hidden, optimizer_name=optimizer_name,
                         lr=lr, seed=seed)
    model.fit(dane.u_ucz, dane.y_ucz,
              epochs=epochs, batch_size=BATCH_SIZE, verbose=0)

    res_ucz = model.evaluate(dane.u_ucz, dane.y_ucz)
    res_wer = model.evaluate(dane.u_wer, dane.y_wer)

    # Krzywa charakterystyki statycznej
    u_curve = np.linspace(dane.u.min(), dane.u.max(), 300)
    y_curve = model.predict(u_curve)

    return {
        'model':        model,
        'n_hidden':     n_hidden,
        'optimizer':    optimizer_name,
        'lr':           lr,
        'seed':         seed,
        'n_params':     model.n_params,
        'mse_ucz':      res_ucz['mse'],
        'rmse_ucz':     res_ucz['rmse'],
        'mse_wer':      res_wer['mse'],
        'rmse_wer':     res_wer['rmse'],
        'y_pred_ucz':   res_ucz['y_pred'],
        'y_pred_wer':   res_wer['y_pred'],
        'u_curve':      u_curve,
        'y_curve':      y_curve,
        'history_loss': model.history_loss,
        'label':        f"opt={optimizer_name}, lr={lr}, seed={seed}",
    }


def print_group_results(group_name: str, results: list[dict]) -> None:
    hdr = f"\n{'─'*70}\n  {group_name}\n{'─'*70}"
    print(hdr)
    print(f"{'Optymalizator':>14}  {'LR':>7}  {'Seed':>5}  "
          f"{'MSE ucz':>12}  {'RMSE ucz':>12}  "
          f"{'MSE wer':>12}  {'RMSE wer':>12}  {'L.par.':>7}")
    print('─' * 100)
    for r in results:
        print(f"{r['optimizer']:>14}  {r['lr']:>7.4f}  {r['seed']:>5}  "
              f"{r['mse_ucz']:>12.6f}  {r['rmse_ucz']:>12.6f}  "
              f"{r['mse_wer']:>12.6f}  {r['rmse_wer']:>12.6f}  {r['n_params']:>7}")
    best = min(results, key=lambda r: r['mse_wer'])
    print(f"\n  Najlepszy: opt={best['optimizer']}, lr={best['lr']}, "
          f"seed={best['seed']} → MSE wer.={best['mse_wer']:.6f}")


# ══════════════════════════════════════════════════════════════════════════════
# Punkt wejścia
# ══════════════════════════════════════════════════════════════════════════════

def main():
    dane    = ImportStaticData(split=0.6)
    plotter = PlotNeural(dane.out_dir)

    print(f"Dane statyczne: {dane}")
    print(f"Keras backend: {os.environ.get('KERAS_BACKEND', '?')}")

    # Zbieramy wyniki per rozmiar (do porównania końcowego)
    size_results_all: dict[str, list[dict]] = {}

    for size_label, n_hidden in HIDDEN_SIZES.items():
        epochs = EPOCHS[size_label]
        print(f"\n\n{'═'*70}")
        print(f"  Sieć: {size_label.upper()}  (hidden={n_hidden}, epochs={epochs})")
        print(f"{'═'*70}")

        all_for_size: list[dict] = []

        # ── a) Zmiana optymalizatora ───────────────────────────────────────────
        group_opt: list[dict] = []
        for opt_name in OPTIMIZERS:
            print(f"  Uczę: opt={opt_name}, lr={FIX_LR}, seed={FIX_SEED} ...", flush=True)
            r = run_experiment(dane, n_hidden, opt_name, FIX_LR, FIX_SEED, epochs)
            group_opt.append(r)
            all_for_size.append(r)
        print_group_results(f"{size_label} | zmiana optymalizatora (lr={FIX_LR}, seed={FIX_SEED})",
                            group_opt)
        plotter.plot_learning_curves(
            group_opt,
            group_title=f"Krzywe uczenia – {size_label} | zmiana optymalizatora",
            fname='krzywe_opt.png',
            size_label=size_label,
        )
        plotter.plot_verification_comparison(
            dane.y_wer, group_opt,
            group_title=f"Weryfikacja – {size_label} | zmiana optymalizatora",
            fname='weryfikacja_opt.png',
            size_label=size_label,
        )

        # ── b) Zmiana współczynnika uczenia ────────────────────────────────────
        group_lr: list[dict] = []
        for lr in LR_LIST:
            print(f"  Uczę: opt={FIX_OPT}, lr={lr}, seed={FIX_SEED} ...", flush=True)
            r = run_experiment(dane, n_hidden, FIX_OPT, lr, FIX_SEED, epochs)
            group_lr.append(r)
            if not any(x['optimizer'] == FIX_OPT and x['lr'] == lr and x['seed'] == FIX_SEED
                       for x in all_for_size):
                all_for_size.append(r)
        print_group_results(f"{size_label} | zmiana LR (opt={FIX_OPT}, seed={FIX_SEED})",
                            group_lr)
        plotter.plot_learning_curves(
            group_lr,
            group_title=f"Krzywe uczenia – {size_label} | zmiana LR",
            fname='krzywe_lr.png',
            size_label=size_label,
        )
        plotter.plot_verification_comparison(
            dane.y_wer, group_lr,
            group_title=f"Weryfikacja – {size_label} | zmiana LR",
            fname='weryfikacja_lr.png',
            size_label=size_label,
        )

        # ── c) Zmiana punktu startowego (seed) ────────────────────────────────
        group_seed: list[dict] = []
        for seed in SEEDS:
            print(f"  Uczę: opt={FIX_OPT}, lr={FIX_LR}, seed={seed} ...", flush=True)
            r = run_experiment(dane, n_hidden, FIX_OPT, FIX_LR, seed, epochs)
            group_seed.append(r)
            if not any(x['optimizer'] == FIX_OPT and x['lr'] == FIX_LR and x['seed'] == seed
                       for x in all_for_size):
                all_for_size.append(r)
        print_group_results(f"{size_label} | zmiana seed (opt={FIX_OPT}, lr={FIX_LR})",
                            group_seed)
        plotter.plot_learning_curves(
            group_seed,
            group_title=f"Krzywe uczenia – {size_label} | zmiana punktu startowego",
            fname='krzywe_seed.png',
            size_label=size_label,
        )
        plotter.plot_verification_comparison(
            dane.y_wer, group_seed,
            group_title=f"Weryfikacja – {size_label} | zmiana punktu startowego",
            fname='weryfikacja_seed.png',
            size_label=size_label,
        )

        # ── Najlepszy model dla tego rozmiaru – charakterystyka y(u) ──────────
        finite = [r for r in all_for_size if np.isfinite(r['mse_wer'])]
        if finite:
            best = min(finite, key=lambda r: r['mse_wer'])
            plotter.plot_characteristic_curve(
                u_all=dane.u, y_all=dane.y,
                u_ucz=dane.u_ucz, y_ucz=dane.y_ucz,
                u_wer=dane.u_wer, y_wer=dane.y_wer,
                u_curve=best['u_curve'], y_curve=best['y_curve'],
                model_label=str(best['model']),
                mse_ucz=best['mse_ucz'], mse_wer=best['mse_wer'],
                fname='charakterystyka_best.png',
                size_label=size_label,
            )
            print(f"\n  → Charakterystyka statyczna zapisana (best: {best['model']})")

        size_results_all[size_label] = all_for_size

    # ── Porównanie najlepszych modeli wszystkich rozmiarów ─────────────────────
    plotter.plot_mse_comparison(
        {sl: [{'label': r['label'], 'mse_ucz': r['mse_ucz'], 'mse_wer': r['mse_wer']}
              for r in rs]
         for sl, rs in size_results_all.items()},
        fname='porownanie_mse.png',
    )
    print("\n\nPorównanie MSE zapisane → porownanie_mse.png")

    # ── Zbiorcze podsumowanie w konsoli ───────────────────────────────────────
    print(f"\n\n{'═'*70}")
    print("  PODSUMOWANIE – najlepsze modele dla każdego rozmiaru")
    print(f"{'═'*70}")
    print(f"{'Rozmiar':>14}  {'hidden':>6}  {'Optym.':>8}  {'LR':>7}  {'Seed':>5}  "
          f"{'MSE ucz':>12}  {'MSE wer':>12}")
    print('─' * 80)
    for size_label, rs in size_results_all.items():
        finite = [r for r in rs if np.isfinite(r['mse_wer'])]
        if finite:
            b = min(finite, key=lambda r: r['mse_wer'])
            print(f"{size_label:>14}  {b['n_hidden']:>6}  {b['optimizer']:>8}  "
                  f"{b['lr']:>7.4f}  {b['seed']:>5}  "
                  f"{b['mse_ucz']:>12.6f}  {b['mse_wer']:>12.6f}")


if __name__ == '__main__':
    main()
