import sys
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).resolve().parent))

from import_danych      import ImportDynamicData
from modele_dynamiczne  import ARXModel, NARXModel
from rysowanie          import PlotDynamic

pretrained = True  # ustaw na True po pierwszym szerokim przeszukiwaniu, aby zawęzić zakres do najlepszych modeli

ORDERS       = [1, 2, 3]          # rzędy modeli ARX
if pretrained == False:  # szerokie przeszukiwanie
    NARX_ORDERS  = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]          # rzędy dynamiki NARX
    NARX_DEGREES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]          # stopnie wielomianów NARX
if pretrained == True:  # zawężone wokół n=20 (najlepszy z pierwszego przeszukiwania)
    NARX_ORDERS  = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,27, 28, 29, 30]  # zawężone wokół nA=20 (najlepszy z drugiego przeszukiwania)
    NARX_DEGREES = [2, 3, 4, 5, 6] # zawężone wokół deg=4

def wyswietl_info(dane: ImportDynamicData) -> None:
    print(f"Zbiór uczący    : {dane.N_ucz} próbek")
    print(f"Zbiór weryf.    : {dane.N_wer} próbek\n")


def rysuj_dane_surowe(dane: ImportDynamicData, plotter: PlotDynamic) -> None:
    plotter.plot_training_data(dane.u_ucz, dane.y_ucz)
    plotter.plot_verification_data(dane.u_wer, dane.y_wer)


def dopasuj_model_arx(dane: ImportDynamicData, plotter: PlotDynamic,
                      order: int) -> dict:
    model = ARXModel(nA=order, nB=order).fit(dane.u_ucz, dane.y_ucz)
    print(model)

    result = {'order': order, 'model': model}
    for recursive in [False, True]:
        tag   = 'r'              if recursive else 'nr'
        label = 'z rekurencją'   if recursive else 'bez rekurencji'

        y_hat_ucz, n = model.predict(dane.u_ucz, dane.y_ucz, recursive=recursive)
        m_ucz = model.evaluate(dane.y_ucz[n:], y_hat_ucz)

        y_hat_wer, n = model.predict(dane.u_wer, dane.y_wer, recursive=recursive)
        m_wer = model.evaluate(dane.y_wer[n:], y_hat_wer)

        print(f"  [{label:>20s}]  "
              f"Ucz: MSE={m_ucz['mse']:.6f}  RMSE={m_ucz['rmse']:.6f}  |  "
              f"Wer: MSE={m_wer['mse']:.6f}  RMSE={m_wer['rmse']:.6f}")

        plotter.plot_verification_results(dane.y_wer[n:], y_hat_wer, n,
                                          m_wer['mse'], m_wer['rmse'],
                                          order, recursive)
        plotter.plot_scatter(dane.y_wer[n:], y_hat_wer, order, recursive)

        result[f'mse_ucz_{tag}']  = m_ucz['mse']
        result[f'rmse_ucz_{tag}'] = m_ucz['rmse']
        result[f'mse_wer_{tag}']  = m_wer['mse']
        result[f'rmse_wer_{tag}'] = m_wer['rmse']

    return result


def drukuj_tabele_metryk(results: list) -> None:
    W = 13
    header = (f"{'n':>3}  "
              f"{'MSE ucz NR':>{W}} {'RMSE ucz NR':>{W}}  "
              f"{'MSE wer NR':>{W}} {'RMSE wer NR':>{W}}  "
              f"{'MSE ucz R':>{W}} {'RMSE ucz R':>{W}}  "
              f"{'MSE wer R':>{W}} {'RMSE wer R':>{W}}")
    sep = '─' * len(header)
    print(f'\n── Tabela metryk ARX ──\n{header}\n{sep}')
    for r in results:
        print(f"{r['order']:>3}  "
              f"{r['mse_ucz_nr']:{W}.6f} {r['rmse_ucz_nr']:{W}.6f}  "
              f"{r['mse_wer_nr']:{W}.6f} {r['rmse_wer_nr']:{W}.6f}  "
              f"{r['mse_ucz_r']:{W}.6f} {r['rmse_ucz_r']:{W}.6f}  "
              f"{r['mse_wer_r']:{W}.6f} {r['rmse_wer_r']:{W}.6f}")
    best = min(results, key=lambda r: r['mse_wer_r'])
    print(f"\nNajlepszy model (MSE wer R): "
          f"n={best['order']}  MSE={best['mse_wer_r']:.6f}  RMSE={best['rmse_wer_r']:.6f}")


def dopasuj_model_narx(dane: ImportDynamicData, plotter: PlotDynamic,
                       nA: int, deg: int) -> dict | None:
    """Dopasowuje model NARX(nA, nA, deg) i zwraca słownik z metrykami.
    Zwraca None dla układów nieoznaczonych (za mało próbek)."""
    n_params  = 2 * nA * deg          # liczba parametrów
    n_offset  = nA                    # max(nA, nB) = nA
    available = dane.N_ucz - n_offset  # liczba równań
    if n_params > available:
        print(f"  NARX(nA={nA}, deg={deg}): pominięto — "
              f"n_params={n_params} > dostępnych próbek={available}")
        return None

    model = NARXModel(nA=nA, nB=nA, deg=deg).fit(dane.u_ucz, dane.y_ucz)
    print(model)

    result = {'nA': nA, 'deg': deg, 'model': model}
    for recursive in [False, True]:
        tag   = 'r'            if recursive else 'nr'
        label = 'z rekurencją' if recursive else 'bez rekurencji'

        y_hat_ucz, n = model.predict(dane.u_ucz, dane.y_ucz, recursive=recursive)
        m_ucz = model.evaluate(dane.y_ucz[n:], y_hat_ucz)

        y_hat_wer, n = model.predict(dane.u_wer, dane.y_wer, recursive=recursive)
        m_wer = model.evaluate(dane.y_wer[n:], y_hat_wer)

        print(f"  [{label:>20s}]  "
              f"Ucz: MSE={m_ucz['mse']:.6f}  RMSE={m_ucz['rmse']:.6f}  |  "
              f"Wer: MSE={m_wer['mse']:.6f}  RMSE={m_wer['rmse']:.6f}")

        if np.isfinite(m_wer['mse']):
            plotter.plot_narx_verification_results(
                dane.y_wer[n:], y_hat_wer, n,
                m_wer['mse'], m_wer['rmse'], nA, deg, recursive)
            plotter.plot_narx_scatter(
                dane.y_wer[n:], y_hat_wer, nA, deg, recursive)

        result[f'mse_ucz_{tag}']  = m_ucz['mse']
        result[f'rmse_ucz_{tag}'] = m_ucz['rmse']
        result[f'mse_wer_{tag}']  = m_wer['mse']
        result[f'rmse_wer_{tag}'] = m_wer['rmse']

    return result


def drukuj_tabele_metryk_narx(results: list) -> None:
    W = 13
    header = (f"{'nA':>3} {'deg':>4}  "
              f"{'MSE ucz NR':>{W}} {'RMSE ucz NR':>{W}}  "
              f"{'MSE wer NR':>{W}} {'RMSE wer NR':>{W}}  "
              f"{'MSE ucz R':>{W}} {'RMSE ucz R':>{W}}  "
              f"{'MSE wer R':>{W}} {'RMSE wer R':>{W}}")
    sep = '─' * len(header)
    print(f'\n── Tabela metryk NARX ──\n{header}\n{sep}')
    for r in results:
        mse_r  = r['mse_wer_r']
        rmse_r = r['rmse_wer_r']
        mse_r_s  = f'{mse_r:{W}.6f}'  if np.isfinite(mse_r)  else f'{"∞":>{W}}'
        rmse_r_s = f'{rmse_r:{W}.6f}' if np.isfinite(rmse_r) else f'{"∞":>{W}}'
        print(f"{r['nA']:>3} {r['deg']:>4}  "
              f"{r['mse_ucz_nr']:{W}.6f} {r['rmse_ucz_nr']:{W}.6f}  "
              f"{r['mse_wer_nr']:{W}.6f} {r['rmse_wer_nr']:{W}.6f}  "
              f"{r['mse_ucz_r']:{W}.6f} {r['rmse_ucz_r']:{W}.6f}  "
              f"{mse_r_s} {rmse_r_s}")
    finite = [r for r in results if np.isfinite(r['mse_wer_r'])]
    if finite:
        best = min(finite, key=lambda r: r['mse_wer_r'])
        print(f"\nNajlepszy model (MSE wer R): "
              f"nA={best['nA']}, deg={best['deg']}  "
              f"MSE={best['mse_wer_r']:.6f}  RMSE={best['rmse_wer_r']:.6f}")


def symulacja_statyczna(model: NARXModel, u_star: float, N_sim: int = 600) -> float:
    """Wyznacza stan ustalony modelu dla stałego wejścia u* metodą symulacyjną."""
    u = np.full(N_sim, u_star)
    y_hat, _ = model.predict(u, np.zeros(N_sim), recursive=True)
    val = float(y_hat[-1])
    return val if np.isfinite(val) else np.nan


def charakterystyka_statyczna_narx(dane: ImportDynamicData, plotter: PlotDynamic,
                                   model: NARXModel) -> None:
    """Wyznacza i rysuje charakterystykę statyczną y(u) na podstawie modelu NARX."""
    stat_path = Path(__file__).resolve().parents[2] / 'danestat28' / 'danestat28.txt'
    stat_data = np.loadtxt(stat_path)
    u_stat = stat_data[:, 0]
    y_stat = stat_data[:, 1]

    # Gęsta siatka u do wykreślenia krzywej
    u_curve = np.linspace(u_stat.min(), u_stat.max(), 200)
    print(f"\n── Charakterystyka statyczna NARX(nA={model.nA}, deg={model.deg}) "
          f"– metoda symulacyjna ──")
    y_curve = np.array([symulacja_statyczna(model, u) for u in u_curve])

    # 3 punkty weryfikacyjne: min u, mediana u, max u
    q_vals   = [np.percentile(u_stat, p) for p in [10, 50, 90]]
    pts_idx  = [int(np.argmin(np.abs(u_stat - q))) for q in q_vals]
    u_pts       = u_stat[pts_idx]
    y_pts_data  = y_stat[pts_idx]
    y_pts_model = np.array([symulacja_statyczna(model, u) for u in u_pts])

    print(f"\n{'u*':>10} {'y* (dane)':>12} {'y* (NARX)':>12} {'|błąd|':>10}")
    print('─' * 50)
    for u, yd, ym in zip(u_pts, y_pts_data, y_pts_model):
        print(f"{u:>10.4f} {yd:>12.4f} {ym:>12.4f} {abs(yd - ym):>10.4f}")

    plotter.plot_static_characteristic_narx(
        u_curve, y_curve, u_stat, y_stat,
        u_pts, y_pts_data, y_pts_model,
        model.nA, model.deg)


def main() -> None:
    dane    = ImportDynamicData()
    plotter = PlotDynamic(dane.out_dir)

    wyswietl_info(dane)
    rysuj_dane_surowe(dane, plotter)

    print('── Modele ARX ──────────────────────────────────────────────────────────')
    all_results = []
    for order in ORDERS:
        res = dopasuj_model_arx(dane, plotter, order)
        all_results.append(res)
        print()

    drukuj_tabele_metryk(all_results)
    plotter.plot_metrics_table(all_results)

    print('\n── Modele NARX (wielomianowe nieliniowe) ───────────────────────────────')
    narx_results = []
    for nA in NARX_ORDERS:
        for deg in NARX_DEGREES:
            res = dopasuj_model_narx(dane, plotter, nA, deg)
            if res is not None:
                narx_results.append(res)
                print()

    drukuj_tabele_metryk_narx(narx_results)
    plotter.plot_narx_metrics_table(narx_results)

    # Charakterystyka statyczna najlepszego modelu NARX
    finite = [r for r in narx_results if np.isfinite(r['mse_wer_r'])]
    if finite:
        best = min(finite, key=lambda r: r['mse_wer_r'])
        print(f"\nNajlepszy model: NARX(nA={best['nA']}, deg={best['deg']}) "
              f"→ MSE wer R = {best['mse_wer_r']:.6f}")
        charakterystyka_statyczna_narx(dane, plotter, best['model'])

    PlotDynamic.show()


if __name__ == '__main__':
    main()
