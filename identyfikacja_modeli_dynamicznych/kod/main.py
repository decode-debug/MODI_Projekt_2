import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from import_danych      import ImportDynamicData
from modele_dynamiczne  import ARXModel
from rysowanie          import PlotDynamic

ORDERS = [1, 2, 3]


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
    PlotDynamic.show()


if __name__ == '__main__':
    main()
