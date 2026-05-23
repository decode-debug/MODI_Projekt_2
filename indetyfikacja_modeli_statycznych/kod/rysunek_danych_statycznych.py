from import_danych import Import_Static_Data
from modele import LinearModel
from rysowanie import PlotData


def main():
    # ── 1. Wczytanie i podział danych ─────────────────────────────────────────
    dane = Import_Static_Data(split=0.6)
    print(f'Łącznie próbek : {dane.N}')
    print(f'Zbiór uczący   : {dane.N_uczacy} ({dane.N_uczacy / dane.N * 100:.0f} %)')
    print(f'Zbiór weryf.   : {dane.N_weryf}  ({dane.N_weryf  / dane.N * 100:.0f} %)')

    # ── 2. Trenowanie modelu liniowego (własny solver MNK) ────────────────────
    model = LinearModel()
    model.fit(dane.u_ucz, dane.y_ucz)
    print(f'\n{model}')

    # ── 3. Ewaluacja ──────────────────────────────────────────────────────────
    wyniki_ucz = model.evaluate(dane.u_ucz, dane.y_ucz)
    wyniki_wer = model.evaluate(dane.u_wer, dane.y_wer)

    print(f'\nBłędy – zbiór uczący   (N={dane.N_uczacy}): '
          f'MSE = {wyniki_ucz["mse"]:.6f}   RMSE = {wyniki_ucz["rmse"]:.6f}')
    print(f'Błędy – zbiór weryf.  (N={dane.N_weryf}):  '
          f'MSE = {wyniki_wer["mse"]:.6f}   RMSE = {wyniki_wer["rmse"]:.6f}')

    # ── 4. Rysowanie ──────────────────────────────────────────────────────────
    plotter = PlotData(dane.out_dir)

    # dane surowe
    plotter.plot_all_data(dane.u, dane.y)
    plotter.plot_static_characteristic(dane.u, dane.y)
    plotter.plot_training_data(dane.u_ucz, dane.y_ucz)
    plotter.plot_training_characteristic(dane.u_ucz, dane.y_ucz)
    plotter.plot_verification_data(dane.u_wer, dane.y_wer, dane.N_uczacy)
    plotter.plot_verification_characteristic(dane.u_wer, dane.y_wer)

    # wyniki modelu
    plotter.plot_model_characteristic(dane.u, dane.y, model.a0, model.a1)
    plotter.plot_verification_results(
        dane.y_wer, wyniki_wer['y_pred'],
        dane.N_uczacy,
        wyniki_wer['mse'], wyniki_wer['rmse']
    )
    plotter.plot_scatter(dane.y_wer, wyniki_wer['y_pred'])

    PlotData.show()


if __name__ == '__main__':
    main()

