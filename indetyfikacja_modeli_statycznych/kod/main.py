from import_danych import Import_Static_Data
from modele import LinearModel, PolynomialModel
from rysowanie import PlotData

# ── Konfiguracja ──────────────────────────────────────────────────────────────
DEGREES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# ══════════════════════════════════════════════════════════════════════════════
# Funkcje pomocnicze
# ══════════════════════════════════════════════════════════════════════════════

def wyswietl_podzial_danych(dane: Import_Static_Data) -> None:
    """Drukuje informację o podziale zbioru na uczący i weryfikujący."""
    print(f'Łącznie próbek : {dane.N}')
    print(f'Zbiór uczący   : {dane.N_uczacy} ({dane.N_uczacy / dane.N * 100:.0f} %)')
    print(f'Zbiór weryf.   : {dane.N_weryf}  ({dane.N_weryf  / dane.N * 100:.0f} %)')


def rysuj_dane_surowe(dane: Import_Static_Data, plotter: PlotData) -> None:
    """Rysuje przebiegi i charakterystyki surowych danych (uczące + weryfikujące)."""
    plotter.plot_all_data(dane.u, dane.y)
    plotter.plot_static_characteristic(dane.u, dane.y)
    plotter.plot_training_data(dane.u_ucz, dane.y_ucz)
    plotter.plot_training_characteristic(dane.u_ucz, dane.y_ucz)
    plotter.plot_verification_data(dane.u_wer, dane.y_wer, dane.N_uczacy)
    plotter.plot_verification_characteristic(dane.u_wer, dane.y_wer)


def dopasuj_model_liniowy(dane: Import_Static_Data, plotter: PlotData) -> dict:
    """
    Trenuje model liniowy (MNK), drukuje metryki i rysuje wyniki.
    Zwraca słownik {'model', 'wyniki_ucz', 'wyniki_wer'}.
    """
    model      = LinearModel().fit(dane.u_ucz, dane.y_ucz)
    wyniki_ucz = model.evaluate(dane.u_ucz, dane.y_ucz)
    wyniki_wer = model.evaluate(dane.u_wer, dane.y_wer)

    print(f'\n{model}')
    print(f'  Zbiór uczący  (N={dane.N_uczacy}): '
          f'MSE = {wyniki_ucz["mse"]:.6f}   RMSE = {wyniki_ucz["rmse"]:.6f}')
    print(f'  Zbiór weryf. (N={dane.N_weryf}):  '
          f'MSE = {wyniki_wer["mse"]:.6f}   RMSE = {wyniki_wer["rmse"]:.6f}')

    plotter.plot_model_characteristic(dane.u, dane.y, model.a0, model.a1)
    plotter.plot_verification_results(
        dane.y_wer, wyniki_wer['y_pred'],
        dane.N_uczacy, wyniki_wer['mse'], wyniki_wer['rmse']
    )
    plotter.plot_scatter(dane.y_wer, wyniki_wer['y_pred'])

    return {'model': model, 'wyniki_ucz': wyniki_ucz, 'wyniki_wer': wyniki_wer}


def dopasuj_modele_wielomianowe(dane: Import_Static_Data, plotter: PlotData,
                                degrees: list) -> tuple:
    """
    Dla każdego stopnia N z listy 'degrees': trenuje model wielomianowy MNK,
    drukuje metryki i rysuje wyniki.
    Zwraca (lista słowników metryk, lista wytrenowanych modeli).
    """
    results = []
    models  = []

    print('\n── Modele wielomianowe ──────────────────────────────────────')
    for deg in degrees:
        model      = PolynomialModel(degree=deg).fit(dane.u_ucz, dane.y_ucz)
        wyniki_ucz = model.evaluate(dane.u_ucz, dane.y_ucz)
        wyniki_wer = model.evaluate(dane.u_wer, dane.y_wer)

        print(model)
        print(f'  Zbiór uczący  (N={dane.N_uczacy}): '
              f'MSE = {wyniki_ucz["mse"]:.6f}   RMSE = {wyniki_ucz["rmse"]:.6f}')
        print(f'  Zbiór weryf. (N={dane.N_weryf}):  '
              f'MSE = {wyniki_wer["mse"]:.6f}   RMSE = {wyniki_wer["rmse"]:.6f}\n')

        plotter.plot_poly_characteristic(dane.u, dane.y, model, deg)
        plotter.plot_poly_verification_results(
            dane.y_wer, wyniki_wer['y_pred'],
            dane.N_uczacy, wyniki_wer['mse'], wyniki_wer['rmse'], deg
        )
        plotter.plot_poly_scatter(dane.y_wer, wyniki_wer['y_pred'], deg)

        results.append({
            'degree':   deg,
            'mse_ucz':  wyniki_ucz['mse'],
            'rmse_ucz': wyniki_ucz['rmse'],
            'mse_wer':  wyniki_wer['mse'],
            'rmse_wer': wyniki_wer['rmse'],
        })
        models.append(model)

    return results, models


def drukuj_tabele_metryk(results: list) -> None:
    """Drukuje zbiorczą tabelę MSE/RMSE w konsoli i wyróżnia najlepszy model."""
    print(f'\n{"N":>4}  {"MSE ucz":>12}  {"RMSE ucz":>12}  {"MSE wer":>12}  {"RMSE wer":>12}')
    print('─' * 60)
    for r in results:
        print(f'{r["degree"]:>4}  {r["mse_ucz"]:>12.6f}  {r["rmse_ucz"]:>12.6f}'
              f'  {r["mse_wer"]:>12.6f}  {r["rmse_wer"]:>12.6f}')
    best = min(results, key=lambda r: r['mse_wer'])
    print(f'\nNajlepszy model (MSE weryf.): N={best["degree"]}  '
          f'MSE={best["mse_wer"]:.6f}  RMSE={best["rmse_wer"]:.6f}')


# ══════════════════════════════════════════════════════════════════════════════
# Punkt wejścia
# ══════════════════════════════════════════════════════════════════════════════

def main():
    dane    = Import_Static_Data(split=0.6)
    plotter = PlotData(dane.out_dir)

    # 1 – informacja o danych
    wyswietl_podzial_danych(dane)

    # 2 – wykresy danych surowych
    rysuj_dane_surowe(dane, plotter)

    # 3 – model liniowy (N=1, oddzielna sekcja w folderze model_liniowy/)
    dopasuj_model_liniowy(dane, plotter)

    # 4 – modele wielomianowe (N=1..20, foldery model_wielomianowy/polyN/)
    results, models = dopasuj_modele_wielomianowe(dane, plotter, DEGREES)

    # 5 – podsumowanie: tabela i wykres zbiorczy
    drukuj_tabele_metryk(results)
    plotter.plot_metrics_table(results)
    plotter.plot_all_poly_characteristics(dane.u, dane.y, models, DEGREES)

    PlotData.show()


if __name__ == '__main__':
    main()

