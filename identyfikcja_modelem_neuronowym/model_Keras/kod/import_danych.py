import os
import numpy as np


class ImportStaticData:
    """Wczytuje dane statyczne i dzieli je na zbiór uczący i weryfikujący.

    Schemat zgodny z indetyfikacja_modeli_statycznych/kod/import_danych.py.
    Dane: danestat28/danestat28.txt (plik z dwiema kolumnami: u  y, 200 wierszy).
    """

    def __init__(self, split: float = 0.6):
        _dir = os.path.dirname(os.path.abspath(__file__))
        _data_path = os.path.join(_dir, '..', '..', '..', 'danestat28', 'danestat28.txt')
        self._out_dir = os.path.join(_dir, '..', 'obrazki')
        os.makedirs(self._out_dir, exist_ok=True)

        data = np.loadtxt(_data_path)
        self.u = data[:, 0]
        self.y = data[:, 1]

        self.N = len(self.u)
        self.N_uczacy = int(np.ceil(split * self.N))   # domyślnie 60 % → 120 próbek
        self.N_weryf  = self.N - self.N_uczacy         # domyślnie 40 % →  80 próbek

        self.u_ucz = self.u[:self.N_uczacy]
        self.y_ucz = self.y[:self.N_uczacy]
        self.u_wer = self.u[self.N_uczacy:]
        self.y_wer = self.y[self.N_uczacy:]

    @property
    def out_dir(self) -> str:
        return self._out_dir

    def __repr__(self) -> str:
        return (f"ImportStaticData(N={self.N}, N_uczacy={self.N_uczacy}, "
                f"N_weryf={self.N_weryf})")
