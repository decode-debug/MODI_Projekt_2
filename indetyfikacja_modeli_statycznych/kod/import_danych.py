import os
import numpy as np


class Import_Static_Data:
    """Wczytuje dane statyczne i dzieli je na zbiór uczący i weryfikujący."""

    def __init__(self, split: float = 0.6):
        _dir = os.path.dirname(os.path.abspath(__file__))
        _data_path = os.path.join(_dir, '..', '..', 'danestat28', 'danestat28.txt')
        self._out_dir = os.path.join(_dir, '..', 'obrazki_charakterystyki_statycznej')
        os.makedirs(self._out_dir, exist_ok=True)

        data = np.loadtxt(_data_path)
        self.u = data[:, 0]   # sygnał wejściowy
        self.y = data[:, 1]   # sygnał wyjściowy

        self.N        = len(self.u)
        self.N_uczacy = int(np.ceil(split * self.N))   # 60% → 120 próbek
        self.N_weryf  = self.N - self.N_uczacy         # 40% → 80 próbek

        self.u_ucz = self.u[:self.N_uczacy]
        self.y_ucz = self.y[:self.N_uczacy]
        self.u_wer = self.u[self.N_uczacy:]
        self.y_wer = self.y[self.N_uczacy:]

    @property
    def out_dir(self) -> str:
        return self._out_dir