import numpy as np
from pathlib import Path


class ImportDynamicData:
    """Wczytuje dane dynamiczne (uczące i weryfikujące) z plików .txt."""

    def __init__(self, base_dir: str | None = None):
        base = Path(base_dir) if base_dir else Path(__file__).resolve().parents[2]
        ucz = np.loadtxt(base / 'danedynucz28' / 'danedynucz28.txt')
        wer = np.loadtxt(base / 'danedynwer28' / 'danedynwer28.txt')
        self.u_ucz = ucz[:, 0];  self.y_ucz = ucz[:, 1]
        self.u_wer = wer[:, 0];  self.y_wer = wer[:, 1]
        self._out_dir = Path(__file__).resolve().parent.parent / 'obrazki'

    @property
    def N_ucz(self) -> int: return len(self.u_ucz)

    @property
    def N_wer(self) -> int: return len(self.u_wer)

    @property
    def out_dir(self) -> Path: return self._out_dir
