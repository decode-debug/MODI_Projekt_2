import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE    = Path(__file__).resolve().parents[2]
out_dir = Path(__file__).resolve().parent.parent / 'obrazki'
out_dir.mkdir(exist_ok=True)

# Wczytanie danych
ucz = np.loadtxt(BASE / 'danedynucz28' / 'danedynucz28.txt')
wer = np.loadtxt(BASE / 'danedynwer28' / 'danedynwer28.txt')

u_ucz = ucz[:, 0]
y_ucz = ucz[:, 1]
N_ucz = len(u_ucz)

u_wer = wer[:, 0]
y_wer = wer[:, 1]
N_wer = len(u_wer)

# ── Rysunek 1: przebiegi czasowe – zbiór uczący ─────────────────────────────
fig1, axes1 = plt.subplots(2, 1, figsize=(10, 6))
fig1.suptitle('Dane dynamiczne – zbiór uczący (N = {})'.format(N_ucz))

axes1[0].plot(range(1, N_ucz + 1), u_ucz, 'b.-', markersize=3)
axes1[0].set_xlabel('Numer próbki')
axes1[0].set_ylabel('u')
axes1[0].set_title('Sygnał wejściowy u – zbiór uczący')
axes1[0].grid(True)

axes1[1].plot(range(1, N_ucz + 1), y_ucz, 'r.-', markersize=3)
axes1[1].set_xlabel('Numer próbki')
axes1[1].set_ylabel('y')
axes1[1].set_title('Sygnał wyjściowy y – zbiór uczący')
axes1[1].grid(True)

fig1.tight_layout()
fig1.savefig(out_dir / 'dane_dynamiczne_uczace.png', dpi=150)

# ── Rysunek 2: przebiegi czasowe – zbiór weryfikujący ───────────────────────
fig2, axes2 = plt.subplots(2, 1, figsize=(10, 6))
fig2.suptitle('Dane dynamiczne – zbiór weryfikujący (N = {})'.format(N_wer))

axes2[0].plot(range(1, N_wer + 1), u_wer, 'b.-', markersize=3)
axes2[0].set_xlabel('Numer próbki')
axes2[0].set_ylabel('u')
axes2[0].set_title('Sygnał wejściowy u – zbiór weryfikujący')
axes2[0].grid(True)

axes2[1].plot(range(1, N_wer + 1), y_wer, 'r.-', markersize=3)
axes2[1].set_xlabel('Numer próbki')
axes2[1].set_ylabel('y')
axes2[1].set_title('Sygnał wyjściowy y – zbiór weryfikujący')
axes2[1].grid(True)

fig2.tight_layout()
fig2.savefig(out_dir / 'dane_dynamiczne_weryfikujace.png', dpi=150)

plt.show()

print(f'Zbiór uczący    : {N_ucz} próbek')
print(f'Zbiór weryf.    : {N_wer} próbek')
