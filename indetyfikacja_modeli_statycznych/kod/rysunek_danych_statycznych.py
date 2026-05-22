import os
import numpy as np
import matplotlib.pyplot as plt

# Ścieżki względem lokalizacji skryptu
_dir = os.path.dirname(os.path.abspath(__file__))
_data_path = os.path.join(_dir, '..', '..', 'danestat28', 'danestat28.txt')
_out_dir   = os.path.join(_dir, '..', 'obrazki_charakterystyki_statycznej')
os.makedirs(_out_dir, exist_ok=True)

# Wczytanie danych
data = np.loadtxt(_data_path)

u = data[:, 0]  # sygnał wejściowy
y = data[:, 1]  # sygnał wyjściowy

N = len(u)
N_uczacy = int(np.ceil(0.6 * N))   # 60% -> 120 próbek
N_weryf  = N - N_uczacy             # 40% -> 80 próbek

u_ucz = u[:N_uczacy]
y_ucz = y[:N_uczacy]

u_wer = u[N_uczacy:]
y_wer = y[N_uczacy:]

# ── Rysunek 1: charakterystyka statyczna (y = f(u)) ─────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.plot(u, y, 'ko', markersize=4)
ax2.set_xlabel('u')
ax2.set_ylabel('y')
ax2.set_title('Charakterystyka statyczna y = f(u) – wszystkie dane')
ax2.grid(True)
fig2.tight_layout()
fig2.savefig(os.path.join(_out_dir, 'charakterystyka_statyczna_wszystkie.png'), dpi=150)

# ── Rysunek 2: charakterystyka statyczna – zbiór uczący ─────────────────────
fig4, ax4 = plt.subplots(figsize=(7, 5))
ax4.plot(u_ucz, y_ucz, 'bo', markersize=4, label='zbiór uczący')
ax4.set_xlabel('u')
ax4.set_ylabel('y')
ax4.set_title('Charakterystyka statyczna – zbiór uczący (60 %)')
ax4.grid(True)
ax4.legend()
fig4.tight_layout()
fig4.savefig(os.path.join(_out_dir, 'charakterystyka_statyczna_uczace.png'), dpi=150)

# ── Rysunek 3: charakterystyka statyczna – zbiór weryfikujący ───────────────
fig6, ax6 = plt.subplots(figsize=(7, 5))
ax6.plot(u_wer, y_wer, 'rs', markersize=4, label='zbiór weryfikujący')
ax6.set_xlabel('u')
ax6.set_ylabel('y')
ax6.set_title('Charakterystyka statyczna – zbiór weryfikujący (40 %)')
ax6.grid(True)
ax6.legend()
fig6.tight_layout()
fig6.savefig(os.path.join(_out_dir, 'charakterystyka_statyczna_weryfikujace.png'), dpi=150)

plt.show()

print(f'Łącznie próbek : {N}')
print(f'Zbiór uczący   : {N_uczacy} ({N_uczacy/N*100:.0f}%)')
print(f'Zbiór weryf.   : {N_weryf} ({N_weryf/N*100:.0f}%)')
