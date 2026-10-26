"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def four_weeks():
    """Vier Wochen: x = (1, 2, 3, 4), y = (2, 4, 5, 8). Mittel 2,5 und 4,75, Σ(x − x̄)(y − ȳ) = 9,5, Σ(x − x̄)² = 5: Steigung 1,9, Achsenabschnitt 4,75 − 1,9 · 2,5 = 0, Rest
    û = y − 1,9 x = (0,1; 0,2; −0,7; 0,4). Dickey–Fuller am Rest: Δû = (0,1; −0,9; 1,1), û_{t−1} = (0,1; 0,2; −0,7), γ̂ = (0,01 − 0,18 − 0,77)/(0,01 + 0,04 + 0,49) = −0,94/0,54 = −1,7407;
    Rest der Regression e = Δû − γ̂ û_{t−1}, Σe² = 0,39376, Standardfehler² = Σe²/(3 − 1)/0,54 = 0,36459, t = −1,7407/0,6038 = −2,8831."""
    return np.array([[1.0, 2.0, 3.0, 4.0]]), np.array([[2.0, 4.0, 5.0, 8.0]])


@pytest.fixture
def de_bruijn_pair():
    """Dreizehn Wochen mit dem Vorzeichenmuster 0, 0, 1, 1 (alle vier Paare aufeinanderfolgender Zustände je dreimal in den zwölf Übergängen) und y_{t+1} = x_t: bei k = 2 Klassen ist
    die Transferentropie I(Y_{t+1}; X_t | Y_t) = H(Y_{t+1} | Y_t) − 0 = ln 2 = 0,6931 nats. Gleichstände werden durch kleine verschiedene Zusätze vermieden; die letzte x-Woche wird nicht benutzt."""
    pattern = np.array([-1.0, -1.0, 1.0, 1.0] * 4)[:13]
    s = pattern * (1.0 + 1e-3 * np.arange(13))
    return np.append(s[1:], 0.0), s            # x_t = s_{t+1}, y_t = s_t
