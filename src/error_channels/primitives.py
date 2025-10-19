import numpy as np
from typing import Sequence

KrausSet = Sequence[np.ndarray]

def pauli_y_channel(p: float) -> KrausSet:
    I = np.eye(2, dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], complex)
    return [np.sqrt(1-p)*I, np.sqrt(p)*Y]