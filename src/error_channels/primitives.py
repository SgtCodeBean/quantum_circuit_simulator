import numpy as np
from typing import Sequence

KrausSet = Sequence[np.ndarray]

# Primitive error channels as Kraus sets
def pauli_y_channel(p: float) -> KrausSet:
    I = np.eye(2, dtype=complex)
    Y = np.array([[0,-1j],[1j,0]], complex)
    return [np.sqrt(1-p)*I, np.sqrt(p)*Y]

def pauli_z_channel(p: float) -> KrausSet:
    I = np.eye(2, dtype=complex)
    Z = np.array([[1,0],[0,-1]], complex)
    return [np.sqrt(1-p)*I, np.sqrt(p)*Z]

def pauli_x_channel(p: float) -> KrausSet:
    I = np.eye(2, dtype=complex)
    X = np.array([[0,1],[1,0]], complex)
    return [np.sqrt(1-p)*I, np.sqrt(p)*X]

def depolarizing_channel(p: float) -> KrausSet:
    I = np.eye(2, dtype=complex)
    X = np.array([[0,1],[1,0]], complex)
    Y = np.array([[0,-1j],[1j,0]], complex)
    Z = np.array([[1,0],[0,-1]], complex)
    return [np.sqrt(1 - 3*p/4)*I, np.sqrt(p/4)*X, np.sqrt(p/4)*Y, np.sqrt(p/4)*Z]

def amplitude_damping_channel(gamma: float) -> KrausSet:
    E0 = np.array([[1,0],[0,np.sqrt(1-gamma)]], complex)
    E1 = np.array([[0,np.sqrt(gamma)],[0,0]], complex)
    return [E0, E1]

def phase_damping_channel(lam: float) -> KrausSet:
    E0 = np.array([[1,0],[0,np.sqrt(1-lam)]], complex)
    E1 = np.array([[0,0],[0,np.sqrt(lam)]], complex)
    return [E0, E1]