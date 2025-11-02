import numpy as np
from Channel.py import Channel

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

def bit_flip(p: float) -> Channel:
    if not (0 <= p <= 1): raise ValueError("p must be in [0,1]")
    return Channel("bit_flip", [np.sqrt(1-p)*I2, np.sqrt(p)*X])

def phase_flip(p: float) -> Channel:
    if not (0 <= p <= 1): raise ValueError("p must be in [0,1]")
    return Channel("phase_flip", [np.sqrt(1-p)*I2, np.sqrt(p)*Z])

def depolarizing(p: float) -> Channel:
    if not (0 <= p <= 1): 
          raise ValueError("p must be in [0,1]")
    return Channel("depolarizing", [
        np.sqrt(1-p)*I2,
        np.sqrt(p/3)*X, np.sqrt(p/3)*Y, np.sqrt(p/3)*Z
    ])

def amplitude_damping(gamma: float) -> Channel:
        if not (0 <= gamma <= 1):
                raise ValueError("gamma must be in [0,1]")
        
        E0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
        E1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
        return Channel("amplitude damping", [E0, E1])

def phase_damping(lam: float) -> Channel:
        if not (0 <= lam <= 1):
                raise ValueError("lambda must be in [0,1]")
        E0 = np.array([[1, 0], [0, np.sqrt(1-lam)]])
        E1 = np.array([[0, 0], [0, np.sqrt(lam)]])

        return Channel("phase damping", [E0, E1])





