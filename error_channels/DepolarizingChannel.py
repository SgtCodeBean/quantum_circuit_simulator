import numpy as np
from .Kraus import Kraus

"""
    Depolarizing channel implementation for single-qubit quantum states.

    The depolarizing channel with parameter p models quantum noise where:
    - With probability (1-p): the state remains unchanged
    - With probability p: the state experiences Pauli errors (X, Y, or Z)
"""
class DepolarizingChannel:

    def __init__(self, p):
        if not 0 <= p <= 1:
            raise ValueError(
                f"Depolarizing probability must be between 0 and 1, got {p}")
        self.p = p

        # pauli matrices
        self.I = np.array([[1, 0], [0, 1]], dtype=complex)
        self.X = np.array([[0, 1], [1, 0]], dtype=complex)
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.Z = np.array([[1, 0], [0, -1]], dtype=complex)

        kraus_ops = self._compute_kraus_operators()
        self.kraus = Kraus(kraus_ops)

    def _compute_kraus_operators(self):
        return [
            np.sqrt(1 - 3*self.p/4) * self.I,
            np.sqrt(self.p/4) * self.X,
            np.sqrt(self.p/4) * self.Y,
            np.sqrt(self.p/4) * self.Z
        ]

    # ε(ρ) = Σᵢ (Eᵢ @ ρ @ Eᵢ†)
    def apply(self, rho):
        result = np.zeros_like(rho, dtype=complex)
        for E in self.kraus.kraus_ops:
            result += E @ rho @ E.T.conj()
        return result

    def __repr__(self):
        return f"DepolarizingChannel(p={self.p})"
