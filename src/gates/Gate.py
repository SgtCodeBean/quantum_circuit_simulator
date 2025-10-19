import numpy as np
from utils.linalg import is_unitary, is_power_of_two_dim

"""
    Generic gate class to provide common implementation across all gates.
"""
class Gate:
    def __init__(self, name, matrix):
        self.name = name
        self.matrix = np.array(matrix, dtype=complex)
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Matrix must be square")
        dim = self.matrix.shape[0]
        if not is_power_of_two_dim(dim):
            raise ValueError("Matrix dimension must be 2^n")
        if not is_unitary(self.matrix):
            raise ValueError("Matrix must be unitary")
        self.arity = int(np.log2(dim))

    def apply(self, state):
        state = np.array(state, dtype=complex)
        n = self.arity
        if state.shape in [(2**n,), (2**n, 1)]:
            return self.matrix @ state
        elif state.shape == (2**n, 2**n):
            return self.matrix @ state @ self.matrix.conj().T
        raise ValueError("Input must be a size-matched state vector or density matrix")

    def __str__(self):
        return f"{self.name} ({self.arity}q) Gate:\n{self.matrix}"