import numpy as np
from typing import Dict, List
from gates.primitives import pauli_gates, hadamard_gate, toffoli_gate
from utils.linalg import is_unitary, is_power_of_two_dim

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

class GateRegistry:
    def __init__(self, preload_defaults: bool = True):
        self._defs: Dict[str, Gate] = {}
        if preload_defaults:
            pauli_x, pauli_y, pauli_z = pauli_gates()
            self.add(gate_name="x", gate_matrix=pauli_x)
            self.add("y", pauli_y)
            self.add("z", pauli_z)
            self.add("h", hadamard_gate())
            self.add("toffoli", toffoli_gate())
            # TODO: add more primitive gates as needed

    def add(self, gate_name, gate_matrix, overwrite: bool = False):
        if not overwrite and gate_name in self._defs:
            raise ValueError(f"Gate '{gate_name}' already exists")
        # basic validation
        g = Gate(gate_name, gate_matrix)
        if g.arity <= 0:
            raise ValueError("arity must be >= 1")
        self._defs[gate_name] = g

    def remove(self, name: str):
        if name not in self._defs:
            raise KeyError(f"Gate '{name}' not found")
        del self._defs[name]

    def get(self, name: str) -> Gate:
        if name not in self._defs:
            raise KeyError(f"Gate '{name}' not found")
        return self._defs[name]

    def list(self) -> List[str]:
        return sorted(self._defs.keys())