import numpy as np

"""
    Generic gate class to provide common implementation across all gates.
"""
class Gate():

    # Constructor to set up Gate class
    def __init__(self, name, matrix):
        self.name = name
        self.matrix = np.array(matrix, dtype=complex)
        dim = self.matrix.shape[0]
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Matrix must be square")
        if not np.log2(dim).is_integer():
            raise ValueError("Matrix dimension must be 2^n")

    def apply(self, state):
        state = np.array(state, dtype=complex)

        num_qubits = int(np.log2(self.matrix.shape[0]))
        if state.shape in [(2**num_qubits,), (2**num_qubits, 1)]:
            return self.matrix @ state
        elif state.shape == (2**num_qubits, 2**num_qubits):
            return self.matrix @ state @ self.matrix.conj().T
        raise ValueError("Input must be in the form of a state vector or density matrix!")
            

    def __str__(self):
        return f"{self.name} Gate: {self.matrix}"