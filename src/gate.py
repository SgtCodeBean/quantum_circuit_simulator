import numpy as np

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

        self.num_qubits = int(np.log2(dim))

    def apply(self, state_vector):
        state_vector = np.array(state_vector, dtype=complex).reshape(-1, 1)

        # Ensure dimensions match
        if state_vector.shape[0] != self.matrix.shape[1]:
            raise ValueError(f"State vector must have dimension {self.matrix.shape[1]}")
        return self.matrix @ state_vector

    def __str__(self):
        return f"{self.name} Gate ({self.num_qubits}-qubit):\n{self.matrix}"