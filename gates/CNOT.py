import numpy as np

class CnotGate:
    def __init__(self):
        self.name = "CNOT gate"
        self.matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)

    def apply(self, state_vector):
        """
        Applies the CNOT gate to a 2-qubit state vector.

        Args:
            state_vector (numpy.ndarray): A 4x1 vector representing the 2-qubit state.

        Returns:
            numpy.ndarray: The new 4x1 state vector after applying the gate.
        """
        if state_vector.shape != (4,):
            raise ValueError("Input must be a 2-qubit state vector of shape (4,).")

        return np.dot(self.matrix, state_vector)


if __name__ == "__main__":
    cnot_gate = CnotGate()
    sample = np.array([0, 0, 1, 0], dtype=complex) # input: |10⟩
    print(cnot_gate.apply(sample))  # output: |11⟩

