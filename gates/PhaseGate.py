import numpy as np

class PhaseGate:
    def __init__(self):
        self.name = "phase gate"
        self.matrix = np.array([
            [1, 0],
            [0, 1j]
        ], dtype=complex)

    def apply(self, state_vector):
        if state_vector.shape != (2,):
            raise ValueError("Input must be a single-qubit state vector of shape (2,).")

        return np.dot(self.matrix, state_vector)


if __name__ == "__main__":
    phase_gate = PhaseGate()
    sample = np.array([0, 1], dtype=complex) # input: |1⟩
    print(phase_gate.apply(sample))  # output: j|1⟩
