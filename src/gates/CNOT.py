import numpy as np
from .Gate import Gate


class CnotGate(Gate):
    def __init__(self):
        cnot_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)

        super().__init__("CNOT gate", cnot_matrix)


if __name__ == "__main__":
    cnot_gate = CnotGate()
    sample = np.array([0, 0, 1, 0], dtype=complex) # input: |10⟩
    print(cnot_gate.apply(sample))  # output: |11⟩
