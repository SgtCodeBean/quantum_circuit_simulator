import numpy as np
from .Gate import Gate

class PhaseGate(Gate):
    def __init__(self):
        matrix = np.array([
            [1, 0],
            [0, 1j]
        ], dtype=complex)

        super().__init__("phase gate", matrix)


if __name__ == "__main__":
    phase_gate = PhaseGate()
    sample = np.array([0, 1], dtype=complex) # input: |1⟩
    print(phase_gate.apply(sample))  # output: j|1⟩
