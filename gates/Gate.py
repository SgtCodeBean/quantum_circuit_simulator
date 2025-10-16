import numpy as np

"""
    Generic gate class to provide common implementation across all gates.
"""
class Gate():

    # Constructor to set up Gate class
    def __init__(self, name, matrix):
        self.name = name
        self.matrix = np.array(matrix, dtype=complex)
        if self.matrix.shape != (2,2):
            raise ValueError("Matrix has to be 2x2")

    def apply(self, state_vector):
        state_vector = np.array(state_vector, dtype=complex)

        num_qbits = int(np.log2(self.matrix.shape[0]))
        if state_vector.shape not in [(2**num_qbits,), (2**num_qbits, 1)]:
            raise ValueError("State vector needs to be a 2x1 vector")
            
        return self.matrix @ state_vector

    def toString(self):
        return f"{self.name} Gate: {self.matrix}"