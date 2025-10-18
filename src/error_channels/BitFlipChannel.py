import numpy as np

class BitFlipChannel:
    def __init__(self, p):
        """
        Args:
            p (float): The probability of a bit flip occurring (must be between 0 and 1).
        """
        if not (0 <= p <= 1):
            raise ValueError("Probability p must be between 0 and 1.")

        self.probability = p
        self.name = f"Bit Flip Channel (probability={p})"
        self.identity_matrix = np.identity(2)
        self.x_matrix = np.array([[0, 1], [1, 0]])

    def apply(self, state_vector):
        if state_vector.shape != (2,):
            raise ValueError("Input must be a single-qubit state vector of shape (2,).")

        if np.random.rand() <= self.probability:
            return self.x_matrix @ state_vector
        else:
            return self.identity_matrix @ state_vector


if __name__ == "__main__":
    p = 0.25
    bfc = BitFlipChannel(p)
    sample = np.array([1, 0], dtype=complex) # input: [1, 0]
    print(bfc.apply(sample)) # output: [1, 0] with probability 75%; [0, 1] with probability 25%