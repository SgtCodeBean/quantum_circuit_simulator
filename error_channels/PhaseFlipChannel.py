import numpy as np

class PhaseFlipChannel:
    def __init__(self, p):
        """
        Args:
            p (float): The probability of a phase flip occurring (must be between 0 and 1).
        """
        if not (0 <= p <= 1):
            raise ValueError("Probability p must be between 0 and 1.")

        self.probability = p
        self.name = f"phase Flip Channel (probability={p})"
        self.identity_matrix = np.identity(2)
        self.z_matrix = np.array([[1, 0], [0, -1]])

    def apply(self, state_vector):
        if state_vector.shape != (2,):
            raise ValueError("Input must be a single-qubit state vector of shape (2,).")

        if np.random.rand() <= self.probability:
            return self.z_matrix @ state_vector
        else:
            return self.identity_matrix @ state_vector


if __name__ == "__main__":
    p = 0.5
    pfc = PhaseFlipChannel(p)
    sample = np.array([1, 1], dtype=complex) # input: [1, 1]
    print(pfc.apply(sample)) # output: [1, -1] with probability 50%; [1, 1] with probability 50%