import numpy as np

"""
    Class to handle Kraus operator storage and application to a density matrix
    or vectorized form of that matrix.

    TODO: Implement application of Kraus operators
    TODO: Implement stochastic/Monte Carlo Kraus operator application (may be separate class)
"""
class Kraus:
    
    def __init__(self, kraus_ops):
        self.kraus_ops = np.array(kraus_ops, dtype=complex)
        self._check_trace()

    # Verify the complete trace of the operators so that it covers full probability space.
    def _check_trace(self):
        dim = self.kraus_ops[0].shape[0]
        total = np.zeros((dim, dim), dtype=complex)
        for K in self.kraus_ops:
            total += K.conj().T @ K
        if not np.allclose(total, np.eye(dim)):
            print("Kraus operators are not trace preserving!")