import numpy as np

"""
    Class to handle Kraus operator storage and application to a density matrix
    or vectorized form of that matrix.

    TODO: Implement stochastic/Monte Carlo Kraus operator application (may be separate class)
"""
class Kraus:

    def __init__(self, kraus_ops):
        self.kraus_ops = np.array(kraus_ops, dtype=complex)
        self._check_trace()

    # Verify the complete trace of the operators so that it covers full probability space.
    def _check_trace(self):
        dim = self.kraus_ops.shape[0]
        total = np.zeros((dim, dim), dtype=complex)
        for K in self.kraus_ops:
            total += K.conj().T @ K
        if not np.allclose(total, np.eye(dim)):
            print("Kraus operators are not trace preserving!")

    # 
    def apply(self, rho):
        rho = np.array(rho, dtype=complex)
        result = np.zeros((rho.shape[0], rho.shape[1]), dtype=complex)
        for K in self.kraus_ops:
            result += K @ rho @ K.conj().T
        return result
    
    # 
    def to_superoperator(self):
        dim = self.kraus_ops.shape[0]
        E = np.zeros((dim**2, dim**2), dtype=complex)
        for K in self.kraus_ops:
            E += np.kron(K, K.conj())
        return E
    
    # 
    def apply_vectorized(self, rho):
        E = self.to_superoperator()
        return E @ rho