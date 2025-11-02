import numpy as np
from utils.utils import KrausSet
from typing import Tuple
from utils.linalg import is_power_of_two_dim

def _validate_kraus(kraus: KrausSet, atol: float = 1e-10) -> Tuple[int, int]:
        if not kraus:
            raise ValueError("Kraus set must be non-empty.")
        d0, d1 = kraus[0].shape
        if d0 != d1:
            raise ValueError("Kraus operators must be square.")
        if not is_power_of_two_dim(d0):
            raise ValueError("Dimension must be 2^n.")
        d = d0
        total = np.zeros((d, d), dtype=complex)
        for K in kraus:
            if K.shape != (d, d):
                raise ValueError("All Kraus ops must have identical shape.")
            total += K.conj().T @ K
        if not np.allclose(total, np.eye(d, dtype=complex), atol=atol):
            raise ValueError("Kraus operators are not complete: Σ K†K != I.")
        arity = int(np.log2(d))
        return d, arity

class Channel:
    """
    Fixed (non-parametric) CPTP map represented by a Kraus set.
    """
    def __init__(self, name: str, kraus_ops: KrausSet):
        self.name = name
        self.kraus_ops = [np.array(K, dtype=complex) for K in kraus_ops]
        self.dim, self.arity = _validate_kraus(self.kraus_ops)

    def apply_density(self, rho: np.ndarray) -> np.ndarray:
        if rho.shape != (self.dim, self.dim):
            raise ValueError(f"ρ must be {self.dim}x{self.dim}.")
        return sum(K @ rho @ K.conj().T for K in self.kraus_ops)

    def apply_statevector(self, psi: np.ndarray, rng=np.random) -> np.ndarray:
        """
        Monte Carlo unraveling: sample an outcome i with prob p_i=||K_i|ψ>||^2,
        then return K_i|ψ>/sqrt(p_i).
        """
        psi = np.asarray(psi, dtype=complex)
        if psi.shape not in [(self.dim,), (self.dim, 1)]:
            raise ValueError(f"|ψ> must be length {self.dim}.")
        if psi.ndim == 2:  # column vector
            psi = psi[:, 0]

        probs = np.array([np.vdot(psi, K.conj().T @ K @ psi).real for K in self.kraus_ops])
        s = probs.sum()
        if s <= 0:
            raise RuntimeError("Numerical issue: total probability is zero.")
        probs /= s
        i = rng.choice(len(self.kraus_ops), p=probs)
        new = self.kraus_ops[i] @ psi
        n = np.linalg.norm(new)
        if n == 0:
            # extremely unlikely unless Kraus set is rank-deficient for this state
            return new
        return new / n

    def __str__(self):
        return f"{self.name} ({self.arity}q) Channel with {len(self.kraus_ops)} Kraus ops"