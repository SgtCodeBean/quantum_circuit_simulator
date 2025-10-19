import numpy as np
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from primitives import *

KrausSet = Sequence[np.ndarray]
Params = Tuple[float, ...]

def _is_power_of_two_dim(dim: int) -> bool:
    return dim > 0 and (dim & (dim - 1)) == 0

def _validate_kraus(kraus: KrausSet, atol: float = 1e-10) -> Tuple[int, int]:
    if not kraus:
        raise ValueError("Kraus set must be non-empty.")
    d0, d1 = kraus[0].shape
    if d0 != d1:
        raise ValueError("Kraus operators must be square.")
    if not _is_power_of_two_dim(d0):
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

# class Channel:
#     """
#     Fixed (non-parametric) CPTP map represented by a Kraus set.
#     """
#     def __init__(self, name: str, kraus_ops: KrausSet):
#         self.name = name
#         self.kraus_ops = [np.array(K, dtype=complex) for K in kraus_ops]
#         self.dim, self.arity = _validate_kraus(self.kraus_ops)
#
#     def apply_density(self, rho: np.ndarray) -> np.ndarray:
#         if rho.shape != (self.dim, self.dim):
#             raise ValueError(f"ρ must be {self.dim}x{self.dim}.")
#         return sum(K @ rho @ K.conj().T for K in self.kraus_ops)
#
#     def apply_statevector(self, psi: np.ndarray, rng=np.random) -> np.ndarray:
#         """
#         Monte Carlo unraveling: sample an outcome i with prob p_i=||K_i|ψ>||^2,
#         then return K_i|ψ>/sqrt(p_i).
#         """
#         psi = np.asarray(psi, dtype=complex)
#         if psi.shape not in [(self.dim,), (self.dim, 1)]:
#             raise ValueError(f"|ψ> must be length {self.dim}.")
#         if psi.ndim == 2:  # column vector
#             psi = psi[:, 0]
#
#         probs = np.array([np.vdot(psi, K.conj().T @ K @ psi).real for K in self.kraus_ops])
#         s = probs.sum()
#         if s <= 0:
#             raise RuntimeError("Numerical issue: total probability is zero.")
#         probs /= s
#         i = rng.choice(len(self.kraus_ops), p=probs)
#         new = self.kraus_ops[i] @ psi
#         n = np.linalg.norm(new)
#         if n == 0:
#             # extremely unlikely unless Kraus set is rank-deficient for this state
#             return new
#         return new / n
#
#     def __str__(self):
#         return f"{self.name} ({self.arity}q) Channel with {len(self.kraus_ops)} Kraus ops"

class ParamChannel:
    """
    Parametric channel: provides kraus_fn(params) -> KrausSet.
    Useful for e.g. depolarizing(p), amplitude_damping(gamma), etc.
    """
    def __init__(self, name: str, arity: int,
                 kraus_fn: Callable[[Params], KrausSet]):
        self.name = name
        self.arity = arity
        self.kraus_fn = kraus_fn

    def instantiate(self, *params: float) -> Channel:
        kraus = self.kraus_fn(params)
        ch = Channel(self.name, kraus)
        if ch.arity != self.arity:
            raise ValueError("Param channel arity mismatch.")
        return ch

class ChannelRegistry:
    def __init__(self, preload_defaults: bool = True):
        self._fixed: Dict[str, Channel] = {}
        self._param: Dict[str, ParamChannel] = {}
        if preload_defaults:
            self._load_defaults()

    def add_fixed(self, channel: Channel, overwrite: bool = False):
        if (not overwrite) and (channel.name in self._fixed or channel.name in self._param):
            raise ValueError(f"Channel '{channel.name}' already exists.")
        self._fixed[channel.name] = channel

    def add_param(self, pch: ParamChannel, overwrite: bool = False):
        if (not overwrite) and (pch.name in self._fixed or pch.name in self._param):
            raise ValueError(f"Channel '{pch.name}' already exists.")
        self._param[pch.name] = pch

    def remove(self, name: str):
        if name in self._fixed:
            del self._fixed[name]
        elif name in self._param:
            del self._param[name]
        else:
            raise KeyError(f"Channel '{name}' not found.")

    def get_fixed(self, name: str) -> Channel:
        if name not in self._fixed:
            raise KeyError(f"Fixed channel '{name}' not found.")
        return self._fixed[name]

    def get_param(self, name: str) -> ParamChannel:
        if name not in self._param:
            raise KeyError(f"Param channel '{name}' not found.")
        return self._param[name]

    def list(self) -> List[str]:
        return sorted(list(self._fixed.keys()) + list(self._param.keys()))

    # ---- defaults ----
    def _load_defaults(self):
        # Register param channels
        self.add_param(ParamChannel("bit_phase_flip", 1, lambda ps: pauli_y_channel(ps[0])))

        # TODO: add more default channels as needed

def test_bit_phase_flip():
    reg = ChannelRegistry()  # loads defaults
    assert "bit_phase_flip" in reg.list(), "bit_phase_flip not found in registry"

    # Instantiate the parametric channel with p
    p = 0.2
    ch = reg.get_param("bit_phase_flip").instantiate(p)

    # --- Density matrix test on |+><+| ---
    # |+> = (|0> + |1>)/sqrt(2)
    ket_plus = (1/np.sqrt(2))*np.array([1, 1], dtype=complex)
    rho_plus = np.outer(ket_plus, ket_plus.conj())

    # Apply the channel
    rho_out = ch.apply_density(rho_plus)

    # Ground truth: (1-p) * rho + p * Y rho Y
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    rho_gt = (1 - p) * rho_plus + p * (Y @ rho_plus @ Y.conj().T)

    assert np.allclose(np.trace(rho_out), 1.0, atol=1e-12), "Trace not preserved"
    assert np.allclose(rho_out, rho_gt, atol=1e-12), "Density result mismatch"

    # Extreme cases p=0 and p=1
    ch0 = reg.get_param("bit_phase_flip").instantiate(0.0)
    ch1 = reg.get_param("bit_phase_flip").instantiate(1.0)
    rho0 = ch0.apply_density(rho_plus)
    rho1 = ch1.apply_density(rho_plus)
    assert np.allclose(rho0, rho_plus, atol=1e-12), "p=0 should be identity"
    assert np.allclose(rho1, Y @ rho_plus @ Y.conj().T, atol=1e-12), "p=1 should be YρY"

    # --- Statevector Monte Carlo test ---
    # Run many trials and estimate flip probability on |0>
    rng = np.random.default_rng(123)
    psi0 = np.array([1, 0], dtype=complex)  # |0>
    nshots = 50_000
    flips = 0
    for _ in range(nshots):
        psi_after = ch.apply_statevector(psi0, rng=rng)
        # A Y error maps |0> -> i|1>, identity keeps |0>
        # Count “flip” if probability of |1> is ~1
        p1 = np.abs(psi_after[1])**2
        flips += (p1 > 0.5)

    est_p = flips / nshots
    assert abs(est_p - p) < 0.01, f"Monte Carlo flip rate off: got {est_p}, expected {p}"

    print("✅ bit_phase_flip channel: density and statevector tests passed.")

if __name__ == '__main__':
    test_bit_phase_flip()