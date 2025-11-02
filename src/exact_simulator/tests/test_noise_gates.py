from gates.registry import Gate
from error_channels.Channel import Channel
import numpy as np

class MockChannel(Channel):
    """Simple deterministic mock of Channel for testing Gate noise integration."""
    def __init__(self, dim=2):
        # identity Kraus op for simplicity
        self.name = "mock"
        self.kraus_ops = [np.eye(dim, dtype=complex)]
        self.dim = dim
        self.arity = int(np.log2(dim))

    def apply_density(self, rho):
        self.last_input = rho
        return rho  # no change

    def apply_statevector(self, psi, rng=np.random):
        self.last_input = psi
        return psi  # no change

def test_apply_statevector_with_mock_noise():
    U = np.array([[0, 1], [1, 0]])  # X gate
    ch = MockChannel(dim=2)
    g = Gate("X", U, noise=[ch])
    psi = np.array([1, 0], dtype=complex)
    out = g.apply(psi)
    # mock noise returns unmodified state
    assert np.allclose(np.abs(out), [0, 1])

