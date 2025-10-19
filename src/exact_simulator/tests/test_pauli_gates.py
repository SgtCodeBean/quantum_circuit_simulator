import numpy as np
from gates.registry import GateRegistry

reg = GateRegistry()

def test_X_gate():
    rho = np.array([[0, 0], [0, 1]], dtype=complex)
    rho_out = reg.get('x').apply(rho)
    expected = np.array([[1, 0],[0, 0]], dtype=complex)
    np.testing.assert_allclose(rho_out, expected)

def test_Z_gate():
    rho = np.array([[0, 0], [0, 1]], dtype=complex)
    rho_out = reg.get('z').apply(rho)
    expected = np.array([[0, 0],[0, 1]], dtype=complex)
    np.testing.assert_allclose(rho_out, expected)

def test_Y_gate():
    rho = np.array([[0, 0], [0, 1]], dtype=complex)
    rho_out = reg.get('y').apply(rho)
    expected = np.array([[1, 0],[0, 0]], dtype=complex)
    np.testing.assert_allclose(rho_out, expected)