import numpy as np
from error_channels.Kraus import Kraus
from gates.PauliGates import PauliGates

def test_bit_flip():
    I = np.eye(2, dtype=complex)
    p = 0.1
    ops = [np.sqrt(1 - p) * I, np.sqrt(p) * PauliGates.X.matrix]
    kraus = Kraus(ops)
    rho = np.array([[1, 0],[0, 0]])
    np.testing.assert_allclose(
        kraus.apply(rho=rho), 
        np.array([[0.9, 0.0], [0.0, 0.1]], dtype=complex)
    )

def test_vector_bit_flip():
    I = np.eye(2, dtype=complex)
    p = 0.1
    ops = [np.sqrt(1 - p) * I, np.sqrt(p) * PauliGates.X.matrix]
    kraus = Kraus(ops)
    rho_vector = np.array([[1, 0], [0, 0]]).flatten(order='F')
    np.testing.assert_allclose(
        kraus.apply_vectorized(rho=rho_vector),
        np.array([[0.9, 0.0], [0.0, 0.1]], dtype=complex).flatten(order='F')
    )

def test_vector_multi_kraus():
    p_X = 0.1
    p_Z = 0.05
    I = np.eye(2, dtype=complex)
    ops_X = [np.sqrt(1 - p_X) * I, np.sqrt(p_X) * PauliGates.X.matrix]
    ops_Z = [np.sqrt(1 - p_Z) * I, np.sqrt(p_Z) * PauliGates.Z.matrix]
    kraus_X = Kraus(ops_X)
    kraus_Z = Kraus(ops_Z)
    rho_vector = (0.5) * np.array([[1, 1], [1, 1]]).flatten(order='F')
    expected = np.array([[0.5, 0.45], [0.45, 0.5]], dtype=complex).flatten(order='F')
    rho_X_out = kraus_X.apply_vectorized(rho_vector)
    rho_Z_out = kraus_Z.apply_vectorized(rho_X_out)
    np.testing.assert_allclose(rho_Z_out, expected)