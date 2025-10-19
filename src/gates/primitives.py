import numpy as np

def pauli_gates():
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    return X, Y, Z

def hadamard_gate():
    return (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])

def toffoli_gate():
    toffoli_matrix = np.eye(8, dtype=complex)
    toffoli_matrix[6, 6] = 0
    toffoli_matrix[7, 7] = 0
    toffoli_matrix[6, 7] = 1
    toffoli_matrix[7, 6] = 1
    return toffoli_matrix

# TODO: Add more primitive gates as needed