import numpy as np

def is_power_of_two_dim(dim: int) -> bool:
    return dim > 0 and (dim & (dim - 1)) == 0

def is_unitary(U: np.ndarray, tol: float = 1e-10) -> bool:
    return np.allclose(U.conj().T @ U, np.eye(U.shape[0], dtype=U.dtype), atol=tol)