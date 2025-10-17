import numpy as np

'''
state - state vector
gate - gate being applied to state vector
q - index of the qubit
n - number of qubits

===============================================Explanation=========================================
Instead of making a 2^n * 2^n state matrix, we will reshape the state matrix to a 2^n tensor vector. By doing this, each
index in the state vector corresponds to one of the computational basis states. A gate acts on only one qubit, but the full
state spans all n qubits. So we need to apply the gate matrix to the correct subspace while keeping other qubits unchanged.

To apply the gate:
1. move target axis to front, multiply, and then mvoe it back
2. Apply the gate along the axis using tensordot
3. move axes back to original order by undoing transpose
4. flatten the tensor

'''
def apply_qubit(state, gate, targets, n):
    """
    Apply an arbitrary k-qubit gate to an n-qubit state vector.

    Args:
      state   : 1D complex ndarray of length 2**n (state vector |psi>)
      gate    : object with .matrix (2**k x 2**k) complex ndarray
      targets : list/tuple of k distinct qubit indices in [0, n-1]
                Order matters: targets[0] is the most-significant qubit
                in the gate's basis ordering |q0 q1 ... q_{k-1}>
      n       : total number of qubits

    Returns:
      New state vector (1D ndarray, length 2**n)
    """
    if isinstance(targets, int):
        targets = [targets]
    elif not isinstance(targets, (list, tuple)):
        raise TypeError("targets must be an int or a list/tuple of ints")
    state = np.asarray(state, dtype=complex)
    if state.ndim != 1 or state.size != 2**n:
        raise ValueError(f"state must be 1D of length 2**n (got shape {state.shape}, n={n})")

    U = np.asarray(gate.matrix, dtype=complex)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("gate.matrix must be a square matrix")
    dim = U.shape[0]
    k = int(np.log2(dim))
    if 2**k != dim:
        raise ValueError("gate.matrix dimension must be 2**k")
    if len(targets) != k:
        raise ValueError(f"len(targets) ({len(targets)}) must equal gate qubit count k={k}")
    if len(set(targets)) != k or any(t < 0 or t >= n for t in targets):
        raise ValueError("targets must be distinct indices in [0, n-1]")

    # 1) Reshape |psi> into an n-axis tensor of shape (2,)*n
    psi = state.reshape((2,)*n)

    # 2) Permute so target axes come first (respecting the order in `targets`)
    front = list(targets)
    rest  = [i for i in range(n) if i not in targets]
    perm  = front + rest
    psi = np.transpose(psi, perm)

    # 3) Collapse front k axes to 2**k and the rest to 2**(n-k)
    psi = psi.reshape(2**k, 2**(n-k))

    # 4) Apply the k-qubit gate on the left
    psi = U @ psi

    # 5) Reshape back to (2,)*n and undo the permutation
    psi = psi.reshape((2,)*k + (2,)*(n-k))
    inv_perm = np.argsort(perm)
    psi = np.transpose(psi, inv_perm)

    return psi.reshape(-1)