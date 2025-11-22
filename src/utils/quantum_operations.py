import numpy as np

'''
state - state
gate - gate being applied to state
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
    Apply an arbitrary k-qubit gate to an n-qubit state (statevector or density matrix).
    Noise-free: pure unitary embedding.
    """
    if isinstance(targets, int):
        targets = [targets]
    elif not isinstance(targets, (list, tuple)):
        raise TypeError("targets must be an int or a list/tuple of ints")

    state = np.asarray(state, dtype=complex)

    U = np.asarray(gate.matrix, dtype=complex)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("gate.matrix must be a square matrix")
    dim = U.shape[0]
    k = int(np.log2(dim))
    if 2**k != dim:
        raise ValueError("gate.matrix dimension must be 2**k")

    # sanity check targets vs gate arity
    if len(targets) != k:
        raise ValueError(f"len(targets)={len(targets)} must equal gate arity k={k}")

    if state.ndim == 1:
        # statevector case
        return _apply_unitary_statevector(state, U, targets, n, k)
    elif state.ndim == 2 and state.shape == (2**n, 2**n):
        # density matrix case
        return _apply_unitary_density(state, U, targets, n, k)
    else:
        raise ValueError(
            f"State has shape {state.shape}, expected vector (2**{n},) "
            f"or density matrix (2**{n}, 2**{n})."
        )

def _apply_unitary_statevector(state, U, targets, n, k):
    U_full = _expand_kraus_operator(U, targets, n)
    return U_full @ state

def _apply_unitary_density(rho, U, targets, n, k):
    U_full = _expand_kraus_operator(U, targets, n)
    return U_full @ rho @ U_full.conj().T

def _apply_channel_to_statevector(psi, channel, targets, n, rng, metrics_callback):
    probs = []
    new_states = []

    for _, K in enumerate(channel.kraus_ops):
        K_full = _expand_kraus_operator(K, targets, n)
        psi_new = K_full @ psi
        p = np.vdot(psi_new, psi_new).real
        probs.append(p)
        new_states.append(psi_new)

    probs = np.array(probs)
    probs /= probs.sum()

    i = rng.choice(len(probs), p=probs)
    psi_out = new_states[i] / np.linalg.norm(new_states[i])

    if metrics_callback:
        metrics_callback(channel.name, kraus_index=i)

    return psi_out


def _apply_channel_to_density(rho, channel, targets, n, metrics_callback):
    rho_new = np.zeros_like(rho)
    for K in channel.kraus_ops:
        K_full = _expand_kraus_operator(K, targets, n)
        rho_new += K_full @ rho @ K_full.conj().T
    return rho_new


def _expand_kraus_operator(op, targets, n):
    """
    Expand a k-qubit operator 'op' to n qubits for arbitrary (possibly non-contiguous) targets.
    """
    k = len(targets)
    targets = list(targets)

    # Build permutation that moves target axes to front
    rest = [i for i in range(n) if i not in targets]
    perm = targets + rest

    # Build full operator using kronecker products of identities
    I = np.eye(2, dtype=complex)
    full = op
    for _ in rest:
        full = np.kron(full, I)

    # Now permute operator axes back to correct positions.
    # Create inverse permutation
    inv = np.argsort(perm)

    # Reshape to 2n × 2n matrix
    full = full.reshape([2]*n + [2]*n)
    full = np.transpose(full, inv.tolist() + (inv+n).tolist())
    full = full.reshape(2**n, 2**n)

    return full

def _get_channel_targets(channel, gate_targets, gate):
    """
    Determine which qubits a noise channel should act on.
    
    Logic:
    - If channel.arity == gate.arity: Apply to entire gate subspace once
    - If channel.arity == 1: Apply to each gate target qubit independently
    - Otherwise: Not yet supported
    
    Args:
        channel: Channel object
        gate_targets: List of qubits the gate acts on
        gate: Gate object
    
    Returns:
        List of target sets, e.g., [[0], [1]] for 1q channel on 2q gate
    """
    if channel.arity == gate.arity:
        # Channel matches gate arity - apply to entire gate subspace
        return [gate_targets]
    
    elif channel.arity == 1:
        # 1-qubit channel - apply independently to each gate target
        return [[q] for q in gate_targets]
    
    else:
        # Future: Could support other combinations like:
        # - 2q channel on 3q gate (apply to first 2 qubits, then last 2, etc.)
        # - 1q channel on specific subset of gate qubits
        raise ValueError(
            f"Channel {channel.name} has arity {channel.arity}, "
            f"gate {gate.name} has arity {gate.arity}. "
            f"Currently only support: "
            f"(1) channel arity == gate arity, or "
            f"(2) channel arity == 1 (applied to each gate qubit)."
        )