import numpy as np

"""
    This is an alternative quantum_operations file that I had initially been making
    when attempting to fix/update our application of gates and error channels to work
    with differing sizes (still not 100%, I think there are edge cases I've missed).
    This file contains very complex operations that involve manual block operations,
    reshaping of the data, etc. This also works with both state vector and density matrix
    forms. I've separated this out because it's complicated, I can't explain it, but it's
    evidently more memory efficient and could be useful.
"""

def apply_qubit(state, gate, targets, n, rng=np.random, metrics_callback=None):
    """
    Apply an arbitrary k-qubit gate to an n-qubit state.

    Args:
      state   : complex ndarray of length 2**n (state vector or density matrix)
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

    U = np.asarray(gate.matrix, dtype=complex)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("gate.matrix must be a square matrix")
    dim = U.shape[0]
    k = int(np.log2(dim))
    if 2**k != dim:
        raise ValueError("gate.matrix dimension must be 2**k")
    
    is_statevector = False
    psi = None
    if state.ndim == 1:
        is_statevector = True
        psi = _apply_unitary_statevector(state, gate.matrix, targets, n, k)
    else:
        psi = _apply_unitary_density(state, gate.matrix, targets, n, k)

    if gate.noise:
        
        for channel in gate.noise:
            channel_targets = _get_channel_targets(channel, targets, gate)
            
            # Apply channel to the full statevector
            for target_set in channel_targets:
                if is_statevector:
                    psi = _apply_channel_to_statevector(
                        psi, channel, target_set, n, rng, metrics_callback
                    )
                else:
                    psi = _apply_channel_to_density(
                        psi, channel, target_set, n
                    )
    
    return psi

def _apply_unitary_statevector(state, U, targets, n, k):
    """
    Apply gate unitary to a state vector state.
    """
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

def _apply_unitary_density(rho, U, targets, n, k):
    """
    Apply unitary to density matrix: ρ' = U ρ U†
    """
    front = list(targets)
    rest = [i for i in range(n) if i not in targets]
    perm = front + rest
    perm_left = perm
    perm_right = [p + n for p in perm]
    full_perm = perm_left + perm_right
    inv_perm = np.argsort(full_perm)
    
    rho_tensor = rho.reshape([2] * (2 * n))
    rho_tensor = np.transpose(rho_tensor, full_perm)
    
    dim_rest = 2 ** (n - k)
    rho_tensor = rho_tensor.reshape(2**k, dim_rest, 2**k, dim_rest)
    
    rho_new = np.zeros_like(rho_tensor)
    for i in range(dim_rest):
        for j in range(dim_rest):
            block = rho_tensor[:, i, :, j]
            rho_new[:, i, :, j] = U @ block @ U.conj().T
    
    rho_new = rho_new.reshape([2] * (2 * n))
    rho_new = np.transpose(rho_new, inv_perm)
    return rho_new.reshape(2**n, 2**n)

def _apply_channel_to_statevector(psi, channel, targets, n, rng, metrics_callback):
    """
    Apply a k-qubit noise channel to specific target qubits in full statevector.
    
    This expands the channel's Kraus operators to act on the full Hilbert space.
    
    Args:
        psi: Full statevector (length 2^n)
        channel: Channel object with Kraus operators
        targets: List of k qubit indices where channel acts
        n: Total number of qubits
        rng: Random number generator
        metrics_callback: Optional callback for metrics
    
    Returns:
        Updated statevector after applying channel
    """
    k = channel.arity
    
    if len(targets) != k:
        raise ValueError(f"Channel requires {k} target qubits, got {len(targets)}")
    
    # Compute probability and resulting state for each Kraus operator
    probs = []
    new_states = []
    
    for K in channel.kraus_ops:
        # Expand K to full Hilbert space
        K_full = _expand_kraus_operator(K, targets, n)
        
        # Apply expanded Kraus operator
        psi_new = K_full @ psi
        
        # Probability = ||K|ψ>||^2
        prob = np.vdot(psi_new, psi_new).real
        probs.append(prob)
        new_states.append(psi_new)
    
    # Normalize probabilities
    total_prob = sum(probs)
    if total_prob <= 0:
        raise RuntimeError("Numerical error: total probability is zero")
    probs = np.array(probs) / total_prob
    
    # Monte Carlo: sample which Kraus operator to apply
    idx = rng.choice(len(probs), p=probs)
    psi_out = new_states[idx]
    
    # Normalize the output state
    norm = np.linalg.norm(psi_out)
    if norm > 0:
        psi_out /= norm
    
    # Record metrics
    if metrics_callback is not None:
        metrics_callback(channel.name, kraus_index=idx)
    
    return psi_out

def _apply_channel_to_density(rho, channel, targets, n, metrics_callback=None):
    """
    Apply noise channel to density matrix: ρ' = Σ_i K_i ρ K_i†
    """
    k = channel.arity
    
    if len(targets) != k:
        raise ValueError(f"Channel requires {k} target qubits, got {len(targets)}")
    
    front = list(targets)
    rest = [i for i in range(n) if i not in targets]
    perm = front + rest
    perm_left = perm
    perm_right = [p + n for p in perm]
    full_perm = perm_left + perm_right
    inv_perm = np.argsort(full_perm)
    
    rho_tensor = rho.reshape([2] * (2 * n))
    rho_tensor = np.transpose(rho_tensor, full_perm)
    
    dim_rest = 2 ** (n - k)
    rho_tensor = rho_tensor.reshape(2**k, dim_rest, 2**k, dim_rest)
    
    rho_new = np.zeros_like(rho_tensor)
    
    for i in range(dim_rest):
        for j in range(dim_rest):
            block = rho_tensor[:, i, :, j]
            
            # Apply ALL Kraus operators
            new_block = np.zeros_like(block)
            for K in channel.kraus_ops:
                new_block += K @ block @ K.conj().T
            
            rho_new[:, i, :, j] = new_block
    
    rho_new = rho_new.reshape([2] * (2 * n))
    rho_new = np.transpose(rho_new, inv_perm)

    return rho_new.reshape(2**n, 2**n)

def _expand_kraus_operator(K, targets, n):
    """
    Expand a k-qubit Kraus operator to act on the full n-qubit Hilbert space.
    
    Args:
        K: Kraus operator (2^k × 2^k matrix)
        targets: List of k qubit indices where K acts (must be contiguous for now)
        n: Total number of qubits
    
    Returns:
        Expanded Kraus operator (2^n × 2^n matrix)
    """
    k = int(np.log2(K.shape[0]))
    
    if len(targets) != k:
        raise ValueError(f"Kraus operator is {k}-qubit but got {len(targets)} targets")
    
    # Check if all qubits are involved (no expansion needed)
    if k == n:
        return K
    
    # Verify targets are contiguous
    targets_sorted = sorted(targets)
    if targets_sorted != list(range(targets_sorted[0], targets_sorted[-1] + 1)):
        raise ValueError(
            f"Non-contiguous target qubits {targets} not supported. "
            f"Targets must be adjacent (e.g., [1,2,3] not [1,3,4])"
        )
    
    result = None
    target_min = targets_sorted[0]
    K_inserted = False
    
    for qubit_idx in range(n):
        if qubit_idx == target_min and not K_inserted:
            current = K
            K_inserted = True
        elif qubit_idx in targets:
            continue
        else:
            current = np.eye(2, dtype=complex)
        
        if result is None:
            result = current
        else:
            result = np.kron(result, current)
    
    return result

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