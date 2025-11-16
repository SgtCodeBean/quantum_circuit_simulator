from typing import Dict, Any, List, Optional
import numpy as np

def _fro_norm_sq(x: np.ndarray) -> float:
    return float(np.sum(np.abs(x) ** 2).real)

def apply_channel_to_qubit(state, channel, target, n_qubits, rng, circuit=None):
    """
    Convenience wrapper for a 1-qubit channel acting on a single target qubit.
    """
    if channel.arity != 1:
        raise ValueError(f"apply_channel_to_qubit expects a 1-qubit channel, got arity={channel.arity}")
    return apply_kqubit_channel(state, channel, [target], n_qubits, rng, circuit)

def apply_kqubit_channel(state, channel, targets, n_qubits, rng, circuit=None):
    """
    Apply a k-qubit channel (arity = len(targets)) to a full n-qubit state.

    Supports:
      * statevector: Monte Carlo (single trajectory)
      * density matrix: deterministic Kraus application
    """
    state = np.asarray(state, dtype=complex)
    d_full = 2 ** n_qubits
    k = channel.arity

    if len(targets) != k:
        raise ValueError(f"Channel arity {k} does not match len(targets)={len(targets)}")

    # ----- STATEVECTOR CASE -----
    if state.shape in [(d_full,), (d_full, 1)]:
        if state.ndim == 2:
            psi = state[:, 0]
        else:
            psi = state

        # reshape psi into tensor of shape (2,)*n, then move target axes to the front
        axes = list(range(n_qubits))
        targets = list(targets)
        front_axes = targets + [i for i in axes if i not in targets]
        psi_tensor = psi.reshape((2,) * n_qubits).transpose(front_axes)

        # treat as a matrix: (2^k, 2^(n-k))
        d_sys = 2 ** k
        d_env = 2 ** (n_qubits - k)
        psi_block = psi_tensor.reshape(d_sys, d_env)

        # Monte Carlo over Kraus operators Ki (each d_sys x d_sys)
        probs = []
        candidates = []
        for K in channel.kraus_ops:
            if K.shape != (d_sys, d_sys):
                raise ValueError(f"Kraus op has shape {K.shape}, expected {(d_sys, d_sys)}")
            tmp = K @ psi_block
            p = _fro_norm_sq(tmp)
            probs.append(p)
            candidates.append(tmp)

        total = sum(probs)
        if total <= 0:
            raise RuntimeError("Total probability from Kraus ops is zero (numerical issue?)")
        probs = [p / total for p in probs]

        idx = rng.choice(len(channel.kraus_ops), p=probs)
        psi_block_out = candidates[idx]
        # normalize
        norm = np.sqrt(_fro_norm_sq(psi_block_out))
        if norm > 0:
            psi_block_out /= norm

        # record metrics if circuit available
        if circuit is not None and hasattr(circuit, "record_channel_hit"):
            circuit.record_channel_hit(channel.name, kraus_index=idx)

        # reshape back to full statevector of length 2^n
        psi_tensor_out = psi_block_out.reshape((2,) * n_qubits)
        # invert the transpose: compute inverse permutation
        inv_axes = np.argsort(front_axes)
        psi_final = psi_tensor_out.transpose(inv_axes).reshape(d_full)

        return psi_final

    # ----- DENSITY MATRIX CASE -----
    elif state.shape == (d_full, d_full):
        rho = state

        # reshape rho into tensor: (2,)*n for rows × (2,)*n for cols
        rho_tensor = rho.reshape((2,) * n_qubits * 2)

        axes = list(range(n_qubits))
        targets = list(targets)
        env_axes = [i for i in axes if i not in targets]

        # new order for row indices: targets first, then env
        row_order = targets + env_axes
        col_order = [a + n_qubits for a in targets] + [a + n_qubits for a in env_axes]
        full_order = row_order + col_order

        rho_perm = rho_tensor.transpose(full_order)
        d_sys = 2 ** k
        d_env = 2 ** (n_qubits - k)
        # reshape into 4-index tensor: (sys, env, sys', env')
        rho_block = rho_perm.reshape(d_sys, d_env, d_sys, d_env)

        rho_out_block = np.zeros_like(rho_block)
        for K in channel.kraus_ops:
            if K.shape != (d_sys, d_sys):
                raise ValueError(f"Kraus op has shape {K.shape}, expected {(d_sys, d_sys)}")
            # Apply K on system index (row) and K† on system index (col)
            # rho_block indices: [s, e, s', e']
            tmp = np.tensordot(K, rho_block, axes=([1], [0]))      # K_{a,s} rho_{s,e,s',e'} -> tmp_{a,e,s',e'}
            tmp2 = np.tensordot(K.conj(), tmp, axes=([1], [2]))    # K*_{b,s'} tmp_{a,e,s',e'} -> out_{a,e,b,e'}
            rho_out_block += tmp2

        # reshape back to full density matrix
        rho_out_perm = rho_out_block.reshape((2,) * n_qubits * 2)
        # invert permutation
        inv_full_order = np.argsort(full_order)
        rho_final = rho_out_perm.transpose(inv_full_order).reshape(d_full, d_full)

        if circuit is not None and hasattr(circuit, "record_channel_hit"):
            # we don’t track specific Kraus index here; it’s averaged
            circuit.record_channel_hit(channel.name)

        return rho_final

    else:
        raise ValueError(
            f"State has shape {state.shape}, expected statevector (2^{n_qubits},) "
            f"or density matrix (2^{n_qubits}, 2^{n_qubits})."
        )

class NoiseModel:
    def __init__(self, channel_registry, default_spec: Optional[Dict[str, Any]] = None,
                 per_gate_specs: Optional[Dict[str, Optional[Dict[str, Any]]]] = None):
        """
        channel_registry: your ChannelRegistry instance
        default_spec: spec used when gate not in per_gate_specs
        per_gate_specs: map gate_name -> spec or None (None = no noise)
          spec example:
            {
              "type": "bit_phase_flip",
              "params": (0.1,),    # passed into ParamChannel.instantiate
              "scope": "per_qubit" # or "per_gate"
            }
        """
        self.chan_reg = channel_registry
        self.default_spec = default_spec or {}
        self.per_gate_specs = per_gate_specs or {}

    def _spec_for_gate(self, gate_name: str) -> Optional[Dict[str, Any]]:
        if gate_name in self.per_gate_specs:
            return self.per_gate_specs[gate_name]
        return self.default_spec or None

    def _build_channel(self, spec: Dict[str, Any]):
        ch_type = spec["type"]              # e.g. "bit_phase_flip"
        params = spec.get("params", ())     # tuple of floats
        pch = self.chan_reg.get_param(ch_type)
        return pch.instantiate(*params)

    def apply_after_gate(self, state: np.ndarray, gate_name: str,
                         targets: List[int], n_qubits: int,
                         rng=None, circuit=None) -> np.ndarray:
        """
        state: full n-qubit statevector or density matrix
        gate_name: e.g. "x", "h", "cx"
        targets: list of qubit indices this gate acted on
        n_qubits: total number of qubits
        rng: numpy Generator for Monte Carlo
        circuit: optional, to record metrics via circuit.record_channel_hit()
        """
        spec = self._spec_for_gate(gate_name)
        if spec is None:
            return state  # no noise for this gate

        scope = spec.get("scope", "per_qubit")
        if rng is None:
            rng = np.random.default_rng()

        # 1) Build the channel (assume 1-qubit for now)
        ch = self._build_channel(spec)

        # 2) Apply it according to scope
        if scope == "per_qubit":
            for q in targets:
                state = apply_channel_to_qubit(state, ch, q, n_qubits, rng, circuit)
            return state
        elif scope == "per_gate":
            # for future multi-qubit channel support
            state = apply_kqubit_channel(state, ch, targets, n_qubits, rng, circuit)
            return state
        else:
            raise ValueError(f"Unknown noise scope: {scope}")