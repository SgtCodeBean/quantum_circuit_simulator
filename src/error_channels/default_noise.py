from error_channels.ChannelRegistry import ChannelRegistry
from error_channels.noise_model import NoiseModel

def build_default_noise_model() -> NoiseModel:
    """
    Reasonable default noise model:

      - Single-qubit gates: mild depolarizing noise (p_1q)
      - Multi-qubit gates (cx, ccx): stronger depolarizing noise (p_2q, p_3q)
      - Scope:
          * single-qubit: "per_qubit" (each target independently)
          * multi-qubit:  "per_gate"  (correlated error on all targets)

      You can tweak p_1q, p_2q, p_3q to make the simulator "noisier" or "cleaner".
    """
    chan_reg = ChannelRegistry(preload_defaults=True)

    # Global knobs
    p_1q = 0.002   # single-qubit depolarizing probability
    p_2q = 0.006   # two-qubit gate depolarizing probability
    p_3q = 0.010   # three-qubit gate depolarizing probability

    # Default: all gates get mild single-qubit depolarizing noise, per target qubit
    default_spec = {
        "type": "depolarizing",
        "params": (p_1q,),
        "scope": "per_qubit",
    }

    # Override for specific gates
    per_gate_specs = {
        "x":  {"type": "depolarizing", "params": (p_1q,), "scope": "per_qubit"},
        "y":  {"type": "depolarizing", "params": (p_1q,), "scope": "per_qubit"},
        "z":  {"type": "depolarizing", "params": (p_1q,), "scope": "per_qubit"},
        "h":  {"type": "depolarizing", "params": (p_1q,), "scope": "per_qubit"},
        "s":  {"type": "phase_damping", "params": (p_1q,), "scope": "per_qubit"},

        # Entangling gates: apply depolarizing noise to *all qubits at once*
        "cx": {
            "type": "depolarizing",
            "params": (p_2q,),
            "scope": "per_gate",
        },
        "ccx": {
            "type": "depolarizing",
            "params": (p_3q,),
            "scope": "per_gate",
        },
    }

    return NoiseModel(
        channel_registry=chan_reg,
        default_spec=default_spec,
        per_gate_specs=per_gate_specs,
    )