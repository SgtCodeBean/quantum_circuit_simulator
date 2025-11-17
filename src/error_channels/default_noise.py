from error_channels.ChannelRegistry import ChannelRegistry
from error_channels.noise_model import NoiseModel

def build_default_noise_model() -> NoiseModel:
    """
    Build a default NoiseModel with:
      - depolarizing noise p=0.05 on all gates by default
      - bit-flip noise p=0.01 on X gates
      - no noise on Toffoli (ccx) gates
    """
    # 1) Build the channel registry with your default param channels
    chan_reg = ChannelRegistry(preload_defaults=True)

    # 2) Define a default noise spec:
    #    - apply a small depolarizing noise p=0.05
    #    - scope "per_qubit" = apply to each target qubit of the gate
    default_spec = {
        "type": "depolarizing",
        "params": (0.05,),      # p = 0.05
        "scope": "per_qubit",
    }

    # 3) Optionally override noise for specific gates
    per_gate_specs = {
        # X gates: pure bit-flip noise with smaller p
        "x": {"type": "bit_flip", "params": (0.01,), "scope": "per_qubit"},
        "ccx": None,
    }

    noise_model = NoiseModel(
        channel_registry=chan_reg,
        default_spec=default_spec,
        per_gate_specs=per_gate_specs,
    )

    return noise_model