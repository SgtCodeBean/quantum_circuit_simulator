import numpy as np
from error_channels.Channel import Channel
from error_channels.ChannelRegistry import ParamChannel, ChannelRegistry

def test_statevector():
    reg = ChannelRegistry()
    assert "bit_flip" in reg.list()
    p = 0.2
    X_channel = reg.get_param("bit_flip").instantiate(p)
    vector = np.array([[1, 0]], dtype="complex").flatten(order='F')
    n_shots = 1000
    flips = 0
    rng = np.random.default_rng(123)
    for _ in range(n_shots):
        psi_after = X_channel.apply_statevector(vector, rng=rng)
        p1 = np.abs(psi_after[1])**2
        flips += (p1 > 0.5)
    flip_p = flips/n_shots
    assert abs(flip_p - p) < 0.01, f"Monte Carlo flip rate off: got {flip_p}, expected {p}"