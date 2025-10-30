import numpy as np
from typing import Callable, Dict, List
from utils.utils import Params, KrausSet
from error_channels.Channel import Channel
from error_channels.primitives import *
from error_channels.BitFlipChannel import BitFlipChannel

class ParamChannel:
    """
    Parametric channel: provides kraus_fn(params) -> KrausSet.
    Useful for e.g. depolarizing(p), amplitude_damping(gamma), etc.
    """
    def __init__(self, name: str, arity: int,
                 kraus_fn: Callable[[Params], KrausSet]):
        self.name = name
        self.arity = arity
        self.kraus_fn = kraus_fn

    def instantiate(self, *params: float) -> Channel:
        kraus = self.kraus_fn(params)
        ch = Channel(self.name, kraus)
        if ch.arity != self.arity:
            raise ValueError("Param channel arity mismatch.")
        return ch

class ChannelRegistry:
    def __init__(self, preload_defaults: bool = True):
        self._fixed: Dict[str, Channel] = {}
        self._param: Dict[str, ParamChannel] = {}
        if preload_defaults:
            self._load_defaults()

    def add_fixed(self, channel: Channel, overwrite: bool = False):
        if (not overwrite) and (channel.name in self._fixed or channel.name in self._param):
            raise ValueError(f"Channel '{channel.name}' already exists.")
        self._fixed[channel.name] = channel

    def add_param(self, pch: ParamChannel, overwrite: bool = False):
        if (not overwrite) and (pch.name in self._fixed or pch.name in self._param):
            raise ValueError(f"Channel '{pch.name}' already exists.")
        self._param[pch.name] = pch

    def remove(self, name: str):
        if name in self._fixed:
            del self._fixed[name]
        elif name in self._param:
            del self._param[name]
        else:
            raise KeyError(f"Channel '{name}' not found.")

    def get_fixed(self, name: str) -> Channel:
        if name not in self._fixed:
            raise KeyError(f"Fixed channel '{name}' not found.")
        return self._fixed[name]

    def get_param(self, name: str) -> ParamChannel:
        if name not in self._param:
            raise KeyError(f"Param channel '{name}' not found.")
        return self._param[name]

    def list(self) -> List[str]:
        return sorted(list(self._fixed.keys()) + list(self._param.keys()))

    # ---- defaults ----
    def _load_defaults(self):
        # Register param channels
        self.add_param(ParamChannel("bit_phase_flip", 1, lambda ps: pauli_y_channel(ps[0])))
        self.add_param(ParamChannel("bit_flip", 1, lambda ps: BitFlipChannel(ps[0]).kraus))
        # TODO: add more default channels as needed