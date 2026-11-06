"""Energy accounting shared by every method.

Two radio profiles are provided:
* ``first_order``: the first-order radio model used by most clustering papers,
  kept so that our numbers can be compared with that literature;
* ``cc2420``: a fixed-power radio whose cost depends on airtime only
  (CC2420 datasheet, 0 dBm, 3 V), used as a robustness check.

Every joule spent in the simulator goes through :class:`EnergyModel`, so the
accounting boundary is the same for all methods.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Packets:
    """Packet sizes in bits (TinyOS-style 11-byte header)."""
    header: int = 11 * 8
    update_payload: int = 5 * 8      # node id 2 B + value 2 B + sequence 1 B
    ack: int = 5 * 8
    control: int = 24 * 8            # status / join / schedule packet

    @property
    def update(self) -> int:
        return self.header + self.update_payload

    def aggregate(self, k: int) -> int:
        """CH-to-sink packet carrying k updates."""
        return self.header + k * self.update_payload


@dataclass
class EnergyModel:
    profile: str = "first_order"
    e_elec: float = 50e-9            # J/bit
    eps_fs: float = 10e-12           # J/bit/m^2
    eps_mp: float = 0.0013e-12       # J/bit/m^4
    e_da: float = 5e-9               # J/bit/signal, aggregation at the CH
    p_tx: float = 3.0 * 17.4e-3      # CC2420 profile, W
    p_rx: float = 3.0 * 18.8e-3
    bitrate: float = 250e3
    cca_fraction: float = 0.10       # idle check of one TDMA slot, share of a packet airtime
    e_cycle: float = 1.5e-9          # MSP430F1611-class MCU, J per cycle
    e_sense: float = 0.0             # per sample, identical for every method
    p_floor: float = 0.0             # sleep floor per round, identical for every method
    packets: Packets = field(default_factory=Packets)

    @property
    def d0(self) -> float:
        return float(np.sqrt(self.eps_fs / self.eps_mp))

    def tx(self, bits, d):
        bits = np.asarray(bits, dtype=float)
        d = np.asarray(d, dtype=float)
        if self.profile == "cc2420":
            return self.p_tx * bits / self.bitrate + 0.0 * d
        amp = np.where(d < self.d0, self.eps_fs * d ** 2, self.eps_mp * d ** 4)
        return bits * (self.e_elec + amp)

    def rx(self, bits):
        bits = np.asarray(bits, dtype=float)
        if self.profile == "cc2420":
            return self.p_rx * bits / self.bitrate
        return bits * self.e_elec

    def listen(self, n_slots):
        return self.cca_fraction * self.rx(self.packets.update) * np.asarray(n_slots, dtype=float)

    def cpu(self, cycles):
        return self.e_cycle * np.asarray(cycles, dtype=float)

    def aggregation(self, n_signals):
        return self.e_da * self.packets.update_payload * np.asarray(n_signals, dtype=float)
