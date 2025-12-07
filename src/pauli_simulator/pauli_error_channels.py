import numpy as np
import time
from .Tableau_Ver2 import Tableau

def build_default_pauli_error_config(
    p_1q: float = 0.001,
    p_2q: float = 0.01,
    p_meas: float = 0.002,
) -> dict:
    """
    Build a reasonable default Pauli error configuration.

    Args:
        p_1q: total Pauli error rate for single-qubit gates
        p_2q: total Pauli error rate (per qubit) for two-qubit gates
        p_meas: total Pauli error rate applied just before measurement

    Returns:
        dict mapping gate name -> (px, py, pz)
    """
    # Symmetric Pauli channels:
    #   p_total = px + py + pz
    p1 = p_1q / 3.0
    p2 = p_2q / 3.0
    pm = p_meas / 3.0

    return {
        # 1-qubit Clifford gates
        "h": (p1, p1, p1),
        "s": (p1, p1, p1),
        "x": (p1, p1, p1),
        "y": (p1, p1, p1),
        "z": (p1, p1, p1),

        # 2-qubit gate (you already call noise on both c and t)
        "cx": (p2, p2, p2),

        # measurement pre-noise
        "measure": (pm, pm, pm),
    }

"""
Class for Pauli error channels
"""

class NoisySimulator:

    def __init__(self, n, error_config=None, enable_metrics=False):
        """
        Args:
            n (int): The number of qubits.
            error_config (dict): A dictionary mapping gate names to error
                rates.
                Example:
                {
                    'h': (0.01, 0.005, 0.005), # (px, py, pz) for H gate
                    'cx': (0.02, 0.02, 0.02),
                    'measure': (0.01, 0, 0)
                }
            enable_metrics (bool): Whether to collect performance metrics
        """
        self.tableau = Tableau(n, enable_metrics=enable_metrics)
        self.n = n

        if error_config is None:
            self.error_config = build_default_pauli_error_config()
        else:
            self.error_config = error_config
        self.rng = np.random.default_rng()

        self._enable_metrics = enable_metrics
        self._metrics = None
        if enable_metrics:
            self._init_metrics()

    def _init_metrics(self):
        """Initialize noise-specific metrics."""
        self._metrics = {
            'noise': {
                'total_noise_events': 0,
                'x_flips': 0,
                'y_flips': 0,
                'z_flips': 0,
                'no_error': 0,
                'noise_time_seconds': 0.0,
                'by_gate': {}
            }
        }

    def _record_noise(self, gate_name, error_type, elapsed):
        """Record a noise application event."""
        if not self._enable_metrics:
            return

        self._metrics['noise']['total_noise_events'] += 1
        self._metrics['noise']['noise_time_seconds'] += elapsed

        if error_type == 'x':
            self._metrics['noise']['x_flips'] += 1
        elif error_type == 'y':
            self._metrics['noise']['y_flips'] += 1
        elif error_type == 'z':
            self._metrics['noise']['z_flips'] += 1
        else:
            self._metrics['noise']['no_error'] += 1

        # Track by gate type
        if gate_name not in self._metrics['noise']['by_gate']:
            self._metrics['noise']['by_gate'][gate_name] = {
                'total': 0, 'x': 0, 'y': 0, 'z': 0, 'none': 0
            }
        self._metrics['noise']['by_gate'][gate_name]['total'] += 1
        if error_type in ['x', 'y', 'z']:
            self._metrics['noise']['by_gate'][gate_name][error_type] += 1
        else:
            self._metrics['noise']['by_gate'][gate_name]['none'] += 1

    def _apply_pauli_noise(self, qubit, probabilities, gate_name='unknown'):
        """
        Applies a Pauli operator to a single qubit.

        Args:
            qubit (int): The qubit to apply noise to.
            probabilities (tuple): A tuple (px, py, pz) of error
                rates.
            gate_name (str): Name of the gate that triggered this noise
        """
        if not probabilities:
            return

        px, py, pz = probabilities
        if px < 0 or py < 0 or pz < 0:
            raise ValueError("Pauli error rates cannot be negative.")

        p_total = px + py + pz
        if p_total == 0:
            return  # No noise
        if p_total > 1:
            raise ValueError("Total Pauli error rates (px+py+pz) cannot exceed 1.")

        start = time.perf_counter() if self._enable_metrics else 0
        error_type = 'none'

        rand_val = self.rng.random()
        if rand_val < px:
            self.tableau.x(qubit)
            error_type = 'x'
        elif rand_val < px + py:
            self.tableau.y(qubit)
            error_type = 'y'
        elif rand_val < p_total:
            self.tableau.z(qubit)
            error_type = 'z'

        if self._enable_metrics:
            self._record_noise(gate_name, error_type, time.perf_counter() - start)

    def h(self, q):
        """Applies a noisy H gate."""
        self.tableau.h(q)
        probs = self.error_config.get('h', (0, 0, 0))
        self._apply_pauli_noise(q, probs, 'h')

    def s(self, q):
        """Applies a noisy S gate."""
        self.tableau.s(q)
        probs = self.error_config.get('s', (0, 0, 0))
        self._apply_pauli_noise(q, probs, 's')

    def cx(self, c, t):
        """Applies a noisy CNOT gate."""
        self.tableau.cx(c, t)
        probs = self.error_config.get('cx', (0, 0, 0))
        self._apply_pauli_noise(c, probs, 'cx')
        self._apply_pauli_noise(t, probs, 'cx')

    def x(self, q):
        """Applies a noisy X gate."""
        self.tableau.x(q)
        probs = self.error_config.get('x', (0, 0, 0))
        self._apply_pauli_noise(q, probs, 'x')

    def y(self, q):
        """Applies a noisy Y gate."""
        self.tableau.y(q)
        probs = self.error_config.get('y', (0, 0, 0))
        self._apply_pauli_noise(q, probs, 'y')

    def z(self, q):
        """Applies a noisy Z gate."""
        self.tableau.z(q)
        probs = self.error_config.get('z', (0, 0, 0))
        self._apply_pauli_noise(q, probs, 'z')

    def measure(self, q):
        """Applies noise before the measurement."""
        probs = self.error_config.get('measure', (0, 0, 0))
        self._apply_pauli_noise(q, probs, 'measure')
        return self.tableau.measure(q)

    def get_metrics(self):
        """
        Get combined metrics from tableau and noise simulation.

        Returns:
            dict: Combined metrics or None if metrics are disabled.
        """
        if not self._enable_metrics:
            return None

        # Get tableau metrics
        tableau_metrics = self.tableau.get_metrics()

        # Merge with noise metrics
        combined = {
            'tableau': tableau_metrics,
            'noise': self._metrics['noise']
        }
        return combined

    def print_metrics(self):
        """Print a formatted summary of all metrics."""
        if not self._enable_metrics:
            print("Metrics collection is disabled.")
            return

        # Print tableau metrics
        self.tableau.print_metrics()

        # Print noise metrics
        n = self._metrics['noise']

        print("\n" + "=" * 50)
        print("NOISE CHANNEL METRICS")
        print("=" * 50)

        print("\n--- Error Summary ---")
        print(f"Total noise events: {n['total_noise_events']}")
        total_errors = n['x_flips'] + n['y_flips'] + n['z_flips']
        print(f"Total errors applied: {total_errors}")
        print(f"  X flips: {n['x_flips']}")
        print(f"  Y flips: {n['y_flips']}")
        print(f"  Z flips: {n['z_flips']}")
        print(f"No error events: {n['no_error']}")

        if n['total_noise_events'] > 0:
            error_rate = total_errors / n['total_noise_events'] * 100
            print(f"Effective error rate: {error_rate:.2f}%")

        if n['by_gate']:
            print("\n--- Errors by Gate Type ---")
            for gate, stats in n['by_gate'].items():
                errors = stats['x'] + stats['y'] + stats['z']
                rate = errors / stats['total'] * 100 if stats['total'] > 0 else 0
                print(f"  {gate.upper()}: {errors}/{stats['total']} ({rate:.1f}%)")

        print(f"\nNoise overhead: {n['noise_time_seconds']*1000:.3f} ms")
        print("=" * 50 + "\n")

    def reset_metrics(self):
        """Reset all metrics to initial values."""
        if self._enable_metrics:
            self._init_metrics()
            self.tableau.reset_metrics()

    def copy(self):
        new_sim = NoisySimulator(self.n, self.error_config, self._enable_metrics)
        new_sim.tableau = self.tableau.copy()

        return new_sim