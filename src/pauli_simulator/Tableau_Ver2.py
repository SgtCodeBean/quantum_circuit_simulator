import numpy as np
import time

"""
Optimized Tableau class for efficient Clifford circuit simulation.
(Measurement logic has been corrected)
"""


class Tableau:
    __slots__ = ('n', '_x', '_z', '_r', '_metrics', '_enable_metrics')

    def __init__(self, n, enable_metrics=False):
        """
        Initialize tableau for n qubits in |0...0⟩ state.

        Args:
            n (int): Number of qubits
            enable_metrics (bool): Whether to collect performance metrics
        """
        self.n = n
        self._x = np.zeros((2 * n, n), dtype=np.uint8)
        self._z = np.zeros((2 * n, n), dtype=np.uint8)
        self._r = np.zeros(2 * n, dtype=np.uint8)
        for i in range(n):
            self._x[i, i] = 1
            self._z[n + i, i] = 1

        self._enable_metrics = enable_metrics
        self._metrics = None
        if enable_metrics:
            self._init_metrics()

    def _init_metrics(self):
        """Initialize metrics dictionary."""
        self._metrics = {
            'execution': {
                'total_time_seconds': 0.0,
                'start_time': None,
                'end_time': None
            },
            'operations': {
                'gate_count': 0,
                'measurement_count': 0,
                'total_operations': 0,
                'gates_by_type': {
                    'h': 0, 's': 0, 'cx': 0, 'x': 0, 'y': 0, 'z': 0
                }
            },
            'timing': {
                'gate_time_seconds': 0.0,
                'measurement_time_seconds': 0.0
            },
            'measurements': {
                'deterministic': 0,
                'probabilistic': 0,
                'outcomes': {0: 0, 1: 0}
            }
        }

    def _record_gate(self, gate_name, elapsed):
        """Record a gate operation."""
        if not self._enable_metrics:
            return
        self._metrics['operations']['gate_count'] += 1
        self._metrics['operations']['total_operations'] += 1
        self._metrics['operations']['gates_by_type'][gate_name] += 1
        self._metrics['timing']['gate_time_seconds'] += elapsed

    def _record_measurement(self, outcome, is_deterministic, elapsed):
        """Record a measurement operation."""
        if not self._enable_metrics:
            return
        self._metrics['operations']['measurement_count'] += 1
        self._metrics['operations']['total_operations'] += 1
        self._metrics['timing']['measurement_time_seconds'] += elapsed
        if is_deterministic:
            self._metrics['measurements']['deterministic'] += 1
        else:
            self._metrics['measurements']['probabilistic'] += 1
        self._metrics['measurements']['outcomes'][outcome] += 1

    def h(self, q):
        """Hadamard gate"""
        start = time.perf_counter() if self._enable_metrics else 0
        for i in range(2 * self.n):
            self._r[i] ^= self._x[i, q] & self._z[i, q]
            self._x[i, q], self._z[i, q] = self._z[i, q], self._x[i, q]
        if self._enable_metrics:
            self._record_gate('h', time.perf_counter() - start)

    def s(self, q):
        """Phase gate"""
        start = time.perf_counter() if self._enable_metrics else 0
        for i in range(2 * self.n):
            self._r[i] ^= self._x[i, q] & self._z[i, q]
            self._z[i, q] ^= self._x[i, q]
        if self._enable_metrics:
            self._record_gate('s', time.perf_counter() - start)

    def cx(self, c, t):
        """CNOT gate: control c, target t"""
        start = time.perf_counter() if self._enable_metrics else 0
        for i in range(2 * self.n):
            self._r[i] ^= self._x[i, c] & self._z[i, t] & (self._x[i, t] ^ self._z[i, c] ^ 1)
            self._x[i, t] ^= self._x[i, c]
            self._z[i, c] ^= self._z[i, t]
        if self._enable_metrics:
            self._record_gate('cx', time.perf_counter() - start)

    def x(self, q):
        """Pauli X: bit flip"""
        start = time.perf_counter() if self._enable_metrics else 0
        for i in range(2 * self.n):
            self._r[i] ^= self._z[i, q]
        if self._enable_metrics:
            self._record_gate('x', time.perf_counter() - start)

    def y(self, q):
        """Pauli Y: bit + phase flip"""
        start = time.perf_counter() if self._enable_metrics else 0
        for i in range(2 * self.n):
            self._r[i] ^= self._x[i, q] ^ self._z[i, q]
        if self._enable_metrics:
            self._record_gate('y', time.perf_counter() - start)

    def z(self, q):
        """Pauli Z: phase flip"""
        start = time.perf_counter() if self._enable_metrics else 0
        for i in range(2 * self.n):
            self._r[i] ^= self._x[i, q]
        if self._enable_metrics:
            self._record_gate('z', time.perf_counter() - start)

    # --- Corrected Measurement Logic (from Aaronson-Gottesman paper) ---

    def measure(self, q):
        """
        Measure qubit q in the Z basis.
        Returns the measurement outcome (0 or 1).
        """
        start = time.perf_counter() if self._enable_metrics else 0

        # Case I: Random outcome
        p = -1
        for i in range(self.n, 2 * self.n):
            if self._x[i, q]:
                p = i
                break

        if p != -1:
            outcome = np.random.randint(0, 2)

            for i in range(2 * self.n):
                if i != p and self._x[i, q]:
                    self._rowsum(i, p)

            self._x[p - self.n] = self._x[p]
            self._z[p - self.n] = self._z[p]
            self._r[p - self.n] = self._r[p]

            self._x[p, :] = 0
            self._z[p, :] = 0
            self._z[p, q] = 1
            self._r[p] = outcome

            if self._enable_metrics:
                self._record_measurement(outcome, False, time.perf_counter() - start)
            return outcome

        # Case II: Deterministic outcome
        else:
            temp_x = np.zeros(self.n, dtype=np.uint8)
            temp_z = np.zeros(self.n, dtype=np.uint8)
            temp_r = 0

            for i in range(self.n):
                if self._x[i, q]:
                    x1 = self._x[i + self.n]
                    z1 = self._z[i + self.n]
                    x2 = temp_x
                    z2 = temp_z

                    g = (x1 & z1) * (x2 ^ z2) + \
                        (x2 & z2) * (x1 ^ z1 ^ 1) + \
                        (x1 & z2) * (x2 & z1)

                    g_sum = (2 * self._r[i + self.n] + 2 * temp_r + np.sum(g)) % 4
                    temp_r = g_sum >> 1
                    temp_x ^= x1
                    temp_z ^= z1

            if self._enable_metrics:
                self._record_measurement(temp_r, True, time.perf_counter() - start)
            return temp_r

    def get_metrics(self):
        """
        Get the collected metrics.

        Returns:
            dict: Metrics dictionary or None if metrics are disabled.
        """
        if not self._enable_metrics:
            return None

        total_time = (self._metrics['timing']['gate_time_seconds'] +
                     self._metrics['timing']['measurement_time_seconds'])
        self._metrics['execution']['total_time_seconds'] = total_time

        return self._metrics.copy()

    def print_metrics(self):
        """Print a formatted summary of metrics."""
        if not self._enable_metrics:
            print("Metrics collection is disabled.")
            return

        m = self._metrics
        total_time = (m['timing']['gate_time_seconds'] +
                     m['timing']['measurement_time_seconds'])

        print("\n" + "=" * 50)
        print("PAULI SIMULATOR METRICS")
        print("=" * 50)

        print(f"\nTableau size: {self.n} qubits")

        print("\n--- Operations ---")
        print(f"Total operations: {m['operations']['total_operations']}")
        print(f"Gate count: {m['operations']['gate_count']}")
        print(f"Measurement count: {m['operations']['measurement_count']}")

        print("\n--- Gates by Type ---")
        for gate, count in m['operations']['gates_by_type'].items():
            if count > 0:
                print(f"  {gate.upper()}: {count}")

        print("\n--- Measurements ---")
        print(f"Deterministic: {m['measurements']['deterministic']}")
        print(f"Probabilistic: {m['measurements']['probabilistic']}")
        print(f"Outcomes: 0={m['measurements']['outcomes'][0]}, 1={m['measurements']['outcomes'][1]}")

        print("\n--- Timing ---")
        print(f"Total time: {total_time*1000:.3f} ms")
        print(f"Gate time: {m['timing']['gate_time_seconds']*1000:.3f} ms")
        print(f"Measurement time: {m['timing']['measurement_time_seconds']*1000:.3f} ms")

        if total_time > 0:
            gate_pct = m['timing']['gate_time_seconds'] / total_time * 100
            meas_pct = m['timing']['measurement_time_seconds'] / total_time * 100
            print(f"Gate time %: {gate_pct:.1f}%")
            print(f"Measurement time %: {meas_pct:.1f}%")

        print("=" * 50 + "\n")

    def reset_metrics(self):
        """Reset all metrics to initial values."""
        if self._enable_metrics:
            self._init_metrics()

    def _rowsum(self, h, i):
        """
        Private helper for measurement.
        Updates row h by multiplying it by row i (h = h * i).
        """
        x1 = self._x[i]
        z1 = self._z[i]
        x2 = self._x[h]
        z2 = self._z[h]

        g = (x1 & z1) * (x2 ^ z2) + \
            (x2 & z2) * (x1 ^ z1 ^ 1) + \
            (x1 & z2) * (x2 & z1)

        g_sum = (2 * self._r[i] + 2 * self._r[h] + np.sum(g)) % 4

        self._r[h] = g_sum >> 1

        self._x[h] ^= x1
        self._z[h] ^= z1

    def copy(self):
        t = Tableau.__new__(Tableau)
        t.n = self.n
        t._x = self._x.copy()
        t._z = self._z.copy()
        t._r = self._r.copy()
        return t
