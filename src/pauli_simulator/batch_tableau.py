import numpy as np
import time

"""
Efficient Pauli Simulator for batch processing.
"""


class BatchTableau:
    __slots__ = ('n', 'batch_size', '_x', '_z', '_r', '_metrics', '_enable_metrics')

    def __init__(self, n, batch_size, enable_metrics=False):
        """
        Initialize a batch of tableaus for n qubits.

        Args:
            n (int): Number of qubits.
            batch_size (int): Number of parallel simulations (shots) to run.
            enable_metrics (bool): Whether to collect performance metrics.
        """
        self.n = n
        self.batch_size = batch_size

        self._x = np.zeros((2 * n, n), dtype=np.uint8)
        self._z = np.zeros((2 * n, n), dtype=np.uint8)
        self._r = np.zeros((batch_size, 2 * n), dtype=np.uint8)

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
            },
            'operations': {
                'gate_count': 0,
                'measurement_count': 0,
            },
            'timing': {
                'gate_time_seconds': 0.0,
                'measurement_time_seconds': 0.0
            }
        }

    def _record_gate(self, gate_name, elapsed):
        """Record a gate operation."""
        if not self._enable_metrics:
            return
        self._metrics['operations']['gate_count'] += 1
        self._metrics['timing']['gate_time_seconds'] += elapsed

    def _record_measurement(self, elapsed):
        """Record a measurement operation."""
        if not self._enable_metrics:
            return
        self._metrics['operations']['measurement_count'] += 1
        self._metrics['timing']['measurement_time_seconds'] += elapsed


    def h(self, q):
        """Hadamard gate"""
        start = time.perf_counter() if self._enable_metrics else 0

        phase_term = self._x[:, q] & self._z[:, q]
        self._r ^= phase_term[np.newaxis, :]

        self._x[:, q], self._z[:, q] = self._z[:, q].copy(), self._x[:, q].copy()

        if self._enable_metrics:
            self._record_gate('h', time.perf_counter() - start)

    def s(self, q):
        """Phase gate"""
        start = time.perf_counter() if self._enable_metrics else 0

        phase_term = self._x[:, q] & self._z[:, q]
        self._r ^= phase_term[np.newaxis, :]

        self._z[:, q] ^= self._x[:, q]

        if self._enable_metrics:
            self._record_gate('s', time.perf_counter() - start)

    def cx(self, c, t):
        """CNOT gate: control c, target t"""
        start = time.perf_counter() if self._enable_metrics else 0

        term = self._x[:, c] & self._z[:, t] & (self._x[:, t] ^ self._z[:, c] ^ 1)
        self._r ^= term[np.newaxis, :]

        self._x[:, t] ^= self._x[:, c]
        self._z[:, c] ^= self._z[:, t]

        if self._enable_metrics:
            self._record_gate('cx', time.perf_counter() - start)

    def x(self, q):
        """Pauli X: bit flip"""
        start = time.perf_counter() if self._enable_metrics else 0
        self._r ^= self._z[:, q][np.newaxis, :]
        if self._enable_metrics:
            self._record_gate('x', time.perf_counter() - start)

    def y(self, q):
        """Pauli Y: bit + phase flip"""
        start = time.perf_counter() if self._enable_metrics else 0
        self._r ^= (self._x[:, q] ^ self._z[:, q])[np.newaxis, :]
        if self._enable_metrics:
            self._record_gate('y', time.perf_counter() - start)

    def z(self, q):
        """Pauli Z: phase flip"""
        start = time.perf_counter() if self._enable_metrics else 0
        self._r ^= self._x[:, q][np.newaxis, :]
        if self._enable_metrics:
            self._record_gate('z', time.perf_counter() - start)

    def measure(self, q):
        """
        Measure qubit q. Returns an array of outcomes of size (batch_size,).
        """
        start = time.perf_counter() if self._enable_metrics else 0

        # Case I: Random outcome (pivot search)
        p = -1
        for i in range(self.n, 2 * self.n):
            if self._x[i, q]:
                p = i
                break

        if p != -1:
            outcomes = np.random.randint(0, 2, size=self.batch_size, dtype=np.uint8)

            for i in range(2 * self.n):
                if i != p and self._x[i, q]:
                    self._rowsum(i, p)

            self._x[p - self.n] = self._x[p]
            self._z[p - self.n] = self._z[p]
            self._r[:, p - self.n] = self._r[:, p]

            self._x[p, :] = 0
            self._z[p, :] = 0
            self._z[p, q] = 1
            self._r[:, p] = outcomes

            if self._enable_metrics:
                self._record_measurement(time.perf_counter() - start)
            return outcomes

        # Case II: Deterministic outcome
        else:
            temp_r = np.zeros(self.batch_size, dtype=np.uint8)
            temp_x = np.zeros(self.n, dtype=np.uint8)
            temp_z = np.zeros(self.n, dtype=np.uint8)

            for i in range(self.n):
                if self._x[i, q]:
                    x1 = self._x[i + self.n]
                    z1 = self._z[i + self.n]
                    x2 = temp_x
                    z2 = temp_z

                    g = np.zeros(self.n, dtype=int)

                    mask_y = (x1 == 1) & (z1 == 1)
                    g[mask_y] = z2[mask_y].astype(int) - x2[mask_y].astype(int)

                    mask_x = (x1 == 1) & (z1 == 0)
                    g[mask_x] = z2[mask_x].astype(int) * (2 * x2[mask_x].astype(int) - 1)

                    mask_z = (x1 == 0) & (z1 == 1)
                    g[mask_z] = x2[mask_z].astype(int) * (1 - 2 * z2[mask_z].astype(int))

                    g_sum = (2 * self._r[:, i + self.n] + 2 * temp_r + np.sum(g)) % 4
                    temp_r = g_sum >> 1
                    temp_x ^= x1
                    temp_z ^= z1

            if self._enable_metrics:
                self._record_measurement(time.perf_counter() - start)
            return temp_r

    def get_metrics(self):
        """Get the collected metrics."""
        if not self._enable_metrics:
            return None
        return self._metrics.copy()

    def _rowsum(self, h, i):
        """
        Updates row h based on row i.
        Updates shared _x/_z and batched _r.
        """
        x1 = self._x[i]
        z1 = self._z[i]
        x2 = self._x[h]
        z2 = self._z[h]

        g = np.zeros(self.n, dtype=int)

        mask_y = (x1 == 1) & (z1 == 1)
        g[mask_y] = z2[mask_y].astype(int) - x2[mask_y].astype(int)

        mask_x = (x1 == 1) & (z1 == 0)
        g[mask_x] = z2[mask_x].astype(int) * (2 * x2[mask_x].astype(int) - 1)

        mask_z = (x1 == 0) & (z1 == 1)
        g[mask_z] = x2[mask_z].astype(int) * (1 - 2 * z2[mask_z].astype(int))

        g_sum = (2 * self._r[:, i] + 2 * self._r[:, h] + np.sum(g)) % 4

        self._r[:, h] = g_sum >> 1

        self._x[h] ^= x1
        self._z[h] ^= z1

