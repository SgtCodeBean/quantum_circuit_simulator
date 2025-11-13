import numpy as np

"""
Optimized Tableau class for efficient Clifford circuit simulation.
(Measurement logic has been corrected)
"""


class Tableau:
    __slots__ = ('n', '_x', '_z', '_r')

    def __init__(self, n):
        """
        Initialize tableau for n qubits in |0...0⟩ state.
        """
        self.n = n
        self._x = np.zeros((2 * n, n), dtype=np.uint8)
        self._z = np.zeros((2 * n, n), dtype=np.uint8)
        self._r = np.zeros(2 * n, dtype=np.uint8)
        for i in range(n):
            self._x[i, i] = 1
            self._z[n + i, i] = 1

    def h(self, q):
        """Hadamard gate"""
        for i in range(2 * self.n):
            self._r[i] ^= self._x[i, q] & self._z[i, q]
            self._x[i, q], self._z[i, q] = self._z[i, q], self._x[i, q]

    def s(self, q):
        """Phase gate"""
        for i in range(2 * self.n):
            self._r[i] ^= self._x[i, q] & self._z[i, q]
            self._z[i, q] ^= self._x[i, q]

    def cx(self, c, t):
        """CNOT gate: control c, target t"""
        for i in range(2 * self.n):
            self._r[i] ^= self._x[i, c] & self._z[i, t] & (self._x[i, t] ^ self._z[i, c] ^ 1)
            self._x[i, t] ^= self._x[i, c]
            self._z[i, c] ^= self._z[i, t]

    def x(self, q):
        """Pauli X: bit flip"""
        for i in range(2 * self.n):
            self._r[i] ^= self._z[i, q]

    def y(self, q):
        """Pauli Y: bit + phase flip"""
        for i in range(2 * self.n):
            self._r[i] ^= self._x[i, q] ^ self._z[i, q]

    def z(self, q):
        """Pauli Z: phase flip"""
        for i in range(2 * self.n):
            self._r[i] ^= self._x[i, q]

    # --- Corrected Measurement Logic (from Aaronson-Gottesman paper) ---

    def measure(self, q):
        """
        Measure qubit q in the Z basis.
        Returns the measurement outcome (0 or 1).
        """

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

            return temp_r

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
