import numpy as np

"""
Optimized Tableau class for efficient Clifford circuit simulation.
"""
class Tableau:
    __slots__ = ('n', '_x', '_z', '_r')

    def __init__(self, n):
        """
        Initialize tableau for n qubits in |0...0⟩ state.
        """
        self.n = n
        self._x = np.zeros((2*n, n), dtype=np.uint8)
        self._z = np.zeros((2*n, n), dtype=np.uint8)
        self._r = np.zeros(2*n, dtype=np.uint8)
        for i in range(n):
            self._x[i, i] = 1
            self._z[n+i, i] = 1

    def h(self, q):
        """Hadamard gate"""
        for i in range(2*self.n):
            self._r[i] ^= self._x[i,q] & self._z[i,q]
            self._x[i,q], self._z[i,q] = self._z[i,q], self._x[i,q]

    def s(self, q):
        """Phase gate"""
        for i in range(2*self.n):
            self._r[i] ^= self._x[i,q] & self._z[i,q]
            self._z[i,q] ^= self._x[i,q]

    def cx(self, c, t):
        """CNOT gate: control c, target t"""
        for i in range(2*self.n):
            self._r[i] ^= self._x[i,c] & self._z[i,t] & (self._x[i,t] ^ self._z[i,c] ^ 1)
            self._x[i,t] ^= self._x[i,c]
            self._z[i,c] ^= self._z[i,t]

    def x(self, q):
        """Pauli X: bit flip"""
        for i in range(2*self.n):
            self._r[i] ^= self._z[i,q]

    def y(self, q):
        """Pauli Y: bit + phase flip"""
        for i in range(2*self.n):
            self._r[i] ^= self._x[i,q] ^ self._z[i,q]

    def z(self, q):
        """Pauli Z: phase flip"""
        for i in range(2*self.n):
            self._r[i] ^= self._x[i,q]

    def measure(self, q):
        """
        Measure qubit q in Z basis.
        """
        # Find first stabilizer with X on qubit q
        p = None
        for i in range(self.n, 2*self.n):
            if self._x[i,q]:
                p = i
                break

        if p is None:
            # qubit in Z eigenstate
            for i in range(self.n, 2*self.n):
                if self._z[i,q]:
                    return (self._r[i] >> 1) & 1
            return 0
        else:
            outcome = np.random.randint(0, 2)
            for i in range(2*self.n):
                if i != p and self._x[i,q]:
                    self._rowsum(i, p)
            self._x[p,:] = 0
            self._z[p,:] = 0
            self._z[p,q] = 1
            self._r[p] = 2 * outcome
            return outcome

    def _rowsum(self, h, i):
        """Multiply row h by row i (Pauli group multiplication)"""
        phase = 2 * np.sum(self._x[h] & self._z[h] & (self._x[i] ^ self._z[i]))
        phase += np.sum(self._x[i] & self._z[h] & ~(self._x[h] ^ self._z[i]))
        phase += 3 * np.sum(~self._x[h] & self._z[h] & self._x[i] & ~self._z[i])
        phase += 3 * np.sum(self._x[h] & ~self._z[h] & ~self._x[i] & self._z[i])
        self._x[h] ^= self._x[i]
        self._z[h] ^= self._z[i]
        self._r[h] = (self._r[h] + self._r[i] + phase) & 3

    def copy(self):
        t = Tableau.__new__(Tableau)
        t.n = self.n
        t._x = self._x.copy()
        t._z = self._z.copy()
        t._r = self._r.copy()
        return t
