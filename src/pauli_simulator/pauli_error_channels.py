import numpy as np
from Tableau_Ver2 import Tableau

"""
Class for Pauli error channels
"""

class NoisySimulator:

    def __init__(self, n, error_config=None):
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
        """
        self.tableau = Tableau(n)
        self.n = n
        self.error_config = error_config if error_config is not None else {}

        self.rng = np.random.default_rng()

    def _apply_pauli_noise(self, qubit, probabilities):
        """
        Applies a Pauli operator to a single qubit.

        Args:
            qubit (int): The qubit to apply noise to.
            probabilities (tuple): A tuple (px, py, pz) of error
                rates.
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

        rand_val = self.rng.random()
        if rand_val < px:
            self.tableau.x(qubit)
        elif rand_val < px + py:
            self.tableau.y(qubit)
        elif rand_val < p_total:
            self.tableau.z(qubit)

    def h(self, q):
        """Applies a noisy H gate."""
        # 1. Apply the perfect gate
        self.tableau.h(q)
        # 2. Apply the noise
        probs = self.error_config.get('h', (0, 0, 0))
        self._apply_pauli_noise(q, probs)

    def s(self, q):
        """Applies a noisy S gate."""
        self.tableau.s(q)
        probs = self.error_config.get('s', (0, 0, 0))
        self._apply_pauli_noise(q, probs)

    def cx(self, c, t):
        """
        Applies a noisy CNOT gate.
        """
        self.tableau.cx(c, t)
        probs = self.error_config.get('cx', (0, 0, 0))
        self._apply_pauli_noise(c, probs)
        self._apply_pauli_noise(t, probs)

    def x(self, q):
        """Applies a noisy X gate."""
        self.tableau.x(q)
        probs = self.error_config.get('x', (0, 0, 0))
        self._apply_pauli_noise(q, probs)

    def y(self, q):
        """Applies a noisy Y gate."""
        self.tableau.y(q)
        probs = self.error_config.get('y', (0, 0, 0))
        self._apply_pauli_noise(q, probs)

    def z(self, q):
        """Applies a noisy Z gate."""
        self.tableau.z(q)
        probs = self.error_config.get('z', (0, 0, 0))
        self._apply_pauli_noise(q, probs)

    def measure(self, q):
        """
        Applies noise before the measurement.
        """
        probs = self.error_config.get('measure', (0, 0, 0))
        self._apply_pauli_noise(q, probs)
        return self.tableau.measure(q)
