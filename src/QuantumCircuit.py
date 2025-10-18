import numpy as np
from .utils.quantum_operations import apply_qubit
import sys
sys.path.append('..')
from gates.Gate import Gate
from gates.PauliGates import PauliGates

"""
QuantumCircuit class for building and simulating quantum circuits in Hilbert space
using unitary gates.

This class manages an n-qubit quantum state and allows sequential application of
unitary quantum gates to evolve the state.
"""
class QuantumCircuit:
    def __init__(self, num_qubits):
        if num_qubits < 1:
            raise ValueError("Number of qubits must be at least 1")

        self.num_qubits = num_qubits
        # initialize state vector in Hilbert space: |00...0⟩
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1.0
        self.gates = []

    def add_gate(self, gate, targets):
        U = np.array(gate.matrix, dtype=complex)
        if not self.is_unitary(U):
            raise ValueError(f"Gate {gate.name} is not unitary")

        self.gates.append((gate, targets))
        return self

    def is_unitary(self, matrix, tol=1e-10):
        U = np.array(matrix, dtype=complex)
        U_dagger = U.T.conj()
        identity = np.eye(U.shape[0])
        product = U_dagger @ U
        return np.allclose(product, identity, atol=tol)

    def execute(self):
        for gate, targets in self.gates:
            self.state = apply_qubit(self.state, gate, targets, self.num_qubits)
        return self

    def get_state(self):
        return self.state.copy()

    def reset_all(self):
        self.state = np.zeros(2**self.num_qubits, dtype=complex)
        self.state[0] = 1.0
        self.gates = []
        return self

    def reset_state_only(self):
        self.state = np.zeros(2**self.num_qubits, dtype=complex)
        self.state[0] = 1.0
        return self

    def measure_probabilities(self):
        return np.abs(self.state)**2

    def __repr__(self):
        return f"QuantumCircuit(num_qubits={self.num_qubits}, gates={len(self.gates)})"

    def __str__(self):
        circuit_str = f"Quantum Circuit with {self.num_qubits} qubit(s)\n"
        circuit_str += f"Number of gates: {len(self.gates)}\n"
        if self.gates:
            circuit_str += "Gate sequence:\n"
            for i, (gate, targets) in enumerate(self.gates):
                circuit_str += f"  {i+1}. {gate.name} on qubit(s) {targets}\n"
        return circuit_str


if __name__ == "__main__":
    # single qubit circuit with Pauli X gate
    print("\n" + "-" * 40)
    qc1 = QuantumCircuit(num_qubits=1)
    print(f"Initial state: {qc1.get_state()}")  # should be [1, 0] = |0⟩
    qc1.add_gate(PauliGates.X, targets=0)
    qc1.execute()
    print(f"After X gate: {qc1.get_state()}")   # should be [0, 1] = |1⟩
    print(f"Probabilities: {qc1.measure_probabilities()}")

    # two qubit circuit with X on qubit 1
    print("\n" + "-" * 40)
    qc2 = QuantumCircuit(num_qubits=2)
    print(f"Initial state: {qc2.get_state()}")       # [1, 0, 0, 0] = |00⟩
    qc2.add_gate(PauliGates.X, targets=1)
    qc2.execute()
    print(f"After X on qubit 1: {qc2.get_state()}")  # [0, 1, 0, 0] = |01⟩

    # multiple gates on different qubits
    print("\n" + "-" * 40)
    qc3 = QuantumCircuit(num_qubits=2)
    qc3.add_gate(PauliGates.X, targets=0)  # X on qubit 0
    qc3.add_gate(PauliGates.X, targets=1)  # X on qubit 1
    qc3.execute()
    print(f"After X on both qubits: {qc3.get_state()}")  # [0, 0, 0, 1] = |11⟩

    # using Z gate (phase flip)
    print("\n" + "-" * 40)
    qc4 = QuantumCircuit(num_qubits=1)
    qc4.add_gate(PauliGates.X, targets=0)
    qc4.add_gate(PauliGates.Z, targets=0)
    qc4.execute()
    print(f"After X then Z: {qc4.get_state()}")  # [0, -1] = -|1⟩

    print("Circuit info:")
    print(qc4)
