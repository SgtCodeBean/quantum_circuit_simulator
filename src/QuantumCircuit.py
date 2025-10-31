import numpy as np
from utils.quantum_operations import apply_qubit
import sys
sys.path.append('..')
from gates.registry import GateRegistry
from cbit import CBit

"""
QuantumCircuit class for building and simulating quantum circuits in Hilbert space
using unitary gates.

This class manages an n-qubit quantum state and allows sequential application of
unitary quantum gates to evolve the state.
"""
class QuantumCircuit:
    def __init__(self, num_qubits, num_cbits=0):
        if num_qubits < 1:
            raise ValueError("Number of qubits must be at least 1")

        self.num_qubits = num_qubits
        # initialize state vector in Hilbert space: |00...0⟩
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1.0
        self.ops = []
        self.cbits = CBit(num_bits=num_cbits)

    def add_gate(self, gate, targets):
        U = np.array(gate.matrix, dtype=complex)
        if not self.is_unitary(U):
            raise ValueError(f"Gate {gate.name} is not unitary")

        self.ops.append(("gate", gate, targets))
        return self

    def is_unitary(self, matrix, tol=1e-10):
        U = np.array(matrix, dtype=complex)
        U_dagger = U.T.conj()
        identity = np.eye(U.shape[0])
        product = U_dagger @ U
        return np.allclose(product, identity, atol=tol)

    def execute(self):
        for op in self.ops:
            if op[0] == "gate":
                _, gate, targets = op
                self.state = apply_qubit(self.state, gate, targets, self.num_qubits)
            elif op[0] == "measure":
                _, qubit, cbit = op
                self._perform_measure(qubit, cbit)
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

    def measure(self, qubit, cbit):
        if qubit not in range(self.num_qubits):
            raise ValueError(f"Qubit index {qubit} out of bounds!")
        elif cbit not in range(self.cbits.__len__()):
            raise ValueError(f"Classic register {cbit} out of bounds!")
        self.ops.append(("measure", qubit, cbit))
        return self
    
    def _perform_measure(self, qubit, cbit):
        bit_mask = 1 << (self.num_qubits - qubit - 1)

        i0 = [i for i in range(2**self.num_qubits) if (i & bit_mask) == 0]
        i1 = [i for i in range(2**self.num_qubits) if (i & bit_mask) != 0]

        p0 = np.sum(np.abs(self.state[i0]) ** 2)
        p1 = np.sum(np.abs(self.state[i1]) ** 2)

        # Check for strong rounding error of probability floats
        if abs(p0 + p1 - 1) > 1e-10:
            norm = np.sqrt(p0 + p1)
            p0 /= norm
            p1 /= norm
        
        collapse = np.random.choice([0, 1], p=[p0, p1])

        if collapse == 0:
            self.state[i1] = 0
            self.state /= np.sqrt(p0)
        else:
            self.state[i0] = 0
            self.state /= np.sqrt(p1)
        
        self.cbits.set_bit(cbit, collapse)

    def reset_qubit(self, collapsed_state, measurement_outcome, target_qubit):
        """
        Resets a qubit to the |0> state after it has been measured.
        This function takes the measurement outcome and the collapsed state,
        and applies a conditional X-gate to ensure the final state is |0>.
    
        Args:
            collapsed_state (numpy.ndarray): The 2^n state vector after measurement.
            measurement_outcome (int): The result of the measurement (0 or 1).
            target_qubit (int): The qubit index that was measured.
    
        Returns:
            numpy.ndarray: The new 2^n state vector after the reset.
        """
    
        if measurement_outcome == 1:
            print(f"Measurement outcome was 1, applying X gate to reset qubit {target_qubit} to |0>.")
            X_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
            X_full = self._get_full_operator(X_matrix, target_qubit)
            final_state = np.dot(X_full, collapsed_state)
        else:
            print(f"Measurement Outcome was 0, no reset needed.")
            final_state = collapsed_state
    
        return final_state
    
    def _get_full_operator(self, gate_matrix, target_qubit):
        """
        Builds the full 2^n x 2^n operator for a gate on a single target qubit.
        Used internally for qubit reset.
        
        Args:
            gate_matrix (numpy.ndarray): The 2x2 matrix for the gate.
            target_qubit (int): The qubit index (from 0 to num_qubits-1) to apply the gate to.
    
        Returns:
            numpy.ndarray: The 2^n x 2^n operator.
        """
        
        op_list = [np.identity(2, dtype=complex) for _ in range(self.num_qubits)]
        op_list[target_qubit] = gate_matrix
        
        full_op = op_list[0]
        for i in range(1, self.num_qubits):
            full_op = np.kron(full_op, op_list[i])
    
        return full_op

    def __repr__(self):
        return f"QuantumCircuit(num_qubits={self.num_qubits}, gates={len(self.gates)})"

    def __str__(self):
        circuit_str = f"Quantum Circuit with {self.num_qubits} qubit(s)\n"
        circuit_str += f"Number of gates: {len(self.gates)}\n"
        if self.gates:
            circuit_str += "Gate sequence:\n"
            for i, (gate, targets) in enumerate(self.gates):
                circuit_str += f"  {i+1}. {gate.name} on qubit(s) {targets}\n"
        if self.measure_ops:
            circuit_str += "Measurement sequence:\n"
            for i, (qubit, cbit) in enumerate(self.measure_ops):
                circuit_str += f"  {i+1}. qubit {qubit} stored in cbit{cbit}"
        return circuit_str

def main():
    reg = GateRegistry()
    # single qubit circuit with Pauli X gate
    print("\n" + "-" * 40)
    qc1 = QuantumCircuit(num_qubits=1, num_cbits=1)
    print(f"Initial state: {qc1.get_state()}")  # should be [1, 0] = |0⟩
    qc1.add_gate(reg.get('x'), targets=0)
    qc1.execute()
    print(f"After X gate: {qc1.get_state()}")   # should be [0, 1] = |1⟩
    print(f"Probabilities: {qc1.measure_probabilities()}")

    # two qubit circuit with X on qubit 1
    print("\n" + "-" * 40)
    qc2 = QuantumCircuit(num_qubits=2)
    print(f"Initial state: {qc2.get_state()}")       # [1, 0, 0, 0] = |00⟩
    qc2.add_gate(reg.get('x'), targets=1)
    qc2.execute()
    print(f"After X on qubit 1: {qc2.get_state()}")  # [0, 1, 0, 0] = |01⟩

    # multiple gates on different qubits
    print("\n" + "-" * 40)
    qc3 = QuantumCircuit(num_qubits=2)
    qc3.add_gate(reg.get('x'), targets=0)  # X on qubit 0
    qc3.add_gate(reg.get('x'), targets=1)  # X on qubit 1
    qc3.execute()
    print(f"After X on both qubits: {qc3.get_state()}")  # [0, 0, 0, 1] = |11⟩

    # using Z gate (phase flip)
    print("\n" + "-" * 40)
    qc4 = QuantumCircuit(num_qubits=1)
    qc4.add_gate(reg.get('x'), targets=0)
    qc4.add_gate(reg.get('z'), targets=0)
    qc4.execute()
    print(f"After X then Z: {qc4.get_state()}")  # [0, -1] = -|1⟩

    print("Circuit info:")
    print(qc4)


if __name__ == "__main__":
    main()
