import numpy as np
from utils.quantum_operations import apply_qubit
import sys
sys.path.append('..')
from gates.registry import GateRegistry
from cbit import CBit
import time
import psutil
import os
from collections import defaultdict
from typing import Optional
from error_channels.noise_model import NoiseModel
from error_channels.default_noise import build_default_noise_model
from output_format import ResultWriter
from circuit_results import ResultManager
from circuit_metrics import CircuitMetrics

"""
QuantumCircuit class for building and simulating quantum circuits in Hilbert space
using unitary gates.

This class manages an n-qubit quantum state and allows sequential application of
unitary quantum gates to evolve the state.
"""
class QuantumCircuit:
    def __init__(self, num_qubits, num_cbits=0, rng_seed=None,
                 enable_metrics=False,
                 num_shots=1024,
                 noise_model: Optional[NoiseModel] = build_default_noise_model(),
                 rng: Optional[np.random.Generator] = None):
        if num_qubits < 1:
            raise ValueError("Number of qubits must be at least 1")

        self.num_qubits = num_qubits
        self.num_cbits = num_cbits
        # initialize state vector in Hilbert space: |00...0⟩
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1.0
        self.ops = []
        self.cbits = CBit(num_bits=num_cbits)
        self.num_shots = num_shots
        self.noise_model = noise_model
        self.rng = rng or np.random.default_rng()
        self._measurements = {}
        if rng_seed is not None:
            np.random.seed(rng_seed)

        # Simulator metrics and results tracking
        self.metrics = CircuitMetrics(enabled=enable_metrics)
        self.results = ResultManager()

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

    def execute(self, verbose=False):
        self.metrics.start_execution()

        for op in self.ops:
            if op[0] == "gate":
                _, gate, targets = op

                if self.metrics.enabled:
                    start_time = time.perf_counter()
                # normalize targets to a list for noise model
                if isinstance(targets, int):
                    tlist = [targets]
                else:
                    tlist = list(targets)

                # 1) ideal unitary evolution
                self.state = apply_qubit(self.state, gate, tlist, self.num_qubits)

                # 2) optional noise after gate
                if self.noise_model is not None:
                    self.state = self.noise_model.apply_after_gate(
                        state=self.state,
                        gate_name=gate.name,  # "x", "h", "cx", "ccx", etc.
                        targets=tlist,
                        n_qubits=self.num_qubits,
                        rng=self.rng,
                        circuit=self,  # so record_channel_hit works
                    )

                if self.metrics.enabled:
                    duration = time.perf_counter() - start_time
                    self.metrics.record_gate(duration)

            elif op[0] == "measure":
                _, qubit, cbit = op

                if self.metrics.enabled:
                    start_time = time.perf_counter()

                self._perform_measure(qubit, cbit)

                if self.metrics.enabled:
                    duration = time.perf_counter() - start_time
                    self.metrics.record_measurement(duration)
            
            elif op[0] == "reset":
                _, qubit = op

                self.state = self._perform_reset(qubit)

        self.metrics.end_execution()

        result = ResultManager.build_execution_result(
            self.state, self.num_qubits, self.num_cbits,
            self.ops, self.cbits, self._measurements,
            self.metrics.get_summary()
        )

        self.results.store_execution_result(result)

        if verbose:
            writer = ResultWriter(format_type='text', verbose=True)
            writer.write_execution_result(result)

        return result

    def run_shots(self, verbose=False):
        """
        Execute the circuit for the specified number of shots and collect measurement statistics.

        Args:
            verbose (bool): If True, print progress and final statistics

        Returns:
            dict: Shot results with counts, probabilities, and aggregate metrics
        """
        if self.num_shots < 1:
            raise ValueError("Number of shots must be at least 1")

        if self.num_cbits == 0:
            raise ValueError("Circuit must have classical bits for shot-based measurement")

        counts = {}

        if verbose:
            print(f"\nRunning {self.num_shots} shots...")
            progress_interval = max(1, self.num_shots // 10)

        for shot in range(self.num_shots):
            self.reset_state_only()

            result = self.execute(verbose=False)

            bitstring = result['classical_bits']['bitstring']

            counts[bitstring] = counts.get(bitstring, 0) + 1

            if verbose and (shot + 1) % progress_interval == 0:
                print(f"  Completed {shot + 1}/{self.num_shots} shots...")

        self.reset_state_only()

        shot_results = ResultManager.build_shot_results(
            counts, self.num_shots, self.num_qubits, self.num_cbits
        )

        self.results.store_shot_results(shot_results)

        if verbose:
            writer = ResultWriter(format_type='text', verbose=True)
            writer.write_shot_results(shot_results)

        return shot_results


    def get_state(self):
        return self.state.copy()

    def reset_all(self):
        self.state = np.zeros(2**self.num_qubits, dtype=complex)
        self.state[0] = 1.0
        self.ops = []
        return self

    def reset_state_only(self):
        self.state = np.zeros(2**self.num_qubits, dtype=complex)
        self.state[0] = 1.0
        return self

    def measure_probabilities(self):
        return np.abs(self.state)**2
    
    def print_cbit(self, cbit):
        print(f"Classical Register {cbit}: {self.cbits.get_bit(cbit)}")

    def get_cbit(self, cbit):
        return self.cbits.get_bit(cbit)
    
    def get_cbits(self):
        return self.cbits.get_bits()

    def set_shots(self, num_shots):
        self.num_shots = num_shots

    def get_execution_result(self):
        """
        Get the result of the most recent single execution.

        Returns:
            dict: Execution results with state, measurements, and metrics.
                Returns None if execute() hasn't been called yet.
        """
        return self.results.get_execution_result()

    def get_shot_results(self):
        """
        Get the results of the most recent shot-based execution.

        Returns:
            dict: Shot results with counts, probabilities, and statistics.
                Returns None if run_shots() hasn't been called yet.
        """
        return self.results.get_shot_results()

    def get_counts(self):
        """
        Get measurement counts from the most recent shot execution.

        Returns:
            dict: Measurement counts {'00': 245, '01': 255, ...}
                Returns None if run_shots() hasn't been called yet.
        """
        return self.results.get_counts()

    def get_statevector(self):
        """
        Get the current quantum state vector.

        Returns:
            numpy.ndarray: Complex state vector of length 2^n
        """
        return self.state

    def get_probabilities(self):
        """
        Get probability distribution of the current state.

        Returns:
            numpy.ndarray: Real-valued probabilities for each basis state
        """
        return self.measure_probabilities()

    def get_measurement_outcomes(self):
        """
        Get the outcomes of all measurements performed in the most recent execution.

        Returns:
            dict: Mapping of qubit indices to their measurement outcomes and storage locations.
                Example: {0: {'outcome': 1, 'cbit': 0}, 1: {'outcome': 0, 'cbit': 1}}
                Returns empty dict if no measurements have been performed.
        """
        return self._measurements

    def get_classical_register(self):
        """
        Get the current values of all classical bits.

        Returns:
            dict: Classical bit values and bitstring representation.
                Example: {'c[0]': 1, 'c[1]': 0, 'bitstring': '10'}
        """
        if self.num_cbits == 0:
            return {'bitstring': ''}

        result = {
            f'c[{i}]': self.cbits.get_bit(i)
            for i in range(self.num_cbits)
        }
        result['bitstring'] = ''.join(
            str(self.cbits.get_bit(i)) for i in range(self.num_cbits)
        )
        return result

    def get_shot_probabilities(self):
        """
        Get probability distribution from the most recent shot execution.

        Returns:
            dict: Probabilities for each measured bitstring.
                Example: {'00': 0.245, '01': 0.255, ...}
                Returns None if run_shots() hasn't been called yet.
        """
        return self.results.get_shot_probabilities()

    def get_expected_value(self, observable):
        """
        Calculate the expected value of an observable for the current state.

        Args:
            observable (numpy.ndarray): Hermitian operator (2^n x 2^n matrix)

        Returns:
            float: Expected value ⟨ψ|O|ψ⟩
        """
        obs = np.asarray(observable, dtype=complex)
        if obs.shape != (2**self.num_qubits, 2**self.num_qubits):
            raise ValueError(f"Observable must be {2**self.num_qubits}x{2**self.num_qubits} matrix")

        return np.real(np.vdot(self.state, obs @ self.state))

    def get_entropy(self):
        """
        Calculate the von Neumann entropy of the current state.
        For a pure state, this should be zero (or near-zero due to numerical errors).

        Returns:
            float: Entropy S = -Σ p_i log(p_i)
        """
        probs = self.measure_probabilities()
        probs_nonzero = probs[probs > 1e-15]
        if len(probs_nonzero) == 0:
            return 0.0
        return -np.sum(probs_nonzero * np.log2(probs_nonzero))

    def get_circuit_depth(self):
        """
        Get the depth of the circuit (number of sequential operations).

        Returns:
            int: Total number of operations in the circuit
        """
        return len(self.ops)

    def get_circuit_width(self):
        """
        Get the width of the circuit (number of qubits).

        Returns:
            int: Number of qubits in the circuit
        """
        return self.num_qubits

    def get_operation_counts(self):
        """
        Get a breakdown of operation types in the circuit.

        Returns:
            dict: Counts of each operation type
        """
        counts = {
            'gates': 0,
            'measurements': 0,
            'resets': 0,
            'total': len(self.ops)
        }

        gate_breakdown = {}

        for op in self.ops:
            if op[0] == "gate":
                counts['gates'] += 1
                gate_name = op[1].name
                gate_breakdown[gate_name] = gate_breakdown.get(gate_name, 0) + 1
            elif op[0] == "measure":
                counts['measurements'] += 1
            elif op[0] == "reset":
                counts['resets'] += 1

        counts['gate_breakdown'] = gate_breakdown
        return counts

    def get_metrics(self):
        """Get performance metrics summary."""
        return self.metrics.get_summary()

    def print_metrics(self):
        """Print formatted metrics summary."""
        self.metrics.print_summary()

    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics.reset()


    def measure(self, qubit, cbit):
        """
        Adds a measurement step to the simulator's set of opperations.

        Args:
            qubit (int) index of the qubit that will have its value read
            cbit (int) index of the classic register that will store the value
        """
        if qubit not in range(self.num_qubits):
            raise ValueError(f"Qubit index {qubit} out of bounds!")
        elif cbit not in range(self.cbits.__len__()):
            raise ValueError(f"Classic register {cbit} out of bounds!")
        self.ops.append(("measure", qubit, cbit))
        return self
    
    def _perform_measure(self, qubit, cbit):
        """
        Executes a measurement of a given qubit and stores it into a given classical
        register. Also collapses the state of that qubit to 0 or 1 based upon the
        value that is measured.

        Args:
            qubit (int) index of the qubit that is being read
            cbit (int) index of the classic register that will store the value
        
        Returns:
            The collapsed state (int).
        """
        # Bit mask to determine probabilities of 0 or 1 within the qubit structure.
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
        self._measurements[qubit] = {
            "outcome": collapse,
            "cbit": cbit
        }
        return collapse

    def reset(self, qubit: int):
        if qubit not in range(self.num_qubits):
            raise ValueError(f"Qubit index {qubit} out of bounds!")
        self.ops.append(("reset", qubit))
        return self

    def reset_qubit(self, qubit):
        """
        Adds a reset call to the operations queue, ops. Stores the
        operation "reset" with the specified qubit so that it will be ran
        when the execute() function is called.
    
        Args:
            qubit(int): The qubit index that was measured.
        """

        if qubit not in range(self.num_qubits):
            raise ValueError(f"Qubit index {qubit} out of bounds!")
        self.ops.append(("reset", qubit))
        return self
    
    def _perform_reset(self, target_qubit):
        """
        Resets a qubit to the |0> state after it has been measured.
        This function takes the measurement outcome and the collapsed state,
        and applies a conditional X-gate to ensure the final state is |0>.
    
        Args:
            target_qubit (int): The qubit index that was measured.
    
        Returns:
            numpy.ndarray: The new 2^n state vector after the reset.
        """

        measurement_record = self._measurements[target_qubit]
        if measurement_record is not None:
            measurement_outcome = measurement_record["outcome"]
        else:
            measurement_outcome = None
        
        if measurement_outcome == 1:
            print(f"Measurement outcome was 1, applying X gate to reset qubit {target_qubit} to |0>.")
            X_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
            X_full = self._get_full_operator(X_matrix, target_qubit)
            final_state = np.dot(X_full, self.state)
        else:
            print(f"Measurement Outcome was 0, no reset needed.")
            final_state = self.state
    
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
        gate_len = len([gate for gate in self.ops if "gate" in gate])
        return f"QuantumCircuit(num_qubits={self.num_qubits}, num_cbits={self.cbits.__len__()}, gates={gate_len})"

    def __str__(self):
        gates = [gate for gate in self.ops if "gate" in gate]
        circuit_str = f"Quantum Circuit with {self.num_qubits} qubit(s) and {self.cbits.__len__()} classic bits\n"
        circuit_str += f"Number of gates: {len(gates)}\n"
        if self.ops:
            circuit_str += "Simulation sequence:\n"
            for i, op in enumerate(self.ops):
                if op[0] == "gate":
                    _, gate, targets = op
                    circuit_str += f"  {i+1}. {gate.name} on qubit(s) {targets}\n"
                elif op[0] == "measure":
                    _, qubit, cbit = op
                    circuit_str += f"  {i+1}. qubit {qubit} stored in cbit {cbit}\n"
                elif op[0] == "reset":
                    _, qubit
                    circuit_str += f"  {i+1}. qubit {qubit} reset to |0⟩."
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
