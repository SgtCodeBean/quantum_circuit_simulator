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

"""
QuantumCircuit class for building and simulating quantum circuits in Hilbert space
using unitary gates.

This class manages an n-qubit quantum state and allows sequential application of
unitary quantum gates to evolve the state.
"""
class QuantumCircuit:
    def __init__(self, num_qubits, num_cbits=0, density=False, enable_metrics=False, num_shots=1024):
        if num_qubits < 1:
            raise ValueError("Number of qubits must be at least 1")

        self.num_qubits = num_qubits
        # initialize state vector in Hilbert space: |00...0⟩
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1.0
        self.ops = []
        self.cbits = CBit(num_bits=num_cbits)
        self.num_shots = num_shots
        self._measurements = {}

        # Simulator metrics tracking
        self.enable_metrics = enable_metrics
        self._init_metrics()

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
        if self.enable_metrics:
            self.metrics['execution_start_time'] = time.perf_counter()
            self.metrics['initial_memory_mb'] = self._get_memory_mb()
            self.metrics['peak_memory_mb'] = self.metrics['initial_memory_mb']

        for op in self.ops:
            if op[0] == "gate":
                _, gate, targets = op

                if self.enable_metrics:
                    start_time = time.perf_counter()
                    mem_before = self._get_memory_mb()

                self.state = apply_qubit(self.state, gate, targets, self.num_qubits)

                if self.enable_metrics:
                    end_time = time.perf_counter()
                    mem_after = self._get_memory_mb()
                    self.metrics['gate_count'] += 1
                    self.metrics['total_gate_time'] += (end_time - start_time)
                    self.metrics['peak_memory_mb'] = max(self.metrics['peak_memory_mb'], mem_after)

            elif op[0] == "measure":
                _, qubit, cbit = op

                if self.enable_metrics:
                    start_time = time.perf_counter()
                    mem_before = self._get_memory_mb()

                self._perform_measure(qubit, cbit)

                if self.enable_metrics:
                    end_time = time.perf_counter()
                    mem_after = self._get_memory_mb()
                    self.metrics['measurement_count'] += 1
                    self.metrics['total_measurement_time'] += (end_time - start_time)
                    self.metrics['peak_memory_mb'] = max(self.metrics['peak_memory_mb'], mem_after)
            
            elif op[0] == "reset":
                _, qubit = op

                self.state = self._perform_reset(qubit)

        if self.enable_metrics:
            self.metrics['execution_end_time'] = time.perf_counter()
            self.metrics['final_memory_mb'] = self._get_memory_mb()
            self.metrics['total_execution_time'] = (
                self.metrics['execution_end_time'] - self.metrics['execution_start_time']
            )

        self.ops = []
        return self

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
        return self.cbits

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

    def _init_metrics(self):
        """Initialize metrics tracking attributes."""
        if not self.enable_metrics:
            self.metrics = None
            return

        try:
            self._process = psutil.Process(os.getpid())
        except:
            self._process = None

        self.metrics = {
            # Execution timing
            'execution_start_time': None,
            'execution_end_time': None,
            'total_execution_time': 0.0,

            # Memory tracking (in MB)
            'initial_memory_mb': 0.0,
            'peak_memory_mb': 0.0,
            'final_memory_mb': 0.0,

            # Operation counts
            'gate_count': 0,
            'measurement_count': 0,
            'channel_count': 0,

            # Timing breakdowns
            'total_gate_time': 0.0,
            'total_measurement_time': 0.0,
            'total_channel_time': 0.0,

            # Error channel statistics
            'channels_applied': defaultdict(lambda: {
                'count': 0,
                'total_time': 0.0,
                'kraus_outcomes': []
            }),

            # Detailed operation log (optional)
            'operation_log': []
        }

    def _get_memory_mb(self):
        if not self.enable_metrics or self._process is None:
            return 0.0
        try:
            return self._process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0

    def reset_metrics(self):
        if self.enable_metrics:
            self._init_metrics()

    def record_channel_hit(self, channel_name, duration=0.0, kraus_index=None):
        """
        Record that an error channel was applied.

        Args:
            channel_name: Name of the error channel (e.g., 'bit_flip', 'depolarizing')
            duration: Time taken to apply the channel (seconds)
            kraus_index: Which Kraus operator was selected (for Monte Carlo)
        """
        if not self.enable_metrics:
            return

        stats = self.metrics['channels_applied'][channel_name]
        stats['count'] += 1
        stats['total_time'] += duration
        if kraus_index is not None:
            stats['kraus_outcomes'].append(kraus_index)

        self.metrics['channel_count'] += 1
        self.metrics['total_channel_time'] += duration

    def get_metrics(self):
        if not self.enable_metrics:
            return None

        # Calculate derived metrics
        total_ops = (self.metrics['gate_count'] +
                    self.metrics['measurement_count'] +
                    self.metrics['channel_count'])

        summary = {
            'execution': {
                'total_time_seconds': self.metrics['total_execution_time'],
                'start_time': self.metrics['execution_start_time'],
                'end_time': self.metrics['execution_end_time'],
            },
            'memory': {
                'initial_mb': self.metrics['initial_memory_mb'],
                'peak_mb': self.metrics['peak_memory_mb'],
                'final_mb': self.metrics['final_memory_mb'],
                'delta_mb': self.metrics['final_memory_mb'] - self.metrics['initial_memory_mb'],
            },
            'operations': {
                'gate_count': self.metrics['gate_count'],
                'measurement_count': self.metrics['measurement_count'],
                'channel_count': self.metrics['channel_count'],
                'total_operations': total_ops,
            },
            'timing': {
                'gate_time_seconds': self.metrics['total_gate_time'],
                'measurement_time_seconds': self.metrics['total_measurement_time'],
                'channel_time_seconds': self.metrics['total_channel_time'],
                'gate_time_percent': self._safe_percent(
                    self.metrics['total_gate_time'],
                    self.metrics['total_execution_time']
                ),
                'measurement_time_percent': self._safe_percent(
                    self.metrics['total_measurement_time'],
                    self.metrics['total_execution_time']
                ),
                'channel_time_percent': self._safe_percent(
                    self.metrics['total_channel_time'],
                    self.metrics['total_execution_time']
                ),
            },
            'channels': {}
        }

        for name, stats in self.metrics['channels_applied'].items():
            avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0.0
            summary['channels'][name] = {
                'hit_count': stats['count'],
                'total_time_seconds': stats['total_time'],
                'avg_time_seconds': avg_time,
                'kraus_outcomes': stats['kraus_outcomes'] if stats['kraus_outcomes'] else None,
            }

        return summary

    def _safe_percent(self, part, total):
        """Calculate percentage safely (handles division by zero)."""
        if total == 0:
            return 0.0
        return (part / total) * 100.0

    def print_metrics(self):
        """Print a formatted metrics summary to console."""
        if not self.enable_metrics:
            print("Metrics tracking is disabled. Enable with enable_metrics=True")
            return

        summary = self.get_metrics()
        if summary is None:
            print("No metrics available")
            return

        print("\n" + "="*25)
        print("QUANTUM SIMULATOR METRICS")
        print("="*25)

        print("\n[EXECUTION]")
        print(f"  Total execution time: {summary['execution']['total_time_seconds']:.6f} seconds")

        print("\n[MEMORY]")
        mem = summary['memory']
        print(f"  Initial: {mem['initial_mb']:.2f} MB")
        print(f"  Peak:    {mem['peak_mb']:.2f} MB")
        print(f"  Final:   {mem['final_mb']:.2f} MB")
        print(f"  Delta:   {mem['delta_mb']:+.2f} MB")

        print("\n[OPERATIONS]")
        ops = summary['operations']
        print(f"  Gates:        {ops['gate_count']}")
        print(f"  Measurements: {ops['measurement_count']}")
        print(f"  Channels:     {ops['channel_count']}")
        print(f"  Total:        {ops['total_operations']}")

        print("\n[TIMING BREAKDOWN]")
        timing = summary['timing']
        print(f"  Gates:        {timing['gate_time_seconds']:.6f}s ({timing['gate_time_percent']:.1f}%)")
        print(f"  Measurements: {timing['measurement_time_seconds']:.6f}s ({timing['measurement_time_percent']:.1f}%)")
        print(f"  Channels:     {timing['channel_time_seconds']:.6f}s ({timing['channel_time_percent']:.1f}%)")

        if summary['channels']:
            print("\n[ERROR CHANNELS]")
            for name, stats in summary['channels'].items():
                print(f"  {name}:")
                print(f"    Hit count:  {stats['hit_count']}")
                print(f"    Total time: {stats['total_time_seconds']:.6f}s")
                print(f"    Avg time:   {stats['avg_time_seconds']:.6f}s")
                if stats['kraus_outcomes']:
                    # Calculate distribution
                    total = len(stats['kraus_outcomes'])
                    unique = set(stats['kraus_outcomes'])
                    dist = {k: stats['kraus_outcomes'].count(k)/total for k in unique}
                    print(f"    Kraus dist: {dist}")

        print("\n" + "="*70)

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
                    circuit_str += f"  {i+1}. qubit {qubit} stored in cbit{cbit}"
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
