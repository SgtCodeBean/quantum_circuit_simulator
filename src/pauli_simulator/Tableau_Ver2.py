import numpy as np
import time
import psutil
import os

"""
Corrected Tableau class for efficient Clifford circuit simulation.
(Measurement logic has been corrected)
"""


class Tableau:
    __slots__ = ('n', '_x', '_z', '_r', '_metrics', '_process', '_enable_metrics', 'ops', 'num_cbits', '_cbits', '_measurements')

    def __init__(self, n, num_cbits=0, enable_metrics=False):
        """
        Initialize tableau for n qubits in |0...0⟩ state.

        Args:
            n (int): Number of qubits
            enable_metrics (bool): Whether to collect performance metrics
        """
        self.n = n
        self.num_cbits = num_cbits
        self._x = np.zeros((2 * n, n), dtype=np.uint8)
        self._z = np.zeros((2 * n, n), dtype=np.uint8)
        self._r = np.zeros(2 * n, dtype=np.uint8)
        for i in range(n):
            self._x[i, i] = 1
            self._z[n + i, i] = 1

        self._cbits = np.zeros(self.num_cbits, dtype=np.uint8)
        self.ops = []
        self._measurements = {}

        self._enable_metrics = enable_metrics
        self._metrics = None
        self._process = None
        if enable_metrics:
            self._init_metrics()

    def _init_metrics(self):
        """Initialize metrics dictionary."""
        try :
            self._process = psutil.Process(os.getpid())
        except :
            self._process = None
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
            },
            'memory': {
                'initial_mb': self._get_memory_mb(),
                'peak_mb': self._get_memory_mb(),
                'final_mb': 0.0,
                'delta_mb': 0.0,
                'gate_memory': {
                    'samples': [],
                    'avg_mb': 0.0,
                    'peak_mb': 0.0
                },
                'measurement_memory': {
                    'samples': [],
                    'avg_mb': 0.0,
                    'peak_mb': 0.0
                },
                'tableau_size_mb': self._calculate_tableau_size_mb()
            }
        }
    
    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        if self._process is None:
            return 0.0
        try:
            return self._process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
        
    def _calculate_tableau_size_mb(self) -> float:
        """Calculate the theoretical memory size of the tableau."""
        # Each array element is uint8 (1 byte)
        # _x: (2n, n), _z: (2n, n), _r: (2n,)
        x_size = self._x.nbytes
        z_size = self._z.nbytes
        r_size = self._r.nbytes
        cbits_size = self._cbits.nbytes if self.num_cbits > 0 else 0
        
        total_bytes = x_size + z_size + r_size + cbits_size
        return total_bytes / (1024 * 1024)
    
    def _record_peak_memory(self):
        current_mem = self._get_memory_mb()
        self._metrics['memory']['gate_memory']['samples'].append(current_mem)
        self._metrics['memory']['gate_memory']['peak_mb'] = max(
            self._metrics['memory']['gate_memory'].get('peak_mb', 0),
            current_mem
        )

    def _record_gate(self, gate_name, elapsed):
        """Record a gate operation."""
        if not self._enable_metrics:
            return
        self._metrics['operations']['gate_count'] += 1
        self._metrics['operations']['total_operations'] += 1
        self._metrics['operations']['gates_by_type'][gate_name] += 1
        self._metrics['timing']['gate_time_seconds'] += elapsed
        self._record_peak_memory()

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
        self._record_peak_memory()

    def _finalize_memory(self):
        """Calculate final memory statistics."""
        if not self._enable_metrics:
            return
        
        # Final memory snapshot
        self._metrics['memory']['final_mb'] = self._get_memory_mb()
        self._metrics['memory']['delta_mb'] = (
            self._metrics['memory']['final_mb'] - 
            self._metrics['memory']['initial_mb']
        )
        
        # Calculate average memory for gates
        gate_samples = self._metrics['memory']['gate_memory']['samples']
        if gate_samples:
            self._metrics['memory']['gate_memory']['avg_mb'] = sum(gate_samples) / len(gate_samples)
        
        # Calculate average memory for measurements
        meas_samples = self._metrics['memory']['measurement_memory']['samples']
        if meas_samples:
            self._metrics['memory']['measurement_memory']['avg_mb'] = sum(meas_samples) / len(meas_samples)


    def h(self, q):
        """Hadamard gate"""
        start = time.perf_counter() if self._enable_metrics else 0
        for i in range(2 * self.n):
            self._r[i] ^= self._x[i, q] & self._z[i, q]
            self._x[i, q], self._z[i, q] = self._z[i, q], self._x[i, q]
        if self._enable_metrics:
            self._record_gate('h', time.perf_counter() - start)
        
        self.ops.append(('gate', 'h', [q]))
        return self

    def s(self, q):
        """Phase gate"""
        start = time.perf_counter() if self._enable_metrics else 0
        for i in range(2 * self.n):
            self._r[i] ^= self._x[i, q] & self._z[i, q]
            self._z[i, q] ^= self._x[i, q]
        if self._enable_metrics:
            self._record_gate('s', time.perf_counter() - start)
        
        self.ops.append(('gate', 's', [q]))
        return self

    def cx(self, c, t):
        """CNOT gate: control c, target t"""
        start = time.perf_counter() if self._enable_metrics else 0
        for i in range(2 * self.n):
            self._r[i] ^= self._x[i, c] & self._z[i, t] & (self._x[i, t] ^ self._z[i, c] ^ 1)
            self._x[i, t] ^= self._x[i, c]
            self._z[i, c] ^= self._z[i, t]
        if self._enable_metrics:
            self._record_gate('cx', time.perf_counter() - start)

        self.ops.append(('gate', 'cx', [c, t]))
        return self

    def x(self, q):
        """Pauli X: bit flip"""
        start = time.perf_counter() if self._enable_metrics else 0
        for i in range(2 * self.n):
            self._r[i] ^= self._z[i, q]
        if self._enable_metrics:
            self._record_gate('x', time.perf_counter() - start)
        
        self.ops.append(('gate', 'x', [q]))
        return self

    def y(self, q):
        """Pauli Y: bit + phase flip"""
        start = time.perf_counter() if self._enable_metrics else 0
        for i in range(2 * self.n):
            self._r[i] ^= self._x[i, q] ^ self._z[i, q]
        if self._enable_metrics:
            self._record_gate('y', time.perf_counter() - start)

        self.ops.append(('gate', 'y', [q]))
        return self

    def z(self, q):
        """Pauli Z: phase flip"""
        start = time.perf_counter() if self._enable_metrics else 0
        for i in range(2 * self.n):
            self._r[i] ^= self._x[i, q]
        if self._enable_metrics:
            self._record_gate('z', time.perf_counter() - start)

        self.ops.append(('gate', 'z', [q]))
        return self

    def measure(self, q, cbit=None):
        """
        Measure qubit q in the Z basis.
        Returns the measurement outcome (0 or 1).
        """
        start = time.perf_counter() if self._enable_metrics else 0

        if cbit is not None:
            if cbit < 0 or cbit >= self.num_cbits:
                raise ValueError(f"Classical bit {cbit} out of range [0, {self.num_cbits})")
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
            
            if cbit is not None:
                self._cbits[cbit] = outcome

            self._measurements[q] = {
                'outcome': outcome,
                'cbit': cbit,
                'deterministic': False
            }

            self.ops.append(('measure', q, cbit, outcome))

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

                    g = np.zeros(self.n, dtype=int)

                    mask_y = (x1 == 1) & (z1 == 1)
                    g[mask_y] = z2[mask_y].astype(int) - x2[mask_y].astype(int)

                    mask_x = (x1 == 1) & (z1 == 0)
                    g[mask_x] = z2[mask_x].astype(int) * (2 * x2[mask_x].astype(int) - 1)

                    mask_z = (x1 == 0) & (z1 == 1)
                    g[mask_z] = x2[mask_z].astype(int) * (1 - 2 * z2[mask_z].astype(int))
                    
                    g_sum = (2 * self._r[i + self.n] + 2 * temp_r + np.sum(g)) % 4
                    temp_r = g_sum >> 1
                    temp_x ^= x1
                    temp_z ^= z1

            if self._enable_metrics:
                self._record_measurement(temp_r, True, time.perf_counter() - start)

            if cbit is not None:
                self._cbits[cbit] = temp_r

            self._measurements[q] = {
                'outcome': temp_r,
                'cbit': cbit,
                'deterministic': True
            }

            self.ops.append(('measure', q, cbit, temp_r))

            return temp_r
    
    def reset(self, q):
        """
        Reset qubit q to |0⟩ state.
        Implemented via measurement + conditional X gate.
        
        Args:
            q (int): Qubit to reset
            
        Returns:
            self: For method chaining
        """
        outcome = self.measure(q, cbit=None)
        
        # If measured 1, apply X to flip to 0
        if outcome == 1:
            self.x(q)

        self.ops.append(('reset', q))
        
        return self
    
    def get_classical_register(self):
        if self.num_cbits == 0:
            return {'bitstring': ''}
        
        result = {
            f'c[{i}]': int(self._cbits[i])
            for i in range(self.num_cbits)
        }
        result['bitstring'] = ''.join(str(int(self._cbits[i])) for i in range(self.num_cbits))
        return result
    
    def get_cbits(self):
        return self._cbits.copy()
    
    def get_cbit(self, index):
        if index < 0 or index >= self.num_cbits:
            raise ValueError(f"Classical bit {index} out of range [0, {self.num_cbits})")
        return self._cbits[index]
    
    def get_measurements(self):
        return self._measurements

    def get_metrics(self):
        """
        Get the collected metrics.

        Returns:
            dict: Metrics dictionary or None if metrics are disabled.
        """
        if not self._enable_metrics:
            return None
        
        self._finalize_memory()

        total_time = (self._metrics['timing']['gate_time_seconds'] +
                     self._metrics['timing']['measurement_time_seconds'])
        self._metrics['execution']['total_time_seconds'] = total_time

        return self._metrics.copy()
    
    def print_circuit(self):
        """Print a readable representation of the circuit structure."""
        print("\n" + "=" * 60)
        print(f"TABLEAU CIRCUIT: {self.n} qubits, {self.num_cbits} classical bits")
        print("=" * 60)
        
        if not self.ops:
            print("(empty circuit)")
        else:
            print(f"\nOperations ({len(self.ops)} total):\n")
            for i, op in enumerate(self.ops, 1):
                if op[0] == 'gate':
                    _, gate_name, qubits = op
                    if len(qubits) == 1:
                        print(f"  {i:3d}. {gate_name.upper():4s} q[{qubits[0]}]")
                    elif len(qubits) == 2:
                        print(f"  {i:3d}. {gate_name.upper():4s} q[{qubits[0]}], q[{qubits[1]}]")
                    else:
                        print(f"  {i:3d}. {gate_name.upper():4s} {qubits}")
                
                elif op[0] == 'measure':
                    _, qubit, cbit, outcome = op
                    if cbit is not None:
                        print(f"  {i:3d}. MEAS q[{qubit}] -> c[{cbit}] (outcome: {outcome})")
                    else:
                        print(f"  {i:3d}. MEAS q[{qubit}] (outcome: {outcome})")
                
                elif op[0] == 'reset':
                    _, qubit = op
                    print(f"  {i:3d}. RESET q[{qubit}]")
        
        # Print final classical register state
        if self.num_cbits > 0:
            print(f"\nFinal classical register: {self.get_classical_register()['bitstring']}")
        
        print("=" * 60 + "\n")

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
        
        print("\n--- Memory (MB) ---")
        print(f"Tableau theoretical size: {m['memory']['tableau_size_mb']:.4f}")
        print(f"Initial (RSS):          {m['memory']['initial_mb']:.2f}")
        print(f"Final (RSS):            {m['memory']['final_mb']:.2f}")
        print(f"Peak (RSS):             {m['memory']['peak_mb']:.2f}")
        print(f"Delta (Final - Initial): {m['memory']['delta_mb']:+.2f}")
        print(f"Avg Gate Op (RSS):      {m['memory']['gate_memory']['avg_mb']:.2f} (Peak: \
              {m['memory']['gate_memory']['peak_mb']:.2f})")
        print(f"Avg Measure Op (RSS):   {m['memory']['measurement_memory']['avg_mb']:.2f} \
              (Peak: {m['memory']['measurement_memory']['peak_mb']:.2f})")

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

        g = np.zeros(self.n, dtype=int)

        # Case: x1=1, z1=1 (Pauli Y) -> g = z2 - x2
        mask_y = (x1 == 1) & (z1 == 1)
        g[mask_y] = z2[mask_y].astype(int) - x2[mask_y].astype(int)

        # Case: x1=1, z1=0 (Pauli X) -> g = z2(2x2 - 1)
        mask_x = (x1 == 1) & (z1 == 0)
        g[mask_x] = z2[mask_x].astype(int) * (2 * x2[mask_x].astype(int) - 1)

        # Case: x1=0, z1=1 (Pauli Z) -> g = x2(1 - 2z2)
        mask_z = (x1 == 0) & (z1 == 1)
        g[mask_z] = x2[mask_z].astype(int) * (1 - 2 * z2[mask_z].astype(int))
        
        g_sum = (2 * self._r[i] + 2 * self._r[h] + np.sum(g)) % 4

        self._r[h] = g_sum >> 1

        self._x[h] ^= x1
        self._z[h] ^= z1

    def copy(self):
        t = Tableau.__new__(Tableau)
        t.n = self.n
        t.num_cbits = self.num_cbits
        t._x = self._x.copy()
        t._z = self._z.copy()
        t._r = self._r.copy()
        t._cbits = self._cbits.copy()
        t.ops = self.ops.copy()
        t._measurements = self._measurements.copy()
        t._enable_metrics = self._enable_metrics
        t._metrics = None
        t._process = None
        if self._enable_metrics and self._metrics:
            import copy
            t._metrics = copy.deepcopy(self._metrics)
        return t
