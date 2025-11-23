"""
This module verifies correctness of the Clifford gate implementation in the Pauli
simulator by comparing measurement statistics against the exact state vector simulator.
It also benchmarks performance differences between the two approaches.
"""

import numpy as np
import time
import sys
sys.path.insert(0, '..')

from Tableau_Ver2 import Tableau
from pauli_error_channels import NoisySimulator
from QuantumCircuit import QuantumCircuit
from gates.registry import GateRegistry

gate_registry = GateRegistry()

def verify_single_qubit_gates():
    tests_passed = 0
    tests_total = 0

    # Test 1: X gate on |0⟩ → |1⟩
    tests_total += 1
    t = Tableau(1)
    t.x(0)
    outcomes = [t.copy().measure(0) for _ in range(100)]
    if all(o == 1 for o in outcomes):
        print("✓ X|0⟩ = |1⟩")
        tests_passed += 1
    else:
        print("✗ X|0⟩ failed")

    # Test 2: H gate creates superposition
    tests_total += 1
    outcomes = {'0': 0, '1': 0}
    for _ in range(1000):
        t = Tableau(1)
        t.h(0)
        outcomes[str(t.measure(0))] += 1
    # Should be roughly 50/50
    ratio = outcomes['0'] / 1000
    if 0.4 < ratio < 0.6:
        print(f"✓ H|0⟩ creates superposition: {outcomes}")
        tests_passed += 1
    else:
        print(f"✗ H|0⟩ superposition failed: {outcomes}")

    # Test 3: HXH = Z (phase flip)
    tests_total += 1
    t = Tableau(1)
    t.h(0)
    t.x(0)
    t.h(0)
    # This should give |0⟩ with phase -1, measurement still gives 0
    outcome = t.measure(0)
    if outcome == 0:
        print("✓ HXH|0⟩ = Z|0⟩ = |0⟩")
        tests_passed += 1
    else:
        print(f"✗ HXH|0⟩ failed: got {outcome}")

    # Test 4: S gate (phase gate)
    tests_total += 1
    t = Tableau(1)
    t.h(0)
    t.s(0)
    t.s(0)  # S² = Z
    t.h(0)
    # H S² H = HZH = X, so X|0⟩ = |1⟩
    outcome = t.measure(0)
    if outcome == 1:
        print("✓ HS²H|0⟩ = X|0⟩ = |1⟩")
        tests_passed += 1
    else:
        print(f"✗ HS²H|0⟩ failed: got {outcome}")

    print(f"\nSingle-qubit tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def verify_bell_states():
    tests_passed = 0
    tests_total = 0

    # Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    tests_total += 1
    outcomes = {'00': 0, '01': 0, '10': 0, '11': 0}
    for _ in range(1000):
        t = Tableau(2)
        t.h(0)
        t.cx(0, 1)
        m0 = t.measure(0)
        m1 = t.measure(1)
        outcomes[f"{m0}{m1}"] += 1

    if outcomes['01'] == 0 and outcomes['10'] == 0:
        ratio = outcomes['00'] / 1000
        if 0.4 < ratio < 0.6:
            print(f"✓ Bell |Φ+⟩: {outcomes}")
            tests_passed += 1
        else:
            print(f"✗ Bell |Φ+⟩ ratio wrong: {outcomes}")
    else:
        print(f"✗ Bell |Φ+⟩ correlations wrong: {outcomes}")

    tests_total += 1
    qc = QuantumCircuit(2, noise_model=None)
    qc.add_gate(gate_registry.get('h'), 0)
    qc.add_gate(gate_registry.get('cx'), [0, 1])
    qc.execute()
    state = qc.state

    expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
    if np.allclose(np.abs(state), np.abs(expected), atol=1e-10):
        print("✓ Exact simulator confirms Bell state amplitudes")
        tests_passed += 1
    else:
        print(f"✗ Exact simulator mismatch: {state}")

    print(f"\nBell state tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def verify_ghz_state():
    tests_passed = 0
    tests_total = 0

    tests_total += 1
    outcomes = {}
    for _ in range(1000):
        t = Tableau(3)
        t.h(0)
        t.cx(0, 1)
        t.cx(1, 2)
        result = ''.join(str(t.measure(i)) for i in range(3))
        outcomes[result] = outcomes.get(result, 0) + 1

    valid = all(k in ['000', '111'] for k in outcomes.keys())
    if valid and len(outcomes) == 2:
        ratio = outcomes.get('000', 0) / 1000
        if 0.4 < ratio < 0.6:
            print(f"✓ 3-qubit GHZ: {outcomes}")
            tests_passed += 1
        else:
            print(f"✗ 3-qubit GHZ ratio wrong: {outcomes}")
    else:
        print(f"✗ 3-qubit GHZ correlations wrong: {outcomes}")

    tests_total += 1
    qc = QuantumCircuit(3, noise_model=None)
    qc.add_gate(gate_registry.get('h'), 0)
    qc.add_gate(gate_registry.get('cx'), [0, 1])
    qc.add_gate(gate_registry.get('cx'), [1, 2])
    qc.execute()
    state = qc.state

    expected = np.zeros(8, dtype=complex)
    expected[0] = 1/np.sqrt(2)
    expected[7] = 1/np.sqrt(2)

    if np.allclose(np.abs(state), np.abs(expected), atol=1e-10):
        print("✓ Exact simulator confirms GHZ state amplitudes")
        tests_passed += 1
    else:
        print(f"✗ Exact simulator mismatch")

    print(f"\nGHZ state tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def verify_measurement_statistics():
    n_qubits = 3
    n_shots = 500

    np.random.seed(42)
    gates = []
    for _ in range(20):
        gate_type = np.random.choice(['h', 's', 'cx', 'x', 'z'])
        if gate_type == 'cx':
            c, t = np.random.choice(n_qubits, 2, replace=False)
            gates.append(('cx', c, t))
        else:
            q = np.random.randint(n_qubits)
            gates.append((gate_type, q))

    pauli_outcomes = {}
    for _ in range(n_shots):
        t = Tableau(n_qubits)
        for g in gates:
            if g[0] == 'h':
                t.h(g[1])
            elif g[0] == 's':
                t.s(g[1])
            elif g[0] == 'x':
                t.x(g[1])
            elif g[0] == 'z':
                t.z(g[1])
            elif g[0] == 'cx':
                t.cx(g[1], g[2])
        result = ''.join(str(t.measure(i)) for i in range(n_qubits))
        pauli_outcomes[result] = pauli_outcomes.get(result, 0) + 1

    exact_outcomes = {}
    for _ in range(n_shots):
        qc = QuantumCircuit(n_qubits, noise_model=None)
        for g in gates:
            if g[0] == 'h':
                qc.add_gate(gate_registry.get('h'), g[1])
            elif g[0] == 's':
                qc.add_gate(gate_registry.get('s'), g[1])
            elif g[0] == 'x':
                qc.add_gate(gate_registry.get('x'), g[1])
            elif g[0] == 'z':
                qc.add_gate(gate_registry.get('z'), g[1])
            elif g[0] == 'cx':
                qc.add_gate(gate_registry.get('cx'), [g[1], g[2]])
        qc.execute()

        probs = np.abs(qc.state)**2
        idx = np.random.choice(len(probs), p=probs)
        result = format(idx, f'0{n_qubits}b')
        exact_outcomes[result] = exact_outcomes.get(result, 0) + 1

    all_keys = set(pauli_outcomes.keys()) | set(exact_outcomes.keys())
    max_diff = 0
    for k in all_keys:
        p1 = pauli_outcomes.get(k, 0) / n_shots
        p2 = exact_outcomes.get(k, 0) / n_shots
        max_diff = max(max_diff, abs(p1 - p2))

    print(f"Pauli outcomes: {dict(sorted(pauli_outcomes.items()))}")
    print(f"Exact outcomes: {dict(sorted(exact_outcomes.items()))}")
    print(f"Max probability difference: {max_diff:.3f}")

    if max_diff < 0.10:
        print("✓ Distributions match within tolerance")
        return True
    else:
        print("✗ Distributions differ - bug in Pauli simulator detected")
        return False 

def benchmark_scaling():
    qubit_counts = [4, 6, 8, 10, 12]
    n_gates = 50

    print(f"\n{'Qubits':<10}{'Pauli (ms)':<15}{'Exact (ms)':<15}{'Speedup':<10}")
    print("-" * 50)

    for n in qubit_counts:
        if n > 14:
            continue
        np.random.seed(42)
        gates = []
        for _ in range(n_gates):
            gate_type = np.random.choice(['h', 's', 'cx'])
            if gate_type == 'cx':
                c, t = np.random.choice(n, 2, replace=False)
                gates.append(('cx', c, t))
            else:
                q = np.random.randint(n)
                gates.append((gate_type, q))

        n_reps = 3
        start = time.perf_counter()
        for _ in range(n_reps):
            t = Tableau(n)
            for g in gates:
                if g[0] == 'h':
                    t.h(g[1])
                elif g[0] == 's':
                    t.s(g[1])
                elif g[0] == 'cx':
                    t.cx(g[1], g[2])
            for i in range(n):
                t.measure(i)
        pauli_time = (time.perf_counter() - start) / n_reps * 1000

        start = time.perf_counter()
        for _ in range(n_reps):
            qc = QuantumCircuit(n, noise_model=None)
            for g in gates:
                if g[0] == 'h':
                    qc.add_gate(gate_registry.get('h'), g[1])
                elif g[0] == 's':
                    qc.add_gate(gate_registry.get('s'), g[1])
                elif g[0] == 'cx':
                    qc.add_gate(gate_registry.get('cx'), [g[1], g[2]])
            qc.execute()
        exact_time = (time.perf_counter() - start) / n_reps * 1000

        speedup = exact_time / pauli_time if pauli_time > 0 else float('inf')
        print(f"{n:<10}{pauli_time:<15.2f}{exact_time:<15.2f}{speedup:<10.1f}x")

    print("\nNote: Pauli simulator is O(n²) per gate, exact is O(2^n)")


def benchmark_gate_operations():
    n = 10
    n_ops = 10000

    t = Tableau(n, enable_metrics=False)

    # H gates
    start = time.perf_counter()
    for _ in range(n_ops):
        t.h(0)
    h_time = (time.perf_counter() - start) * 1000

    # S gates
    start = time.perf_counter()
    for _ in range(n_ops):
        t.s(0)
    s_time = (time.perf_counter() - start) * 1000

    # CNOT gates
    start = time.perf_counter()
    for _ in range(n_ops):
        t.cx(0, 1)
    cx_time = (time.perf_counter() - start) * 1000

    # X gates
    start = time.perf_counter()
    for _ in range(n_ops):
        t.x(0)
    x_time = (time.perf_counter() - start) * 1000

    print(f"\n{n_ops} operations each:")
    print(f"  H gate:    {h_time:.2f} ms ({h_time/n_ops*1000:.2f} µs/op)")
    print(f"  S gate:    {s_time:.2f} ms ({s_time/n_ops*1000:.2f} µs/op)")
    print(f"  CNOT gate: {cx_time:.2f} ms ({cx_time/n_ops*1000:.2f} µs/op)")
    print(f"  X gate:    {x_time:.2f} ms ({x_time/n_ops*1000:.2f} µs/op)")


def benchmark_with_metrics():
    print("\n" + "="*60)
    print("BENCHMARK: Circuit Execution with Metrics")
    print("="*60)

    n = 8
    t = Tableau(n, enable_metrics=True)

    # apply 100 random Clifford gates
    np.random.seed(42)
    for _ in range(100):
        gate_type = np.random.choice(['h', 's', 'cx', 'x', 'y', 'z'])
        if gate_type == 'cx':
            c, t_q = np.random.choice(n, 2, replace=False)
            t.cx(c, t_q)
        elif gate_type == 'h':
            t.h(np.random.randint(n))
        elif gate_type == 's':
            t.s(np.random.randint(n))
        elif gate_type == 'x':
            t.x(np.random.randint(n))
        elif gate_type == 'y':
            t.y(np.random.randint(n))
        elif gate_type == 'z':
            t.z(np.random.randint(n))

    # measure all qubits
    for i in range(n):
        t.measure(i)

    t.print_metrics()


def benchmark_noisy_simulator():
    """Benchmark NoisySimulator with error channels."""
    print("\n" + "="*60)
    print("BENCHMARK: Noisy Simulator Performance")
    print("="*60)

    n = 8
    error_config = {
        'h': (0.01, 0.01, 0.01),
        's': (0.01, 0.01, 0.01),
        'cx': (0.02, 0.02, 0.02),
    }

    ns = NoisySimulator(n, error_config=error_config, enable_metrics=True)

    # apply 100 random gates
    np.random.seed(42)
    for _ in range(100):
        gate_type = np.random.choice(['h', 's', 'cx'])
        if gate_type == 'cx':
            c, t_q = np.random.choice(n, 2, replace=False)
            ns.cx(c, t_q)
        elif gate_type == 'h':
            ns.h(np.random.randint(n))
        elif gate_type == 's':
            ns.s(np.random.randint(n))

    # Measure all qubits
    for i in range(n):
        ns.measure(i)

    ns.print_metrics()


def run_all():
    v1 = verify_single_qubit_gates()
    v2 = verify_bell_states()
    v3 = verify_ghz_state()
    v4 = verify_measurement_statistics()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if v1 and v2 and v3 and v4:
        print("✓ All verification tests PASSED")
    else:
        print("✗ Some verification tests FAILED")
        return False

    benchmark_scaling()
    benchmark_gate_operations()
    benchmark_with_metrics()
    benchmark_noisy_simulator()
    return True


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
