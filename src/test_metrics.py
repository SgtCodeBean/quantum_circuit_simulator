"""
Test script for simulator metrics functionality.

- Wall time computation for gates and measurements
- Memory usage tracking
- Error channel statistics
- Operation counts
"""

import sys
sys.path.append('.')

import numpy as np
from QuantumCircuit import QuantumCircuit
from gates.registry import GateRegistry
from error_channels.Channel import Channel
import time

def test_basic_metrics():
    """Test basic metrics collection for gates and measurements."""
    print("TEST 1: Basic Metrics (Gates and Measurements)")

    reg = GateRegistry()

    qc = QuantumCircuit(num_qubits=3, num_cbits=2, enable_metrics=True)

    qc.add_gate(reg.get('h'), targets=0)  # Hadamard on qubit 0
    qc.add_gate(reg.get('x'), targets=1)  # X gate on qubit 1
    qc.add_gate(reg.get('z'), targets=2)  # Z gate on qubit 2
    qc.add_gate(reg.get('h'), targets=0)  # Another Hadamard
    qc.measure(0, 0)  # Measure qubit 0
    qc.measure(1, 1)  # Measure qubit 1

    qc.execute()
    qc.print_metrics()
    metrics = qc.get_metrics()

    print("\n[Verification]")
    print(f"  Expected 4 gates, got: {metrics['operations']['gate_count']}")
    print(f"  Expected 2 measurements, got: {metrics['operations']['measurement_count']}")
    print(f"  Total operations: {metrics['operations']['total_operations']}")

    assert metrics['operations']['gate_count'] == 4, "Gate count mismatch"
    assert metrics['operations']['measurement_count'] == 2, "Measurement count mismatch"
    print("  ✓ All assertions passed!")


def test_large_circuit_metrics():
    """Test metrics on a larger circuit to see timing differences."""
    print("TEST 2: Large Circuit Metrics")

    reg = GateRegistry()

    num_qubits = 10
    qc = QuantumCircuit(num_qubits=num_qubits, enable_metrics=True)

    for i in range(num_qubits):
        qc.add_gate(reg.get('h'), targets=i)

    for i in range(num_qubits - 1):
        qc.add_gate(reg.get('x'), targets=i)

    qc.execute()
    qc.print_metrics()

    metrics = qc.get_metrics()
    print(f"\n[Performance Info]")
    print(f"  Simulated {num_qubits} qubits")
    print(f"  State vector size: 2^{num_qubits} = {2**num_qubits} complex amplitudes")
    print(f"  Memory per state vector: ~{2**num_qubits * 16 / 1024:.2f} KB (complex128)")
    print(f"  Average time per gate: {metrics['timing']['gate_time_seconds'] / metrics['operations']['gate_count']:.6f}s")


def test_channel_metrics():
    """Test metrics collection for error channels."""
    print("TEST 3: Error Channel Metrics")

    qc = QuantumCircuit(num_qubits=2, enable_metrics=True)
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    bit_flip = Channel("bit_flip", [np.sqrt(0.7)*I, np.sqrt(0.3)*X])

    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    p = 0.1
    depolarizing = Channel("depolarizing", [
        np.sqrt(1-p)*I,
        np.sqrt(p/3)*X,
        np.sqrt(p/3)*Y,
        np.sqrt(p/3)*Z
    ])

    psi = qc.state[:2]  # Single qubit state (|0>)

    rng = np.random.default_rng(42)

    # apply bit flip channel 50 times
    for _ in range(50):
        start = time.perf_counter()
        psi = bit_flip.apply_statevector(
            psi,
            rng=rng,
            metrics_callback=lambda name, kraus_index: qc.record_channel_hit(
                name,
                duration=time.perf_counter() - start,
                kraus_index=kraus_index
            )
        )

    # apply depolarizing channel 30 times
    for _ in range(30):
        start = time.perf_counter()
        psi = depolarizing.apply_statevector(
            psi,
            rng=rng,
            metrics_callback=lambda name, kraus_index: qc.record_channel_hit(
                name,
                duration=time.perf_counter() - start,
                kraus_index=kraus_index
            )
        )

    qc.print_metrics()
    metrics = qc.get_metrics()
    print("\n[Verification]")
    print(f"  Bit flip applications: {metrics['channels']['bit_flip']['hit_count']}")
    print(f"  Depolarizing applications: {metrics['channels']['depolarizing']['hit_count']}")
    print(f"  Total channel applications: {metrics['operations']['channel_count']}")


def test_metrics_disabled():
    """Test that metrics can be disabled for zero overhead."""
    print("TEST 4: Metrics Disabled (Default Behavior)")

    reg = GateRegistry()
    qc = QuantumCircuit(num_qubits=2, enable_metrics=False)

    qc.add_gate(reg.get('h'), targets=0)
    qc.add_gate(reg.get('x'), targets=1)
    qc.execute()

    qc.print_metrics()
    metrics = qc.get_metrics()

    assert metrics is None, "Metrics should be None when disabled"
    print("\n  ✓ Metrics correctly disabled!")


def test_metrics_reset():
    """Test that metrics can be reset."""
    print("TEST 5: Metrics Reset")

    reg = GateRegistry()
    qc = QuantumCircuit(num_qubits=2, enable_metrics=True)

    qc.add_gate(reg.get('h'), targets=0)
    qc.execute()
    print("\nFirst execution:")
    qc.print_metrics()

    first_metrics = qc.get_metrics()
    first_gate_count = first_metrics['operations']['gate_count']

    qc.reset_metrics()
    print("\n\n[After reset]")

    qc.ops = []
    qc.add_gate(reg.get('x'), targets=0)
    qc.add_gate(reg.get('x'), targets=1)
    qc.execute()

    print("\nSecond execution (after reset):")
    qc.print_metrics()

    second_metrics = qc.get_metrics()
    second_gate_count = second_metrics['operations']['gate_count']

    print(f"\n[Verification]")
    print(f"  First run: {first_gate_count} gates")
    print(f"  Second run: {second_gate_count} gates (after reset)")
    print(f"  Metrics were properly reset: {second_gate_count == 2}")


def main():
    """Run all metrics tests."""

    try:
        test_basic_metrics()
        test_large_circuit_metrics()
        test_channel_metrics()
        test_metrics_disabled()
        test_metrics_reset()

        print("\n ALL TESTS PASSED ✓")

    except Exception as e:
        print(f"\n TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
