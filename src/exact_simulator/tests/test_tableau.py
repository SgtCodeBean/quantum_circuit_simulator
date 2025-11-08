import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import numpy as np
from exact_simulator.Tableau import Tableau
from QuantumCircuit import QuantumCircuit
from gates.registry import GateRegistry

def test_basic():
    """Test basic tableau operations"""
    t = Tableau(2)
    t.h(0)
    t.cx(0, 1)  # Bell state
    print("✓ Basic operations")

def test_vs_statevector():
    """Compare with state vector simulation"""
    np.random.seed(42)

    tests = [
        ("Bell state", 2, [('h', 0), ('cx', (0,1))]),
        ("GHZ state", 3, [('h', 0), ('cx', (0,1)), ('cx', (1,2))]),
        ("Random", 5, [('h', 0), ('s', 1), ('cx', (0,2)), ('h', 3)])
    ]

    for name, n, circuit in tests:
        # Tableau
        t = Tableau(n)
        for gate, *args in circuit:
            if gate == 'h':
                t.h(args[0])
            elif gate == 's':
                t.s(args[0])
            elif gate == 'cx':
                t.cx(*args[0])

        # State vector
        qc = QuantumCircuit(n)
        reg = GateRegistry()
        for gate, *args in circuit:
            if gate == 'cx':
                qc.add_gate(reg.get('cx'), args[0])
            else:
                qc.add_gate(reg.get(gate), args[0])
        qc.execute()

        print(f"✓ {name}")

def benchmark():
    """Quick performance test"""
    import time

    n = 15
    depth = 20

    # Build random circuit
    np.random.seed(123)
    gates_1q = ['h', 's', 'x', 'z']

    # Tableau
    t0 = time.perf_counter()
    t = Tableau(n)
    for _ in range(depth):
        for q in range(n):
            if np.random.random() < 0.5:
                gate = np.random.choice(gates_1q)
                getattr(t, gate)(q)
        for _ in range(n//2):
            c, tgt = np.random.randint(0, n, 2)
            if c != tgt:
                t.cx(c, tgt)
    t1 = time.perf_counter()

    print(f"✓ {n} qubits, depth {depth}: {(t1-t0)*1000:.2f}ms")

if __name__ == "__main__":
    print("Testing optimized Tableau class...")
    test_basic()
    test_vs_statevector()
    benchmark()
