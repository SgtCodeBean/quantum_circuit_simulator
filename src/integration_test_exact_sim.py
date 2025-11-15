import numpy as np
import pytest
import sys

sys.path.append('')

try:
    from QuantumCircuit import QuantumCircuit
    from gates.registry import GateRegistry
    from error_channels.ChannelRegistry import ChannelRegistry
    from gates.registry import Gate
except ImportError:
    print("Error: Could not import simulator components.")
    print("Please ensure 'src' is in the Python path and contains:")
    print("QuantumCircuit.py, gates/registry.py, error_channels/ChannelRegistry.py")
    sys.exit(1)


@pytest.fixture(scope='module')
def registries():
    """Provides the Gate and Channel registries to the tests."""
    try:
        gate_reg = GateRegistry()
        chan_reg = ChannelRegistry()
        return gate_reg, chan_reg
    except Exception as e:
        pytest.fail(f"Failed to initialize registries: {e}")


def s(state_vector):
    """Helper to format a state vector for printing."""
    return "[" + ", ".join([f"{c.real:+.2f}{c.imag:+.2f}j" for c in state_vector]) + "]"


# --- Test Suite ---

def test_pauli_x_gate(registries):
    """Tests the X gate on |0⟩."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=1)

    # State |0⟩ -> X -> |1⟩
    qc.add_gate(gate_reg.get('x'), targets=0)
    qc.execute()

    actual_state = qc.get_state()
    expected_state = np.array([0.0, 1.0], dtype=complex)

    print(f"Pauli X Test: Expected {s(expected_state)}, Got {s(actual_state)}")
    assert np.allclose(actual_state, expected_state), "X gate failed: |0⟩ did not become |1⟩"


def test_pauli_z_gate(registries):
    """Tests the Z gate on |1⟩."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=1)

    # State |1⟩ -> Z -> -|1⟩
    qc.add_gate(gate_reg.get('x'), targets=0)
    qc.add_gate(gate_reg.get('z'), targets=0)
    qc.execute()

    actual_state = qc.get_state()
    expected_state = np.array([0, -1], dtype=complex)

    print(f"Pauli Z Test: Expected {s(expected_state)}, Got {s(actual_state)}")
    assert np.allclose(actual_state, expected_state), "Z gate failed: |1⟩ did not become -|1⟩"


def test_hadamard_gate(registries):
    """Tests the H gate on |0⟩."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=1)

    # State |0⟩ -> H -> |0⟩ + |1⟩ / √2
    qc.add_gate(gate_reg.get('h'), targets=0)
    qc.execute()

    actual_state = qc.get_state()
    expected_state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)

    print(f"Hadamard Test: Expected {s(expected_state)}, Got {s(actual_state)}")
    assert np.allclose(actual_state, expected_state), "H gate failed: |0⟩ did not become |+⟩"


def test_cnot_gate(registries):
    """Tests the CNOT gate on all 4 computational basis states."""
    gate_reg, _ = registries

    # Test |00⟩ -> |00⟩
    qc = QuantumCircuit(num_qubits=2)
    qc.add_gate(gate_reg.get('cx'), targets=[0, 1])
    qc.execute()
    assert np.allclose(qc.get_state(), [1, 0, 0, 0]), "CNOT failed on |00⟩"

    # Test |01⟩ -> |01⟩
    qc = QuantumCircuit(num_qubits=2)
    qc.add_gate(gate_reg.get('x'), targets=1)
    qc.add_gate(gate_reg.get('cx'), targets=[0, 1])
    qc.execute()
    assert np.allclose(qc.get_state(), [0, 1, 0, 0]), "CNOT failed on |01⟩"

    # Test |10⟩ -> |11⟩
    qc = QuantumCircuit(num_qubits=2)
    qc.add_gate(gate_reg.get('x'), targets=0)
    qc.add_gate(gate_reg.get('cx'), targets=[0, 1])
    qc.execute()
    assert np.allclose(qc.get_state(), [0, 0, 0, 1]), "CNOT failed on |10⟩"

    # Test |11⟩ -> |10⟩
    qc = QuantumCircuit(num_qubits=2)
    qc.add_gate(gate_reg.get('x'), targets=0)
    qc.add_gate(gate_reg.get('x'), targets=1)
    qc.add_gate(gate_reg.get('cx'), targets=[0, 1])
    qc.execute()
    assert np.allclose(qc.get_state(), [0, 0, 1, 0]), "CNOT failed on |11⟩"


def test_phase_gate(registries):
    """Tests the S (phase) gate on |+⟩."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=1)

    # State |0⟩ -> H -> S -> (|0⟩ + i|1⟩)/√2
    qc.add_gate(gate_reg.get('h'), targets=0)
    qc.add_gate(gate_reg.get('s'), targets=0)
    qc.execute()

    actual_state = qc.get_state()
    expected_state = np.array([1 / np.sqrt(2), 1j / np.sqrt(2)], dtype=complex)

    print(f"S Test: Expected {s(expected_state)}, Got {s(actual_state)}")
    assert np.allclose(actual_state, expected_state), "S gate failed: |+⟩ did not become (|0⟩ + i|1⟩)/√2"


def test_qubit_reset(registries):
    """Tests that reset_qubit correctly resets to |0⟩."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=1, num_cbits=1)

    # Create |1⟩ state
    qc.add_gate(gate_reg.get('x'), targets=0)
    qc.execute()
    assert np.allclose(qc.get_state(), [0, 1]), "Failed to create |1⟩"

    qc.measure(qubit=0, cbit=0)

    # Reset the qubit
    qc.reset_qubit(qubit=0)
    qc.execute() 

    actual_state = qc.get_state()
    expected_state = np.array([1.0, 0.0], dtype=complex)

    print(f"Reset Test: Expected {s(expected_state)}, Got {s(actual_state)}")
    assert np.allclose(actual_state, expected_state), "reset_qubit failed"

  
# add more tests...
def test_noisy_cnot(registries):
    gate_reg, error_reg = registries
    qc = QuantumCircuit(num_qubits=2, num_cbits=2)
    phase_error = error_reg.get_param('bit_phase_flip').instantiate(1)

    noisy_cnot = Gate('ncx', gate_reg.get('cx').matrix, [phase_error])
    qc.add_gate(gate_reg.get('x'), 0)
    qc.add_gate(noisy_cnot, (0, 1))
    qc.measure(1, 0)
    qc.execute()

    actual_value = qc.get_cbit(0)
    actuale_state = qc.get_state()
    expected_value = 0
    expected_state = np.array([-1, 0, 0, 0], dtype=complex)

    assert actual_value == expected_value, "noisy_cnot test failure"
    assert np.allclose(actuale_state, expected_state), "noisy_cnot test failure: states are not equal"

if __name__ == "__main__":
    pytest.main([__file__])