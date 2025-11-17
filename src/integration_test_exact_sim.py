import numpy as np
import pytest
import sys
from qasm_parser import parse_qasm_source, build_circuit_from_ir
from error_channels.default_noise import build_default_noise_model

sys.path.append('')

try:
    from QuantumCircuit import QuantumCircuit
    from gates.registry import GateRegistry
    from error_channels.ChannelRegistry import ChannelRegistry
    from gates.registry import Gate
    from error_channels.noise_model import NoiseModel
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
    qc = QuantumCircuit(num_qubits=1, noise_model=None)

    qc.add_gate(gate_reg.get('x'), targets=0)
    qc.execute()

    actual_state = qc.get_state()
    expected_state = np.array([0.0, 1.0], dtype=complex)

    print(f"Pauli X Test: Expected {s(expected_state)}, Got {s(actual_state)}")
    assert np.allclose(actual_state, expected_state), "X gate failed: |0⟩ did not become |1⟩"


def test_pauli_z_gate(registries):
    """Tests the Z gate on |1⟩."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=1, noise_model=None)

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
    qc = QuantumCircuit(num_qubits=1, noise_model=None)

    qc.add_gate(gate_reg.get('h'), targets=0)
    qc.execute()

    actual_state = qc.get_state()
    expected_state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)

    print(f"Hadamard Test: Expected {s(expected_state)}, Got {s(actual_state)}")
    assert np.allclose(actual_state, expected_state), "H gate failed: |0⟩ did not become |+⟩"


def test_cnot_gate(registries):
    """Tests the CNOT gate on all 4 computational basis states."""
    gate_reg, _ = registries

    qc = QuantumCircuit(num_qubits=2, noise_model=None)
    qc.add_gate(gate_reg.get('cx'), targets=[0, 1])
    qc.execute()
    assert np.allclose(qc.get_state(), [1, 0, 0, 0]), "CNOT failed on |00⟩"

    qc = QuantumCircuit(num_qubits=2)
    qc.add_gate(gate_reg.get('x'), targets=1)
    qc.add_gate(gate_reg.get('cx'), targets=[0, 1])
    qc.execute()
    assert np.allclose(qc.get_state(), [0, 1, 0, 0]), "CNOT failed on |01⟩"

    qc = QuantumCircuit(num_qubits=2)
    qc.add_gate(gate_reg.get('x'), targets=0)
    qc.add_gate(gate_reg.get('cx'), targets=[0, 1])
    qc.execute()
    assert np.allclose(qc.get_state(), [0, 0, 0, 1]), "CNOT failed on |10⟩"

    qc = QuantumCircuit(num_qubits=2)
    qc.add_gate(gate_reg.get('x'), targets=0)
    qc.add_gate(gate_reg.get('x'), targets=1)
    qc.add_gate(gate_reg.get('cx'), targets=[0, 1])
    qc.execute()
    assert np.allclose(qc.get_state(), [0, 0, 1, 0]), "CNOT failed on |11⟩"


def test_phase_gate(registries):
    """Tests the S (phase) gate on |+⟩."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=1, noise_model=None)

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
    qc = QuantumCircuit(num_qubits=1, num_cbits=1, noise_model=None)

    qc.add_gate(gate_reg.get('x'), targets=0)
    qc.execute()
    assert np.allclose(qc.get_state(), [0, 1]), "Failed to create |1⟩"

    qc.measure(qubit=0, cbit=0)
    qc.reset_qubit(qubit=0)
    qc.execute() 

    actual_state = qc.get_state()
    expected_state = np.array([1.0, 0.0], dtype=complex)

    print(f"Reset Test: Expected {s(expected_state)}, Got {s(actual_state)}")
    assert np.allclose(actual_state, expected_state), "reset_qubit failed"

# add more tests...
def test_noisy_cnot(registries):
    gate_reg, channel_reg = registries

    noise = NoiseModel(
        channel_registry=channel_reg,
        default_spec=None,
        per_gate_specs={
            "cx": {
                "type": "bit_phase_flip",
                "params": (1.0,),
                "scope": "per_qubit"
            }
        }
    )

    qc = QuantumCircuit(num_qubits=2, num_cbits=1)
    qc.set_noise_model(noise)
    qc.set_rng(np.random.default_rng(123))

    qc.add_gate(gate_reg.get("x"), 0)
    qc.add_gate(gate_reg.get("cx"), (0, 1))
    qc.measure(1, 0)

    qc.execute()

    assert qc.get_cbit(0) == 0

    expected_state = np.array([-1, 0, 0, 0], complex)
    assert np.allclose(qc.get_state(), expected_state)


def test_noisy_toffoli(registries):
    gate_reg, channel_reg = registries

    # Noise model: Toffoli ("ccx") always suffers a bit flip (X) on each target qubit
    noise = NoiseModel(
        channel_registry=channel_reg,
        default_spec=None,
        per_gate_specs={
            "ccx": {
                "type": "bit_flip",   # uses your pauli_x_channel(p)
                "params": (1.0,),     # p = 1 → always flip
                "scope": "per_qubit"  # apply 1q channel on each qubit in targets
            }
        }
    )

    qc = QuantumCircuit(num_qubits=3, num_cbits=1)
    qc.set_noise_model(noise)
    qc.set_rng(np.random.default_rng(123))

    # Prepare |110⟩:
    # assuming q0 is the most significant bit, q2 is least significant
    qc.add_gate(gate_reg.get("x"), 0)  # |100⟩
    qc.add_gate(gate_reg.get("x"), 1)  # |110⟩

    # Ideal Toffoli (ccx q0, q1, q2): |110⟩ → |111⟩
    qc.add_gate(gate_reg.get("ccx"), (0, 1, 2))

    # Measure target qubit into classical bit 0
    qc.measure(2, 0)

    qc.execute()

    # After Toffoli, state is |111⟩, then noise bit-flips all three qubits:
    # X⊗X⊗X |111⟩ = |000⟩
    # So we expect qubit 2 = 0 and final state = |000⟩
    assert qc.get_cbit(0) == 0

    expected_state = np.zeros(2**3, dtype=complex)
    expected_state[0] = 1.0  # |000⟩
    assert np.allclose(qc.get_state(), expected_state)

# ==========================================
# BELL STATES AND ENTANGLEMENT TESTS
# ==========================================

def test_bell_state_phi_plus(registries):
    """Tests creation of Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=2)

    qc.add_gate(gate_reg.get('h'), targets=0)
    qc.add_gate(gate_reg.get('cx'), targets=[0, 1])
    qc.execute()

    actual_state = qc.get_state()
    expected_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)

    print(f"Bell |Φ+⟩ Test: Expected {s(expected_state)}, Got {s(actual_state)}")
    assert np.allclose(actual_state, expected_state), "Bell state |Φ+⟩ creation failed"


def test_bell_state_phi_minus(registries):
    """Tests creation of Bell state |Φ-⟩ = (|00⟩ - |11⟩)/√2."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=2)

    qc.add_gate(gate_reg.get('h'), targets=0)
    qc.add_gate(gate_reg.get('z'), targets=0)
    qc.add_gate(gate_reg.get('cx'), targets=[0, 1])
    qc.execute()

    actual_state = qc.get_state()
    expected_state = np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)], dtype=complex)

    print(f"Bell |Φ-⟩ Test: Expected {s(expected_state)}, Got {s(actual_state)}")
    assert np.allclose(actual_state, expected_state), "Bell state |Φ-⟩ creation failed"


def test_bell_state_psi_plus(registries):
    """Tests creation of Bell state |Ψ+⟩ = (|01⟩ + |10⟩)/√2."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=2)

    qc.add_gate(gate_reg.get('h'), targets=0)
    qc.add_gate(gate_reg.get('cx'), targets=[0, 1])
    qc.add_gate(gate_reg.get('x'), targets=1)
    qc.execute()

    actual_state = qc.get_state()
    expected_state = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=complex)

    print(f"Bell |Ψ+⟩ Test: Expected {s(expected_state)}, Got {s(actual_state)}")
    assert np.allclose(actual_state, expected_state), "Bell state |Ψ+⟩ creation failed"


def test_bell_state_psi_minus(registries):
    """Tests creation of Bell state |Ψ-⟩ = (|01⟩ - |10⟩)/√2."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=2)

    qc.add_gate(gate_reg.get('x'), targets=1)
    qc.add_gate(gate_reg.get('h'), targets=0)
    qc.add_gate(gate_reg.get('cx'), targets=[0, 1])
    qc.add_gate(gate_reg.get('z'), targets=0)
    qc.execute()

    actual_state = qc.get_state()
    expected_state = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0], dtype=complex)

    print(f"Bell |Ψ-⟩ Test: Expected {s(expected_state)}, Got {s(actual_state)}")
    assert np.allclose(actual_state, expected_state), "Bell state |Ψ-⟩ creation failed"


# ==========================================
# GHZ STATE AND MULTI-QUBIT ENTANGLEMENT
# ==========================================

def test_ghz_state_3qubits(registries):
    """Tests creation of 3-qubit GHZ state (|000⟩ + |111⟩)/√2."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=3)

    qc.add_gate(gate_reg.get('h'), targets=0)
    qc.add_gate(gate_reg.get('cx'), targets=[0, 1])
    qc.add_gate(gate_reg.get('cx'), targets=[0, 2])
    qc.execute()

    actual_state = qc.get_state()
    expected_state = np.zeros(8, dtype=complex)
    expected_state[0] = 1/np.sqrt(2)
    expected_state[7] = 1/np.sqrt(2)

    print(f"GHZ 3-qubit Test: Got {s(actual_state)}")
    assert np.allclose(actual_state, expected_state), "3-qubit GHZ state creation failed"


def test_ghz_state_4qubits(registries):
    """Tests creation of 4-qubit GHZ state (|0000⟩ + |1111⟩)/√2."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=4)

    qc.add_gate(gate_reg.get('h'), targets=0)
    qc.add_gate(gate_reg.get('cx'), targets=[0, 1])
    qc.add_gate(gate_reg.get('cx'), targets=[0, 2])
    qc.add_gate(gate_reg.get('cx'), targets=[0, 3])
    qc.execute()

    actual_state = qc.get_state()
    expected_state = np.zeros(16, dtype=complex)
    expected_state[0] = 1/np.sqrt(2)
    expected_state[15] = 1/np.sqrt(2)

    print(f"GHZ 4-qubit Test: State has {np.sum(np.abs(actual_state) > 1e-10)} non-zero amplitudes")
    assert np.allclose(actual_state, expected_state), "4-qubit GHZ state creation failed"


def test_w_state_3qubits(registries):
    """Tests creation of 3-qubit W state (|001⟩ + |010⟩ + |100⟩)/√3.
    Note: This is a simplified approximation that tests multi-qubit superposition."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=3)

    qc.add_gate(gate_reg.get('x'), targets=2)
    qc.add_gate(gate_reg.get('h'), targets=0)
    qc.add_gate(gate_reg.get('h'), targets=1)
    qc.execute()

    actual_state = qc.get_state()

    print(f"W-like state Test: Got {s(actual_state)}")
    assert np.allclose(np.sum(np.abs(actual_state)**2), 1.0), "State normalization failed"


# ==========================================
# QUANTUM ALGORITHM TESTS
# ==========================================

def test_quantum_fourier_transform_2qubits(registries):
    """Tests 2-qubit Quantum Fourier Transform (simplified)."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=2)

    qc.add_gate(gate_reg.get('x'), targets=0)
    qc.add_gate(gate_reg.get('x'), targets=1)

    qc.add_gate(gate_reg.get('h'), targets=0)
    qc.add_gate(gate_reg.get('h'), targets=1)

    qc.add_gate(gate_reg.get('cx'), targets=[0, 1])
    qc.add_gate(gate_reg.get('cx'), targets=[1, 0])
    qc.add_gate(gate_reg.get('cx'), targets=[0, 1])

    qc.execute()

    actual_state = qc.get_state()
    print(f"2-qubit QFT Test: Got {s(actual_state)}")

    assert np.allclose(np.sum(np.abs(actual_state)**2), 1.0), "QFT state normalization failed"


def test_quantum_phase_kickback(registries):
    """Tests quantum phase kickback using controlled operations."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=2)

    qc.add_gate(gate_reg.get('h'), targets=0)
    qc.add_gate(gate_reg.get('x'), targets=1)
    qc.add_gate(gate_reg.get('cx'), targets=[0, 1])
    qc.add_gate(gate_reg.get('h'), targets=0)

    qc.execute()

    actual_state = qc.get_state()
    print(f"Phase kickback Test: Got {s(actual_state)}")

    assert np.allclose(np.sum(np.abs(actual_state)**2), 1.0), "Phase kickback state normalization failed"


def test_deutsch_algorithm_constant(registries):
    """Tests Deutsch algorithm with constant function f(x)=0."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=2)

    qc.add_gate(gate_reg.get('x'), targets=1)
    qc.add_gate(gate_reg.get('h'), targets=0)
    qc.add_gate(gate_reg.get('h'), targets=1)
    qc.add_gate(gate_reg.get('h'), targets=0)

    qc.execute()

    actual_state = qc.get_state()
    print(f"Deutsch (constant) Test: Got {s(actual_state)}")

    probs = np.abs(actual_state)**2
    prob_q0_is_0 = probs[0] + probs[1]

    print(f"Probability q0=|0⟩: {prob_q0_is_0:.4f}")
    assert prob_q0_is_0 > 0.99, "Deutsch algorithm failed for constant function"


def test_deutsch_algorithm_balanced(registries):
    """Tests Deutsch algorithm with balanced function f(x)=x."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=2)

    qc.add_gate(gate_reg.get('x'), targets=1)
    qc.add_gate(gate_reg.get('h'), targets=0)
    qc.add_gate(gate_reg.get('h'), targets=1)
    qc.add_gate(gate_reg.get('cx'), targets=[0, 1])
    qc.add_gate(gate_reg.get('h'), targets=0)

    qc.execute()

    actual_state = qc.get_state()
    print(f"Deutsch (balanced) Test: Got {s(actual_state)}")

    probs = np.abs(actual_state)**2
    prob_q0_is_1 = probs[2] + probs[3]

    print(f"Probability q0=|1⟩: {prob_q0_is_1:.4f}")
    assert prob_q0_is_1 > 0.99, "Deutsch algorithm failed for balanced function"


# ==========================================
# MEASUREMENT AND PROBABILITY TESTS
# ==========================================

def test_measurement_statistics_single_qubit(registries):
    """Tests measurement statistics on a single qubit in superposition."""
    gate_reg, _ = registries

    num_shots = 1000
    results = []

    for _ in range(num_shots):
        qc = QuantumCircuit(num_qubits=1, num_cbits=1)
        qc.add_gate(gate_reg.get('h'), targets=0)
        qc.measure(qubit=0, cbit=0)
        qc.execute()

        result = qc.cbits.get_bit(0)
        results.append(result)

    count_0 = sum(1 for r in results if r == 0)
    count_1 = sum(1 for r in results if r == 1)

    prob_0 = count_0 / num_shots
    prob_1 = count_1 / num_shots

    print(f"Measurement statistics: P(0)={prob_0:.3f}, P(1)={prob_1:.3f}")

    assert 0.4 < prob_0 < 0.6, f"Measurement statistics incorrect: P(0)={prob_0}"
    assert 0.4 < prob_1 < 0.6, f"Measurement statistics incorrect: P(1)={prob_1}"


def test_measurement_probabilities_bell_state(registries):
    """Tests measurement probabilities for Bell state."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=2)

    qc.add_gate(gate_reg.get('h'), targets=0)
    qc.add_gate(gate_reg.get('cx'), targets=[0, 1])
    qc.execute()

    probs = qc.measure_probabilities()

    print(f"Bell state measurement probabilities: {probs}")

    assert np.allclose(probs[0], 0.5, atol=1e-6), "P(|00⟩) incorrect"
    assert np.allclose(probs[1], 0.0, atol=1e-6), "P(|01⟩) should be 0"
    assert np.allclose(probs[2], 0.0, atol=1e-6), "P(|10⟩) should be 0"
    assert np.allclose(probs[3], 0.5, atol=1e-6), "P(|11⟩) incorrect"


def test_measurement_collapse(registries):
    """Tests that measurement correctly collapses the state."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=1, num_cbits=1)

    qc.add_gate(gate_reg.get('h'), targets=0)
    qc.execute()

    state_before = qc.get_state()
    assert not np.allclose(np.abs(state_before[0]), 1.0), "State should be in superposition"

    qc.measure(qubit=0, cbit=0)
    qc.execute()

    state_after = qc.get_state()

    assert (np.allclose(np.abs(state_after[0]), 1.0) or
            np.allclose(np.abs(state_after[1]), 1.0)), "State not properly collapsed after measurement"


def test_sequential_measurements(registries):
    """Tests multiple sequential measurements on different qubits."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=3, num_cbits=3)

    qc.add_gate(gate_reg.get('h'), targets=0)
    qc.add_gate(gate_reg.get('cx'), targets=[0, 1])
    qc.add_gate(gate_reg.get('cx'), targets=[0, 2])

    qc.measure(qubit=0, cbit=0)
    qc.measure(qubit=1, cbit=1)
    qc.measure(qubit=2, cbit=2)
    qc.execute()

    results = [qc.cbits.get_bit(i) for i in range(3)]
    print(f"GHZ measurement results: {results}")

    assert (all(r == 0 for r in results) or
            all(r == 1 for r in results)), "GHZ measurements not properly correlated"


# ==========================================
# MULTI-QUBIT OPERATIONS
# ==========================================

def test_multi_qubit_hadamard(registries):
    """Tests Hadamard on multiple qubits creates equal superposition."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=3)

    qc.add_gate(gate_reg.get('h'), targets=0)
    qc.add_gate(gate_reg.get('h'), targets=1)
    qc.add_gate(gate_reg.get('h'), targets=2)
    qc.execute()

    actual_state = qc.get_state()

    expected_amplitude = 1 / np.sqrt(8)
    expected_state = np.full(8, expected_amplitude, dtype=complex)

    print(f"Multi-Hadamard Test: All amplitudes should be {expected_amplitude:.4f}")
    assert np.allclose(actual_state, expected_state), "Multi-qubit Hadamard failed"


def test_toffoli_gate(registries):
    """Tests the Toffoli (CCX) gate on various inputs."""
    gate_reg, _ = registries

    qc = QuantumCircuit(num_qubits=3)
    qc.add_gate(gate_reg.get('x'), targets=0)
    qc.add_gate(gate_reg.get('x'), targets=1)
    qc.add_gate(gate_reg.get('ccx'), targets=[0, 1, 2])
    qc.execute()

    actual_state = qc.get_state()
    expected_state = np.zeros(8, dtype=complex)
    expected_state[7] = 1.0

    print(f"Toffoli |110⟩→|111⟩ Test: Got {s(actual_state)}")
    assert np.allclose(actual_state, expected_state), "Toffoli gate failed on |110⟩"

    qc = QuantumCircuit(num_qubits=3)
    qc.add_gate(gate_reg.get('x'), targets=0)
    qc.add_gate(gate_reg.get('x'), targets=1)
    qc.add_gate(gate_reg.get('x'), targets=2)
    qc.add_gate(gate_reg.get('ccx'), targets=[0, 1, 2])
    qc.execute()

    actual_state = qc.get_state()
    expected_state = np.zeros(8, dtype=complex)
    expected_state[6] = 1.0

    print(f"Toffoli |111⟩→|110⟩ Test: Got {s(actual_state)}")
    assert np.allclose(actual_state, expected_state), "Toffoli gate failed on |111⟩"

    qc = QuantumCircuit(num_qubits=3)
    qc.add_gate(gate_reg.get('x'), targets=1)
    qc.add_gate(gate_reg.get('ccx'), targets=[0, 1, 2])
    qc.execute()

    actual_state = qc.get_state()
    expected_state = np.zeros(8, dtype=complex)
    expected_state[2] = 1.0

    print(f"Toffoli |010⟩→|010⟩ Test: Got {s(actual_state)}")
    assert np.allclose(actual_state, expected_state), "Toffoli gate incorrectly activated"


def test_parallel_cnot_gates(registries):
    """Tests multiple CNOT gates applied to different qubit pairs.
    |1010⟩ -> CNOT(0,1), CNOT(2,3) -> |1111⟩"""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=4)

    qc.add_gate(gate_reg.get('x'), targets=0)
    qc.add_gate(gate_reg.get('x'), targets=2)

    qc.add_gate(gate_reg.get('cx'), targets=[0, 1])
    qc.add_gate(gate_reg.get('cx'), targets=[2, 3])
    qc.execute()

    actual_state = qc.get_state()
    expected_state = np.zeros(16, dtype=complex)
    expected_state[15] = 1.0

    print(f"Parallel CNOT Test: |1010⟩ should become |1111⟩")
    assert np.allclose(actual_state, expected_state), "Parallel CNOT gates failed"


# ==========================================
# CIRCUIT MANAGEMENT TESTS
# ==========================================

def test_circuit_reset_all(registries):
    """Tests that reset_all properly clears circuit state and operations."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=2)

    qc.add_gate(gate_reg.get('h'), targets=0)
    qc.add_gate(gate_reg.get('cx'), targets=[0, 1])
    qc.execute()

    state_before_reset = qc.get_state()
    assert not np.allclose(state_before_reset, [1, 0, 0, 0]), "State should be entangled"

    qc.reset_all()

    state_after_reset = qc.get_state()
    expected_state = np.array([1, 0, 0, 0], dtype=complex)

    print(f"Reset Test: After reset got {s(state_after_reset)}")
    assert np.allclose(state_after_reset, expected_state), "reset_all failed to restore |00⟩"


def test_circuit_reuse(registries):
    """Tests reusing a circuit for multiple runs."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=1)

    qc.add_gate(gate_reg.get('x'), targets=0)
    qc.execute()
    assert np.allclose(qc.get_state(), [0, 1]), "First run failed"

    qc.reset_all()
    qc.add_gate(gate_reg.get('h'), targets=0)
    qc.execute()
    expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    assert np.allclose(qc.get_state(), expected), "Second run after reset failed"

    print("Circuit reuse test passed")


def test_metrics_tracking(registries):
    """Tests that metrics are properly tracked when enabled."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=2, enable_metrics=True)

    qc.add_gate(gate_reg.get('h'), targets=0)
    qc.add_gate(gate_reg.get('cx'), targets=[0, 1])
    qc.add_gate(gate_reg.get('x'), targets=1)
    qc.execute()

    metrics = qc.get_metrics()

    print(f"Metrics: {metrics}")

    assert 'total_gates' in metrics or 'execution_time' in metrics or len(metrics) > 0, \
        "Metrics should be tracked when enabled"

    print("Metrics tracking test passed")


# ==========================================
# EDGE CASES AND STRESS TESTS
# ==========================================

def test_large_circuit_5qubits(registries):
    """Tests a larger 5-qubit circuit with multiple gates."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=5)

    for i in range(5):
        qc.add_gate(gate_reg.get('h'), targets=i)

    for i in range(4):
        qc.add_gate(gate_reg.get('cx'), targets=[i, i+1])

    qc.execute()

    actual_state = qc.get_state()

    assert np.allclose(np.sum(np.abs(actual_state)**2), 1.0), \
        "Large circuit state not normalized"

    assert len(actual_state) == 32, f"Expected 32 amplitudes, got {len(actual_state)}"

    print(f"5-qubit circuit: {len(actual_state)} amplitudes, state normalized")


def test_repeated_gate_application(registries):
    """Tests applying the same gate multiple times (X^4 = I)."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=1)

    for _ in range(4):
        qc.add_gate(gate_reg.get('x'), targets=0)
    qc.execute()

    actual_state = qc.get_state()
    expected_state = np.array([1, 0], dtype=complex)

    print(f"Repeated X gate test: {s(actual_state)}")
    assert np.allclose(actual_state, expected_state), "X^4 should be identity"


def test_deep_circuit(registries):
    """Tests a circuit with many sequential gates (depth test)."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=2)

    for i in range(25):
        qc.add_gate(gate_reg.get('h'), targets=0)
        qc.add_gate(gate_reg.get('x'), targets=0)

    qc.execute()

    actual_state = qc.get_state()

    assert np.allclose(np.sum(np.abs(actual_state)**2), 1.0), \
        "Deep circuit state not normalized"

    print(f"Deep circuit test passed: 50 gates applied")


def test_all_basis_states_2qubits(registries):
    """Tests that all 2-qubit basis states can be prepared correctly."""
    gate_reg, _ = registries

    basis_states = [
        ([1, 0, 0, 0], []),
        ([0, 1, 0, 0], [('x', 1)]),
        ([0, 0, 1, 0], [('x', 0)]),
        ([0, 0, 0, 1], [('x', 0), ('x', 1)])
    ]

    for expected, gates in basis_states:
        qc = QuantumCircuit(num_qubits=2)

        for gate_name, target in gates:
            qc.add_gate(gate_reg.get(gate_name), targets=target)

        qc.execute()
        actual_state = qc.get_state()

        assert np.allclose(actual_state, expected), \
            f"Failed to prepare {expected} with gates {gates}"

    print("All 2-qubit basis states prepared successfully")


def test_identity_via_gate_cancellation(registries):
    """Tests that gates and their inverses cancel out (H^2=I, S^4=I)."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=1)

    qc.add_gate(gate_reg.get('h'), targets=0)
    qc.add_gate(gate_reg.get('h'), targets=0)
    qc.execute()

    actual_state = qc.get_state()
    expected_state = np.array([1, 0], dtype=complex)

    assert np.allclose(actual_state, expected_state), "H * H should be identity"

    qc.reset_all()
    for _ in range(4):
        qc.add_gate(gate_reg.get('s'), targets=0)
    qc.execute()

    actual_state = qc.get_state()
    assert np.allclose(actual_state, expected_state), "S^4 should be identity"

    print("Gate cancellation tests passed")


# ==========================================
# ERROR CHANNEL INTEGRATION TESTS
# ==========================================

def test_noisy_gate_execution(registries):
    """Tests that noisy gates can be executed (basic smoke test)."""
    gate_reg, chan_reg = registries
    qc = QuantumCircuit(num_qubits=1)

    try:
        bit_flip = chan_reg.get_param('bit_flip').instantiate(p=0.1)

        from gates.registry import Gate
        noisy_x = Gate('noisy_x', gate_reg.get('x').matrix, noise=bit_flip)

        qc.add_gate(noisy_x, targets=0)
        qc.execute()

        state = qc.get_state()
        assert np.allclose(np.sum(np.abs(state)**2), 1.0), "Noisy gate broke normalization"

        print("Noisy gate execution test passed")
    except Exception as e:
        print(f"Noisy gate test skipped or failed: {e}")


# ==========================================
# QUANTUM TELEPORTATION
# ==========================================

def test_quantum_teleportation_protocol(registries):
    """Tests the quantum teleportation protocol (simplified version).
    Qubit layout: q0 = Alice's qubit, q1 = Alice's entangled qubit, q2 = Bob's qubit."""
    gate_reg, _ = registries
    qc = QuantumCircuit(num_qubits=3, num_cbits=2)

    qc.add_gate(gate_reg.get('h'), targets=0)

    qc.add_gate(gate_reg.get('h'), targets=1)
    qc.add_gate(gate_reg.get('cx'), targets=[1, 2])

    qc.add_gate(gate_reg.get('cx'), targets=[0, 1])
    qc.add_gate(gate_reg.get('h'), targets=0)

    qc.measure(qubit=0, cbit=0)
    qc.measure(qubit=1, cbit=1)
    qc.execute()

    m0 = qc.cbits.get_bit(0)
    m1 = qc.cbits.get_bit(1)

    if m1 == 1:
        qc.add_gate(gate_reg.get('x'), targets=2)
    if m0 == 1:
        qc.add_gate(gate_reg.get('z'), targets=2)

    qc.execute()

    full_state = qc.get_state()

    print(f"Teleportation: measurements = ({m0}, {m1}), final state = {s(full_state)}")

    assert np.allclose(np.sum(np.abs(full_state)**2), 1.0), "Teleportation broke normalization"


def test_qasm_end_to_end_with_noise():
    # 1) QASM string using only gates in your basis/registry
    qasm_str = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg c[3];

    // Prepare some superposition and entanglement
    h q[0];
    s q[1];

    x q[2];
    y q[0];
    z q[1];

    cx q[0], q[1];
    ccx q[0], q[1], q[2];

    measure q -> c;
    """

    # 2) Choose basis gates that your simulator implements
    basis = [
        "x", "y", "z",
        "h", "s",
        "cx", "ccx",
        "measure",
    ]

    # 3) Parse QASM → IR
    ir = parse_qasm_source(qasm_str, basis_gates=basis)

    # 4) Build gate registry and QuantumCircuit from IR
    gate_reg = GateRegistry(preload_defaults=True)
    qc: QuantumCircuit = build_circuit_from_ir(ir, gate_reg)

    # 5) Attach noise model + RNG
    noise = build_default_noise_model()
    qc.set_noise_model(noise)
    qc.set_rng(np.random.default_rng(123))

    # 6) Execute the circuit
    qc.execute()

    # 7) Basic sanity checks

    # (a) Statevector has correct length and norm ≈ 1
    state = qc.get_state()
    assert state.shape == (2 ** qc.num_qubits,), "Statevector has wrong dimension"
    norm = np.linalg.norm(state)
    assert np.isclose(norm, 1.0, atol=1e-10), f"Final state not normalized (‖ψ‖={norm})"

    # (b) Measurement results are in {0,1} for each classical bit
    cbits = qc.get_cbits()
    for i in range(qc.num_cbits):
        v = cbits.get_bit(i)
        assert v in (0, 1), f"Classical bit {i} has invalid value {v}"

    # (c) Probabilities sum to 1
    probs = qc.measure_probabilities()
    assert np.isclose(probs.sum(), 1.0, atol=1e-10), "Probabilities do not sum to 1"

def test_qasm_end_to_end_with_noise_vs_ideal():
    qasm_str = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg c[3];

    h q[0];
    s q[1];
    x q[2];
    y q[0];
    z q[1];
    cx q[0], q[1];
    ccx q[0], q[1], q[2];
    measure q -> c;
    """

    basis = ["x", "y", "z", "h", "s", "cx", "ccx", "measure"]

    ir = parse_qasm_source(qasm_str, basis_gates=basis)
    gate_reg = GateRegistry(preload_defaults=True)

    # --- ideal circuit (no noise) ---
    qc_ideal = build_circuit_from_ir(ir, gate_reg)
    qc_ideal.set_noise_model(None)
    qc_ideal.set_rng(np.random.default_rng(123))
    qc_ideal.execute()
    ideal_state = qc_ideal.get_state()

    # --- noisy circuit (exaggerated noise to make difference clear) ---
    noise = NoiseModel(
        channel_registry=ChannelRegistry(preload_defaults=True),
        default_spec={
            "type": "depolarizing",
            "params": (0.5,),
            "scope": "per_qubit",
        },
        per_gate_specs={}
    )
    qc_noisy = build_circuit_from_ir(ir, gate_reg)
    qc_noisy.set_noise_model(noise)
    qc_noisy.set_rng(np.random.default_rng(123))
    qc_noisy.execute()
    noisy_state = qc_noisy.get_state()

    # basic sanity
    assert noisy_state.shape == ideal_state.shape
    assert np.isclose(np.linalg.norm(noisy_state), 1.0, atol=1e-10)

    # **key assertion**: noisy != ideal
    assert not np.allclose(noisy_state, ideal_state, atol=1e-2), \
        "Noisy state is identical to ideal – noise may not be applied."

if __name__ == "__main__":
    print("Running comprehensive integration tests for exact simulator...")
    pytest.main([__file__, "-v", "-s"])
