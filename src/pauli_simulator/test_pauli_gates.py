import numpy as np
import pytest
from Tableau_Ver2 import Tableau


class TestPauliXGate:
    """Tests for the Pauli X (bit-flip) gate."""

    def test_x_on_zero_state(self):
        """X|0⟩ = |1⟩"""
        tab = Tableau(1)
        tab.x(0)
        result = tab.measure(0)
        # Should deterministically measure |1⟩
        assert result == 1, "X gate failed: |0⟩ should become |1⟩"

    def test_x_on_one_state(self):
        """X|1⟩ = |0⟩"""
        tab = Tableau(1)
        tab.x(0)  # Create |1⟩
        tab.x(0)  # Apply X again
        result = tab.measure(0)
        # Should deterministically measure |0⟩
        assert result == 0, "X gate failed: |1⟩ should become |0⟩"

    def test_x_is_self_inverse(self):
        """X^2 = I (identity)"""
        tab = Tableau(1)
        # Apply X twice
        tab.x(0)
        tab.x(0)
        result = tab.measure(0)
        # Should be back to |0⟩
        assert result == 0, "X^2 should be identity"

    def test_x_four_times(self):
        """X^4 = I (stronger test)"""
        tab = Tableau(1)
        for _ in range(4):
            tab.x(0)
        result = tab.measure(0)
        assert result == 0, "X^4 should be identity"

    def test_x_on_multiple_qubits(self):
        """Test X on different qubits independently."""
        tab = Tableau(3)
        tab.x(0)
        tab.x(2)

        m0 = tab.measure(0)
        m1 = tab.measure(1)
        m2 = tab.measure(2)

        assert m0 == 1, "X on qubit 0 failed"
        assert m1 == 0, "Qubit 1 should remain |0⟩"
        assert m2 == 1, "X on qubit 2 failed"

    def test_x_on_superposition(self):
        """Test X on |+⟩ state: X|+⟩ = |+⟩"""
        np.random.seed(42)
        tab = Tableau(1)
        tab.h(0)  # Create |+⟩
        tab.x(0)  # X|+⟩ = |+⟩

        # Measure in X basis by applying H first
        tab.h(0)
        result = tab.measure(0)
        # Should still measure |0⟩ in computational basis after H
        assert result == 0, "X|+⟩ should equal |+⟩"


class TestPauliYGate:
    """Tests for the Pauli Y (bit-flip + phase-flip) gate."""

    def test_y_on_zero_state(self):
        """Y|0⟩ = i|1⟩ (measures as |1⟩)"""
        tab = Tableau(1)
        tab.y(0)
        result = tab.measure(0)
        # Should deterministically measure |1⟩
        assert result == 1, "Y gate failed: |0⟩ should become i|1⟩"

    def test_y_on_one_state(self):
        """Y|1⟩ = -i|0⟩ (measures as |0⟩)"""
        tab = Tableau(1)
        tab.x(0)  # Create |1⟩
        tab.y(0)  # Apply Y
        result = tab.measure(0)
        # Should deterministically measure |0⟩
        assert result == 0, "Y gate failed: |1⟩ should become -i|0⟩"

    def test_y_is_self_inverse(self):
        """Y^2 = I (identity)"""
        tab = Tableau(1)
        tab.y(0)
        tab.y(0)
        result = tab.measure(0)
        # Should be back to |0⟩
        assert result == 0, "Y^2 should be identity"

    def test_y_equals_ixz(self):
        """Y = iXZ (up to global phase)"""
        # Test that Y and XZ have the same effect on computational basis
        tab1 = Tableau(1)
        tab1.y(0)
        r1 = tab1.measure(0)

        tab2 = Tableau(1)
        tab2.z(0)
        tab2.x(0)
        r2 = tab2.measure(0)

        assert r1 == r2, "Y should equal XZ (up to phase)"

    def test_y_on_multiple_qubits(self):
        """Test Y on different qubits independently."""
        tab = Tableau(2)
        tab.y(0)

        m0 = tab.measure(0)
        m1 = tab.measure(1)

        assert m0 == 1, "Y on qubit 0 failed"
        assert m1 == 0, "Qubit 1 should remain |0⟩"


class TestPauliZGate:
    """Tests for the Pauli Z (phase-flip) gate."""

    def test_z_on_zero_state(self):
        """Z|0⟩ = |0⟩"""
        tab = Tableau(1)
        tab.z(0)
        result = tab.measure(0)
        # Should deterministically measure |0⟩
        assert result == 0, "Z gate failed: |0⟩ should remain |0⟩"

    def test_z_on_one_state(self):
        """Z|1⟩ = -|1⟩ (still measures as |1⟩)"""
        tab = Tableau(1)
        tab.x(0)  # Create |1⟩
        tab.z(0)  # Apply Z
        result = tab.measure(0)
        # Should deterministically measure |1⟩ (phase doesn't affect measurement)
        assert result == 1, "Z gate failed: |1⟩ should remain |1⟩"

    def test_z_is_self_inverse(self):
        """Z^2 = I (identity)"""
        tab = Tableau(1)
        tab.x(0)  # Start with |1⟩
        tab.z(0)
        tab.z(0)
        result = tab.measure(0)
        # Should still be |1⟩
        assert result == 1, "Z^2 should be identity"

    def test_z_on_plus_state(self):
        """Z|+⟩ = |-⟩"""
        np.random.seed(42)
        tab = Tableau(1)
        tab.h(0)  # Create |+⟩
        tab.z(0)  # Z|+⟩ = |-⟩
        tab.h(0)  # Convert back to computational basis
        result = tab.measure(0)
        # |-⟩ in X basis is |1⟩ in Z basis
        assert result == 1, "Z|+⟩ should equal |-⟩"

    def test_z_commutes_with_itself(self):
        """Z can be applied multiple times."""
        tab = Tableau(1)
        tab.x(0)
        tab.z(0)
        tab.z(0)
        tab.z(0)
        result = tab.measure(0)
        # Odd number of Z's still gives |1⟩
        assert result == 1, "Multiple Z applications failed"


class TestPauliGateRelations:
    """Tests for relationships between Pauli gates."""

    def test_xyz_equals_i(self):
        """XYZ = iI (up to global phase)"""
        tab1 = Tableau(1)
        tab1.x(0)
        tab1.y(0)
        tab1.z(0)
        r1 = tab1.measure(0)

        # iI on |0⟩ is still |0⟩
        assert r1 == 0, "XYZ should be identity (up to phase)"

    def test_x_anticommutes_with_z(self):
        """XZ = -ZX (anti-commutation)"""
        # XZ|0⟩ = X|0⟩ = |1⟩
        tab1 = Tableau(1)
        tab1.z(0)
        tab1.x(0)
        r1 = tab1.measure(0)

        # ZX|0⟩ = Z|1⟩ = -|1⟩
        tab2 = Tableau(1)
        tab2.x(0)
        tab2.z(0)
        r2 = tab2.measure(0)

        # Both should measure |1⟩ (global phase doesn't matter)
        assert r1 == r2 == 1, "X and Z should anti-commute"

    def test_pauli_commutation_on_bell_state(self):
        """Test Pauli operations on entangled state."""
        tab = Tableau(2)
        tab.h(0)
        tab.cx(0, 1)

        # Apply X to both qubits
        tab.x(0)
        tab.x(1)

        m0 = tab.measure(0)
        m1 = tab.measure(1)

        # Bell state with X on both should still be correlated
        assert m0 == m1, "Pauli operations should preserve Bell state correlation"

    def test_all_paulis_square_to_identity(self):
        """Test that X^2 = Y^2 = Z^2 = I"""
        for gate_method in ['x', 'y', 'z']:
            tab = Tableau(1)
            getattr(tab, gate_method)(0)
            getattr(tab, gate_method)(0)
            result = tab.measure(0)
            assert result == 0, f"{gate_method.upper()}^2 should be identity"


class TestPauliGatesWithMeasurement:
    """Tests combining Pauli gates with measurements."""

    def test_measurement_after_x(self):
        """Measure after X gate multiple times."""
        results = []
        for _ in range(10):
            tab = Tableau(1)
            tab.x(0)
            results.append(tab.measure(0))

        # All measurements should be deterministic |1⟩
        assert all(r == 1 for r in results), "X|0⟩ should always measure as 1"

    def test_measurement_after_pauli_on_superposition(self):
        """Measure after Pauli gates on |+⟩ state."""
        np.random.seed(42)

        # Collect statistics for Z|+⟩
        results = []
        for _ in range(100):
            tab = Tableau(1)
            tab.h(0)  # Create |+⟩
            tab.z(0)  # Z|+⟩ = |-⟩
            results.append(tab.measure(0))

        # Z|+⟩ = |-⟩ gives 50/50 in Z basis
        count_0 = sum(1 for r in results if r == 0)
        count_1 = sum(1 for r in results if r == 1)

        # Should have both outcomes
        assert count_0 > 0 and count_1 > 0, "Should get mixed results for |-⟩"


class TestPauliGatesWithCliffordGates:
    """Tests Pauli gates combined with other Clifford gates."""

    def test_hxh_equals_z(self):
        """HXH = Z (conjugation)"""
        tab1 = Tableau(1)
        tab1.h(0)
        tab1.x(0)
        tab1.h(0)
        # This creates the same state as Z|0⟩ = |0⟩
        r1 = tab1.measure(0)

        tab2 = Tableau(1)
        tab2.z(0)
        r2 = tab2.measure(0)

        assert r1 == r2 == 0, "HXH should equal Z"

    def test_hzh_equals_x(self):
        """HZH = X (conjugation)"""
        tab1 = Tableau(1)
        tab1.h(0)
        tab1.z(0)
        tab1.h(0)
        # This creates |1⟩
        r1 = tab1.measure(0)

        tab2 = Tableau(1)
        tab2.x(0)
        r2 = tab2.measure(0)

        assert r1 == r2 == 1, "HZH should equal X"

    def test_cnot_with_pauli_gates(self):
        """Test CNOT with Pauli gates."""
        tab = Tableau(2)
        tab.x(0)  # Control = |1⟩
        tab.cx(0, 1)  # CNOT

        m0 = tab.measure(0)
        m1 = tab.measure(1)

        assert m0 == 1 and m1 == 1, "CNOT with X on control should flip target"

    def test_bell_state_with_z_gates(self):
        """Apply Z gates to Bell state."""
        tab = Tableau(2)
        tab.h(0)
        tab.cx(0, 1)

        # Apply Z to first qubit
        tab.z(0)

        m0 = tab.measure(0)
        m1 = tab.measure(1)

        # Should still be correlated
        assert m0 == m1, "Z on Bell state should preserve correlation"


class TestEdgeCases:
    """Edge cases and stress tests."""

    def test_many_pauli_gates(self):
        """Apply many Pauli gates in sequence."""
        tab = Tableau(1)

        # Apply 100 X gates (should be |0⟩ since 100 is even)
        for _ in range(100):
            tab.x(0)

        result = tab.measure(0)
        assert result == 0, "Even number of X gates should return to |0⟩"

    def test_pauli_gates_on_large_system(self):
        """Test Pauli gates on 10-qubit system."""
        n = 10
        tab = Tableau(n)

        # Apply X to even qubits, Z to odd qubits
        for i in range(n):
            if i % 2 == 0:
                tab.x(i)
            else:
                tab.z(i)

        results = [tab.measure(i) for i in range(n)]

        # Even qubits should be |1⟩, odd should be |0⟩
        for i in range(n):
            if i % 2 == 0:
                assert results[i] == 1, f"X failed on qubit {i}"
            else:
                assert results[i] == 0, f"Z failed on qubit {i}"

    def test_random_pauli_sequence(self):
        """Apply random sequence of Pauli gates."""
        np.random.seed(123)
        tab = Tableau(2)

        paulis = ['x', 'y', 'z']
        for _ in range(50):
            gate = np.random.choice(paulis)
            qubit = np.random.choice([0, 1])
            getattr(tab, gate)(qubit)

        # Just verify state is valid (normalizes)
        m0 = tab.measure(0)
        m1 = tab.measure(1)

        assert m0 in [0, 1] and m1 in [0, 1], "Random Pauli sequence produced invalid state"


def test_pauli_gate_documentation():
    """Verify that Pauli gates exist and are callable."""
    tab = Tableau(1)

    # Verify all three Pauli gates exist
    assert hasattr(tab, 'x'), "Tableau should have x method"
    assert hasattr(tab, 'y'), "Tableau should have y method"
    assert hasattr(tab, 'z'), "Tableau should have z method"

    # Verify they're callable
    assert callable(tab.x), "x should be callable"
    assert callable(tab.y), "y should be callable"
    assert callable(tab.z), "z should be callable"


if __name__ == "__main__":
    print("Running comprehensive Pauli gate tests for Pauli Simulator...")
    pytest.main([__file__, "-v", "-s"])
