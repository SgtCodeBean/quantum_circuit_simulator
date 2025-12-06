"""
Result management for quantum circuit simulation.
"""
import numpy as np
from typing import Dict, Any, Optional


class ResultManager:
    """
    Manages and formats quantum circuit execution results.
    """
    
    def __init__(self):
        """Initialize result manager."""
        self._last_execution_result = None
        self._last_shot_results = None
    
    def store_execution_result(self, result: Dict[str, Any]):
        """Store the most recent execution result."""
        self._last_execution_result = result.copy()
    
    def store_shot_results(self, results: Dict[str, Any]):
        """Store the most recent shot results."""
        self._last_shot_results = results.copy()
    
    def get_execution_result(self) -> Optional[Dict[str, Any]]:
        """
        Get the result of the most recent single execution.
        
        Returns:
            Execution results dict or None if not available
        """
        if self._last_execution_result is None:
            return None
        return self._last_execution_result.copy()
    
    def get_shot_results(self) -> Optional[Dict[str, Any]]:
        """
        Get the results of the most recent shot-based execution.
        
        Returns:
            Shot results dict or None if not available
        """
        if self._last_shot_results is None:
            return None
        return self._last_shot_results.copy()
    
    def get_counts(self) -> Optional[Dict[str, int]]:
        """
        Get measurement counts from the most recent shot execution.
        
        Returns:
            Measurement counts dict or None if not available
        """
        if self._last_shot_results is None:
            return None
        return self._last_shot_results['counts'].copy()
    
    def get_shot_probabilities(self) -> Optional[Dict[str, float]]:
        """
        Get probability distribution from the most recent shot execution.
        
        Returns:
            Probabilities dict or None if not available
        """
        if self._last_shot_results is None:
            return None
        return self._last_shot_results['probabilities'].copy()
    
    @staticmethod
    def build_execution_result(state, num_qubits: int, num_cbits: int, 
                               ops: list, cbits, measurements: dict,
                               metrics: Optional[Dict] = None,
                               use_density_matrix: bool = False) -> Dict[str, Any]:
        """
        Build execution result dictionary from circuit state.
        
        Args:
            state: State vector (1D array) or density matrix (2D array)
            num_qubits: Number of qubits
            num_cbits: Number of classical bits
            ops: List of operations performed
            cbits: Classical bit register object
            measurements: Dictionary of measurement outcomes
            metrics: Optional performance metrics
            use_density_matrix: Whether state is a density matrix
            
        Returns:
            Dictionary containing execution results
        """
        # Detect if density matrix based on shape
        is_density = len(state.shape) == 2
        
        if is_density:
            return ResultManager._build_density_result(
                state, num_qubits, num_cbits, ops, cbits, measurements, metrics
            )
        else:
            return ResultManager._build_statevector_result(
                state, num_qubits, num_cbits, ops, cbits, measurements, metrics
            )
        
    @staticmethod
    def _build_statevector_result(state, num_qubits, num_cbits, ops, cbits, 
                                   measurements, metrics):
        """Build result for statevector simulation."""
        # Calculate probabilities
        probabilities = np.abs(state) ** 2
        
        # Build classical register
        classical_bits = {}
        if num_cbits > 0:
            for i in range(num_cbits):
                classical_bits[f'c[{i}]'] = cbits.get_bit(i)
            classical_bits['bitstring'] = ''.join(
                str(cbits.get_bit(i)) for i in range(num_cbits)
            )
        else:
            classical_bits['bitstring'] = ''
        
        # Build measurements dict
        measurements_dict = {}
        for qubit, info in measurements.items():
            measurements_dict[f'q[{qubit}]'] = {
                'outcome': info['outcome'],
                'stored_in': f"c[{info['cbit']}]" if 'cbit' in info else None
            }
        
        # Count operations
        gate_count = sum(1 for op in ops if op[0] == 'gate')
        meas_count = sum(1 for op in ops if op[0] == 'measure')
        reset_count = sum(1 for op in ops if op[0] == 'reset')
        
        result = {
            'simulation_type': 'statevector',
            'circuit_info': {
                'num_qubits': num_qubits,
                'num_cbits': num_cbits,
                'num_operations': len(ops),
                'gate_count': gate_count,
                'measurement_count': meas_count,
                'reset_count': reset_count,
            },
            'state_vector': state.copy(),
            'probabilities': probabilities,
            'classical_bits': classical_bits,
            'measurements': measurements_dict,
        }
        
        if metrics:
            result['metrics'] = metrics
        
        return result
    
    @staticmethod
    def _build_density_result(rho, num_qubits, num_cbits, ops, cbits, 
                              measurements, metrics):
        """Build result for density matrix simulation."""
        # Extract probabilities from diagonal
        probabilities = np.real(np.diag(rho))
        
        # Calculate purity: Tr(ρ²)
        purity = float(np.real(np.trace(rho @ rho)))
        
        # Calculate von Neumann entropy: -Tr(ρ log₂ ρ)
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = np.real(eigenvalues)
        eigenvalues = eigenvalues[eigenvalues > 1e-15]  # Filter numerical noise
        entropy = float(-np.sum(eigenvalues * np.log2(eigenvalues))) if len(eigenvalues) > 0 else 0.0
        
        # Calculate trace (should be 1)
        trace = float(np.real(np.trace(rho)))
        
        # Find significant off-diagonal elements (coherences)
        coherence_threshold = 0.01
        significant_coherences = []
        dim = rho.shape[0]
        
        for i in range(dim):
            for j in range(i + 1, dim):  # Only upper triangle
                magnitude = abs(rho[i, j])
                if magnitude > coherence_threshold:
                    significant_coherences.append({
                        'i': i,
                        'j': j,
                        'value': rho[i, j],
                        'magnitude': magnitude
                    })
        
        # Sort by magnitude
        significant_coherences.sort(key=lambda x: x['magnitude'], reverse=True)
        
        # Build classical register
        classical_bits = {}
        if num_cbits > 0:
            for i in range(num_cbits):
                classical_bits[f'c[{i}]'] = cbits.get_bit(i)
            classical_bits['bitstring'] = ''.join(
                str(cbits.get_bit(i)) for i in range(num_cbits)
            )
        else:
            classical_bits['bitstring'] = ''
        
        # Build measurements dict
        measurements_dict = {}
        for qubit, info in measurements.items():
            measurements_dict[f'q[{qubit}]'] = {
                'outcome': info['outcome'],
                'stored_in': f"c[{info['cbit']}]" if 'cbit' in info else None
            }
        
        # Count operations
        gate_count = sum(1 for op in ops if op[0] == 'gate')
        meas_count = sum(1 for op in ops if op[0] == 'measure')
        reset_count = sum(1 for op in ops if op[0] == 'reset')
        
        result = {
            'simulation_type': 'density_matrix',
            'circuit_info': {
                'num_qubits': num_qubits,
                'num_cbits': num_cbits,
                'num_operations': len(ops),
                'gate_count': gate_count,
                'measurement_count': meas_count,
                'reset_count': reset_count,
            },
            'density_matrix': rho.copy(),  # Include full matrix for programmatic access
            'probabilities': probabilities,
            'purity': purity,
            'entropy': entropy,
            'trace': trace,
            'significant_coherences': significant_coherences,
            'coherence_threshold': coherence_threshold,
            'classical_bits': classical_bits,
            'measurements': measurements_dict,
        }
        
        if metrics:
            result['metrics'] = metrics
        
        return result
    
    @staticmethod
    def build_shot_results(counts: Dict[str, int], num_shots: int,
                          num_qubits: int, num_cbits: int) -> Dict[str, Any]:
        """Build a comprehensive shot results dictionary."""
        return {
            'counts': counts,
            'shots': num_shots,
            'probabilities': {k: v / num_shots for k, v in counts.items()},
            'num_qubits': num_qubits,
            'num_cbits': num_cbits
        }