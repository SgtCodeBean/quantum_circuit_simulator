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
    def build_execution_result(state: np.ndarray, num_qubits: int, num_cbits: int,
                               ops: list, cbits, measurements: dict,
                               metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build a comprehensive execution result dictionary."""
        result = {
            'success': True,
            'state_vector': state.copy(),
            'probabilities': np.abs(state)**2,
            'classical_bits': {},
            'measurements': {},
            'circuit_info': {
                'num_qubits': num_qubits,
                'num_cbits': num_cbits,
                'num_operations': len(ops),
                'gate_count': len([op for op in ops if op[0] == "gate"]),
                'measurement_count': len([op for op in ops if op[0] == "measure"]),
                'reset_count': len([op for op in ops if op[0] == "reset"])
            }
        }
        
        # Add classical bit values
        if num_cbits > 0:
            result['classical_bits'] = {
                f'c[{i}]': cbits.get_bit(i) for i in range(num_cbits)
            }
            result['classical_bits']['bitstring'] = ''.join(
                str(cbits.get_bit(i)) for i in range(num_cbits)
            )
        
        # Add measurement details
        if measurements:
            result['measurements'] = {
                f'q[{qubit}]': {
                    'outcome': info['outcome'],
                    'stored_in': f"c[{info['cbit']}]"
                }
                for qubit, info in measurements.items()
            }
        
        # Add metrics if provided
        if metrics is not None:
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