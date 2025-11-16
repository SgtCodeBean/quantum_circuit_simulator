import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional, TextIO
import sys

class OutputFormatter:
    """
    Handles formatting and output of quantum circuit simulation results.
    Supports console printing and file writing in multiple formats.

    100% written by AI because I'm tired.
    """
    
    @staticmethod
    def format_execution_result(result: Dict[str, Any], format_type: str = 'text') -> str:
        """
        Format a single execution result.
        
        Args:
            result: Execution result dictionary
            format_type: 'text', 'json', or 'minimal'
        
        Returns:
            Formatted string
        """
        if format_type == 'json':
            return OutputFormatter._format_execution_json(result)
        elif format_type == 'minimal':
            return OutputFormatter._format_execution_minimal(result)
        else:  # text
            return OutputFormatter._format_execution_text(result)
    
    @staticmethod
    def format_shot_results(results: Dict[str, Any], format_type: str = 'text') -> str:
        """
        Format shot-based execution results.
        
        Args:
            results: Shot results dictionary
            format_type: 'text', 'json', 'csv', or 'minimal'
        
        Returns:
            Formatted string
        """
        if format_type == 'json':
            return OutputFormatter._format_shots_json(results)
        elif format_type == 'csv':
            return OutputFormatter._format_shots_csv(results)
        elif format_type == 'minimal':
            return OutputFormatter._format_shots_minimal(results)
        else:  # text
            return OutputFormatter._format_shots_text(results)
    
    @staticmethod
    def write_to_file(content: str, filepath: str, mode: str = 'w'):
        """
        Write content to a file.
        
        Args:
            content: String content to write
            filepath: Path to output file
            mode: Write mode ('w' or 'a')
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, mode, encoding='utf-8') as f:
            f.write(content)
    
    @staticmethod
    def print_to_console(content: str, file: TextIO = sys.stdout):
        """
        Print content to console or specified stream.
        
        Args:
            content: String content to print
            file: Output stream (default: stdout)
        """
        print(content, file=file)
    
    # === Private formatting methods ===
    
    @staticmethod
    def _format_execution_text(result: Dict[str, Any]) -> str:
        """Format execution result as human-readable text."""
        lines = []
        lines.append("=" * 60)
        lines.append("QUANTUM CIRCUIT EXECUTION RESULT")
        lines.append("=" * 60)
        
        # Circuit info
        lines.append("\n[CIRCUIT INFO]")
        info = result['circuit_info']
        lines.append(f"  Qubits:       {info['num_qubits']}")
        lines.append(f"  Classical:    {info['num_cbits']}")
        lines.append(f"  Operations:   {info['num_operations']} "
                    f"({info['gate_count']} gates, {info['measurement_count']} measurements, "
                    f"{info['reset_count']} resets)")
        
        # Classical bits
        if result.get('classical_bits'):
            lines.append("\n[CLASSICAL REGISTER]")
            bitstring = result['classical_bits']['bitstring']
            lines.append(f"  Bitstring: |{bitstring}⟩")
            for cbit, value in sorted(result['classical_bits'].items()):
                if cbit != 'bitstring':
                    lines.append(f"  {cbit} = {value}")
        
        # Measurements
        if result.get('measurements'):
            lines.append("\n[MEASUREMENTS]")
            for qubit, info in sorted(result['measurements'].items()):
                lines.append(f"  {qubit} → {info['outcome']} (stored in {info['stored_in']})")
        
        # State vector (show only non-zero amplitudes)
        lines.append("\n[FINAL STATE]")
        probs = result['probabilities']
        state_vector = result['state_vector']
        num_qubits = result['circuit_info']['num_qubits']
        threshold = 1e-10
        
        non_zero_states = [(i, state_vector[i], probs[i]) 
                          for i in range(len(probs)) if probs[i] > threshold]
        
        if len(non_zero_states) <= 8:
            for i, amplitude, prob in non_zero_states:
                bitstring = format(i, f'0{num_qubits}b')
                real = amplitude.real
                imag = amplitude.imag
                amp_str = f"{real:+.6f}{imag:+.6f}j" if abs(imag) > 1e-10 else f"{real:+.6f}"
                lines.append(f"  |{bitstring}⟩: {amp_str}  (P = {prob:.6f})")
        else:
            non_zero_states.sort(key=lambda x: x[2], reverse=True)
            lines.append(f"  Showing top 8 of {len(non_zero_states)} non-zero states:")
            for i, amplitude, prob in non_zero_states[:8]:
                bitstring = format(i, f'0{num_qubits}b')
                real = amplitude.real
                imag = amplitude.imag
                amp_str = f"{real:+.6f}{imag:+.6f}j" if abs(imag) > 1e-10 else f"{real:+.6f}"
                lines.append(f"  |{bitstring}⟩: {amp_str}  (P = {prob:.6f})")
        
        # Metrics
        if result.get('metrics'):
            lines.append("\n[PERFORMANCE METRICS]")
            metrics = result['metrics']
            lines.append(f"  Execution time: {metrics['execution']['total_time_seconds']:.6f}s")
            lines.append(f"  Peak memory:    {metrics['memory']['peak_mb']:.2f} MB")
            lines.append(f"  Memory delta:   {metrics['memory']['delta_mb']:+.2f} MB")
        
        lines.append("\n" + "=" * 60 + "\n")
        return "\n".join(lines)
    
    @staticmethod
    def _format_execution_minimal(result: Dict[str, Any]) -> str:
        """Format execution result as minimal summary."""
        lines = []
        if result.get('classical_bits'):
            lines.append(f"Result: |{result['classical_bits']['bitstring']}⟩")
        
        # Show top 3 most probable states
        probs = result['probabilities']
        state_vector = result['state_vector']
        num_qubits = result['circuit_info']['num_qubits']
        
        non_zero = [(i, probs[i]) for i in range(len(probs)) if probs[i] > 1e-10]
        non_zero.sort(key=lambda x: x[1], reverse=True)
        
        lines.append("Top states:")
        for i, prob in non_zero[:3]:
            bitstring = format(i, f'0{num_qubits}b')
            lines.append(f"  |{bitstring}⟩: {prob:.4f}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_execution_json(result: Dict[str, Any]) -> str:
        """Format execution result as JSON."""
        # Convert numpy arrays to lists for JSON serialization
        json_result = result.copy()
        
        if 'state_vector' in json_result:
            json_result['state_vector'] = [
                {'real': float(c.real), 'imag': float(c.imag)} 
                for c in json_result['state_vector']
            ]
        
        if 'probabilities' in json_result:
            json_result['probabilities'] = [float(p) for p in json_result['probabilities']]
        
        return json.dumps(json_result, indent=2)
    
    @staticmethod
    def _format_shots_text(results: Dict[str, Any]) -> str:
        """Format shot results as human-readable text."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"SHOT-BASED SIMULATION RESULTS ({results['shots']} shots)")
        lines.append("=" * 60)
        
        lines.append("\n[MEASUREMENT STATISTICS]")
        sorted_counts = sorted(results['counts'].items(), 
                              key=lambda x: x[1], reverse=True)
        
        for bitstring, count in sorted_counts:
            prob = results['probabilities'][bitstring]
            bar_length = int(40 * prob)
            bar = '█' * bar_length
            lines.append(f"  |{bitstring}⟩: {count:5d} ({prob:6.2%}) {bar}")
        
        lines.append("\n" + "=" * 60 + "\n")
        return "\n".join(lines)
    
    @staticmethod
    def _format_shots_minimal(results: Dict[str, Any]) -> str:
        """Format shot results as minimal summary."""
        lines = []
        lines.append(f"Shots: {results['shots']}")
        
        sorted_counts = sorted(results['counts'].items(), 
                              key=lambda x: x[1], reverse=True)
        
        for bitstring, count in sorted_counts[:5]:  # Top 5
            prob = results['probabilities'][bitstring]
            lines.append(f"|{bitstring}⟩: {count} ({prob:.2%})")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_shots_json(results: Dict[str, Any]) -> str:
        """Format shot results as JSON."""
        return json.dumps(results, indent=2)
    
    @staticmethod
    def _format_shots_csv(results: Dict[str, Any]) -> str:
        """Format shot results as CSV."""
        lines = []
        lines.append("bitstring,count,probability")
        
        for bitstring, count in sorted(results['counts'].items()):
            prob = results['probabilities'][bitstring]
            lines.append(f"{bitstring},{count},{prob}")
        
        return "\n".join(lines)


class ResultWriter:
    """
    High-level interface for writing quantum simulation results.
    """
    
    def __init__(self, output_path: Optional[str] = None, format_type: str = 'text', 
                 verbose: bool = True):
        """
        Initialize result writer.
        
        Args:
            output_path: Path to output file (None for console only)
            format_type: Output format ('text', 'json', 'csv', 'minimal')
            verbose: Whether to also print to console
        """
        self.output_path = output_path
        self.format_type = format_type
        self.verbose = verbose
    
    def write_execution_result(self, result: Dict[str, Any]):
        """Write or print execution result."""
        content = OutputFormatter.format_execution_result(result, self.format_type)
        
        if self.verbose:
            OutputFormatter.print_to_console(content)
        
        if self.output_path:
            OutputFormatter.write_to_file(content, self.output_path)
    
    def write_shot_results(self, results: Dict[str, Any]):
        """Write or print shot results."""
        content = OutputFormatter.format_shot_results(results, self.format_type)
        
        if self.verbose:
            OutputFormatter.print_to_console(content)
        
        if self.output_path:
            OutputFormatter.write_to_file(content, self.output_path)
    
    def append_to_file(self, content: str):
        """Append content to the output file."""
        if self.output_path:
            OutputFormatter.write_to_file(content, self.output_path, mode='a')


# Convenience functions
def save_results(result: Dict[str, Any], filepath: str, format_type: str = 'text'):
    """
    Save quantum circuit results to a file.
    
    Args:
        result: Result dictionary (from execute() or run_shots())
        filepath: Output file path
        format_type: Format type ('text', 'json', 'csv', 'minimal')
    """
    # Detect result type
    if 'counts' in result and 'shots' in result:
        content = OutputFormatter.format_shot_results(result, format_type)
    else:
        content = OutputFormatter.format_execution_result(result, format_type)
    
    OutputFormatter.write_to_file(content, filepath)


def print_results(result: Dict[str, Any], format_type: str = 'text'):
    """
    Print quantum circuit results to console.
    
    Args:
        result: Result dictionary (from execute() or run_shots())
        format_type: Format type ('text', 'json', 'minimal')
    """
    # Detect result type
    if 'counts' in result and 'shots' in result:
        content = OutputFormatter.format_shot_results(result, format_type)
    else:
        content = OutputFormatter.format_execution_result(result, format_type)
    
    OutputFormatter.print_to_console(content)