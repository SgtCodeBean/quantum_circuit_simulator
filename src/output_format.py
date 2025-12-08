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
        is_density = result.get('simulation_type') == 'density_matrix'

        if format_type == 'json':
            return OutputFormatter._format_execution_json(result)
        elif format_type == 'minimal':
            return OutputFormatter._format_execution_minimal(result)
        else:
            if is_density:
                return OutputFormatter._format_density_text(result)
            else:
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
        else:
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
    def _format_density_text(result: Dict[str, Any]) -> str:
        """Format density matrix execution result as human-readable text."""
        lines = []
        lines.append("=" * 60)
        lines.append("QUANTUM CIRCUIT EXECUTION RESULT (Density Matrix)")
        lines.append("=" * 60)
        
        # Circuit info
        lines.append("\n[CIRCUIT INFO]")
        info = result['circuit_info']
        lines.append(f"  Qubits:       {info['num_qubits']}")
        lines.append(f"  Classical:    {info['num_cbits']}")
        lines.append(f"  Operations:   {info['num_operations']} "
                    f"({info['gate_count']} gates, {info['measurement_count']} measurements, "
                    f"{info['reset_count']} resets)")
        lines.append(f"  Matrix size:  {2**info['num_qubits']} × {2**info['num_qubits']}")
        
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
        
        # Density matrix properties
        lines.append("\n[DENSITY MATRIX PROPERTIES]")
        
        # Purity
        if 'purity' in result:
            lines.append(f"  Purity: {result['purity']:.6f}")
            if result['purity'] > 0.99:
                lines.append(f"    (≈ 1, nearly pure state)")
            else:
                lines.append(f"    (< 1, mixed state)")
        
        # Von Neumann entropy
        if 'entropy' in result:
            lines.append(f"  Von Neumann Entropy: {result['entropy']:.6f}")
        
        # Trace (should be 1)
        if 'trace' in result:
            lines.append(f"  Trace: {result['trace']:.6f} (should be 1.0)")
        
        # Show diagonal (probabilities)
        lines.append("\n[PROBABILITY DISTRIBUTION]")
        probs = result['probabilities']
        num_qubits = result['circuit_info']['num_qubits']
        threshold = 1e-10
        
        non_zero_probs = [(i, probs[i]) for i in range(len(probs)) if probs[i] > threshold]
        
        if len(non_zero_probs) <= 8:
            for i, prob in non_zero_probs:
                bitstring = format(i, f'0{num_qubits}b')
                lines.append(f"  |{bitstring}⟩: P = {prob:.6f}")
        else:
            non_zero_probs.sort(key=lambda x: x[1], reverse=True)
            lines.append(f"  Showing top 8 of {len(non_zero_probs)} non-zero probabilities:")
            for i, prob in non_zero_probs[:8]:
                bitstring = format(i, f'0{num_qubits}b')
                lines.append(f"  |{bitstring}⟩: P = {prob:.6f}")
        
        # Show coherences (off-diagonal elements) if significant
        if 'significant_coherences' in result and result['significant_coherences']:
            lines.append("\n[SIGNIFICANT COHERENCES]")
            lines.append(f"  (Off-diagonal elements with |ρ_ij| > {result.get('coherence_threshold', 0.01)})")
            coherences = result['significant_coherences'][:10]  # Show top 10
            for coh in coherences:
                i, j, value = coh['i'], coh['j'], coh['value']
                i_bits = format(i, f'0{num_qubits}b')
                j_bits = format(j, f'0{num_qubits}b')
                real = value.real
                imag = value.imag
                val_str = f"{real:+.4f}{imag:+.4f}j" if abs(imag) > 1e-10 else f"{real:+.4f}"
                lines.append(f"  ρ[{i_bits},{j_bits}] = {val_str}")
            
            if len(result['significant_coherences']) > 10:
                lines.append(f"  ... and {len(result['significant_coherences']) - 10} more")
        
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
    def _format_execution_minimal(result: Dict[str, Any], is_density: bool = False) -> str:
        """Format execution result as minimal summary."""
        lines = []
        
        sim_type = "Density Matrix" if is_density else "Statevector"
        lines.append(f"Simulation: {sim_type}")
        
        if result.get('classical_bits'):
            lines.append(f"Result: |{result['classical_bits']['bitstring']}⟩")
        
        # Show top 3 most probable states
        probs = result['probabilities']
        num_qubits = result['circuit_info']['num_qubits']
        
        non_zero = [(i, probs[i]) for i in range(len(probs)) if probs[i] > 1e-10]
        non_zero.sort(key=lambda x: x[1], reverse=True)
        
        lines.append("Top states:")
        for i, prob in non_zero[:3]:
            bitstring = format(i, f'0{num_qubits}b')
            lines.append(f"  |{bitstring}⟩: {prob:.4f}")
        
        if is_density and 'purity' in result:
            lines.append(f"Purity: {result['purity']:.4f}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_execution_json(result: Dict[str, Any], is_density: bool = False) -> str:
        """Format execution result as JSON."""
        json_result = result.copy()
        
        if is_density:
            # For density matrix, don't include the full matrix (too large)
            # Just include summary statistics
            if 'density_matrix' in json_result:
                del json_result['density_matrix']
            
            # Convert probabilities
            if 'probabilities' in json_result:
                json_result['probabilities'] = [float(p) for p in json_result['probabilities']]
            
            # Convert coherences if present
            if 'significant_coherences' in json_result:
                coherences_list = []
                for coh in json_result['significant_coherences']:
                    coherences_list.append({
                        'i': int(coh['i']),
                        'j': int(coh['j']),
                        'value': {
                            'real': float(coh['value'].real),
                            'imag': float(coh['value'].imag)
                        },
                        'magnitude': float(abs(coh['value']))
                    })
                json_result['significant_coherences'] = coherences_list
        else:
            # Convert statevector
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


class TableauOutputFormatter:
    """Formatting methods specifically for Tableau simulation results."""
    
    @staticmethod
    def format_tableau_result(tableau, format_type: str = 'text') -> str:
        """
        Format Tableau simulation results.
        
        Args:
            tableau: Tableau instance after simulation
            format_type: 'text', 'json', or 'minimal'
        
        Returns:
            Formatted string
        """
        if format_type == 'json':
            return TableauOutputFormatter._format_tableau_json(tableau)
        elif format_type == 'minimal':
            return TableauOutputFormatter._format_tableau_minimal(tableau)
        else:  # text
            return TableauOutputFormatter._format_tableau_text(tableau)
    
    @staticmethod
    def _format_tableau_text(tableau) -> str:
        """Format Tableau result as human-readable text."""
        lines = []
        lines.append("=" * 60)
        lines.append("TABLEAU SIMULATION RESULT (Clifford Circuit)")
        lines.append("=" * 60)
        
        # Circuit info
        lines.append("\n[CIRCUIT INFO]")
        lines.append(f"  Qubits:       {tableau.n}")
        lines.append(f"  Classical:    {tableau.num_cbits}")
        lines.append(f"  Operations:   {len(tableau.ops)}")
        
        # Count operation types
        gate_count = sum(1 for op in tableau.ops if op[0] == 'gate')
        meas_count = sum(1 for op in tableau.ops if op[0] == 'measure')
        reset_count = sum(1 for op in tableau.ops if op[0] == 'reset')
        lines.append(f"                ({gate_count} gates, {meas_count} measurements, "
                    f"{reset_count} resets)")
        
        # Classical register
        if tableau.num_cbits > 0:
            lines.append("\n[CLASSICAL REGISTER]")
            classical_reg = tableau.get_classical_register()
            bitstring = classical_reg['bitstring']
            lines.append(f"  Bitstring: |{bitstring}⟩")
            for i in range(tableau.num_cbits):
                lines.append(f"  c[{i}] = {classical_reg[f'c[{i}]']}")
        
        # Measurements
        if tableau._measurements:
            lines.append("\n[MEASUREMENTS]")
            for qubit, info in sorted(tableau._measurements.items()):
                det_str = "deterministic" if info['deterministic'] else "probabilistic"
                cbit_str = f"c[{info['cbit']}]" if info['cbit'] is not None else "not stored"
                lines.append(f"  q[{qubit}] → {info['outcome']} ({det_str}, stored in {cbit_str})")
        
        # Stabilizer state info
        lines.append("\n[STABILIZER STATE]")
        lines.append(f"  Tableau dimensions: {2 * tableau.n} × {tableau.n}")
        lines.append(f"  State type: Clifford (exponentially compressed)")
        
        # Show a few operations
        if tableau.ops:
            lines.append("\n[CIRCUIT OPERATIONS] (last 10 shown)")
            for i, op in enumerate(tableau.ops[-10:], start=max(1, len(tableau.ops) - 9)):
                if op[0] == 'gate':
                    _, gate_name, qubits = op
                    if len(qubits) == 1:
                        lines.append(f"  {i:3d}. {gate_name.upper():4s} q[{qubits[0]}]")
                    elif len(qubits) == 2:
                        lines.append(f"  {i:3d}. {gate_name.upper():4s} q[{qubits[0]}], q[{qubits[1]}]")
                
                elif op[0] == 'measure':
                    _, qubit, cbit, outcome = op
                    if cbit is not None:
                        lines.append(f"  {i:3d}. MEAS q[{qubit}] → c[{cbit}] (outcome: {outcome})")
                    else:
                        lines.append(f"  {i:3d}. MEAS q[{qubit}] (outcome: {outcome})")
                
                elif op[0] == 'reset':
                    _, qubit = op
                    lines.append(f"  {i:3d}. RESET q[{qubit}]")
        
        lines.append("\n" + "=" * 60 + "\n")
        return "\n".join(lines)
    
    @staticmethod
    def _format_tableau_minimal(tableau) -> str:
        """Format Tableau result as minimal summary."""
        lines = []
        
        if tableau.num_cbits > 0:
            bitstring = tableau.get_classical_register()['bitstring']
            lines.append(f"Result: |{bitstring}⟩")
        else:
            lines.append("Result: No measurements performed")
        
        lines.append(f"Operations: {len(tableau.ops)}")
        
        if tableau._measurements:
            lines.append(f"Measurements: {len(tableau._measurements)}")
            det_count = sum(1 for m in tableau._measurements.values() if m['deterministic'])
            prob_count = len(tableau._measurements) - det_count
            lines.append(f"  - Deterministic: {det_count}")
            lines.append(f"  - Probabilistic: {prob_count}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_tableau_json(tableau) -> str:
        """Format Tableau result as JSON."""
        result = {
            'circuit_info': {
                'num_qubits': tableau.n,
                'num_cbits': tableau.num_cbits,
                'num_operations': len(tableau.ops),
                'gate_count': sum(1 for op in tableau.ops if op[0] == 'gate'),
                'measurement_count': sum(1 for op in tableau.ops if op[0] == 'measure'),
                'reset_count': sum(1 for op in tableau.ops if op[0] == 'reset'),
            },
            'classical_register': tableau.get_classical_register() if tableau.num_cbits > 0 else {},
            'measurements': {
                str(q): info for q, info in tableau._measurements.items()
            },
            'operations': []
        }
        
        # Add operations
        for op in tableau.ops:
            if op[0] == 'gate':
                result['operations'].append({
                    'type': 'gate',
                    'gate': op[1],
                    'qubits': op[2]
                })
            elif op[0] == 'measure':
                result['operations'].append({
                    'type': 'measure',
                    'qubit': op[1],
                    'cbit': op[2],
                    'outcome': int(op[3])
                })
            elif op[0] == 'reset':
                result['operations'].append({
                    'type': 'reset',
                    'qubit': op[1]
                })
        
        return json.dumps(result, indent=2)
    
    @staticmethod
    def format_tableau_metrics(tableau, format_type: str = 'text') -> str:
        """
        Format Tableau performance metrics.
        
        Args:
            tableau: Tableau instance with metrics enabled
            format_type: 'text' or 'json'
        
        Returns:
            Formatted string
        """
        metrics = tableau.get_metrics()
        
        if metrics is None:
            return "Metrics collection was not enabled."
        
        if format_type == 'json':
            return json.dumps(metrics, indent=2)
        
        # Text format
        lines = []
        lines.append("\n" + "=" * 50)
        lines.append("TABLEAU SIMULATOR METRICS")
        lines.append("=" * 50)
        
        lines.append(f"\nTableau size: {tableau.n} qubits")
        
        lines.append("\n--- Operations ---")
        lines.append(f"Total operations: {metrics['operations']['total_operations']}")
        lines.append(f"Gate count: {metrics['operations']['gate_count']}")
        lines.append(f"Measurement count: {metrics['operations']['measurement_count']}")
        
        lines.append("\n--- Gates by Type ---")
        for gate, count in metrics['operations']['gates_by_type'].items():
            if count > 0:
                lines.append(f"  {gate.upper()}: {count}")
        
        lines.append("\n--- Measurements ---")
        lines.append(f"Deterministic: {metrics['measurements']['deterministic']}")
        lines.append(f"Probabilistic: {metrics['measurements']['probabilistic']}")
        lines.append(f"Outcomes: 0={metrics['measurements']['outcomes'][0]}, "
                    f"1={metrics['measurements']['outcomes'][1]}")
        
        lines.append("\n--- Timing ---")
        total_time = metrics['execution']['total_time_seconds']
        lines.append(f"Total time: {total_time*1000:.3f} ms")
        lines.append(f"Gate time: {metrics['timing']['gate_time_seconds']*1000:.3f} ms")
        lines.append(f"Measurement time: {metrics['timing']['measurement_time_seconds']*1000:.3f} ms")
        
        if total_time > 0:
            gate_pct = metrics['timing']['gate_time_seconds'] / total_time * 100
            meas_pct = metrics['timing']['measurement_time_seconds'] / total_time * 100
            lines.append(f"Gate time %: {gate_pct:.1f}%")
            lines.append(f"Measurement time %: {meas_pct:.1f}%")
        
        memory = metrics['memory']
        lines.append("\n--- Memory (MB) ---")
        lines.append(f"Tableau theoretical size: {memory['tableau_size_mb']:.5f}")
        lines.append(f"Initial (RSS):          {memory['initial_mb']:.2f}")
        lines.append(f"Final (RSS):            {memory['final_mb']:.2f}")
        lines.append(f"Peak (RSS):             {memory['peak_mb']:.2f}")
        lines.append(f"Delta (Final - Initial): {memory['delta_mb']:+.2f}")
        
        if memory['gate_memory']['samples']:
            lines.append(f"Avg Gate Op (RSS):      {memory['gate_memory']['avg_mb']:.2f}\
                          (Peak: {memory['gate_memory']['peak_mb']:.2f})")
        
        if memory['measurement_memory']['samples']:
             lines.append(f"Avg Measure Op (RSS):   {memory['measurement_memory']['avg_mb']:.2f}\
                           (Peak: {memory['measurement_memory']['peak_mb']:.2f})")
        
        lines.append("=" * 50 + "\n")
        return "\n".join(lines)


class TableauResultWriter:
    """
    Result writer specifically for Tableau simulations.
    """
    
    def __init__(self, output_path=None, format_type='text', verbose=True):
        """
        Initialize Tableau result writer.
        
        Args:
            output_path: Path to output file (None for console only)
            format_type: Output format ('text', 'json', 'minimal')
            verbose: Whether to also print to console
        """
        self.output_path = output_path
        self.format_type = format_type
        self.verbose = verbose
    
    def write_result(self, tableau):
        """Write or print Tableau simulation result."""
        content = TableauOutputFormatter.format_tableau_result(tableau, self.format_type)
        
        if self.verbose:
            print(content)
        
        if self.output_path:
            self._write_to_file(content)
    
    def write_metrics(self, tableau):
        """Write or print Tableau metrics."""
        content = TableauOutputFormatter.format_tableau_metrics(tableau, self.format_type)
        
        if self.verbose:
            print(content)
        
        if self.output_path:
            self._append_to_file(content)
    
    def write_circuit_structure(self, tableau):
        """Write or print detailed circuit structure."""
        if self.verbose:
            tableau.print_circuit()
        
        if self.output_path:
            # Capture print_circuit output
            import io
            import sys
            
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            
            tableau.print_circuit()
            
            sys.stdout = old_stdout
            content = buffer.getvalue()
            
            self._append_to_file(content)
    
    def _write_to_file(self, content):
        """Write content to file."""
        from pathlib import Path
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _append_to_file(self, content):
        """Append content to file."""
        from pathlib import Path
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'a', encoding='utf-8') as f:
            f.write(content)


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

def save_tableau_results(tableau, filepath: str, format_type: str = 'text', 
                         include_metrics: bool = False):
    """
    Save Tableau simulation results to a file.
    
    Args:
        tableau: Tableau instance after simulation
        filepath: Output file path
        format_type: Format type ('text', 'json', 'minimal')
        include_metrics: Whether to include performance metrics
    """
    writer = TableauResultWriter(output_path=filepath, format_type=format_type, verbose=False)
    writer.write_result(tableau)
    
    if include_metrics and tableau.get_metrics():
        writer.write_metrics(tableau)


def print_tableau_results(tableau, format_type: str = 'text', include_metrics: bool = False):
    """
    Print Tableau simulation results to console.
    
    Args:
        tableau: Tableau instance after simulation
        format_type: Format type ('text', 'json', 'minimal')
        include_metrics: Whether to include performance metrics
    """
    writer = TableauResultWriter(output_path=None, format_type=format_type, verbose=True)
    writer.write_result(tableau)
    
    if include_metrics and tableau.get_metrics():
        writer.write_metrics(tableau)