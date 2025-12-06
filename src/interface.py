import argparse
import sys
import os
import qasm_parser
from gates.registry import GateRegistry
from circuit_results import ResultManager
from output_format import ResultWriter, TableauResultWriter

def verify_file(file_path):
    abspath = os.path.abspath(file_path)
    if not os.path.isfile(abspath):
        print(f"Error: QASM file '{file_path}' does not exist.")
        sys.exit(1)

def determine_output_format(output_path):
    """Determine output format from file extension."""
    if output_path is None:
        return 'text'
    
    ext = os.path.splitext(output_path)[1].lower()
    format_map = {
        '.json': 'json',
        '.csv': 'csv',
        '.txt': 'text',
        '.md': 'text'
    }
    return format_map.get(ext, 'text')

def run_statevector_simulator(file, shots, metrics=False, output=None, seed=None, verbose=True, format_type=None):
    print(f"Running statevector simulation for {file} with {shots} shots...")
    if output:
        print(f"Output will be saved to: {output}")

    qc = qasm_parser.qasm_file_to_exactsim(file, num_shots=shots, enable_metrics=metrics, rng_seed=seed)

    if format_type is None:
        format_type = determine_output_format(output)
    
    writer = ResultWriter(
        output_path=output,
        format_type=format_type,
        verbose=verbose
    )

    if shots > 1 and qc.num_cbits > 0:
        shot_results = qc.run_shots(verbose=False)
        writer.write_shot_results(shot_results)
        
        if metrics and verbose:
            print("\n" + "="*60)
            qc.print_metrics()
    else:
        exec_result = qc.execute(verbose=False)
        writer.write_execution_result(exec_result)
        
        if metrics and verbose:
            print("\n" + "="*60)
            qc.print_metrics()
    
    if verbose and output:
        print(f"\n✓ Results saved to {output}")

def run_density_simulator(file, shots, metrics=None, seed=None, output=None, verbose=True, format_type=None):
    print(f"Running density matrix simulation for {file}...")

    if output:
        print(f"Output will be saved to: {output}")

    qc = qasm_parser.qasm_file_to_exactsim(file, num_shots=shots, use_density_matrix=True, enable_metrics=metrics, rng_seed=seed)

    if format_type is None:
        format_type = determine_output_format(output)
    
    writer = ResultWriter(
        output_path=output,
        format_type=format_type,
        verbose=verbose
    )

    if shots > 1 and qc.num_cbits > 0:
        print("Note: Shot-based execution with density matrices...")
        shot_results = qc.run_shots(verbose=False)
        writer.write_shot_results(shot_results)
        
        if metrics and verbose:
            print("\n" + "="*60)
            qc.print_metrics()
    else:
        exec_result = qc.execute(verbose=False)
        writer.write_execution_result(exec_result)
        
        if metrics and verbose:
            print("\n" + "="*60)
            qc.print_metrics()
    
    if verbose and output:
        print(f"\n✓ Results saved to {output}")

def run_tableau_simulator(file, shots, metrics=None, output=None, verbose=False, format_type=None):
    print(f"Running tableau structure simulation for {file}...")
    if output:
        print(f"Output will be saved to: {output}")

    clifford_basis = ['h', 's', 'x', 'y', 'z', 'cx', 'measure', 'reset']
    qc = qasm_parser.qasm_file_to_tableau(file, basis_gates=clifford_basis, num_shots=shots, metrics=metrics)

    if format_type is None:
        format_type = determine_output_format(output)
    
    writer = TableauResultWriter(
        output_path=output,
        format_type=format_type,
        verbose=verbose
    )

    writer.write_result(qc)

    if metrics:
        if verbose:
            print("\n" + "="*60)
        writer.write_metrics(qc)
    
    if verbose and output:
        print(f"\n✓ Results saved to {output}")

def main():
    parser = argparse.ArgumentParser(
        description="Quantum Circuit Simulator CLI"
    )

    parser.add_argument(
        "mode",
        choices=["statevector", "density", "tableau"],
        help="Select simulation mode"
    )

    parser.add_argument(
        "-f", "--file",
        required=True,
        help="Path to QASM file"
    )

    parser.add_argument(
        "-r", "--root",
        help="Root directory for QASM files"
    )

    parser.add_argument(
        "-s", "--shots",
        type=int,
        default=1024,
        help="Number of shots for state vector simulation (default: 1024)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Integer seed to be used in random generation for state vector simulation (default: unseeded)"
    )

    parser.add_argument(
        "-o", "--output",
        help="Path to file where simulator output will be saved"
    )

    parser.add_argument(
        "-m", "--metrics",
        action="store_true",
        default=False,
        help="Flag to include simulator metrics (default: False)"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        default=False,
        help="Quiet mode - only write to file, don't print to console"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Verbose mode - print detailed execution information"
    )

    parser.add_argument(
        "--format",
        choices=["text", "json", "csv", "minimal"],
        help="Output format (auto-detected from file extension if not specified)"
    )

    args = parser.parse_args()

    verify_file(args.file)

    verbose = not args.quiet
    if args.verbose:
        verbose = True

    if args.mode == "statevector":
        run_statevector_simulator(
            file=args.file, 
            shots=args.shots, 
            metrics=args.metrics, 
            output=args.output, 
            seed=args.seed,
            verbose=verbose,
            format_type=args.format
            )
    elif args.mode == "density":
        run_density_simulator(args.file, args.metrics, args.output)
    elif args.mode == "tableau":
        run_tableau_simulator(
            file=args.file,
            shots=args.shots,
            metrics=args.metrics,
            output=args.output,
            verbose=verbose,
            format_type=args.format
            )
    else:
        print("Unknown mode selected.")
        sys.exit(1)

if __name__ == "__main__":
    main()