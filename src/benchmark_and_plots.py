import numpy as np
import time
import psutil
import os
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
from pauli_simulator.Tableau_Ver2 import Tableau 
from QuantumCircuit import QuantumCircuit
from src.gates.registry import GateRegistry

gate_registry = GateRegistry()

QUBIT_RANGE = [4, 8, 10, 12, 16, 20] 
CIRCUIT_DEPTH = 50
FAILED = -1.0

def generate_random_clifford_circuit(n: int, depth: int) -> List[Tuple[str, int, int]]:
    """Generates a list of random Clifford gates for an n-qubit circuit."""
    circuit = []
    rng = np.random.default_rng(42)
    
    num_ops = n * depth
    for _ in range(num_ops):
        gate_type = rng.choice(['h', 's', 'cx'])
        
        if gate_type == 'cx':
            targets = rng.choice(n, 2, replace=False)
            circuit.append(('cx', int(targets[0]), int(targets[1])))
        else:
            target = rng.choice(n)
            circuit.append((gate_type, int(target), -1))
            
    return circuit

def measure_peak_rss(pid: int) -> float:
    """Measures the current Resident Set Size (RSS) in MB."""
    process = psutil.Process(pid)
    return process.memory_info().rss / (1024 * 1024)

def run_benchmark(qubits: int, circuit: List[Tuple], simulator_class: type, name: str) -> Dict[str, float]:
    """Runs a single benchmark for a given simulator and returns time and peak memory."""
    
    start_rss = measure_peak_rss(os.getpid())
    peak_rss = start_rss
    run_time = FAILED
    success = True 

    try:
        if name == "Pauli (Tableau)":
            sim = simulator_class(qubits, enable_metrics=False) 
            start_time = time.perf_counter()
            
            for gate, c, t in circuit:
                current_rss = measure_peak_rss(os.getpid())
                peak_rss = max(peak_rss, current_rss)

                if gate == 'h':
                    sim.h(c)
                elif gate == 's':
                    sim.s(c)
                elif gate == 'cx':
                    sim.cx(c, t)
                
            for i in range(qubits):
                sim.measure(i)

            end_time = time.perf_counter()
            run_time = end_time - start_time

        elif name == "Exact (State Vector)":
            sim = simulator_class(qubits, num_cbits=qubits)
            start_time = time.perf_counter()

            for gate, c, t in circuit:
                current_rss = measure_peak_rss(os.getpid())
                peak_rss = max(peak_rss, current_rss)

                if gate == 'cx':
                    sim.add_gate(gate_registry.get('cx'), targets=[c,t])
                else:
                    sim.add_gate(gate_registry.get(gate), targets=c)
            
            sim.execute()
            for i in range(qubits):
                sim.measure(i, i)
            
            end_time = time.perf_counter()
            run_time = end_time - start_time
            
    except MemoryError:
        print(f"FAILURE: {name} ran out of memory at {qubits} qubits.")
        success = False
    except Exception as e:
         print(f"ERROR: {name} run at {qubits} qubits failed: {e}")
         success = False
    
    current_rss = measure_peak_rss(os.getpid())
    peak_rss = sim.get_state_memory_mb()
    
    if not success:
        run_time = FAILED
        peak_rss = FAILED
        
    return {
        'time_s': run_time, 
        'peak_rss_mb': peak_rss
    }

def collect_data():
    """Iterates through qubit sizes and collects benchmark data for both simulators."""
    
    results = []
    
    print("-" * 50)
    print(f"Starting Benchmark (Depth: {CIRCUIT_DEPTH} ops/qubit)")
    print("-" * 50)

    for n in QUBIT_RANGE:
        print(f"Benchmarking n={n} qubits...")
        
        circuit = generate_random_clifford_circuit(n, CIRCUIT_DEPTH)

        pauli_results = run_benchmark(n, circuit, Tableau, "Pauli (Tableau)")
        exact_results = run_benchmark(n, circuit, QuantumCircuit, "Exact (State Vector)")

        print(f"  Pauli Time: {pauli_results['time_s']:.4f} s | Exact Time: {exact_results['time_s']:.4f} s")
        print(f"  Pauli Mem: {pauli_results['peak_rss_mb']:.5f} MB | Exact Mem: {exact_results['peak_rss_mb']:.5f} MB")

        time_diff = exact_results['time_s'] - pauli_results['time_s'] if exact_results['time_s'] != FAILED else FAILED
        mem_diff = exact_results['peak_rss_mb'] - pauli_results['peak_rss_mb'] if exact_results['peak_rss_mb'] != FAILED else FAILED

        time_diff_percent = (time_diff / exact_results['time_s'] * 100) if exact_results['time_s'] != FAILED else FAILED
        mem_diff_percent = (mem_diff / exact_results['peak_rss_mb'] * 100) if exact_results['peak_rss_mb'] != FAILED else FAILED

        print(f"  Time Diff: {time_diff:.4f} s ({time_diff_percent:.2f}%) | Mem Diff: {mem_diff:.2f} MB ({mem_diff_percent:.2f}%)")

        results.append({
            'Qubits': n,
            'Pauli_Time_s': pauli_results['time_s'],
            'Pauli_Mem_MB': pauli_results['peak_rss_mb'],
            'Exact_Time_s': exact_results['time_s'],
            'Exact_Mem_MB': exact_results['peak_rss_mb']
        })
        
    print("-" * 50)
    print("Benchmark Complete.")
    return pd.DataFrame(results)


def plot_results(df: pd.DataFrame):
    """Generates and saves the two comparison plots."""
    
    df_plot = df.replace(FAILED, np.nan)
    
    plt.figure(figsize=(10, 6))
    plt.rcParams["grid.color"] = "black"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.linewidth"] = 1.5
    plt.rcParams["grid.alpha"] = 1.0

    plt.plot(df_plot['Qubits'], df_plot['Pauli_Time_s'], marker='o', label='Pauli (Tableau) $\mathcal{O}(n^k)$', linestyle='-', color='blue', markersize=12)
    plt.plot(df_plot['Qubits'], df_plot['Exact_Time_s'], marker='s', label='Exact (State Vector) $\mathcal{O}(2^n)$', linestyle='--', color='red', markersize=12)

    # plt.grid(visible=True, color='black', linestyle='-', linewidth=1.5, alpha=1.0) 
    plt.yscale('log')
    plt.title('Execution Time Growth: Pauli (Polynomial) vs. Exact (Exponential)', fontsize=14)
    plt.xlabel('Number of Qubits ($n$)', fontsize=12)
    plt.ylabel('Execution Time (Seconds, Log Scale)', fontsize=12)
    plt.xticks(df['Qubits'])
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend(title='Simulator Type and Complexity')
    plt.tight_layout()
    plt.savefig('timing_growth_comparison_final.png')
    print("Saved: timing_growth_comparison_final.png")

    plt.figure(figsize=(10, 6))

    plt.plot(df_plot['Qubits'], df_plot['Pauli_Mem_MB'], marker='o', label='Pauli (Tableau) $\mathcal{O}(n^2)$', linestyle='-', color='blue', markersize=12)
    plt.plot(df_plot['Qubits'], df_plot['Exact_Mem_MB'], marker='s', label='Exact (State Vector) $\mathcal{O}(2^n)$', linestyle='--', color='red', markersize=12)

    # plt.grid(visible=True, color='black', linestyle='--', linewidth=1.5, alpha=1.0) 
    plt.yscale('log')
    plt.title('Peak Memory Growth: Pauli (Polynomial) vs. Exact (Exponential)', fontsize=14)
    plt.xlabel('Number of Qubits ($n$)', fontsize=12)
    plt.ylabel('Peak RSS (MB, Log Scale)', fontsize=12)
    plt.xticks(df['Qubits'])
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend(title='Simulator Type and Complexity')
    
    failed_n = df[df['Exact_Mem_MB'] == FAILED]['Qubits']
    if not failed_n.empty:
        first_fail_n = failed_n.iloc[0]
        plt.axvline(x=first_fail_n, color='gray', linestyle=':', linewidth=1.5, label='Approx. Failure Point')
        plt.text(first_fail_n + 0.5, plt.ylim()[1] / 2, 'Resource Limit Reached',
                 color='gray', rotation=90, va='center', ha='left', fontsize=9)

    plt.tight_layout()
    plt.savefig('memory_growth_comparison_final.png')
    print("Saved: memory_growth_comparison_final.png")


if __name__ == "__main__":
    import psutil
        
    results_df = collect_data()
    
    results_df.to_csv('benchmark_results.csv', index=False)
    print("\nSaved raw data to benchmark_results.csv")
    
    plot_results(results_df)

    print("\nFinished generating plots. Check your directory for PNG files.")