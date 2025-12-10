import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def create_memory_context_plot():
    """
    Creates two graphs: one log scale (for true comparison) and one linear scale
    (to show the exponential curve visually).

    AI Generated to help with creating the contextual memory scaling charts.
    """
    # 1. Configuration and Data Points
    QUBITS_MAX = 52 
    qubits = np.arange(1, QUBITS_MAX + 1)
    
    # Constants
    BYTES_PER_COMPLEX = 16
    GB_IN_BYTES = 1024**3
    
    # Exact Simulator Memory: O(2^n)
    # Memory in Gigabytes = (16 * 2^n) / (1024^3)
    exact_memory_gb = (BYTES_PER_COMPLEX * 2**qubits) / GB_IN_BYTES
    
    # Pauli Simulator Memory: O(n^2)
    # Tableau size: 2*n rows * n columns (for X) + 2*n rows * n columns (for Z) = 4*n^2 elements
    # Using np.uint8 (1 byte per element)
    # Added a small constant (100 bytes) for auxiliary structure overhead to keep line slightly visible
    TABLEAU_OVERHEAD_BYTES = 100 
    pauli_memory_gb = (4 * qubits**2 + TABLEAU_OVERHEAD_BYTES) / GB_IN_BYTES

    # 2. Define Real-World Memory Milestones (in Gigabytes)
    milestones = [
        (128, 'High-End Workstation (128 GB)', 'b', 16),
        (4096, 'Large Enterprise Server (4 TB)', 'g', 22),
        (5500000, 'El Capitan Total RAM (~5.5 PB)', '#a40014', 30) 
    ]
    
    # ======================================================================
    # A. LOG SCALE PLOT (Mathematically Correct for Comparison)
    # ======================================================================
    plt.figure(figsize=(12, 8))

    plt.rcParams["grid.color"] = "black"
    plt.rcParams["grid.linestyle"] = "-"
    plt.rcParams["grid.linewidth"] = 1.5
    plt.rcParams["grid.alpha"] = 0.3
    
    plt.plot(qubits, exact_memory_gb, 
             label=r'Exact Memory ($\mathcal{O}(2^n)$)', 
             color='red', linewidth=3)
             
    plt.plot(qubits, pauli_memory_gb, 
             label=r'Pauli Memory ($\mathcal{O}(n^2)$)', 
             color='blue', linestyle='--', linewidth=2)
             
    # Add Milestones
    for mem_gb, label, color, n_qubits_approx in milestones:
        plt.axhline(y=mem_gb, color=color, linestyle=':', linewidth=2.0, alpha=0.5)
        
        if mem_gb >= 1024**2: 
            label_text = label.replace("PB", f"({mem_gb/1024**2:.1f} PB)")
        elif mem_gb >= 1024: 
            label_text = label.replace("TB", f"({mem_gb/1024:.1f} TB)")
        else:
            label_text = label
            
        plt.text(n_qubits_approx - 2, mem_gb * 1.5, label_text, 
                 color=color, fontsize=15, verticalalignment='bottom')
        
    # Final Plot Aesthetics (Log)
    plt.title(r'Memory Wall: Exponential vs. Polynomial Scaling', fontsize=16)
    plt.gca().tick_params(axis='both', which='major', labelsize=15, direction='in', length=5, width=1.5)
    plt.xlabel('Number of Qubits ($N$)', fontsize=15)
    plt.ylabel('Memory Cost (GB)', fontsize=15)
    plt.yscale('log')
    plt.xlim(1, QUBITS_MAX)
    # Set lower limit low enough to capture the O(n^2) growth curve
    plt.ylim(10**(-6), exact_memory_gb[-1] * 2) 
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend(loc='upper left', fontsize=15, frameon=False, ncol=2, handlelength=0.5, columnspacing=0.6)
    plt.tight_layout()
    plt.savefig('memory_wall_context_plot_log.png')

    # ======================================================================
    # B. LINEAR SCALE PLOT (Visually Dramatic, but Impractical)
    # ======================================================================
    
    # We must limit the range for the linear plot
    LINEAR_QUBITS_MAX = 38 
    linear_qubits = np.arange(1, LINEAR_QUBITS_MAX + 1)
    linear_exact_memory_gb = (BYTES_PER_COMPLEX * 2**linear_qubits) / GB_IN_BYTES
    linear_pauli_memory_gb = (4 * linear_qubits**2 + TABLEAU_OVERHEAD_BYTES) / GB_IN_BYTES

    plt.figure(figsize=(15, 6))

    plt.rcParams["grid.color"] = "black"
    plt.rcParams["grid.linestyle"] = "-"
    plt.rcParams["grid.linewidth"] = 1.5
    plt.rcParams["grid.alpha"] = 1.0
    
    plt.plot(linear_qubits, linear_exact_memory_gb, 
             label=r'Exact Simulator Memory ($\mathcal{O}(2^n)$)', 
             color='red', linewidth=3)
             
    # Plotting the tiny Pauli line here is still pointless, but we include it.
    plt.plot(linear_qubits, linear_pauli_memory_gb, 
             label=r'Pauli Simulator Memory ($\mathcal{O}(n^2)$)', 
             color='blue', linestyle='--', linewidth=2)
    
    plt.title(r'Memory Growth (Linear Scale): $\mathcal{O}(2^n)$ vs. $\mathcal{O}(n^2)$', fontsize=14)
    plt.xlabel('Number of Qubits ($n$)', fontsize=12)
    plt.ylabel('Theoretical Peak Memory (Gigabytes, Linear Scale)', fontsize=12)
    plt.xlim(1, LINEAR_QUBITS_MAX)
    plt.ylim(0, linear_exact_memory_gb[-1] * 1.1)
    plt.grid(True, which="both", ls="--", linewidth=0.5, alpha=0.7)
    plt.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig('memory_wall_context_plot_linear.png')

create_memory_context_plot()