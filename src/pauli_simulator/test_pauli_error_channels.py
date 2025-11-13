from pauli_error_channels import NoisySimulator
from collections import defaultdict

"""
This script verifies the Pauli error channels by 
creating a Bell state many times and observing 
the measurement outcomes.

A perfect Bell state should only ever result in 
'00' or '11' outcomes.

With noise, we expect to see '01' and '10' outcomes.
"""


ERROR_CONFIG = {
    'h': (0.01, 0.01, 0.01),
    'cx': (0.01, 0.01, 0.01)
}

N_RUNS = 10000


def run_bell_experiment(error_config):
    print(f"--- Running experiment with config: {error_config} ---")

    counts = defaultdict(int)

    for _ in range(N_RUNS):
        sim = NoisySimulator(2, error_config)

        # Create the Bell state
        sim.h(0)
        sim.cx(0, 1)

        # Measure both qubits
        m0 = sim.measure(0)
        m1 = sim.measure(1)

        # Record the outcome
        outcome_str = f"{m0}{m1}"
        counts[outcome_str] += 1

    # Print the results
    print(f"Total runs: {N_RUNS}")
    for outcome in sorted(counts.keys()):
        percent = (counts[outcome] / N_RUNS) * 100
        print(f"  Outcome '{outcome}': {counts[outcome]:>5} runs ({percent:5.2f}%)")
    print("-" * 40 + "\n")


# --- Run the Experiments ---

# Run with NO noise to verify the baseline
print("VERIFYING NOISELESS SIMULATION...")
run_bell_experiment(error_config={})

# Run WITH noise to verify the error channels
print("VERIFYING NOISY SIMULATION...")
run_bell_experiment(error_config=ERROR_CONFIG)
