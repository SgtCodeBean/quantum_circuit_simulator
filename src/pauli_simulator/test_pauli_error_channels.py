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
def test_noiseless_bell_state():
    print("VERIFYING NOISELESS SIMULATION...")
    run_bell_experiment(error_config={})

# Run WITH noise to verify the error channels
def test_pauli_noise_on_bell_state():
    print("VERIFYING NOISY SIMULATION...")
    run_bell_experiment(error_config=ERROR_CONFIG)

def test_pauli_noise_off_means_no_flips():
    # Turn noise completely off
    zero_cfg = {
        "h": (0.0, 0.0, 0.0),
        "s": (0.0, 0.0, 0.0),
        "x": (0.0, 0.0, 0.0),
        "y": (0.0, 0.0, 0.0),
        "z": (0.0, 0.0, 0.0),
        "cx": (0.0, 0.0, 0.0),
        "measure": (0.0, 0.0, 0.0),
    }

    sim = NoisySimulator(n=2, error_config=zero_cfg, enable_metrics=True)

    # Do a bunch of gates + measurements
    num_trials = 500
    for _ in range(num_trials):
        # simple little Clifford circuit
        sim.tableau.h(0)
        sim.tableau.cx(0, 1)
        sim.tableau.measure(0)
        sim.tableau.measure(1)

    metrics = sim.get_metrics()
    noise = metrics["noise"]

    # Sanity: no noise events, no flips
    assert noise["total_noise_events"] == 0
    assert noise["x_flips"] == 0
    assert noise["y_flips"] == 0
    assert noise["z_flips"] == 0
    assert noise["no_error"] == 0

def test_default_pauli_error_config_produces_some_flips():
    from pauli_error_channels import NoisySimulator, build_default_pauli_error_config

    cfg = build_default_pauli_error_config()

    sim = NoisySimulator(n=2, error_config=cfg, enable_metrics=True)

    num_trials = 5000
    for _ in range(num_trials):
        # random-ish small circuit
        sim.h(0)
        sim.cx(0, 1)
        sim.s(1)
        sim.measure(0)
        sim.measure(1)

    metrics = sim.get_metrics()
    noise = metrics["noise"]

    total_errors = noise["x_flips"] + noise["y_flips"] + noise["z_flips"]
    total_events = noise["total_noise_events"]

    # Sanity: we should see some errors, but not 100%
    assert total_errors > 0
    assert total_errors < total_events

def test_measurement_noise_changes_outcomes():
    from Tableau_Ver2 import Tableau
    # No noise case
    zero_cfg = {"measure": (0.0, 0.0, 0.0)}
    sim_ideal = NoisySimulator(n=1, error_config=zero_cfg)

    ideal_ones = 0
    trials = 1000
    for _ in range(trials):
        sim_ideal.tableau = Tableau(1)  # reset to |0>
        m = sim_ideal.measure(0)
        ideal_ones += m
    ideal_rate = ideal_ones / trials

    # Now with a bit of X noise before measurement
    noisy_cfg = {"measure": (0.2, 0.0, 0.0)}  # 20% chance of X flip
    sim_noisy = NoisySimulator(n=1, error_config=noisy_cfg)

    noisy_ones = 0
    for _ in range(trials):
        sim_noisy.tableau = Tableau(1)  # reset to |0>
        m = sim_noisy.measure(0)
        noisy_ones += m
    noisy_rate = noisy_ones / trials

    # Ideal should be ~0 (we’re measuring |0⟩)
    assert ideal_rate < 0.05
    # Noisy should be notably higher
    assert noisy_rate > 0.1

def test_single_qubit_x_noise_stats():
    cfg = {
        "h": (0.1, 0.0, 0.0),
    }

    sim = NoisySimulator(n=1, error_config=cfg, enable_metrics=True)
    # sim.rng = np.random.default_rng(123)

    num_trials = 10_000
    for _ in range(num_trials):
        sim.h(0)

    metrics = sim.get_metrics()
    noise = metrics["noise"]

    total_events = noise["total_noise_events"]
    x_flips = noise["x_flips"]
    y_flips = noise["y_flips"]
    z_flips = noise["z_flips"]

    total_errors = x_flips + y_flips + z_flips

    # We should at least have recorded some events
    assert total_events > 0

    empirical_error_rate = total_errors / total_events
    # Expect ~0.1 within some tolerance
    assert abs(empirical_error_rate - 0.1) < 0.02, (
        f"Expected ~0.1 error rate, got {empirical_error_rate}"
    )

    # And almost all of them should be X flips
    assert y_flips == 0
    assert z_flips == 0

    assert total_events == num_trials

def test_single_qubit_y_noise_stats():
    cfg = {
        "h": (0.0, 0.1, 0.0),
    }

    sim = NoisySimulator(n=1, error_config=cfg, enable_metrics=True)
    # sim.rng = np.random.default_rng(123)

    num_trials = 10_000
    for _ in range(num_trials):
        sim.h(0)

    metrics = sim.get_metrics()
    noise = metrics["noise"]

    total_events = noise["total_noise_events"]
    x_flips = noise["x_flips"]
    y_flips = noise["y_flips"]
    z_flips = noise["z_flips"]

    total_errors = x_flips + y_flips + z_flips

    # We should at least have recorded some events
    assert total_events > 0

    empirical_error_rate = total_errors / total_events
    # Expect ~0.1 within some tolerance
    assert abs(empirical_error_rate - 0.1) < 0.02, (
        f"Expected ~0.1 error rate, got {empirical_error_rate}"
    )

    # And almost all of them should be X flips
    assert x_flips == 0
    assert z_flips == 0

    assert total_events == num_trials

def test_single_qubit_z_noise_stats():
    cfg = {
        "h": (0.0, 0.0, 0.1),
    }

    sim = NoisySimulator(n=1, error_config=cfg, enable_metrics=True)
    # sim.rng = np.random.default_rng(123)

    num_trials = 10_000
    for _ in range(num_trials):
        sim.h(0)

    metrics = sim.get_metrics()
    noise = metrics["noise"]

    total_events = noise["total_noise_events"]
    x_flips = noise["x_flips"]
    y_flips = noise["y_flips"]
    z_flips = noise["z_flips"]

    total_errors = x_flips + y_flips + z_flips

    # We should at least have recorded some events
    assert total_events > 0

    empirical_error_rate = total_errors / total_events
    # Expect ~0.1 within some tolerance
    assert abs(empirical_error_rate - 0.1) < 0.02, (
        f"Expected ~0.1 error rate, got {empirical_error_rate}"
    )

    # And almost all of them should be X flips
    assert x_flips == 0
    assert y_flips == 0

    assert total_events == num_trials