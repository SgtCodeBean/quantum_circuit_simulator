from pauli_error_channels import NoisySimulator

error_config = {
    'h': (0.1, 0.1, 0.1),  # 30% total error rate
    'cx': (0.05, 0.05, 0.05),
}

ns = NoisySimulator(2, error_config=error_config, enable_metrics=True)

for _ in range(100):
    ns.h(0)
    ns.cx(0, 1)

ns.measure(0)
ns.measure(1)

m = ns.get_metrics()
noise = m['noise']

print(f"Total noise events: {noise['total_noise_events']}")
print(f"X flips: {noise['x_flips']}")
print(f"Y flips: {noise['y_flips']}")
print(f"Z flips: {noise['z_flips']}")
print(f"No error: {noise['no_error']}")

for gate, stats in noise['by_gate'].items():
    print(f"{gate}: {stats}")

ns.print_metrics()