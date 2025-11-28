from typing import Dict

from Tableau_Ver2 import Tableau
from pauli_error_channels import NoisySimulator


def run_shots(t: Tableau | NoisySimulator, n_shots: int = 1000) -> Dict[str, int]:
    """
    Executes measurement shots on a Tableau or NoisySimulator instance.

    Args:
        t: Instance of Tableau or NoisySimulator.
        n_shots (int): Number of measurement shots to perform.

    Returns:
        dict: A dictionary mapping bitstrings (e.g., '011') to counts.
    """

    if not isinstance(t, (Tableau, NoisySimulator)):
        raise TypeError(
            f"run_shots only supports 'Tableau' or 'NoisySimulator'. "
            f"Got '{type(t).__name__}' instead. "
        )

    counts = {}
    n = t.n
    for _ in range(n_shots):
        t_shot = t.copy()

        outcome_bits = ''
        for q in range(n):
            bit = t_shot.measure(q)
            outcome_bits += str(bit)

        counts[outcome_bits] = counts.get(outcome_bits, 0) + 1

    return counts


def format_counts(counts: Dict[str, int], n_shots: int = 1000, title: str = "Simulation Results") -> str:
    """
    Formats the measurement counts into a simple probability table.
    """
    sorted_counts = dict(sorted(counts.items()))
    lines = []
    lines.append("")
    lines.append("=" * 50)
    lines.append(f"{title:^50}")
    lines.append("=" * 50)
    lines.append(f"{'Bitstring':<15} | {'Count':<15} | {'Probability':<15}")
    lines.append("-" * 50)

    for bitstring, count in sorted_counts.items():
        prob = count / n_shots
        lines.append(f"{bitstring:<15} | {count:<15} | {prob:<15.4f}")

    lines.append("-" * 50)
    lines.append(f"Total Shots: {n_shots}")
    lines.append("=" * 50)

    return "\n".join(lines)


if __name__ == "__main__":
    # t = Tableau(2)
    
    error_config = {
        'h': (0.3, 0.3, 0.4),
        'cx': (0.3, 0.3, 0.4)
    }
    t = NoisySimulator(n=2, error_config=error_config)


    t.h(0)
    t.cx(0, 1)

    counts = run_shots(t)

    print(format_counts(counts))