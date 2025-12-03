from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

from numbers import Number
from QuantumCircuit import QuantumCircuit
from gates.registry import GateRegistry
from pauli_simulator.Tableau_Ver2 import Tableau
import qiskit as q
from qiskit import transpile
from qiskit.circuit.parameterexpression import ParameterExpression
import numpy as np

# --- helpers ---
def _param_to_float_if_numeric(p: Any) -> Any:
    """

    """
    if isinstance(p, Number):
        return float(p)
    if isinstance(p, ParameterExpression):
        if len(p.parameters) == 0:
            return float(p)
        return str(p)
    return p

def _qc_to_ir(qc: q.QuantumCircuit) -> Dict[str, Any]:
    qubit_index  = {qb: i for i, qb in enumerate(qc.qubits)}
    clbit_index  = {cb: i for i, cb in enumerate(qc.clbits)}
    ops: List[Dict[str, Any]] = []

    for inst in qc.data:
        op   = inst.operation
        qids = [qubit_index[qb] for qb in inst.qubits]
        cids = [clbit_index[cb] for cb in inst.clbits]
        params = [_param_to_float_if_numeric(p) for p in getattr(op, "params", [])]

        ops.append({
            "name": op.name,
            "qargs": qids,
            "cargs": cids,
            "params": params,
            "condition": getattr(op, "condition", None),
        })

    return {
        "n_qubits": qc.num_qubits,
        "n_clbits": qc.num_clbits,
        "ops": ops,
    }

# --- public entry points ---
def parse_qasm_source(qasm: str,
                      basis_gates: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Parse OpenQASM 2.0 source to your IR. Optionally transpile to a basis
    (expands custom/opaque gates into standard gates your simulator supports).
    """
    qc = q.QuantumCircuit.from_qasm_str(qasm)
    if basis_gates:
        qc = transpile(qc, basis_gates=basis_gates, optimization_level=0)
    return _qc_to_ir(qc)

def parse_qasm_file(path: Union[str, Path],
                    basis_gates: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load a .qasm file and return IR."""
    path = Path(path)
    qc = q.QuantumCircuit.from_qasm_file(str(path))
    if basis_gates:
        qc = transpile(qc, basis_gates=basis_gates, optimization_level=0)
    return _qc_to_ir(qc)

def build_circuit_from_ir(ir: Dict[str, Any], reg: GateRegistry, num_shots=1024, metrics=False, rng_seed=None) -> QuantumCircuit:
    n = int(ir["n_qubits"])
    m = int(ir.get("n_clbits", 0))
    qc = QuantumCircuit(num_qubits=n, num_cbits=m, num_shots=num_shots, enable_metrics=metrics, rng_seed=rng_seed)

    for op in ir["ops"]:
        name = op["name"]
        qargs = op.get("qargs", []) or []
        cargs = op.get("cargs", []) or []
        params = op.get("params", []) or []

        if name in reg.list():
            gate = reg.get(name)
            targets: List[int] = list(qargs)
            qc.add_gate(gate, targets if len(targets) > 1 else targets[0])
        elif name == "measure":
            if len(qargs) == len(cargs) and len(qargs) > 0:
                for qb, cb in zip(qargs, cargs):
                    qc.measure(qb, cb)
            else:
                raise ValueError(f"Unsupported measure form: nargs={qargs}, params={params}")
        elif name == "reset":
            if len(qargs) == 1:
                qc.reset(qargs[0])
            elif len(qargs) > 1:
                for qb in qargs:
                    qc.reset(qb)
            else:
                raise ValueError("reset requires at least one qubit index")
        else:
            raise NotImplementedError(f"Op '{name}' not supported by backend (qargs={qargs}, cargs={cargs})")

    return qc

def build_tableau_from_ir(ir: Dict[str, Any], num_shots=1024, metrics=False) -> Tableau:
    """
    Build a Tableau circuit from intermediate representation.
    
    This function creates a Tableau-based quantum circuit and queues all operations
    from the IR. The circuit can then be executed with the execute() method.
    
    Args:
        ir (dict): Intermediate representation with n_qubits, n_clbits, and ops
        num_shots (int): Number of measurement shots (not used in Tableau, kept for API consistency)
        enable_metrics (bool): Whether to enable performance metrics collection
        rng_seed (int, optional): Random number generator seed
        
    Returns:
        Tableau: Configured tableau circuit ready for execution
    """
    n = int(ir["n_qubits"])
    m = int(ir.get("n_clbits", 0))
    qc = Tableau(n, num_cbits=m, enable_metrics=metrics)

    gate_map = {
        'h': lambda qargs: qc.h(qargs[0]),
        's': lambda qargs: qc.s(qargs[0]),
        'x': lambda qargs: qc.x(qargs[0]),
        'y': lambda qargs: qc.y(qargs[0]),
        'z': lambda qargs: qc.z(qargs[0]),
        'cx': lambda qargs: qc.cx(qargs[0], qargs[1]),
    }

    for op in ir["ops"]:
        name = op["name"]
        qargs = op.get("qargs", []) or []
        cargs = op.get("cargs", []) or []
        params = op.get("params", []) or []

        if name in gate_map:
            gate_map[name](qargs)
        
        elif name == "measure":
            if len(qargs) == len(cargs) and len(qargs) > 0:
                for qb, cb in zip(qargs, cargs):
                    qc.measure(qb, cb)
            else:
                raise ValueError(f"Unsupported measure form: nargs={qargs}, params={params}")
        
        elif name == "reset":
            for qb in qargs:
                qc.reset(qb)

        else:
            raise NotImplementedError(f"Op '{name}' not supported by backend (qargs={qargs}, cargs={cargs})")
    return qc

def qasm_to_tableau(qasm: str, basis_gates: Optional[List[str]] = None, num_shots=1024, metrics: bool = False) -> Tableau:
    """
    One-step conversion from QASM string to Tableau circuit.
    
    Args:
        qasm (str): OpenQASM 2.0 source code
        basis_gates (list, optional): Basis gates for transpilation
        num_shots (int): Number of simulator executions
        enable_metrics (bool): Enable performance metrics
        
    Returns:
        Tableau: Ready-to-execute tableau circuit
    """

    ir = parse_qasm_source(qasm, basis_gates=basis_gates)
    return build_tableau_from_ir(ir, num_shots=num_shots, metrics=metrics)

def qasm_file_to_tableau(path: Union[str, Path], basis_gates: Optional[List[str]] = None, num_shots=1024, metrics: bool = False) -> Tableau:
    """
    One-step conversion from QASM file to Tableau circuit.
    
    Args:
        path (str or Path): Path to QASM file
        basis_gates (list, optional): Basis gates for transpilation
        num_shots (int): Number of simulator executions
        enable_metrics (bool): Enable performance metrics
        
    Returns:
        Tableau: Ready-to-execute tableau circuit
    """
    ir = parse_qasm_file(path, basis_gates=basis_gates)
    return build_tableau_from_ir(ir, num_shots=num_shots, metrics=metrics)

def qasm_to_exactsim(qasm: str, basis_gates: Optional[List[str]] = None, gate_reg = GateRegistry(preload_defaults=True), num_shots = 1024, enable_metrics = False, rng_seed = None):
    """
    One-step conversion from QASM string to an Exact Circuit Simulator.

    Args:
        qasm (str): OpenQASM 2.0 source code
        basis_gates (list, optional): Basis gates for transpilation
        gate_reg (GateRegistry): Set of gates allowed in circuit building
        num_shots (int): Number of simulator executions
        enable_metrics (bool): Enable performance metrics
        rng_seed (int, optional): Random seed

    Returns:
        QuantumCircuit: Ready-to-execute Quantum Circuit
    """
    ir = parse_qasm_source(qasm, basis_gates=basis_gates)
    return build_circuit_from_ir(ir=ir, reg=gate_reg, num_shots=num_shots, metrics= enable_metrics, rng_seed=rng_seed)

def qasm_file_to_exactsim(path: str, basis_gates: Optional[List[str]] = None, gate_reg = GateRegistry(preload_defaults=True), num_shots = 1024, enable_metrics = False, rng_seed = None):
    """
    One-step conversion from QASM file to an Exact Circuit Simulator.

    Args:
        qasm (str or Path): OpenQASM 2.0 source code
        basis_gates (list, optional): Basis gates for transpilation
        gate_reg (GateRegistry): Set of gates allowed in circuit building
        num_shots (int): Number of simulator executions
        enable_metrics (bool): Enable performance metrics
        rng_seed (int, optional): Random seed

    Returns:
        QuantumCircuit: Ready-to-execute Quantum Circuit
    """
    ir = parse_qasm_file(path, basis_gates=basis_gates)
    return build_circuit_from_ir(ir=ir, reg=gate_reg, num_shots=num_shots, metrics= enable_metrics, rng_seed=rng_seed)

# Test
if __name__ == "__main__":
    qasm_str = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];
        
        h q[0];
        s q[1];
        
        x q[2];
        y q[0];
        z q[1];
        
        cx q[0], q[1];
        ccx q[0], q[1], q[2];
        
        measure q -> c;
        
        reset q;
        
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    """

    # Choose a basis simulator implements (edit as needed):
    basis = [
        "x","y","z",
        "h",
        "s",
        "cx",
        "ccx",
        "measure",
        "reset"
    ]

    # Test
    str_ir = parse_qasm_source(qasm_str, basis_gates=basis)
    print("From QASM string:")
    print(json.dumps(str_ir, indent=2))

    file_ir = parse_qasm_file("test_qasm.qasm", basis_gates=basis)
    print("\nFrom QASM file:")
    print(json.dumps(file_ir, indent=2))
    assert file_ir == str_ir, "IR from string and file should match"

    reg = GateRegistry(preload_defaults=True)
    qc = build_circuit_from_ir(str_ir, reg)
    print("\nReconstructed QuantumCircuit:")
    print("Number of qubits: ", qc.num_qubits)
    print("Number of classical bits: ", qc.num_cbits)
    print("Initial state: ", qc.state)

    rng = np.random.default_rng(456)

    # 3) Execute with noise
    qc.execute()

    print("\nFinal noisy statevector:")
    print(qc.get_state())

    print("\nMeasurement probabilities (from final state):")
    print(qc.measure_probabilities())