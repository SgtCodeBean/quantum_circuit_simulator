from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

from numbers import Number
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.parameterexpression import ParameterExpression

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

def _qc_to_ir(qc: QuantumCircuit) -> Dict[str, Any]:
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
    qc = QuantumCircuit.from_qasm_str(qasm)
    if basis_gates:
        qc = transpile(qc, basis_gates=basis_gates, optimization_level=0)
    return _qc_to_ir(qc)

def parse_qasm_file(path: Union[str, Path],
                    basis_gates: Optional[List[str]] = None) -> Dict[str, Any]:
    """Load a .qasm file and return IR."""
    path = Path(path)
    qc = QuantumCircuit.from_qasm_file(str(path))
    if basis_gates:
        qc = transpile(qc, basis_gates=basis_gates, optimization_level=0)
    return _qc_to_ir(qc)

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