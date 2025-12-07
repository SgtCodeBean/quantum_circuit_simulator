OPENQASM 2.0;

include 'qelib1.inc'; // stdgates.inc for OPENQASM 3, qelib1.inc for OPENQASM 2.0

//qubit[2] q1;
qreg q1[2];
//bit[2] c;
creg c[2];

h q1[0];
cx q1[0], q1[1];
// c[0] = measure q1[0];
// c[1] = measure q1[1];
measure q1[0] -> c[0];
measure q1[1] -> c[1];