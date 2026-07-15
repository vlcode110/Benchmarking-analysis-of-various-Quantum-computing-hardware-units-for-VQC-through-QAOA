// 10-qubit XEB random circuit, depth d=1, instance k=11.
// Same connectivity / single-qubit gate set as xeb-10qubits.ipynb.
// Ideal probabilities for this circuit: p_ideal.npz['d01_c011'].

OPENQASM 3.0;
include "stdgates.inc";
gate r(p0, p1) _gate_q_0 {
  U(p0, -pi/2 + p1, pi/2 - p1) _gate_q_0;
}
bit[10] meas;
qubit[10] q;
r(pi/2, pi/4) q[0];
rx(pi/2) q[1];
rx(pi/2) q[2];
r(pi/2, pi/4) q[3];
r(pi/2, pi/4) q[4];
rx(pi/2) q[5];
rx(pi/2) q[6];
r(pi/2, pi/4) q[7];
rx(pi/2) q[8];
rx(pi/2) q[9];
cx q[0], q[1];
cx q[1], q[2];
cx q[2], q[3];
cx q[3], q[4];
cx q[5], q[6];
cx q[6], q[7];
cx q[7], q[8];
cx q[8], q[9];
ry(pi/2) q[0];
r(pi/2, pi/4) q[1];
r(pi/2, pi/4) q[2];
rx(pi/2) q[3];
rx(pi/2) q[4];
r(pi/2, pi/4) q[5];
ry(pi/2) q[6];
rx(pi/2) q[7];
ry(pi/2) q[8];
r(pi/2, pi/4) q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
meas[0] = measure q[0];
meas[1] = measure q[1];
meas[2] = measure q[2];
meas[3] = measure q[3];
meas[4] = measure q[4];
meas[5] = measure q[5];
meas[6] = measure q[6];
meas[7] = measure q[7];
meas[8] = measure q[8];
meas[9] = measure q[9];
