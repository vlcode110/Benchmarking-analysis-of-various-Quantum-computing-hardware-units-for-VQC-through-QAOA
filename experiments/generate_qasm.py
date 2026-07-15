"""
Generate OpenQASM 3 files for the 10-qubit QVC circuit.

Outputs:
  qvc_qnn_10qubits_template.qasm  - parameterized circuit (30 `input` params)
  qvc_qnn_10qubits_example.qasm   - one fully-bound concrete circuit
                                    (example image + random weights)

Why two files?
- Some software stacks (Qiskit, OpenQASM 3 simulators, Braket via OpenQASM via
  Pulser/QIR, etc.) accept parameterized OpenQASM 3 with `input` declarations.
- Most actual hardware queue endpoints (IonQ, IBM, Rigetti, Quantinuum, ...)
  want a fully numeric circuit. The "example" file is one such snapshot, and
  you can regenerate one snapshot per (image, weights) pair from this script.

A `Z^{⊗10}` expectation is recovered classically from the measurement counts as
    <Z...Z> = sum_b ((-1)^|b|) * counts[b] / shots
"""

import numpy as np
from qiskit import qasm3

from qvc_qnn_10qubits_run import build_full_circuit, SIZE


def _add_measurements(qc):
    qc = qc.copy()
    qc.measure_all()
    return qc


def main() -> None:
    full, ansatz, _obs = build_full_circuit(SIZE)
    full_meas = _add_measurements(full)

    with open("qvc_qnn_10qubits_template.qasm", "w") as f:
        f.write(
            "// QVC / QNN — 10 qubits — parameterized OpenQASM 3 template\n"
            "// 10 inputs a[0..9]   = image-pixel angles (encoded by z_feature_map)\n"
            "// 20 inputs θ[0..19]  = trainable weights (Ry: θ[0..9], Rx: θ[10..19])\n"
            "// Final measurement: c[i] = Z-basis outcome on q[i].\n"
            "// Recover <Z⊗...⊗Z> = (1/shots) Σ_b (-1)^(|b|) counts[b].\n\n"
        )
        f.write(qasm3.dumps(full_meas))

    np.random.seed(0)
    example_input = np.random.rand(SIZE) * np.pi / 4
    np.random.seed(42)
    example_weights = np.random.rand(2 * SIZE) * 2 * np.pi

    binding = {}
    for p in full_meas.parameters:
        idx = int(p.name.split("[")[1].rstrip("]"))
        if p.name.startswith("a"):
            binding[p] = float(example_input[idx])
        else:
            binding[p] = float(example_weights[idx])
    bound = full_meas.assign_parameters(binding)

    with open("qvc_qnn_10qubits_example.qasm", "w") as f:
        f.write(
            "// QVC / QNN — 10 qubits — fully-bound OpenQASM 3 example\n"
            "// All 30 parameters are bound to concrete values:\n"
            f"//   image angles a   = {example_input.tolist()}\n"
            f"//   weights θ        = {example_weights.tolist()}\n"
            "// Re-run generate_qasm.py with new values to produce a new snapshot.\n\n"
        )
        f.write(qasm3.dumps(bound))

    print("wrote qvc_qnn_10qubits_template.qasm")
    print("wrote qvc_qnn_10qubits_example.qasm")


if __name__ == "__main__":
    main()
