"""
Pure-OpenQASM 3 deployment pipeline for 10-qubit XEB on IBM Quantum.

Sibling of `ibm_qasm_pipeline.py` (which deploys the trained VQC). XEB is a
*hybrid* protocol: the QPU only samples bitstrings from random circuits, but
all of the actual benchmarking is classical post-processing against ideal
probabilities computed locally by statevector simulation.

Three subcommands, exactly mirroring `ibm_qasm_pipeline.py`:

    generate    one bound OpenQASM 3 file per random circuit, plus a
                companion `p_ideal.npz` of classically-simulated ideal
                probabilities and a `manifest.csv`. These are the
                IBM-facing artifacts.
    submit      load each QASM back, transpile, submit in one Job via
                qiskit-ibm-runtime SamplerV2; save per-circuit counts
                as JSON in `counts/`.
    evaluate    load counts + ideal probs, compute the linear-XEB
                fidelity per circuit, average per depth, and fit
                F_XEB(d) = A * f**d to extract per-cycle fidelity.

Typical run:

    python ibm_xeb_qasm_pipeline.py generate
    python ibm_xeb_qasm_pipeline.py submit   --backend ibm_brisbane
    python ibm_xeb_qasm_pipeline.py evaluate --plot xeb_decay.png

The XEB random-circuit conventions (single-qubit gate set, no-repeat rule,
trailing single-qubit layer, fixed 2x5-grid CNOT entangler, depths
1..6) match `xeb-10qubits.ipynb` exactly, so the per-cycle fidelity you
recover here is the noise budget for the QVC ansatz on the same backend.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np

from qiskit import QuantumCircuit, qasm3
from qiskit.quantum_info import Statevector


# ----------------------------------------------------------------------------
# XEB configuration (must match xeb-10qubits.ipynb)
# ----------------------------------------------------------------------------
NUM_QUBITS = 10

CNOT_LAYER = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8), (8, 9),
]

DEPTHS = [1, 2, 3, 4, 5, 6]
NUM_CIRCUITS_PER_DEPTH = 20
SHOTS = 4000
RNG_SEED = 1234

SQ_GATES = ("sx", "sy", "sw")


# ----------------------------------------------------------------------------
# Random-circuit generator (identical conventions to the notebook)
# ----------------------------------------------------------------------------
def _apply_sq(qc: QuantumCircuit, qubit: int, name: str) -> None:
    if name == "sx":
        qc.rx(np.pi / 2, qubit)
    elif name == "sy":
        qc.ry(np.pi / 2, qubit)
    elif name == "sw":
        # sqrt(W), W = (X+Y)/sqrt(2): rotate by pi/2 about (X+Y)/sqrt(2).
        qc.r(np.pi / 2, np.pi / 4, qubit)
    else:
        raise ValueError(name)


def random_xeb_circuit(num_qubits: int, num_cycles: int,
                       rng: np.random.Generator) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    last = [None] * num_qubits
    for _ in range(num_cycles):
        for q in range(num_qubits):
            choices = [g for g in SQ_GATES if g != last[q]]
            g = rng.choice(choices)
            _apply_sq(qc, q, g)
            last[q] = g
        for a, b in CNOT_LAYER:
            qc.cx(a, b)
    for q in range(num_qubits):
        choices = [g for g in SQ_GATES if g != last[q]]
        g = rng.choice(choices)
        _apply_sq(qc, q, g)
    return qc


def ideal_probabilities(circuit: QuantumCircuit) -> np.ndarray:
    return np.abs(Statevector.from_instruction(circuit).data) ** 2


# ----------------------------------------------------------------------------
# `generate`: emit one QASM per random circuit + p_ideal.npz + manifest.csv
# ----------------------------------------------------------------------------
def cmd_generate(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    depths = [int(d) for d in args.depths.split(",")] if args.depths else list(DEPTHS)
    n_per_d = args.num_circuits

    rng = np.random.default_rng(args.seed)
    manifest = []
    p_ideals: dict[str, np.ndarray] = {}

    for d in depths:
        for k in range(n_per_d):
            qc = random_xeb_circuit(NUM_QUBITS, d, rng)
            p_ideals_key = f"d{d:02d}_c{k:03d}"
            p_ideals[p_ideals_key] = ideal_probabilities(qc).astype(np.float64)

            qc_meas = qc.copy()
            qc_meas.measure_all()

            fname = f"xeb_{p_ideals_key}.qasm"
            header = (
                f"// 10-qubit XEB random circuit, depth d={d}, instance k={k}.\n"
                f"// Same connectivity / single-qubit gate set as xeb-10qubits.ipynb.\n"
                f"// Ideal probabilities for this circuit: p_ideal.npz['{p_ideals_key}'].\n\n"
            )
            (out_dir / fname).write_text(header + qasm3.dumps(qc_meas))
            manifest.append({"file": fname, "depth": d, "k": k, "key": p_ideals_key})

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "depth", "k", "key"])
        w.writeheader()
        w.writerows(manifest)

    np.savez_compressed(out_dir / "p_ideal.npz", **p_ideals)

    print(f"wrote {len(manifest)} QASM files + p_ideal.npz + manifest.csv to {out_dir}/")
    print(f"depths = {depths},  circuits/depth = {n_per_d},  total = {len(manifest)}")


# ----------------------------------------------------------------------------
# `submit`: send the QASMs to IBM, save counts
# ----------------------------------------------------------------------------
def cmd_submit(args: argparse.Namespace) -> None:
    out_dir = Path(args.dir)
    manifest_path = out_dir / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")

    rows = list(csv.DictReader(manifest_path.open()))
    if not rows:
        raise SystemExit("manifest is empty")

    try:
        circuits = [qasm3.load(str(out_dir / r["file"])) for r in rows]
    except Exception as e:
        raise SystemExit(
            "could not load QASM 3 files. You probably need:\n"
            "    pip install qiskit_qasm3_import\n"
            f"Underlying error: {e}"
        )

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    service = QiskitRuntimeService()
    backend = (service.backend(args.backend) if args.backend
               else service.least_busy(operational=True, simulator=False))
    print(f"backend: {backend.name}")

    pm = generate_preset_pass_manager(target=backend.target,
                                      optimization_level=args.optimization_level)
    transpiled = [pm.run(qc) for qc in circuits]

    sampler = SamplerV2(mode=backend, options={"default_shots": args.shots})
    print(f"submitting {len(transpiled)} circuits as one Job ({args.shots} shots each)...")
    job = sampler.run([(qc,) for qc in transpiled])
    print(f"job id: {job.job_id()}    waiting for completion...")
    result = job.result()

    counts_dir = out_dir / "counts"
    counts_dir.mkdir(exist_ok=True)
    for row, pub_res in zip(rows, result):
        bits = pub_res.data.meas.get_bitstrings()
        counts: dict[str, int] = {}
        for b in bits:
            counts[b] = counts.get(b, 0) + 1
        stem = Path(row["file"]).stem
        (counts_dir / f"{stem}.json").write_text(json.dumps(counts))
    print(f"saved {len(rows)} count files to {counts_dir}/")


# ----------------------------------------------------------------------------
# `evaluate`: F_XEB per circuit -> average per depth -> exponential fit
# ----------------------------------------------------------------------------
def _xeb_fidelity(p_ideal: np.ndarray, counts: dict[str, int]) -> float:
    D = len(p_ideal)
    p_meas = np.zeros(D, dtype=np.float64)
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    for bitstr, c in counts.items():
        b = bitstr.replace(" ", "")
        idx = int(b, 2)
        p_meas[idx] += c
    p_meas /= total
    num = float(np.sum(p_meas * p_ideal) - 1.0 / D)
    den = float(np.sum(p_ideal ** 2) - 1.0 / D)
    return num / den


def cmd_evaluate(args: argparse.Namespace) -> None:
    out_dir = Path(args.dir)
    manifest_path = out_dir / "manifest.csv"
    counts_dir = out_dir / "counts"
    p_ideal_path = out_dir / "p_ideal.npz"

    if not manifest_path.exists():
        raise SystemExit(f"missing manifest: {manifest_path}")
    if not counts_dir.exists():
        raise SystemExit(f"missing counts directory: {counts_dir}")
    if not p_ideal_path.exists():
        raise SystemExit(f"missing ideal-probabilities file: {p_ideal_path}")

    rows = list(csv.DictReader(manifest_path.open()))
    p_ideals = np.load(p_ideal_path)

    by_depth: dict[int, list[float]] = {}
    per_circuit_rows = []
    for row in rows:
        stem = Path(row["file"]).stem
        depth = int(row["depth"])
        cpath = counts_dir / f"{stem}.json"
        if not cpath.exists():
            print(f"  skipping {stem} (no counts file)")
            continue
        counts = json.loads(cpath.read_text())
        f_xeb = _xeb_fidelity(p_ideals[row["key"]], counts)
        by_depth.setdefault(depth, []).append(f_xeb)
        per_circuit_rows.append({"file": row["file"], "depth": depth,
                                 "k": int(row["k"]), "F_XEB": f_xeb})

    if not by_depth:
        raise SystemExit("no circuits could be evaluated.")

    depths_sorted = sorted(by_depth)
    print(f"\n{'depth':>5}  {'<F_XEB>':>9}  {'SEM':>7}  {'N':>4}")
    mean_f, sem_f = [], []
    for d in depths_sorted:
        vals = np.array(by_depth[d], dtype=np.float64)
        m = float(np.mean(vals))
        s = float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        mean_f.append(m)
        sem_f.append(s)
        print(f"{d:>5}  {m:>9.4f}  {s:>7.4f}  {len(vals):>4d}")
    mean_f = np.array(mean_f)
    sem_f = np.array(sem_f)
    depths_arr = np.array(depths_sorted, dtype=float)

    A_fit = f_fit = A_err = f_err = None
    if len(depths_arr) >= 2:
        from scipy.optimize import curve_fit

        def model(d, A, f):
            return A * f ** d

        try:
            popt, pcov = curve_fit(
                model, depths_arr, mean_f, p0=[1.0, 0.99],
                sigma=np.maximum(sem_f, 1e-3), absolute_sigma=False,
                bounds=([0, 0], [1.5, 1.0]),
            )
            A_fit, f_fit = popt
            A_err, f_err = np.sqrt(np.diag(pcov))
            print(f"\nfit  F_XEB(d) = A * f^d:")
            print(f"   per-cycle fidelity f = {f_fit:.4f} +/- {f_err:.4f}")
            print(f"   SPAM prefactor    A = {A_fit:.4f} +/- {A_err:.4f}")
        except Exception as e:
            print(f"fit failed: {e}")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "depth", "k", "F_XEB"])
            w.writeheader()
            w.writerows(per_circuit_rows)
        print(f"per-circuit fidelities -> {args.out_csv}")

    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        plt.errorbar(depths_sorted, mean_f, yerr=sem_f, fmt="s-",
                     label=r"$F_\mathrm{XEB}$")
        if f_fit is not None:
            dd = np.linspace(min(depths_sorted), max(depths_sorted), 100)
            plt.plot(dd, A_fit * f_fit ** dd, "k:",
                     label=f"fit: A={A_fit:.3f}, f={f_fit:.4f}")
        plt.axhline(0, color="gray", lw=0.5)
        plt.axhline(1, color="gray", lw=0.5)
        plt.xlabel("cycles d")
        plt.ylabel(r"$F_\mathrm{XEB}$")
        plt.title("10-qubit XEB on IBM (from saved QASM counts)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)
        print(f"plot -> {args.plot}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Pure-OpenQASM 3 IBM deployment of 10-qubit XEB"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="emit random-circuit QASMs + ideal probs")
    g.add_argument("--out", default="ibm_xeb_qasm_jobs",
                   help="output directory for .qasm files, p_ideal.npz, manifest.csv")
    g.add_argument("--num-circuits", type=int, default=NUM_CIRCUITS_PER_DEPTH,
                   dest="num_circuits",
                   help="random circuits per depth (more = tighter error bars).")
    g.add_argument("--depths", default=None,
                   help="comma-separated depths, e.g. '1,2,3,4,5,6'. "
                        f"Default: {DEPTHS}.")
    g.add_argument("--seed", type=int, default=RNG_SEED,
                   help="RNG seed for circuit generation.")
    g.set_defaults(fn=cmd_generate)

    s = sub.add_parser("submit", help="submit the QASMs to IBM via qiskit-ibm-runtime")
    s.add_argument("--dir", default="ibm_xeb_qasm_jobs")
    s.add_argument("--backend", default=None,
                   help="IBM backend name (e.g. ibm_brisbane). "
                        "Default: least busy non-simulator.")
    s.add_argument("--shots", type=int, default=SHOTS)
    s.add_argument("--optimization-level", type=int, default=3,
                   dest="optimization_level",
                   help="preset pass manager level (default 3).")
    s.set_defaults(fn=cmd_submit)

    e = sub.add_parser("evaluate", help="F_XEB and per-cycle fidelity from counts")
    e.add_argument("--dir", default="ibm_xeb_qasm_jobs")
    e.add_argument("--out-csv", default=None,
                   help="optional CSV: per-circuit (file, depth, k, F_XEB).")
    e.add_argument("--plot", default=None,
                   help="optional path to save the F_XEB(d) decay plot.")
    e.set_defaults(fn=cmd_evaluate)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
