"""
Pure-OpenQASM deployment pipeline for the 10-qubit VQC on IBM Quantum.

Three subcommands:

    generate    one bound OpenQASM 3 file per test image (artifacts
                you would upload to IBM as the "QASM-only" job)
    submit      load each QASM back, transpile, submit in one Job
                via qiskit-ibm-runtime; save per-circuit counts as JSON
    evaluate    compute <Z⊗...⊗Z> from saved counts and report accuracy

Typical end-to-end run (after `python qvc_qnn_10qubits_run.py --provider
simulator` has produced `weights_10q.npy`):

    python ibm_qasm_pipeline.py generate --weights weights_10q.npy
    python ibm_qasm_pipeline.py submit   --backend ibm_brisbane
    python ibm_qasm_pipeline.py evaluate --out-csv predictions.csv

The IBM-facing artifact is the directory of `.qasm` files. If you don't want
to use this submitter at all, you can upload those files to IBM Quantum
Composer / the IBM Quantum REST API directly — see the README at the bottom.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from qiskit import QuantumCircuit, qasm3
from qvc_qnn_10qubits_run import (
    DATA_SEED,
    NUM_IMAGES,
    SIZE,
    SPLIT_SEED,
    TEST_FRACTION,
    build_full_circuit,
    generate_dataset,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _parameter_index(name: str) -> int:
    return int(name.split("[")[1].rstrip("]"))


def _prepare_test_set(num_images: int):
    np.random.seed(DATA_SEED)
    images, labels = generate_dataset(num_images)
    _, test_images, _, test_labels = train_test_split(
        images, labels, test_size=TEST_FRACTION, random_state=SPLIT_SEED
    )
    return test_images, test_labels


# ----------------------------------------------------------------------------
# `generate`: bound QASM per test image
# ----------------------------------------------------------------------------
def cmd_generate(args: argparse.Namespace) -> None:
    full, _, _ = build_full_circuit(SIZE)
    full = full.copy()
    full.measure_all()

    weights = np.load(args.weights)
    if weights.shape != (2 * SIZE,):
        raise SystemExit(
            f"weights file {args.weights}: expected shape ({2*SIZE},), got {weights.shape}"
        )

    test_images, test_labels = _prepare_test_set(args.num_images)
    if args.limit is not None:
        test_images = test_images[: args.limit]
        test_labels = test_labels[: args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    a_params = {p.name: p for p in full.parameters if p.name.startswith("a")}
    t_params = {p.name: p for p in full.parameters if p.name.startswith("θ")}

    manifest = []
    for i, (image, label) in enumerate(zip(test_images, test_labels)):
        binding = {}
        for name, param in a_params.items():
            binding[param] = float(image[_parameter_index(name)])
        for name, param in t_params.items():
            binding[param] = float(weights[_parameter_index(name)])
        bound = full.assign_parameters(binding)

        fname = f"qvc_test_{i:03d}.qasm"
        (out_dir / fname).write_text(
            f"// 10-qubit VQC inference for test image {i} (true label {int(label)}).\n"
            f"// All 30 parameters bound: 10 image angles + 20 trained weights.\n"
            f"// Recover <Z^10> = (1/shots) Σ_b (-1)^|b| counts[b].\n\n"
            + qasm3.dumps(bound)
        )
        manifest.append({"file": fname, "index": i, "true_label": int(label)})

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "index", "true_label"])
        w.writeheader()
        w.writerows(manifest)

    print(f"wrote {len(manifest)} QASM files + {manifest_path.name} to {out_dir}/")


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
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            f"could not load QASM files. You probably need to install the "
            f"OpenQASM 3 importer:\n    pip install qiskit_qasm3_import\n"
            f"Underlying error: {e}"
        )

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    service = QiskitRuntimeService()
    backend = (service.backend(args.backend) if args.backend
               else service.least_busy(operational=True, simulator=False))
    print(f"backend: {backend.name}")

    pm = generate_preset_pass_manager(target=backend.target, optimization_level=3)
    transpiled = [pm.run(qc) for qc in circuits]

    sampler = SamplerV2(mode=backend, options={"default_shots": args.shots})
    print(f"submitting {len(transpiled)} circuits as one Job...")
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
# `evaluate`: compute <Z^10> -> accuracy
# ----------------------------------------------------------------------------
def cmd_evaluate(args: argparse.Namespace) -> None:
    out_dir = Path(args.dir)
    manifest_path = out_dir / "manifest.csv"
    counts_dir = out_dir / "counts"
    if not manifest_path.exists():
        raise SystemExit(f"missing manifest: {manifest_path}")
    if not counts_dir.exists():
        raise SystemExit(f"missing counts directory: {counts_dir}")

    rows = list(csv.DictReader(manifest_path.open()))
    predictions = []
    correct = 0

    for row in rows:
        stem = Path(row["file"]).stem
        path = counts_dir / f"{stem}.json"
        if not path.exists():
            print(f"  skipping {stem} (no counts file)")
            continue
        counts = json.loads(path.read_text())

        total = sum(counts.values())
        if total == 0:
            continue
        ev = 0.0
        for bitstr, c in counts.items():
            parity = bitstr.replace(" ", "").count("1") % 2
            ev += ((-1) ** parity) * c
        ev /= total

        pred = 1 if ev >= 0 else -1
        true = int(row["true_label"])
        predictions.append({"index": int(row["index"]), "ev": ev,
                            "pred": pred, "true": true})
        correct += int(pred == true)

    n = len(predictions)
    if n == 0:
        raise SystemExit("no predictions could be evaluated.")
    print(f"accuracy = {correct}/{n} = {100 * correct / n:.2f}%")
    mean_abs_ev = float(np.mean([abs(p["ev"]) for p in predictions]))
    print(f"mean |<Z^10>| over test images = {mean_abs_ev:.4f}   "
          f"(close to 1 => clean, close to 0 => noise-dominated)")

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["index", "ev", "pred", "true"])
            w.writeheader()
            w.writerows(predictions)
        print(f"per-image predictions -> {args.out_csv}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Pure-OpenQASM IBM deployment of the 10-qubit VQC")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="emit one bound QASM per test image")
    g.add_argument("--weights", default="weights_10q.npy",
                   help="numpy file of trained weights (length 20).")
    g.add_argument("--num-images", type=int, default=NUM_IMAGES,
                   help="size of the original dataset (the test split is then "
                        f"{int(TEST_FRACTION*100)}%% of this).")
    g.add_argument("--limit", type=int, default=None,
                   help="if given, generate only the first N test images.")
    g.add_argument("--out", default="ibm_qasm_jobs",
                   help="directory to write QASM files + manifest.csv into.")
    g.set_defaults(fn=cmd_generate)

    s = sub.add_parser("submit", help="submit the QASMs to IBM via qiskit-ibm-runtime")
    s.add_argument("--dir", default="ibm_qasm_jobs")
    s.add_argument("--backend", default=None,
                   help="IBM backend name (e.g. ibm_brisbane). "
                        "Default: least busy non-simulator.")
    s.add_argument("--shots", type=int, default=1024)
    s.set_defaults(fn=cmd_submit)

    e = sub.add_parser("evaluate", help="accuracy from saved counts")
    e.add_argument("--dir", default="ibm_qasm_jobs")
    e.add_argument("--out-csv", default=None,
                   help="optional CSV: per-image (index, <Z^10>, pred, true).")
    e.set_defaults(fn=cmd_evaluate)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
