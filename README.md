# Multi-device QML benchmarking

Source code and data for a **10-qubit variational quantum classifier (VQC)** and **linear cross-entropy benchmarking (XEB)** study across IBM, Amazon Braket, and IonQ backends.

## Repository layout

```
├── data/
│   ├── circuits/       # OpenQASM circuits + trained weights
│   ├── ideal/          # Classical reference tables (statevector)
│   ├── processed/      # Summary CSVs used in the paper figures
│   └── raw/            # Device histogram JSON and XEB summaries
├── experiments/        # Run circuits on cloud backends (optional re-runs)
└── analysis/           # Turn raw JSON into CSVs; multi-device comparison plots
```

See [`data/README.md`](data/README.md) for the raw-data folder map.

## Reproduce the paper figures

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook analysis/vqc_multi_device_comparison.ipynb
```

Run from the **repository root** or from `analysis/`. The notebook reads summary CSVs from `data/processed/` and XEB JSON from `data/raw/`.

### Regenerate summary CSVs from raw histograms

Run the matching notebook in `analysis/` for each backend (e.g. `vqc_braket_iqm_emerald_analysis.ipynb`). Each writes a CSV to `data/processed/`.

### Regenerate ideal references (no cloud account needed)

```bash
jupyter notebook analysis/ideal_reference_vqc_from_qasm.ipynb
jupyter notebook analysis/ideal_reference_xeb_from_qasm.ipynb
```

## Devices included

**VQC (10 backends):** IBM Kingston, Fez, Marrakesh · IQM Emerald & Garnet · Rigetti Cepheus · Braket SV1 & TN1 · IonQ simulator & Forte-1.

**Linear XEB (Section 9):** IQM Emerald · Braket TN1 · IonQ simulator · IonQ Forte-1.

Device-to-path mapping is defined in `analysis/vqc_multi_device_comparison.ipynb` (`DEVICE_REGISTRY` and `XEB_DEVICE_REGISTRY`).

## Re-running cloud jobs (optional)

Set credentials via environment variables only — never commit API keys.

| Provider | Env vars | Entry point |
|----------|----------|-------------|
| IBM | `QISKIT_IBM_TOKEN` | `experiments/ibm_qasm_pipeline.py`, `ibm_xeb_qasm_pipeline.py` |
| IonQ | `IONQ_API_KEY` | `experiments/ionq-native-qaoa-and-vqc-from-qasm.ipynb`, `ionq-xeb-from-qasm.ipynb` |
| Braket | AWS credentials | `experiments/braket-benchmarks-from-qasm.ipynb` |

Save new outputs under `data/raw/` using the same folder naming as the existing runs.

## Citation

If you use this dataset or code, please cite the accompanying paper (details TBD).
