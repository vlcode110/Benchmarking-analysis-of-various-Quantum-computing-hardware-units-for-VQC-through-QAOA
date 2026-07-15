"""
QVC / QNN — 10 qubits, runnable on Amazon Braket or Azure Quantum.

Same algorithm as `qvc-qnn-10qubits.ipynb`, consolidated into a single script.
A `--provider` flag switches between:

    simulator   - local Qiskit StatevectorEstimator (default, free, fast)
    braket      - Amazon Braket via qiskit-braket-provider
    azure       - Azure Quantum via azure-quantum[qiskit]
    ibm         - IBM Quantum via qiskit-ibm-runtime

Install only the provider package(s) you actually need:

    pip install "qiskit>=1.2" qiskit-machine-learning scikit-learn scipy matplotlib
    # Amazon Braket
    pip install qiskit-braket-provider amazon-braket-sdk
    # Azure Quantum
    pip install "azure-quantum[qiskit]"
    # IBM Quantum
    pip install qiskit-ibm-runtime

Authentication (set up once outside this script):
    Braket  -> `aws configure`  (AWS access key/secret/region)
    Azure   -> `az login` and set AZURE_QUANTUM_RESOURCE_ID + AZURE_QUANTUM_LOCATION
    IBM     -> QiskitRuntimeService.save_account(token=..., channel="ibm_quantum")

Example:
    python qvc_qnn_10qubits_run.py --provider simulator
    python qvc_qnn_10qubits_run.py --provider braket  --backend SV1
    python qvc_qnn_10qubits_run.py --provider azure   --backend ionq.simulator
    python qvc_qnn_10qubits_run.py --provider ibm     # picks least-busy
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import z_feature_map
from qiskit.quantum_info import SparsePauliOp


# ----------------------------------------------------------------------------
# Problem configuration (matches qvc-qnn-10qubits.ipynb)
# ----------------------------------------------------------------------------
SIZE = 10               # number of qubits / pixels
VERT_SIZE = 2           # one image dimension; must divide SIZE
LINE_SIZE = 2           # length of the line to detect
HOR_SIZE = SIZE // VERT_SIZE   # = 5  ->  images are 2x5

NUM_IMAGES = 200
TEST_FRACTION = 0.3
DATA_SEED = 42
SPLIT_SEED = 246

BATCH_SIZE = 140
NUM_EPOCHS = 1
MAXITER = 100           # COBYLA iterations per batch


# ----------------------------------------------------------------------------
# Data generation (same as the notebook)
# ----------------------------------------------------------------------------
def generate_dataset(
    num_images: int,
    size: int = SIZE,
    vert_size: int = VERT_SIZE,
    line_size: int = LINE_SIZE,
) -> Tuple[list[np.ndarray], list[int]]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    hor_array = np.zeros((size - (line_size - 1) * vert_size, size))
    ver_array = np.zeros((round(size / vert_size) * (vert_size - line_size + 1), size))

    j = 0
    for i in range(0, size - 1):
        if i % (size / vert_size) <= (size / vert_size) - line_size:
            for p in range(0, line_size):
                hor_array[j][i + p] = np.pi / 2
            j += 1

    j = 0
    for i in range(0, round(size / vert_size) * (vert_size - line_size + 1)):
        for p in range(0, line_size):
            ver_array[j][i + p * round(size / vert_size)] = np.pi / 2
        j += 1

    for _ in range(num_images):
        rng = np.random.randint(0, 2)
        if rng == 0:
            labels.append(-1)
            random_image = np.random.randint(0, len(hor_array))
            images.append(np.array(hor_array[random_image]))
        else:
            labels.append(1)
            random_image = np.random.randint(0, len(ver_array))
            images.append(np.array(ver_array[random_image]))

        for i in range(size):
            if images[-1][i] == 0:
                images[-1][i] = np.random.rand() * np.pi / 4
    return images, labels


# ----------------------------------------------------------------------------
# Circuit construction
# ----------------------------------------------------------------------------
def build_full_circuit(num_qubits: int = SIZE):
    """Return (full_circuit, ansatz, observable) for the 10-qubit QVC."""
    feature_map = z_feature_map(num_qubits, parameter_prefix="a")

    ansatz = QuantumCircuit(num_qubits)
    weights = ParameterVector("θ", length=2 * num_qubits)

    for i in range(num_qubits):
        ansatz.ry(weights[i], i)

    # CNOTs covering both rows of the 2x5 image grid
    cnot_list = [
        [0, 1], [1, 2], [2, 3], [3, 4],     # top row
        [5, 6], [6, 7], [7, 8], [8, 9],     # bottom row
    ]
    for a, b in cnot_list:
        ansatz.cx(a, b)

    for i in range(num_qubits):
        ansatz.rx(weights[num_qubits + i], i)

    full = QuantumCircuit(num_qubits)
    full.compose(feature_map, range(num_qubits), inplace=True)
    full.compose(ansatz, range(num_qubits), inplace=True)

    observable = SparsePauliOp.from_list([("Z" * num_qubits, 1)])
    return full, ansatz, observable


# ----------------------------------------------------------------------------
# Provider / Estimator selection
# ----------------------------------------------------------------------------
def get_estimator(provider: str, backend_name: str | None, shots: int):
    """
    Return a tuple (estimator, backend, transpile_fn).

    transpile_fn(circuit, observable) -> (circuit_for_run, observable_for_run)
    is provider-specific so we run the appropriate transpile pass.
    """
    provider = provider.lower()

    # ---------------- Simulator -----------------
    if provider == "simulator":
        from qiskit.primitives import StatevectorEstimator
        estimator = StatevectorEstimator()
        return estimator, None, lambda qc, obs: (qc, obs)

    # ---------------- Amazon Braket -----------------
    if provider == "braket":
        # See: https://github.com/qiskit-community/qiskit-braket-provider
        from qiskit.primitives import BackendEstimatorV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_braket_provider import BraketProvider, BraketLocalBackend

        if backend_name is None or backend_name.lower() == "local":
            backend = BraketLocalBackend()  # local simulator, no AWS calls
        else:
            # e.g. "SV1", "DM1", "TN1", "Aria-1", "Forte 1", "Garnet", ...
            backend = BraketProvider().get_backend(backend_name)

        pm = generate_preset_pass_manager(target=backend.target, optimization_level=1)
        estimator = BackendEstimatorV2(backend=backend, options={"default_shots": shots})

        def _transpile(qc, obs):
            qc_t = pm.run(qc)
            obs_t = obs.apply_layout(qc_t.layout)
            return qc_t, obs_t

        return estimator, backend, _transpile

    # ---------------- Azure Quantum -----------------
    if provider == "azure":
        # See: https://learn.microsoft.com/azure/quantum/quickstart-microsoft-qiskit
        from azure.quantum.qiskit import AzureQuantumProvider
        from qiskit.primitives import BackendEstimatorV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        resource_id = os.environ.get("AZURE_QUANTUM_RESOURCE_ID")
        location = os.environ.get("AZURE_QUANTUM_LOCATION")
        if not resource_id or not location:
            raise RuntimeError(
                "Set AZURE_QUANTUM_RESOURCE_ID and AZURE_QUANTUM_LOCATION env vars "
                "for Azure Quantum (see Azure portal -> your workspace)."
            )
        az_provider = AzureQuantumProvider(resource_id=resource_id, location=location)
        backend = az_provider.get_backend(backend_name or "ionq.simulator")

        pm = generate_preset_pass_manager(target=backend.target, optimization_level=1)
        estimator = BackendEstimatorV2(backend=backend, options={"default_shots": shots})

        def _transpile(qc, obs):
            qc_t = pm.run(qc)
            obs_t = obs.apply_layout(qc_t.layout)
            return qc_t, obs_t

        return estimator, backend, _transpile

    # ---------------- IBM Quantum (kept for parity with the notebook) -----------------
    if provider == "ibm":
        from qiskit_ibm_runtime import EstimatorV2 as IBMEstimator, QiskitRuntimeService
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        service = QiskitRuntimeService()
        backend = (service.backend(backend_name) if backend_name
                   else service.least_busy(operational=True, simulator=False))
        pm = generate_preset_pass_manager(target=backend.target, optimization_level=3)
        estimator = IBMEstimator(mode=backend, options={"default_shots": shots})

        def _transpile(qc, obs):
            qc_t = pm.run(qc)
            obs_t = obs.apply_layout(qc_t.layout)
            return qc_t, obs_t

        return estimator, backend, _transpile

    raise ValueError(f"Unknown provider: {provider}")


# ----------------------------------------------------------------------------
# Forward pass + loss
# ----------------------------------------------------------------------------
def forward(circuit, input_params, weight_params, estimator, observable) -> np.ndarray:
    num_samples = input_params.shape[0]
    weights = np.broadcast_to(weight_params, (num_samples, len(weight_params)))
    params = np.concatenate((input_params, weights), axis=1)
    pub = (circuit, observable, params)
    job = estimator.run([pub])
    return job.result()[0].data.evs


def mse_loss(predict: np.ndarray, target: np.ndarray) -> float:
    if len(predict.shape) > 1:
        raise AssertionError("input should be 1d-array")
    return float(((predict - target) ** 2).mean())


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="10-qubit QVC on Braket / Azure / IBM / sim")
    parser.add_argument(
        "--provider",
        default="simulator",
        choices=["simulator", "braket", "azure", "ibm"],
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="Backend name within the chosen provider "
             "(e.g. SV1 for Braket, ionq.simulator for Azure).",
    )
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--maxiter", type=int, default=MAXITER)
    parser.add_argument("--num-images", type=int, default=NUM_IMAGES)
    parser.add_argument(
        "--save-weights",
        default="weights_10q.npy",
        help="Where to save the trained weight vector (.npy).",
    )
    args = parser.parse_args()

    # ---- data ----
    np.random.seed(DATA_SEED)
    images, labels = generate_dataset(args.num_images)
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=TEST_FRACTION, random_state=SPLIT_SEED
    )

    # ---- circuit ----
    full_circuit, ansatz, observable = build_full_circuit(SIZE)
    print(f"Circuit: {full_circuit.num_qubits} qubits, "
          f"depth (pre-transpile) = {full_circuit.decompose().depth()}, "
          f"2q-depth = {full_circuit.decompose().depth(lambda i: len(i.qubits) > 1)}, "
          f"params = {len(full_circuit.parameters)}")

    # ---- backend / estimator ----
    estimator, backend, transpile_fn = get_estimator(
        args.provider, args.backend, args.shots
    )
    if backend is not None:
        print(f"Using backend: {backend.name}")

    run_circuit, run_observable = transpile_fn(full_circuit, observable)
    print(f"Run circuit: depth = {run_circuit.decompose().depth()}, "
          f"2q-depth = {run_circuit.decompose().depth(lambda i: len(i.qubits) > 1)}")

    # ---- training loop ----
    np.random.seed(DATA_SEED)
    weight_params = np.random.rand(len(ansatz.parameters)) * 2 * np.pi

    objective_func_vals: list[float] = []

    def cost_fn(w: np.ndarray) -> float:
        preds = forward(run_circuit, input_params, w, estimator, run_observable)
        c = mse_loss(preds, target_arr)
        objective_func_vals.append(c)
        if cost_fn.iter % 10 == 0:
            print(f"  iter {cost_fn.iter:4d}  loss = {c:.5f}")
        cost_fn.iter += 1
        return c

    cost_fn.iter = 0  # type: ignore[attr-defined]

    num_samples = len(train_images)
    t0 = time.time()
    for epoch in range(NUM_EPOCHS):
        for b in range((num_samples - 1) // BATCH_SIZE + 1):
            print(f"epoch {epoch}, batch {b}")
            start_i = b * BATCH_SIZE
            end_i = start_i + BATCH_SIZE
            input_params = np.array(train_images[start_i:end_i])
            target_arr = np.array(train_labels[start_i:end_i])
            cost_fn.iter = 0
            res = minimize(
                cost_fn, weight_params, method="COBYLA",
                options={"maxiter": args.maxiter},
            )
            weight_params = res["x"]
    print(f"training time: {time.time() - t0:.1f}s")

    np.save(args.save_weights, weight_params)
    print(f"saved weights -> {args.save_weights}")

    # ---- evaluation ----
    def predict(images_arr: np.ndarray) -> np.ndarray:
        preds = forward(run_circuit, images_arr, weight_params, estimator, run_observable)
        return np.where(preds >= 0, 1, -1)

    train_pred = predict(np.array(train_images))
    test_pred = predict(np.array(test_images))
    print(f"Train accuracy: {accuracy_score(train_labels, train_pred) * 100:.2f}%")
    print(f"Test  accuracy: {accuracy_score(test_labels, test_pred) * 100:.2f}%")


if __name__ == "__main__":
    main()
