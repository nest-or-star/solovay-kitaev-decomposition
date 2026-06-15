import numpy as np
from sklearn.neighbors import BallTree

import os
import pickle
import logging
from pathlib import Path

from modules.decomposition import balanced_group_commutator
from modules.utils import compare_su2, build_cnot, vectorize_unitary, inverse_circuit, compose_circuit_gate, \
    remove_global_phase


def initialise_gates(n: int, *gate_names: str) -> dict[str, np.ndarray]:
    """
    Returns a dictionary 'gate name: matrix form' for the specified number of qubits and gate names.
    Builds different gates for different qubits, e.g., H.0 is H applied to 0th qubit, H.2 is applied on the 2nd.
    Allowed gate names 'H', 'T', 'Tdg', 'CNOT'.

    :param n: number of gates
    :param gate_names: list of name strings
    :returns: name-matrix form pair dictionary
    """
    gate_dict = {}
    dim = 2 ** n
    eye = np.eye(2, dtype=complex)

    for name in gate_names:
        match name:
            case "H":
                gate = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)
            case "T":
                gate = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
            case "Tdg":
                gate = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)
            # single qubit gate generation follows after the default case

            case _: # for multiqubit gates, specifically CNOT
                if not name == "CNOT":
                    raise ValueError("Unsupported gate name.")

                # CNOT
                else:
                    for i in range(n): # controlling qubit
                        for j in range(n): # controlled qubit
                            # CNOT from self to self pass
                            if i == j:
                                continue
                            gate_dict["CX." + str(i) + "." + str(j)] = build_cnot(n, i, j)
                    break

        for i in range(n):
            m = np.array([1], dtype=complex)
            # "applies" gate to the i-th qubit, identity to others
            for _ in range(i):
                m = np.kron(m, eye)
            m = np.kron(m, gate)
            for _ in range(i + 1, n):
                m = np.kron(m, eye)

            gate_dict[name + "." + str(i)], _ = remove_global_phase(m)

    logging.info(f"Initialised gate set of {len(gate_dict)} matrices.")
    return gate_dict


def generate_basic_circuits(gate_dict: dict[str, np.ndarray], l: int) -> list[tuple[np.ndarray, str]]:
    """
    Generates a list of all possible circuits from the allowed gate set up to the specified length.
    Used to find basic approximations at the 0 recursion level of Solovay-Kitaev.

    The result could be very large and is not suitable for many qubits or large lengths.
    For fast search it should be converted into a BallTree, for example, and used as a reference,
    to look up the full circuit.

    :param gate_dict: see initialise_gates
    :param l: maximum length
    :return: list of pairs of circuit matrix form, and gate name sequence as string
    """
    logging.info("Building basic circuits from scratch...")
    basic_circuits = []
    current_level = []

    # Level 1: single gate_dict gates
    for gate_name, gate_u in gate_dict.items():
        current_level.append((gate_u, gate_name))
    basic_circuits.extend(current_level)

    for i in range(1, l + 1):
        next_level = []
        for u, seq in current_level:
            last_gate = seq.split()[-1] # reminder: gate names are: (C) + single capital letter + (dg) + . + qubit

            for new_gate, gate_u in gate_dict.items():
                is_new, new_seq = compose_circuit_gate(seq, new_gate)

                if is_new:
                    # print(new_seq)
                    next_level.append((gate_u @ u, new_seq))

        logging.info(f"\t\tLevel {i+1}: {len(next_level)} circuits")
        basic_circuits.extend(next_level)
        current_level = next_level

    logging.info("Circuit generation complete!")
    return basic_circuits


def load_basic_circuits(n: int, l: int, *gate_names) -> list[tuple[np.ndarray, str]]:
    """
    Returns a list of basic short circuits for n qubits up to length l of specified gates.
    Allowed gate names 'H', 'T', 'Tdg', 'CNOT'. See function initialise_gates.
    Tries to load from a file first, if exists.
    Otherwise, generates new and saves. See function generate_basic_circuits.

    :param n: number of qubits
    :param l: maximum length of circuits
    :param gate_names: names of gates in the gate set
    :return: list of pairs of circuit matrix form, and gate name sequence as string
    """
    logging.info(f"Loading basic circuits for {n} qubits up to length {l} with gate set {gate_names}...")
    filename = ""
    gates_sorted = sorted(gate_names)
    for gate in gates_sorted:
        filename += gate + "_"
    filename += f"q{n}_mxl{l}.pkl"

    root_dir = Path(__file__).resolve().parent.parent
    pickle_dir = root_dir / "pickles"
    pickle_dir.mkdir(parents=True, exist_ok=True)

    file_path = pickle_dir / filename

    logging.info(f"Looking up file: {file_path}")
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            basic_circuits = pickle.load(f)
    else:
        logging.info("No file found.")
        gate_dict = initialise_gates(n, *gate_names)
        basic_circuits = generate_basic_circuits(gate_dict, l)
        logging.info("Saving...")
        with open(file_path, 'wb') as f:
            pickle.dump(basic_circuits, f)

    logging.info(f"{len(basic_circuits)} basic circuits loaded successfully!\n")
    return basic_circuits


def solovay_kitaev_decomposition(u_target: np.ndarray,
                                 depth: int,
                                 base: list[tuple[np.ndarray, str]],
                                 tree: BallTree,
                                 progress)\
        -> tuple[np.ndarray, str, list[str]]:
    """

    :param u_target:
    :param depth:
    :param base:
    :param tree:
    :param progress:
    :return:
    """

    _, phi = remove_global_phase(u_target)
    logging.info(f"SK Algorithm, recursion level {depth}:")
    logging.info(f"Phase: {phi}")

    if depth == 0:
        u_approx, seq = base_approximation(u_target, base, tree)
        progress[2] += 1
        progress[0].progress(progress[2] / progress[1])
        return u_approx, seq, [seq]
    
    u_approx, seq, history = solovay_kitaev_decomposition(u_target, depth - 1, base, tree, progress)

    # If operators are basically the same
    if compare_su2(u_target, u_approx) < 1e-10:
        return u_approx, seq, history

    v, w = balanced_group_commutator(u_target @ u_approx.conj().T)

    v, seq_v, _ = solovay_kitaev_decomposition(v, depth - 1, base, tree, progress)
    w, seq_w, _ = solovay_kitaev_decomposition(w, depth - 1, base, tree, progress)

    # Extends the circuit with the BCG
    seq = " ".join([seq, inverse_circuit(seq_w), inverse_circuit(seq_v), seq_w, seq_v])
    u_approx, _ = remove_global_phase(v @ w @ v.conj().T @ w.conj().T @ u_approx)

    # For review
    history.append(seq)
    progress[2] += 1
    progress[0].progress(progress[2] / progress[1])

    return u_approx, seq, history


# WIP to refactor for 2+ qubits and new data structures

# def solovay_kitaev_reverse(u_target: np.ndarray,
#                            qc: QuantumCircuit,
#                            epsilon: float,
#                            base_circuits: list[tuple[QuantumCircuit, np.ndarray]],
#                            progress,
#                            depth: int = 0,
#                            max_depth: int = 7
#                            ) -> tuple[QuantumCircuit, np.float64, list[QuantumCircuit]]:
#
#     u = Operator(qc).data
#     if compare_su2(u_target, u) < epsilon:
#         return qc, compare_su2(u_target, Operator(qc).data), [qc.copy()]
#
#     if depth >= max_depth:
#         return qc, compare_su2(u_target, Operator(qc).data), [qc.copy()]
#
#     v, w = balanced_group_commutator(u_target @ u.conj().T)
#
#     qc_v, _, _ = solovay_kitaev_decomposition(v, depth, base_circuits, progress)
#     qc_w, _, _ = solovay_kitaev_decomposition(w, depth, base_circuits, progress)
#
#     qc_historic = qc.copy()
#
#     # Extends the circuit with BCG
#     qc.compose(qc_w.inverse(), [0], inplace=True)
#     qc.compose(qc_v.inverse(), [0], inplace=True)
#     qc.compose(qc_w, [0], inplace=True)
#     qc.compose(qc_v, [0], inplace=True)
#
#     progress[2] += 1
#     progress[0].progress(progress[2] / progress[1])
#
#     result = solovay_kitaev_reverse(u_target, qc, epsilon, base_circuits, progress, depth + 1, max_depth)
#     result[2].insert(0, qc_historic)
#
#     return result


def base_approximation(u_target: np.ndarray,
                       base: list[tuple[np.ndarray, str]],
                       tree: BallTree,
                       k: int = 50)\
        -> tuple[np.ndarray, str]:
    """
    Vectorizes a target unitary and finds the closest match in the Ball Tree.

    :param u_target:
    :param base:
    :param tree:
    :param k:
    :return:
    """
    best_seq = None
    best_u = None
    min_error = float('inf')

    logging.info("BallTree search...")
    target_vec = vectorize_unitary(u_target)
    query_point = target_vec.reshape(1, -1)
    ind = tree.query(query_point, k=k, return_distance=False)

    # The BallTree uses Frobenius norm to find the nearest neighbours,
    # but the closest 2-norm is approximately near the best Frobenius

    for idx in ind[0]:
        candidate_u, candidate_seq = base[idx]

        error = compare_su2(candidate_u, u_target)

        if error < min_error:
            min_error = error
            best_seq = candidate_seq
            best_u = candidate_u

    # logging.info("Linear search...")
    # for candidate_u, candidate_seq in base:
    #     error = compare_su2(candidate_u, u_target)
    #
    #     if error < min_error:
    #         min_error = error
    #         best_seq = candidate_seq
    #         best_u = candidate_u

    logging.info(f"Found best approximation from {k} candidates at distance: {compare_su2(u_target, best_u)} - {best_seq}")
    return best_u, best_seq


# Testing
if __name__ == "__main__":
    base = load_basic_circuits(1, 12, "H", "T", "Tdg")
    # print([x for _, x in base])