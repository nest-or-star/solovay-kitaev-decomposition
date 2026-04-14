import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.linalg import expm, logm, polar
import streamlit as st

import os
import pickle

# CONSTANTS
# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)
PAULI_BASE = [X, Y, Z]


# Function
# Initiates the H+T gate set to make the base approximation list
def ht_gate_set() -> list[QuantumCircuit]:
    gate_set = []

    gate_set.append(QuantumCircuit(1))
    gate_set[-1].h(0)
    gate_set[-1].name = "H"
    gate_set.append(QuantumCircuit(1))
    gate_set[-1].t(0)
    gate_set[-1].name = "T"
    gate_set.append(QuantumCircuit(1))
    gate_set[-1].tdg(0)
    gate_set[-1].name = "-T"

    return gate_set


# Function
# Generates short 1 qubit circuits up to a defined length from the given gate set
# Each element is a tuple:
# - QuantumCircuit - to build the decomposition
# - matrix form (numpy array) - to perform calculations
def generate_base_circuits(gate_set: list, max_length: int) -> list[tuple[QuantumCircuit, np.ndarray]]:
    base_circuits = []
    groups_by_len = [[] for (_) in range(max_length)]

    # 1 gate circuits
    for gate_qc in gate_set:
        u = Operator(gate_qc).data
        groups_by_len[0].append((gate_qc, u))

    # 2-Max gate circuits
    for i in range(1, max_length):
        # Builds on the previous level
        for (qc, _) in groups_by_len[i-1].copy():
            for gate_qc in gate_set:
                name_list = qc.name.split(' ')

                # H @ H = I
                if name_list[-1] == gate_qc.name == 'H':
                    continue

                # V @ V.inv = I
                # Inverse matrix names begin with '-'
                if name_list[-1] == gate_qc.name[1:] or name_list[-1][1:] == gate_qc.name:
                    continue

                # No more than 4 T or T.inv in a row
                # since the more would have a shorter form if using inverses
                if gate_qc.name in ('T', '-T') and len(name_list) >= 4:
                    if gate_qc.name == name_list[-1] == name_list[-2] == name_list[-3] == name_list[-4]:
                        continue
                
                new_qc = qc.compose(gate_qc, [0], inplace=False)
                new_qc.name = qc.name + ' ' + gate_qc.name
                new_u = Operator(new_qc).data

                groups_by_len[i].append((new_qc, new_u))

    for i in range(max_length):
        base_circuits.extend(groups_by_len[i])

    return base_circuits


# Function
# Loads base circuits from file or generates new
def load_base_circuits(gate_set: list[QuantumCircuit], max_length: int) -> list[tuple[QuantumCircuit, np.ndarray]]:
    filename = ""
    for (gate) in gate_set:
        filename += gate.name + "_"
    filename += f"mxl_{max_length}.pkl"
    file_path = os.path.join("pickles", filename)

    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            base_circuits = pickle.load(f)
    else:
        base_circuits = generate_base_circuits(gate_set, max_length)
        with open(file_path, 'wb') as f:
            pickle.dump(base_circuits, f)

    return base_circuits


# Function
# 1 qubit rotation decomposition:
# U = exp(i alpha) Rz(beta) H Rz(gamma) H Rz(delta)
def rotation_decomposition(u_target: np.ndarray) -> tuple[QuantumCircuit, np.float64]:

    r, alpha = remove_global_phase(u_target)

    gamma = 2 * np.arccos(abs(r[0, 0]))

    main_d = np.angle(r[1, 1]) - np.angle(r[0, 0])
    off_d = np.angle(r[1, 0]) - np.angle(r[0, 1])
    beta = (main_d + off_d) / 2
    delta = (main_d - off_d) / 2

    qc = QuantumCircuit(1)
    if delta != 0:
        qc.rz(delta, 0)
    qc.h(0)
    if gamma != 0:
        qc.rz(gamma, 0)
    qc.h(0)
    if beta != 0:
        qc.rz(beta, 0)

    qc.global_phase = alpha

    return qc, compare_su2(u_target, Operator(qc).data)


# Function
# Solovay-Kitaev algorithm for 1 qubit
def solovay_kitaev_decomposition(u_target: np.ndarray,
                                 depth: int,
                                 base_circuits: list[tuple[QuantumCircuit, np.ndarray]],
                                 progress
                                 ) -> tuple[QuantumCircuit, np.float64, list[QuantumCircuit]]:

    if depth == 0:
        qc, error = base_approximation(u_target, base_circuits)
        progress[2] += 1
        progress[0].progress(progress[2] / progress[1])
        return qc, error, [qc.copy()]
    
    qc, _, history = solovay_kitaev_decomposition(u_target, depth - 1, base_circuits, progress)

    # If operators are basically the same
    if compare_su2(u_target, Operator(qc).data) < 1e-10:
        return qc, np.float64(0.), history

    u_approx = Operator(qc).data
    v, w = balanced_group_commutator(u_target @ u_approx.conj().T)

    qc_v, _, _ = solovay_kitaev_decomposition(v, depth - 1, base_circuits, progress)
    qc_w, _, _ = solovay_kitaev_decomposition(w, depth - 1, base_circuits, progress)

    # Extends the circuit with the BCG
    qc.compose(qc_w.inverse(), [0], inplace=True)
    qc.compose(qc_v.inverse(), [0], inplace=True)
    qc.compose(qc_w, [0], inplace=True)
    qc.compose(qc_v, [0], inplace=True)

    # For review
    history.append(qc.copy())
    progress[2] += 1
    progress[0].progress(progress[2] / progress[1])

    return qc, compare_su2(u_target, Operator(qc).data), history


# Function
# Reverses the first recursion branch of Solovay-Kitaev algorithm
# to increase the recursion level until the given precision
def solovay_kitaev_reverse(u_target: np.ndarray,
                           qc: QuantumCircuit,
                           epsilon: float,
                           base_circuits: list[tuple[QuantumCircuit, np.ndarray]],
                           progress,
                           depth: int = 0,
                           max_depth: int = 7
                           ) -> tuple[QuantumCircuit, np.float64, list[QuantumCircuit]]:

    u = Operator(qc).data
    if compare_su2(u_target, u) < epsilon:
        return qc, compare_su2(u_target, Operator(qc).data), [qc.copy()]

    if depth >= max_depth:
        return qc, compare_su2(u_target, Operator(qc).data), [qc.copy()]

    v, w = balanced_group_commutator(u_target @ u.conj().T)

    qc_v, _, _ = solovay_kitaev_decomposition(v, depth, base_circuits, progress)
    qc_w, _, _ = solovay_kitaev_decomposition(w, depth, base_circuits, progress)

    qc_historic = qc.copy()

    # Extends the circuit with BCG
    qc.compose(qc_w.inverse(), [0], inplace=True)
    qc.compose(qc_v.inverse(), [0], inplace=True)
    qc.compose(qc_w, [0], inplace=True)
    qc.compose(qc_v, [0], inplace=True)

    progress[2] += 1
    progress[0].progress(progress[2] / progress[1])

    result = solovay_kitaev_reverse(u_target, qc, epsilon, base_circuits, progress, depth + 1, max_depth)
    result[2].insert(0, qc_historic)

    return result

# Function
# Finds the best approximation from a pre-made list of base circuits
def base_approximation(u_target: np.ndarray, base_circuits: list[tuple[QuantumCircuit, np.ndarray]]) -> tuple[QuantumCircuit, np.float64]:
    min_error = np.float64('inf')
    best_circuit = base_circuits[0][0].copy()

    for entry in base_circuits:
        error = compare_su2(u_target, entry[1])
        if error < min_error:
            min_error = error
            best_circuit = entry[0].copy()

    return best_circuit, min_error # QuantumCircuit and error


# Function
# Finds A and B such that U = ABA⁺B⁺,
# Method described in Dawson & Nielsen paper
def balanced_group_commutator(u_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    # Remove global phase
    u, _ = remove_global_phase(u_target)

    # Extract rotation angle
    _, u_th = extract_axis_angle(u)

    # Builds rotations V and W
    # U = S(VWV⁺W⁺)S⁺
    phi = 2 * np.arcsin(((1 - np.cos(u_th / 2)) / 2) ** 0.25)

    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])

    v = rotation_matrix(x_axis, phi)
    w = rotation_matrix(y_axis, phi)
    com = v @ w @ v.conj().T @ w.conj().T

    # Solves S(VWV⁺W⁺)S⁺ = U
    s = solve_unitary_conjugate(u, com)
    s, _ = remove_global_phase(s)

    v_sol = s @ v @ s.conj().T
    w_sol = s @ w @ s.conj().T
    
    return v_sol, w_sol

# Function
# Solves V = S W S⁺ for S
def solve_unitary_conjugate(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    eigvals_v, s_v = np.linalg.eig(v)
    eigvals_w, s_w = np.linalg.eig(w)

    # Eigenvalues should match, but their order may be different
    if not np.allclose(eigvals_v[0], eigvals_w[0]):
        eigvals_w = eigvals_w[::-1]
        s_w = s_w[:, ::-1]
    if not np.allclose(eigvals_v, eigvals_w, atol=1e-10):
        raise ValueError("Eigenvalues do not match")
    
    s = s_v @ s_w.conj().T
    return s


# Function
# Extracts rotation axis and angle from an SU(2) gate
def extract_axis_angle(r: np.ndarray) -> tuple[np.ndarray, np.float64]:
    # Checks if SU(2)
    if not np.isclose(np.linalg.det(r), 1, atol=1e-10):
        raise ValueError("Operator not in SU(2)")

    # Computes angle
    trace = np.trace(r)
    theta = np.arccos(np.real(trace) / 2) * 2

    # If angle is negligible
    if np.isclose(theta, 0, atol=1e-12):
        return np.array([1, 0, 0]), np.float64(0.)

    # Computes axis
    a = (r - np.cos(theta / 2) * I) / (-1j * np.sin(theta / 2))
    nx = np.real(a[1, 0])
    ny = np.imag(a[1, 0])
    nz = np.real(a[0, 0])
    axis = np.array([nx, ny, nz]) / np.linalg.norm([nx, ny, nz])

    return axis, theta


# Function
# Builds a rotation from axis and angle
def rotation_matrix(axis: np.ndarray, theta: float | np.float64) -> np.ndarray:
    return expm(-1j * theta / 2 * (axis[0] * X + axis[1] * Y + axis[2] * Z))


# Function
# Self-explanatory?
def remove_global_phase(u: np.ndarray) -> tuple[np.ndarray, np.float64]:
    phase = np.angle(np.linalg.det(u)) / 2
    v = u / np.exp(1j * phase)
    w, _ = polar(v)
    if np.linalg.det(w) < 0:
        w = -w
        phase += np.pi
    return w, phase


# Function
# Self-explanatory?
def add_global_phase(u, phase):
    return u * np.exp(1j * phase)


# Function
# Compares two gates without global phase
def compare_su2(v: np.ndarray, w: np.ndarray) -> np.float64:
    v, _ = remove_global_phase(v)
    w, _ = remove_global_phase(w)
    return min(np.linalg.norm(v - w, 2), np.linalg.norm(v + w, 2))


# UNUSED : not applicable with the starting precision of usable base circuits
# Function
# Finds the necessary recursion depths to achieve the given precision

# def approximate_depth(u_target: np.ndarray, target_error: float, base_circuits: list[tuple[QuantumCircuit, np.ndarray]]) -> int | None:
#
#     u, _ = remove_global_phase(u_target)
#
#     qc_approx = base_approximation(u, base_circuits)
#     u_approx = Operator(qc_approx).data
#
#     error = compare_su2(u, u_approx)
#     c_approx = 4 * np.sqrt(2)
#
#     if error < 1 / 32:
#         n = 0
#         while error > target_error:
#             error = c_approx * error ** 1.5
#             n += 1
#         return n
#
#     return None


# Function
# Self-explanatory?
def is_unitary(u: np.ndarray, tol: float = 1e-10) -> bool:
    return np.allclose(u.conj().T @ u, np.eye(u.shape[0]), atol=tol)


# Function
# Matches the global phase of U to Target
def align_phase(u: np.ndarray, target: np.ndarray) -> np.ndarray:
    target_phase = np.angle(np.linalg.det(target)) / 2
    V, _ = remove_global_phase(u)
    return add_global_phase(V, target_phase)


# Testing
if __name__ == "__main__":
    gate_set = ht_gate_set()
    max_length = 10
    short_circuits = load_base_circuits(gate_set, max_length)