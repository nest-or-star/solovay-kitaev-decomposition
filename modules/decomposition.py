import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import scipy.linalg as la

from modules.utils import solve_unitary_conjugate, extract_axis_angle, rotation_matrix, remove_global_phase, compare_su2, PAULI_X, PAULI_Y, PAULI_Z

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
# Finds A and B such that U = ABA⁺B⁺,
# Method described in Dawson & Nielsen paper
def balanced_group_commutator(u_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    # Remove global phase
    u, _ = remove_global_phase(u_target)

    if u.ndim == 2:
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

    d = u_target.shape[0]

    # 1. Step into the Lie algebra using the matrix logarithm
    # U = exp(iH) -> iH = logm(U) -> H = -1j * logm(U)
    iH = la.logm(u_target)
    H = -1j * iH

    # Ensure H is strictly Hermitian and traceless (floating-point cleanup)
    H = 0.5 * (H + np.conj(H).T)
    H -= np.trace(H) / d * np.eye(d)

    # 2. Diagonalize H to solve the commutator in a simplified basis
    # H = M @ D @ M^\dagger
    eigenvalues, M = la.eigh(H)

    # 3. Solve [A_diag, B_diag] = iD
    # We choose A_diag to be a diagonal matrix with a linear spread of values
    # This prevents zero-divisions and structural degeneracy during the B_diag solve.
    a_diag_elements = np.linspace(-1, 1, d)
    A_diag = np.diag(a_diag_elements)

    B_diag = np.zeros((d, d), dtype=complex)
    for i in range(d):
        for j in range(d):
            if i != j:
                # Since [A, B]_ij = B_ij * (A_ii - A_jj) = i * D_ij
                # For a diagonal matrix D, D_ij = 0 when i != j.
                B_diag[i, j] = 0.0
            else:
                # The diagonal elements of [A, B] are always 0.
                # We place the eigenvalue components on the off-diagonals of B_diag safely.
                B_diag[i, (i + 1) % d] = 1j * eigenvalues[i] / (a_diag_elements[i] - a_diag_elements[(i + 1) % d])

    # 4. Enforce the Balancing Constraint
    # Scale A and B so their Frobenius norms are identical, preserving [A, B] = iD
    norm_A = la.norm(A_diag, 'fro')
    norm_B = la.norm(B_diag, 'fro')

    scale_factor = np.sqrt(norm_B / norm_A)
    A_diag_balanced = A_diag * scale_factor
    B_diag_balanced = B_diag / scale_factor

    # 5. Transform back to the original basis
    A = M @ A_diag_balanced @ np.conj(M).T
    b_prime = M @ B_diag_balanced @ np.conj(M).T

    # Ensure structural components remain zero-trace
    A -= np.trace(A) / d * np.eye(d)
    b_prime -= np.trace(b_prime) / d * np.eye(d)

    # 6. Exponentiate back to the Lie Group SU(d)
    V = la.expm(A)
    W = la.expm(b_prime)

    return V, W



# def cartan_kak_decomp(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, k]:
#     pauli_xx = np.kron(PAULI_X, PAULI_X)
#     pauli_yy = np.kron(PAULI_Y, PAULI_Y)
#     pauli_zz = np.kron(PAULI_Z, PAULI_Z)
#
#     u_re = (u + u.conj()) / 2
#     u_im = (u - u.conj()) / 2j
#
#     np.linalg.svd()
