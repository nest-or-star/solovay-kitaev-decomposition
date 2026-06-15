import numpy as np
from qiskit import QuantumCircuit
from scipy.linalg import expm, polar

# CONSTANTS
# Pauli matrices
PAULI_X = np.array([[0, 1], [1, 0]], complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], complex)
PAULI_Z = np.array([[1, 0], [0, -1]], complex)


def solve_unitary_conjugate(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Solves the equation V = S @ W @ Sh.
    Currently, orders eigenvalues as if the unitary is 2x2.

    :param v:
    :param w:
    :return:
    """
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


def extract_axis_angle(r: np.ndarray) -> tuple[np.ndarray, np.float64]:
    """
    Based on the fact that SU(2) gates can be expressed as a rotation matrix about some axis and some angle.

    :param r: SU(2) unitary
    :return: pair of the 3D np.ndarray axis vector and the rotation angle
    """

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
    a = (r - np.cos(theta / 2) * np.eye(2, dtype=complex)) / (-1j * np.sin(theta / 2))
    nx = np.real(a[1, 0])
    ny = np.imag(a[1, 0])
    nz = np.real(a[0, 0])
    axis = np.array([nx, ny, nz]) / np.linalg.norm([nx, ny, nz])

    return axis, theta


def rotation_matrix(axis: np.ndarray, theta: float | np.float64) -> np.ndarray:
    """
    Based on the fact that SU(2) gates can be expressed as a rotation matrix about some axis and some angle.

    :param axis: 3D np.ndarray axis vector
    :param theta: rotation angle in radians
    :return: SU(2) rotation operator
    """
    return expm(-1j * theta / 2 * (axis[0] * PAULI_X + axis[1] * PAULI_Y + axis[2] * PAULI_Z))


def remove_global_phase(u: np.ndarray) -> tuple[np.ndarray, np.float64]:
    """
    :param u:
    :return:
    """
    phase = np.angle(np.linalg.det(u)) / 2
    v = u / np.exp(1j * phase)
    w, _ = polar(v)
    if np.linalg.det(w) < 0:
        w = -w
        phase += np.pi
    return w, phase


def add_global_phase(u, phase):
    return u * np.exp(1j * phase)


def compare_su2(v: np.ndarray, w: np.ndarray) -> np.float64:
    v, _ = remove_global_phase(v)
    w, _ = remove_global_phase(w)
    return min(np.linalg.norm(v - w, 2), np.linalg.norm(v + w, 2))


def is_unitary(u: np.ndarray, tol: float = 1e-10) -> bool:
    return np.allclose(u.conj().T @ u, np.eye(u.shape[0]), atol=tol)


def align_phase(u: np.ndarray, target: np.ndarray) -> np.ndarray:
    target_phase = np.angle(np.linalg.det(target)) / 2
    v, _ = remove_global_phase(u)
    return add_global_phase(v, target_phase)


def eckart_young(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u_a, a_derived, vh_a = np.linalg.svd(a)
    idx = np.linalg.matrix_rank(a)
    d = a_derived[:idx, :idx]

    b_derived = u_a.conj().T @ b @ vh_a.conj().T

    pass

def kron_factor():
    pass

def vectorize_unitary(matrix):
    """
    Converts a complex matrix into a real-valued vector for spatial tree storage.
    For a 2x2 complex matrix, this returns an 8-dimensional real vector.
    """
    flat = matrix.flatten()
    return np.concatenate((np.real(flat), np.imag(flat)))

def build_cnot(n: int, i: int, j: int) -> np.ndarray:
    """
    Builds the matrix form of the described CNOT.

    Starts from n*n identity and swaps columns where i-th qubit is 1 and j-th qubit is 0/1 and other qubits are the same.
    Let's examine column indexes as binary numbers where k-th digit from the left is the state of the k-th qubit.
    Then i-th qubit is 1 in the indexes that have odd whole parts when divided by 2^(n-i-1),
    and pairs of indexes where all the digits are the same except j-th are exactly 2^(n-j-1) apart.

    :param n: number of qubits
    :param i: controlling qubit
    :param j: controlled qubit
    :return: matrix form
    """
    dim = 2 ** n
    u = np.eye(dim, dtype=complex)

    for k in range(2 ** n):
        if (k // 2 ** (n-i-1)) % 2 == 1 and (k // 2 ** (n-j-1)) % 2 == 0: # i-th binary digit is 1 and j-th is 0
            swap = (k + 2 ** (n-j-1)) # % n might not be needed because indexes with j-th digit is 0 are def smaller than if j-th digit is 1
            u[:, [k, swap]] = u[:, [swap, k]]

    return u


def inverse_circuit(seq: str) -> str:
    """
    Returns the inverse of the given quantum circuit in string format.

    :param seq: quantum circuit
    :return:
    """
    inv = ""
    circuit = seq.split()

    for gate in circuit:
        name, qubit = gate.split('.', 1)
        if not (name == "H" or name == "CX"):
            # Inverse the gate
            if "dg" in name:
                name = name.replace("dg", "")
            else:
                name += "dg"
        # Inverse the order
        inv = name + "." + qubit + " " + inv

    return inv


def str_to_circuit(seq: str, n: int) -> QuantumCircuit:
    """

    :param seq:
    :param n:
    :return:
    """
    circuit = seq.split()
    qc = QuantumCircuit(n)

    for gate in circuit:
        name, qubit = gate.split('.', 1)
        match name:
            case "H":
                qc.h(int(qubit))
            case "T":
                qc.t(int(qubit))
            case "Tdg":
                qc.tdg(int(qubit))
            case "CX":
                control, target = qubit.split('.')
                qc.cx(int(control), int(target))
            case "_":
                raise ValueError("Unknown gate in circuit " + name)

    return qc


def expand_circuit_str(seq: str, n: int) -> list[list[str]]:

    circuit = seq.split()
    regs = [[] for _ in range(n)]

    for gate in circuit:
        name, qubit = gate.split('.', 1)

        if name[0] == "C":
            control, target = qubit.split('.')
            regs[int(control)].append("C." + target)
            regs[int(target)].append(name[1:] + "." + control)

        else:
            regs[int(qubit)].append(name)

    return regs


def compose_circuit_gate(seq: str, gate: str) -> tuple[bool, str]:

    _, qubit = gate.split('.', 1)

    if gate[0] == "C":
        qubit = qubit.split('.')[-1] # target qubit

    gates = list(filter(lambda s : s.endswith('.' + qubit), seq.split())) # filters out only target register
    print(gates, gate)
    if len(gates) > 1:
        # H @ H = X @ X = I
        if gates[-1] == gate and ("H" in gate or "X" in gate):
            return False, seq
        # V @ Vdg = I
        if gates[-1] == gate.replace("dg", "") or gates[-1].replace("dg", "") == gate:
            return False, seq

    if len(gates) > 4:
        # T @ T @ T @ T = Tdg @ Tdg @ Tdg @ Tdg
        if gate == gates[-1] == gates[-2] == gates[-3] == gates[-4] and "T" in gate:
            return False, seq

    return True, seq + " " + gate


# Testing
if __name__ == "__main__":
    print(build_cnot(3, 0, 1))