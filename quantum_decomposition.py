import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.visualization import plot_bloch_vector
from scipy.linalg import expm, logm, polar
import streamlit as st

import os
import pickle


# Konstantes
# Pauli matricas 
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)
PAULI_BASE = [X, Y, Z]


# Funkcija "izveidot H+T vārtu kopu"
# izveido H+T vārtu kopu 1 kubita dekompozīcijai.
def create_h_t_gate_set() -> list[QuantumCircuit]:
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


# Funkcija "ģenerēt īsas ķēdes"
# izveido īsas 1 kubita ķēdes līdz noteiktam garumam, izmantojot doto vārtu kopu.
# Katrs elements ir kortežs:
# - QuantumCircuit objekts, tiek izmantots dekompozīcijas saglabāšanai;
# - matricas forma (numpy array), tiek izmantota dekompozīcijas aprēķināšanai.
def generate_short_circuits(gate_set: list, max_length: int, status) -> list[tuple[QuantumCircuit, np.ndarray]]:
    status.write("Ģenerē īsās ķēdes...")

    short_circuits = []
    short_circuits_by_len = [[] for (_) in range(max_length)]

    # Sāk ar īsām ķēdēm ar vienu operatoru
    for (gate_qc) in gate_set:
        U = Operator(gate_qc).data
        short_circuits_by_len[0].append((gate_qc, U))

    # Visas iespējamās kombinācijas līdz max_length garumam
    for (i) in range(1, max_length):
        # Balstās tikai uz iepriekšējo līmeni
        for (qc, _) in short_circuits_by_len[i-1].copy():
            for (gate_qc) in gate_set:
                name_list = qc.name.split(' ')

                # Izvairās no diviem H pēc kārtas (jo H*H=I)
                if (name_list[-1] == gate_qc.name == 'H'):
                    continue

                # Izvairās no diviem pretējiem vārtiem pēc kārtas
                # pretējo vārtu nosaukumā ir prefikss '-'
                if (name_list[-1] == gate_qc.name[1:]) or (name_list[-1][1:] == gate_qc.name):
                    continue

                # Izvairās no vairāk kā četrām T rotācijām pēc kārtas
                if (gate_qc.name in ('T', '-T')) and (len(name_list) >= 4):
                    if (gate_qc.name == name_list[-1] == name_list[-2] == name_list[-3] == name_list[-4]):
                        continue
                
                new_qc = qc.compose(gate_qc, [0], inplace=False)
                new_qc.name = qc.name + ' ' + gate_qc.name
                new_U = Operator(new_qc).data

                short_circuits_by_len[i].append((new_qc, new_U))
        status.write(f"Pievienoja {len(short_circuits_by_len[i])} ķēdes garumā {i+1}.")

    for (i) in range(max_length):
        short_circuits.extend(short_circuits_by_len[i])

    return short_circuits


# Funkcija "īso ķēžu ielādēšana"
# pārbauda, vai eksistē saglabātie īso ķēžu dati, un tos ielādē no faila.
# Ja neeksistē, ģenerē jaunus datus un saglabā tos failā.
def load_short_circuits(gate_set: list[QuantumCircuit], max_length: int, status) -> list[tuple[QuantumCircuit, np.ndarray]]:
    filename = ""
    for (gate) in gate_set:
        filename += gate.name + "_"
    filename += f"mxl_{max_length}.pkl"

    status.write("Meklē īsās ķēdes atmiņā...")

    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            short_circuits = pickle.load(f)
        status.write("Īsās ķēdes ielādētas no faila.")
    else:
        short_circuits = generate_short_circuits(gate_set, max_length, status)
        with open(filename, 'wb') as f:
            pickle.dump(short_circuits, f)
        status.write("Īsās ķēdes uzģenerētas un saglabātas failā.")

    return short_circuits


# Funkcija "rotācijas dekompozīcija"
# izsaka doto unitāro operatoru ar rotācijām ap Z asi un Hadamarda vārtiem.
# Balstās uz fakta, ka jebkuru SU(2) operatoru var izteikt kā
# U = expm(i alpha) Rz(beta) H Rz(gamma) H Rz(delta)
def rotation_decomposition(U_target: np.ndarray) -> tuple[QuantumCircuit, np.float64, np.float64]:

    R, alpha = remove_global_phase(U_target)

    gamma = 2 * np.arccos(abs(R[0, 0]))

    main_d = np.angle(R[1, 1]) - np.angle(R[0, 0])
    off_d = np.angle(R[1, 0]) - np.angle(R[0, 1])
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

    return (qc, alpha, compare_su2(U_target, Operator(qc).data))


# Funkcija "Soloveja-Kitaeva dekompozīcija"
# implementē Soloveja-Kitajeva dekompozīcijas algoritmu 1 kubitam
def solovay_kitaev_decomposition(U_target: np.ndarray, depth: int, short_circuits: list[tuple[QuantumCircuit, np.ndarray]], status) -> tuple[QuantumCircuit, np.float64, list[QuantumCircuit]]:

    status.write(f"Rekursijas dziļums: {depth}...")

    if (depth == 0):
        qc, error = short_circuits_approximation(U_target, short_circuits, status)
        return (qc, error, [qc.copy()])
    
    qc, _, history = solovay_kitaev_decomposition(U_target, depth-1, short_circuits, status)

    # Ja operatori it faktiski vienādi, atgriež to, kas ir
    if (compare_su2(U_target, Operator(qc).data) < 1e-10):
        return (qc, np.float64(0.), history)

    U_approx = Operator(qc).data
    A, B = gc_decomposition(U_target @ U_approx.conj().T, status)

    qc_A, _, _ = solovay_kitaev_decomposition(A, depth-1, short_circuits, status)
    qc_B, _, _ = solovay_kitaev_decomposition(B, depth-1, short_circuits, status)
    qc_A_inv = qc_A.inverse()
    qc_B_inv = qc_B.inverse()

    # Papildina ķēdi ar komutatoru
    qc.compose(qc_B_inv, [0], inplace=True)
    qc.compose(qc_A_inv, [0], inplace=True)
    qc.compose(qc_B, [0], inplace=True)
    qc.compose(qc_A, [0], inplace=True)

    status.write(f"Dziļums {depth} pabeigts.")
    history.append(qc.copy())

    return (qc, compare_su2(U_target, Operator(qc).data), history)


# Funkcija "Īso ķēžu tuvinājums"
# atrod īsāko ķēdi no dotā saraksta, kas vislabāk aproksimē doto operatoru
def short_circuits_approximation(U_target: np.ndarray, short_circuits: list[tuple[QuantumCircuit, np.ndarray]], status) -> tuple[QuantumCircuit, np.float64]:
    min_error = np.float64('inf')
    best_circuit = short_circuits[0][0].copy()

    status.write("Meklē labāko īso ķēdi...")

    for (entry) in short_circuits:
        error = compare_su2(U_target, entry[1])
        if (error < min_error):
            min_error = error
            best_circuit = entry[0].copy()

    status.write(f"Atrada [{best_circuit.name}] ar kļūdu {min_error:.2e}.")

    return best_circuit, min_error # atgriež QuantumCircuit un kļūdu


# Funkcija "grupas komutatora dekompozīcija"
# nosaka divus tādus unitārus operatorus A un B, ka U = ABA⁺B⁺,
# izmantojot metodi no informācijas avota
def gc_decomposition(U_target: np.ndarray, status) -> tuple[np.ndarray, np.ndarray]:
    status.write("Veic grupas komutatora dekompozīciju...")

    # Noņem globālo fāzi
    U, _ = remove_global_phase(U_target)

    # Izvelk rotācijas asi un leņķi
    _, U_th = extract_axis_angle(U)

    # Rotācijas V un W
    # U = S(VWV⁺W⁺)S⁺
    phi = 2 * np.arcsin(((1 - np.cos(U_th / 2)) / 2) ** 0.25)

    X_xs = np.array([1, 0, 0])
    Y_xs = np.array([0, 1, 0])

    V = rotation_matrix(X_xs, phi)
    W = rotation_matrix(Y_xs, phi)
    Com = V @ W @ V.conj().T @ W.conj().T

    # Atrisina vienādojumu S * (VWV⁺W⁺) * S⁺ = U
    K = np.kron(I, Com.T) - np.kron(U, I)
    _, _, Vh = np.linalg.svd(K)
    S_vec = Vh.conj().T[:, -1]
    S_raw = S_vec.reshape((2, 2))

    S, _ = polar(S_raw)
    S, _ = remove_global_phase(S)

    A = S @ V @ S.conj().T
    B = S @ W @ S.conj().T

    error = compare_su2(U_target, A @ B @ A.conj().T @ B.conj().T)
    status.write(f"Komutatora dekompozīcijas kļūda: {error:.2e}.")
    
    return (A, B)


# Funkcija "iegūt asi un leņķi"
# izvelk rotācijas asi un leņķi no dotā SU(2) operatora
def extract_axis_angle(R: np.ndarray) -> tuple[np.ndarray, np.float64]:
    # Pārbauda, vai R pieder SU(2)
    if not np.isclose(np.linalg.det(R), 1, atol=1e-10):
        raise ValueError("Operators nav SU(2) grupa.")

    # Aprēķina leņķi
    trace = np.trace(R)
    theta = np.arccos(np.real(trace) / 2) * 2

    # Ja leņķis ir tuvu nullei, atgriež standarta asi un nulles leņķi
    if np.isclose(theta, 0, atol=1e-12):
        return (np.array([1, 0, 0]), np.float64(0.))

    # Aprēķina rotācijas asi
    A = (R - np.cos(theta / 2) * I) / (-1j * np.sin(theta / 2))
    nx = np.real(A[1, 0])
    ny = np.imag(A[1, 0])
    nz = np.real(A[0, 0])
    axis = np.array([nx, ny, nz]) / np.linalg.norm([nx, ny, nz])

    return (axis, theta)


# Funkcija "rotācijas matrica"
# izveido rotācijas matricu ap doto asi un leņķi
def rotation_matrix(axis: np.ndarray, theta: float | np.float64) -> np.ndarray:
    return expm(-1j * theta / 2 * (axis[0] * X + axis[1] * Y + axis[2] * Z))


# Funkcija "noņemt globālo fāzi"
def remove_global_phase(U: np.ndarray) -> tuple[np.ndarray, np.float64]:
    phase = np.angle(np.linalg.det(U)) / 2
    V = U / np.exp(1j * phase)
    W, _ = polar(V)
    return W, phase


# Funkcija "pievienot globālo fāzi"
def add_global_phase(U, phase):
    return U * np.exp(1j * phase)


# Funkcija "salīdzināt SU(2) operātorus"
# konvertē dotos unitāros operātorus uz SU(2)
# un aprēķina to atšķirību ar operātoru 2-normu
def compare_su2(U1: np.ndarray, U2: np.ndarray) -> np.float64:
    U1, _ = remove_global_phase(U1)
    U2, _ = remove_global_phase(U2)
    return np.linalg.norm(U1 - U2, 2)


# Funkcija "aptuvenais dziļums"
# aprēķina nepieciešamo Soloveja-Kitaeva dekompozīcijas dziļumu,
# lai sasniegtu doto kļūdas robežu, pēc formulas no informācijas avota.
def approximate_depth(U_target: np.ndarray, target_error: float, short_circuits: list[tuple[QuantumCircuit, np.ndarray]]) -> int | None:

    U, _ = remove_global_phase(U_target)

    qc_approx = short_circuits_approximation(U, short_circuits, None)
    U_approx = Operator(qc_approx).data

    error = compare_su2(U, U_approx)
    c_approx = 4 * np.sqrt(2)

    if error < 1 / 32:
        n = 0
        while error > target_error:
            error = c_approx * error ** 1.5
            n += 1
        return n
    
    return None


# Funkcija "pārbaude, vai matrica ir unitāra"
def is_unitary(U: np.ndarray, tol: float = 1e-10) -> bool:
    return np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=tol)