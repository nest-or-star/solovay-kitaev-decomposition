import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
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


# Funkcija, kas izveido H+T vārtu kopu 1 kubita dekompozīcijai.
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


# Funkcija, kas izveido īsas 1 kubita ķēdes līdz noteiktam garumam, izmantojot doto vārtu kopu.
# Katrs elements ir kortežs:
# - QuantumCircuit objekts, tiek izmantots dekompozīcijas saglabāšanai;
# - matricas forma (numpy array), tiek izmantota dekompozīcijas aprēķināšanai.
def generate_short_circuits(gate_set: list, max_length: int) -> list[tuple[QuantumCircuit, np.ndarray]]:
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

    for (i) in range(max_length):
        short_circuits.extend(short_circuits_by_len[i])

    return short_circuits


# Funkcija, kas ielādē īsās ķēdes no faila vai ģenerē jaunas:
# pārbauda, vai eksistē saglabātie īso ķēžu dati, un tos ielādē no faila.
# Ja neeksistē, ģenerē jaunus datus un saglabā tos failā.
def load_short_circuits(gate_set: list[QuantumCircuit], max_length: int) -> list[tuple[QuantumCircuit, np.ndarray]]:
    filename = ""
    for (gate) in gate_set:
        filename += gate.name + "_"
    filename += f"mxl_{max_length}.pkl"

    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            short_circuits = pickle.load(f)
    else:
        short_circuits = generate_short_circuits(gate_set, max_length)
        with open(filename, 'wb') as f:
            pickle.dump(short_circuits, f)

    return short_circuits


# Funkcija, kas veic rotāciju dekompozīciju 1 kubitam
# U = exp(i alpha) Rz(beta) H Rz(gamma) H Rz(delta)
def rotation_decomposition(U_target: np.ndarray) -> tuple[QuantumCircuit, np.float64]:

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

    qc.global_phase = alpha

    return (qc, compare_su2(U_target, Operator(qc).data))


# Funkcija, kas implementē Soloveja-Kitajeva dekompozīcijas algoritmu 1 kubitam
def solovay_kitaev_decomposition(U_target: np.ndarray, depth: int, short_circuits: list[tuple[QuantumCircuit, np.ndarray]], progress) -> tuple[QuantumCircuit, np.float64, list[QuantumCircuit]]:

    if (depth == 0):
        qc, error = base_approximation(U_target, short_circuits)
        progress[2] += 1
        progress[0].progress(progress[2] / progress[1])
        return (qc, error, [qc.copy()])
    
    qc, _, history = solovay_kitaev_decomposition(U_target, depth-1, short_circuits, progress)

    # Ja operatori it faktiski vienādi, atgriež to, kas ir
    if (compare_su2(U_target, Operator(qc).data) < 1e-10):
        return (qc, np.float64(0.), history)

    U_approx = Operator(qc).data
    A, B = gc_decomposition(U_target @ U_approx.conj().T)

    qc_A, _, _ = solovay_kitaev_decomposition(A, depth-1, short_circuits, progress)
    qc_B, _, _ = solovay_kitaev_decomposition(B, depth-1, short_circuits, progress)
    qc_A_inv = qc_A.inverse()
    qc_B_inv = qc_B.inverse()

    # Papildina ķēdi ar komutatoru
    qc.compose(qc_B_inv, [0], inplace=True)
    qc.compose(qc_A_inv, [0], inplace=True)
    qc.compose(qc_B, [0], inplace=True)
    qc.compose(qc_A, [0], inplace=True)

    history.append(qc.copy())
    progress[2] += 1
    progress[0].progress(progress[2] / progress[1])

    return (qc, compare_su2(U_target, Operator(qc).data), history)

# Funkcija, kas implementē Soloveja-Kitajeva dekompozīcijas reverso algoritmu 1 kubitam,
# lai sasniegtu dotu precizitāti epsilon
def solovay_kitaev_reverse(U_target: np.ndarray, qc: QuantumCircuit, epsilon: float, short_circuits: list[tuple[QuantumCircuit, np.ndarray]], progress, depth: int = 0, max_depth: int = 7) -> tuple[QuantumCircuit, np.float64, list[QuantumCircuit]]:

    U = Operator(qc).data
    if compare_su2(U_target, U) < epsilon:
        return (qc, compare_su2(U_target, Operator(qc).data), [qc.copy()])

    if depth >= max_depth:
        return qc, compare_su2(U_target, Operator(qc).data), [qc.copy()]

    A, B = gc_decomposition(U_target @ U.conj().T)

    qc_A, _, _ = solovay_kitaev_decomposition(A, depth, short_circuits, progress)
    qc_B, _, _ = solovay_kitaev_decomposition(B, depth, short_circuits, progress)
    qc_A_inv = qc_A.inverse()
    qc_B_inv = qc_B.inverse()

    qc_historic = qc.copy()

    # Papildina ķēdi ar komutatoru
    qc.compose(qc_B_inv, [0], inplace=True)
    qc.compose(qc_A_inv, [0], inplace=True)
    qc.compose(qc_B, [0], inplace=True)
    qc.compose(qc_A, [0], inplace=True)

    progress[2] += 1
    progress[0].progress(progress[2] / progress[1])

    result = solovay_kitaev_reverse(U_target, qc, epsilon, short_circuits, progress, depth + 1, max_depth)
    result[2].insert(0, qc_historic)

    return result

# Funkcija, kas atrod īsāko ķēdi no dotā saraksta, kas vislabāk aproksimē doto operatoru
def base_approximation(U_target: np.ndarray, short_circuits: list[tuple[QuantumCircuit, np.ndarray]]) -> tuple[QuantumCircuit, np.float64]:
    min_error = np.float64('inf')
    best_circuit = short_circuits[0][0].copy()

    for (entry) in short_circuits:
        error = compare_su2(U_target, entry[1])
        if (error < min_error):
            min_error = error
            best_circuit = entry[0].copy()

    return best_circuit, min_error # atgriež QuantumCircuit un kļūdu


# Funkcija, kas nosaka divus tādus unitārus operatorus A un B, ka U = ABA⁺B⁺,
# izmantojot metodi no informācijas avota
def gc_decomposition(U_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

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
    S = solve_unitary_conjugate(U, Com)
    S, _ = remove_global_phase(S)

    A = S @ V @ S.conj().T
    B = S @ W @ S.conj().T
    
    return (A, B)

# Funkcija, kas atrisina V = S W S⁺
# atrod atbilstošo S, ja ir zināmi V un W
def solve_unitary_conjugate(V: np.ndarray, W: np.ndarray) -> np.ndarray:
    eigvals_V, S_V = np.linalg.eig(V)
    eigvals_W, S_W = np.linalg.eig(W)

    # īpašvērtībām jāskarīt, bet tās var būt dažādā secībā
    if not np.allclose(eigvals_V[0], eigvals_W[0]):
        eigvals_W = eigvals_W[::-1]
        S_W = S_W[:, ::-1]
    if not np.allclose(eigvals_V, eigvals_W, atol=1e-10):
        raise ValueError("Īpašvērtības nesakrīt.")
    
    S = S_V @ S_W.conj().T
    return S


# Funkcija, kas izvelk rotācijas asi un leņķi no SU(2) operatora
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


# Funkcija, kas izveido SU(2) rotācijas matricu
def rotation_matrix(axis: np.ndarray, theta: float | np.float64) -> np.ndarray:
    return expm(-1j * theta / 2 * (axis[0] * X + axis[1] * Y + axis[2] * Z))


# Funkcija, kas sadala unitāro operatoru
# globālajā fāzē un SU(2) daļā
def remove_global_phase(U: np.ndarray) -> tuple[np.ndarray, np.float64]:
    phase = np.angle(np.linalg.det(U)) / 2
    V = U / np.exp(1j * phase)
    W, _ = polar(V)
    if np.linalg.det(W) < 0:
        W = -W
        phase += np.pi
    return W, phase


# Funkcija, kas pievieno globālo fāzi
def add_global_phase(U, phase):
    return U * np.exp(1j * phase)


# Funkcija, kas salīdzina divus operatorus up to globālās fāzes
def compare_su2(U1: np.ndarray, U2: np.ndarray) -> np.float64:
    U1, _ = remove_global_phase(U1)
    U2, _ = remove_global_phase(U2)
    return min(np.linalg.norm(U1 - U2, 2), np.linalg.norm(U1 + U2, 2))

# Funkcija, kas aprēķina nepieciešamo Soloveja-Kitaeva dekompozīcijas dziļumu,
# lai sasniegtu dotu precizitāti, pēc formulas no avota.
# Neder biežāk lietojamiem gadījumiem. NETIEK LIETOTA
def approximate_depth(U_target: np.ndarray, target_error: float, short_circuits: list[tuple[QuantumCircuit, np.ndarray]]) -> int | None:

    U, _ = remove_global_phase(U_target)

    qc_approx = base_approximation(U, short_circuits, None)
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


# Funkcija, kas pārbauda, vai dotā matrica ir unitāra
def is_unitary(U: np.ndarray, tol: float = 1e-10) -> bool:
    return np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=tol)


# Funkcija, kas pielāgo U matricas globālo fāzi tā,
# lai tā sakristu ar target matricas globālo fāzi
def align_phase(U: np.ndarray, target: np.ndarray) -> np.ndarray:
    target_phase = np.angle(np.linalg.det(target)) / 2
    V, _ = remove_global_phase(U)
    return add_global_phase(V, target_phase)


# Testa bloks
if __name__ == "__main__":
    V = X
    W = Z
    print(solve_unitary_conjugate(V, W))