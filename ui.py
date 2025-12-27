import streamlit as st
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt
import quantum_decomposition as qd


# Funkcija "galvenā"
# ielādē Streamlit virsrakstu un inicializē lapas komponentes.
def main() -> None:
    st.set_page_config(page_title="Kvantu dekompozīcija", layout="wide")
    st.title("Kvantu operatoru dekompozīcija")

    load_target_unitary()

    load_sidebar()

    load_results()


# Funkcija "mērķa unitārās matricas ielāde"
# inicializē vai ielādē mērķa unitāro matricu no sesijas stāvokļa.
def load_target_unitary() -> None:
    global U_target

    if "U_target" not in st.session_state:
        U_target = np.eye(2, dtype=complex) # Noklusējuma unitārā matrica
        st.session_state.U_target = U_target
    else:
        U_target = st.session_state.U_target


# Funkcija "ielādēt sānu joslu"
# izveido sānu joslu/izvēlni ar dekompozīcijas iestatījumiem,
# kas pielāgojas izvēlētajam režīmam.
def load_sidebar() -> None:
    st.sidebar.header("Iestatījumi")
    mode = st.sidebar.radio("Dekompozīcijas veids", ["Rotācija (H + Rz)", "Soloveja-Kitaeva (H + T)"])

    with st.sidebar.expander("Skatīt unitāro matricu"):
        st.write(U_target)

        if st.button("Rediģēt"):
            input_unitary(1)

    if mode == "Rotācija (H + Rz)":
        if st.sidebar.button("Palaist"):
            launch_rotation_decomposition()
        
    else:
        st.sidebar.subheader("SK Parametri")
        recursion_depth = st.sidebar.number_input("Rekursijas dziļums", min_value=0, max_value=5, value=2)
        max_length = st.sidebar.number_input("Maksimālais sākotnējais garums", min_value=1, max_value=20, value=12)

        if st.sidebar.button("Palaist"):
            launch_solovay_kitaev_decomposition(recursion_depth, max_length)


# Funkcija "palaist rotācijas dekompozīciju"
def launch_rotation_decomposition() -> None:
    qc, phase, precision = qd.rotation_decomposition(U_target)
    qc.global_phase = phase

    st.success("Rotācijas dekompozīcija pabeigta.")
    st.session_state.qc = qc
    st.session_state.precision = precision
    st.session_state.history = []


# Funkcija "palaist Soloveja-Kitaeva dekompozīciju"
def launch_solovay_kitaev_decomposition(recursion_depth: int, max_length: int) -> None:
    with st.status("Soloveja-Kitaeva dekompozīcija...") as status:
        gate_set = qd.create_h_t_gate_set()
        status.write("H+T vārtu kopa izveidota.")
        short_circuits = qd.load_short_circuits(gate_set, max_length, status)
        qc, precision, history = qd.solovay_kitaev_decomposition(U_target, recursion_depth, short_circuits, status)

    st.success("Soloveja-Kitaeva dekompozīcija pabeigta.")
    st.session_state.qc = qc
    st.session_state.precision = precision
    st.session_state.history = history


# Funkcija "Unitārās matricas ievade"
# izveido dialoga logu, kurā lietotājs var ievadīt unitāro operatoru,
# norādot katram elementam reālo un imagināro daļu.
@st.dialog("Unitārās matricas ievade", width="medium")
def input_unitary(num_qubits: int) -> None:
    global U_target
    dim = 2**num_qubits

    for (i) in range(dim):
        row = st.columns(dim)
        for (j, col) in enumerate(row):
            cell = col.container(border=True, horizontal=True, vertical_alignment="center", horizontal_alignment="distribute")

            real_part = cell.number_input(label=f"Reālā daļa U[{i},{j}]", min_value=-1., max_value=1., key=f"real_{i}_{j}",
                                          value=float(U_target[i, j].real), format="%.6f", width=90, label_visibility="collapsed")
            cell.markdown("**+**", width="content")
            imag_part = cell.number_input(label=f"Imaginārā daļa U[{i},{j}]", min_value=-1., max_value=1., key=f"imag_{i}_{j}",
                                          value=float(U_target[i, j].imag), format="%.6f", width=90, label_visibility="collapsed")
            cell.markdown("**i**", width="content")

    if st.button("Apstiprināt"):
        U_temp = np.zeros((dim, dim), dtype=complex)
        for (i) in range(dim):
            for (j) in range(dim):
                real_part = st.session_state[f"real_{i}_{j}"]
                imag_part = st.session_state[f"imag_{i}_{j}"]
                U_temp[i, j] = complex(real_part, imag_part)

        if qd.is_unitary(U_temp):
            U_target = U_temp
            st.session_state.U_target = U_target
            st.rerun()
        else:
            st.error("Ievadītā matrica nav unitāra.")

# Funkcija "ielādēt rezultātus"
def load_results() -> None:
    if 'qc' in st.session_state:
        history = st.session_state.get('history', [])

        if history:
            idx = st.slider("Dekompozīcijas solis", 0, len(history)-1, len(history)-1)
            qc = history[idx]
            precision = qd.compare_su2(st.session_state.U_target, Operator(qc).data)
        else:
            qc = st.session_state.qc
            precision = st.session_state.precision

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Ķēdes diagramma")
            fig_circuit = qc.draw(output='mpl')
            st.pyplot(fig_circuit)

        with col2:
            st.subheader("Bloha sfēras attēlojums")
            state = Statevector(qc)
            fig_bloch = plot_bloch_multivector(state)
            st.pyplot(fig_bloch)

        st.divider()
        st.subheader("Veiktspējas rādītāji")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        m_col1.metric("Vārtu skaits", qc.size())
        m_col2.metric("Ķēdes dziļums", qc.depth())
        m_col3.metric("Kubiti", qc.num_qubits)
        m_col4.metric("Kļūda", f"{precision:.2e}")

        with st.expander("Skatīt unitāro matricu"):
            U = qd.align_phase(Operator(qc).data, U_target)
            st.write(U)
    else:
        st.info("Lūdzu, palaidiet dekompozīciju, izmantojot sānu izvēlni!")


# Galvenais izpildes punkts
if __name__ == "__main__":
    main()