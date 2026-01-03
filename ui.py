import streamlit as st
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import matplotlib.pyplot as plt
import quantum_decomposition as qd
import json
import os


# Galvenā funkcija, kas inicializē pamatdatus un lietotāja saskarni.
def main() -> None:

    load_target_unitary()
    load_language()

    st.set_page_config(
        page_title=get_text("page_title"),
        layout="wide")
    st.title(get_text("main_title"))

    load_sidebar()
    load_results()


# Funkcija, kas ielādē tulkojumus no JSON failiem
# un kešatmiņā saglabā tos, lai uzlabotu veiktspēju.
# tulkojumi atrodas mapē "locales"
@st.cache_data
def load_translations(language_code):
    file_path = os.path.join("locales", f"{language_code}.json")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
        
    except FileNotFoundError:
        st.error(f"Tulkojumi nav atrasti: {file_path}")
        return {}


# Funkcija, kas iegūst tulkoto tekstu pēc atslēgas
def get_text(key):
    return st.session_state.translations.get(key, f"NAV TULKOJUMA: {key}")


# Funkcija, kas uzstāda noklusējuma unitāro matricu
def load_target_unitary() -> None:
    if "U_target" not in st.session_state:
        st.session_state.U_target = np.eye(2, dtype=complex) # Noklusējuma unitārā matrica


# Funkcija, kas ielādē un pārvalda valodas izvēli
def load_language() -> None:
    # Noklusējuma valoda ir latviešu
    if "lang" not in st.session_state:
        st.session_state["lang"] = "lv"

    # Valodas izvēle sānjoslā
    lang_choice = st.sidebar.selectbox(
        "Valoda / Language",
        ["lv", "en"], 
        index = 0 if st.session_state["lang"] == "lv" else 1)

    # Ja valoda ir mainīta, atjaunina sesijas stāvokli un pārlādē lapu
    if lang_choice != st.session_state["lang"]:
        st.session_state["lang"] = lang_choice
        st.rerun()

    # Saglabā tulkojumus sesijas stāvoklī
    st.session_state.translations = load_translations(st.session_state["lang"])


# Funkcija, kas izveido sānjoslu ar lietotāja ievadi
def load_sidebar() -> None:
    st.sidebar.header(
        get_text("sidebar_header"))
    mode = st.sidebar.radio( # dekompozīcijas režīma izvēle
        get_text("decomp_type"),
        [get_text("rotation_h_rz"), get_text("solovay_kitaev_h_t")])

    # Matricas skatīšana un rediģēšana
    with st.sidebar.expander(get_text("view_unitary")):
        st.write(st.session_state.U_target)

        if st.button(get_text("edit")):
            input_unitary(1)

    # Rotācijas dekompozīcijas režīms
    if mode == get_text("rotation_h_rz"):
        if st.sidebar.button(get_text("run")):
            launch_rotation_decomp()
    
    # Soloveja-Kitaeva dekompozīcijas režīms
    else:
        st.sidebar.subheader(get_text("sk_params"))
        precision_mode = st.sidebar.radio( # precizitātes režīma izvēle
            get_text("precision_mode"),
            [get_text("precision"), get_text("recursion_depth")],
            label_visibility="collapsed")

        if precision_mode == get_text("precision"):
            target_epsilon = st.sidebar.number_input( # precizitātes ievade
                get_text("precision"),
                min_value=0.00001, max_value=2.0,
                value=0.01,
                format="%.5f",
                label_visibility="collapsed")
            recursion_depth = None

        else:
            recursion_depth = st.sidebar.number_input( # rekursijas dziļuma ievade
                get_text("recursion_depth"),
                min_value=0, max_value=5,
                value=2,
                label_visibility="collapsed")
            target_epsilon = None

        max_length = st.sidebar.number_input( # maksimālā pamatķēžu garuma ievade
            get_text("max_length"),
            min_value=1, max_value=20,
            value=12)

        if st.sidebar.button(get_text("run")):
            launch_solovay_kitaev(target_epsilon, recursion_depth, max_length)


# Funkcija, kas pārvalda rotācijas dekompozīcijas palaišanu un rezultātu saglabāšanu
def launch_rotation_decomp() -> None:
    qc, precision = qd.rotation_decomposition(st.session_state.U_target)

    st.success(get_text("success_rotation"))
    # Saglabā rezultātus sesijas stāvoklī
    st.session_state.qc = qc
    st.session_state.precision = precision
    st.session_state.history = []


# Funkcija, kas pārvalda Soloveja-Kitaeva dekompozīcijas palaišanu un rezultātu saglabāšanu
def launch_solovay_kitaev(epsilon: float|None, recursion_depth: int|None, max_length: int) -> None:

    bar = st.progress(
        value=0.0,
        text=get_text("running_sk"))

    gate_set = qd.create_h_t_gate_set()
    bar.progress(0.0, text=get_text("success_h_t"))

    short_circuits = qd.load_short_circuits(gate_set, max_length)
    bar.progress(0.0, text=f"{get_text("loaded")} {len(short_circuits)} {get_text("short_circuits")}") # "Ielādētas X pamatķēdes"

    # Dota precizitāte
    if epsilon is not None:
        progress_info = [bar, sum([3**i for i in range(7)]), 0] # progresa josla + kopējais soļu skaits + sākums
        qc_0, _ = qd.base_approximation(st.session_state.U_target, short_circuits)
        qc, precision, history = qd.solovay_kitaev_reverse(st.session_state.U_target, qc_0, epsilon, short_circuits, progress_info)

    # Dots rekursijas dziļums
    if recursion_depth is not None:
        progress_info = [bar, sum([3**i for i in range(recursion_depth+1)]), 0] # progresa josla + kopējais soļu skaits + sākums
        qc, precision, history = qd.solovay_kitaev_decomposition(st.session_state.U_target, recursion_depth, short_circuits, progress_info)

    st.success(get_text("success_sk"))
    # Saglabā rezultātus sesijas stāvoklī
    st.session_state.qc = qc
    st.session_state.precision = precision
    st.session_state.history = history


# Funkcija, kas parāda rezultātus lietotājam
def load_results() -> None:
    # Ja rezultāti ir pieejami, parāda tos lietotājam
    if 'qc' in st.session_state:
        history = st.session_state.get('history', [])

        if len(history) > 1: # ja ir vēsture, ļauj izvēlēties soli
            idx = st.slider(
                get_text("decomp_step"),
                min_value=0, max_value=len(history)-1,
                value=len(history)-1)
            qc = history[idx]
            precision = qd.compare_su2(st.session_state.U_target, Operator(qc).data)

        else:
            qc = st.session_state.qc
            precision = st.session_state.precision

        _, mid, _ = st.columns([1, 6, 1]) # centrē diagrammu
        with mid:
            # Kvantu ķēde
            st.subheader(get_text("circuit_diagram"))
            fig_circuit = qc.draw(output='mpl')
            st.pyplot(fig_circuit)

            # Precizitātes grafiks
            if len(history) > 1:
                st.divider()
                st.subheader(get_text("precision_change"))
                precisions = [qd.compare_su2(st.session_state.U_target, Operator(circ).data) for circ in history]
                fig, ax = plt.subplots()
                ax.plot(range(len(precisions)), precisions, marker='o')
                ax.set_xlabel(get_text("decomp_step"))
                ax.set_ylabel(get_text("precision"))
                ax.set_yscale("log")
                ax.set_xticks(range(len(precisions)))
                st.pyplot(fig)

        st.divider()
        # Veiktspējas rādītāji
        st.subheader(get_text("performance_metrics"))
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        m_col1.metric(get_text("num_gates"), qc.size())
        m_col2.metric(get_text("circuit_depth"), qc.depth())
        m_col3.metric(get_text("num_qubits"), qc.num_qubits)
        m_col4.metric(get_text("precision_achieved"), f"{precision:.2e}")

        with st.expander(get_text("view_unitary")):
            U = qd.align_phase(Operator(qc).data, st.session_state.U_target)
            U = np.round(U, decimals=6) # noapaļo, lai uzlabotu lasāmību
            st.write(U)

    # Ja rezultāti nav pieejami, parāda informācijas ziņojumu
    else:
        st.info(get_text("please_start"))


# Funkcija, kas definē logu unitārās matricas ievadei
# un tās validāciju
@st.dialog("Ievads / Input", width="medium")
def input_unitary(num_qubits: int) -> None:
    st.subheader(get_text("input_unitary"))

    U_target = st.session_state.U_target
    dim = 2**num_qubits # pamats paplašināšanai līdz vairākiem kubitiem

    for (i) in range(dim):
        row = st.columns(dim)
        # Definē tabulus ar reālo un imagināro daļu ievadi
        for (j, col) in enumerate(row):
            cell = col.container(border=True, horizontal=True, vertical_alignment="center", horizontal_alignment="distribute")

            real_part = cell.number_input(label=f"{get_text("real")} U[{i},{j}]", min_value=-1., max_value=1., key=f"real_{i}_{j}",
                                        value=float(U_target[i, j].real), format="%.6f", width=90, label_visibility="collapsed")
            cell.markdown("**+**", width="content")
            imag_part = cell.number_input(label=f"{get_text("imaginary")} U[{i},{j}]", min_value=-1., max_value=1., key=f"imag_{i}_{j}",
                                        value=float(U_target[i, j].imag), format="%.6f", width=90, label_visibility="collapsed")
            cell.markdown("**i**", width="content")

    # Validācija
    if st.button(get_text("apply")):
        U_temp = np.zeros((dim, dim), dtype=complex)
        # Izveido matricu no ievadītajām vērtībām
        for (i) in range(dim):
            for (j) in range(dim):
                real_part = st.session_state[f"real_{i}_{j}"]
                imag_part = st.session_state[f"imag_{i}_{j}"]
                U_temp[i, j] = complex(real_part, imag_part)

        if qd.is_unitary(U_temp):
            st.session_state.U_target = U_temp
            if 'qc' in st.session_state: # notīra iepriekšējos rezultātus
                del st.session_state['qc']
                del st.session_state['precision']
                del st.session_state['history']
            st.rerun()

        else:
            st.error(get_text("not_unitary"))

# Galvenais izpildes punkts
if __name__ == "__main__":
    main()