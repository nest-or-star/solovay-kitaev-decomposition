from typing import Any

import streamlit as st
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
import matplotlib.pyplot as plt
import quantum_decomposition as qd
import json
import os

# Function
# Fetches the translation with the given key
def _loc(key: str) -> str:
    return st.session_state.translations.get(key, f"NO TRANSLATION: {key}")

# Function
# Loads JSON translation file and stores in cache memory
# Translations are stored in "/locales"
@st.cache_data
def load_translations(language_code: str) -> Any:
    file_path = os.path.join("locales", f"{language_code}.json")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except FileNotFoundError:
        st.error(f"No locales found: {file_path}")
        return {}


# MAIN Function
# Initializes the UI
def main() -> None:

    load_target_unitary()
    load_language()

    st.set_page_config(
        page_title=_loc("page_title"),
        layout="wide")
    st.title(_loc("main_title"))

    # Input fields
    load_sidebar()
    # Main page
    load_results()

# Function
# If no target unitary stored, loads an identity matrix
def load_target_unitary() -> None:
    if "U_target" not in st.session_state:
        st.session_state.u_target = np.eye(2, dtype=complex)

# Function
# Manages the choice of localization
def load_language() -> None:
    # Default is Latvian
    if "lang" not in st.session_state:
        st.session_state["lang"] = "lv"

    lang_choice = str(st.sidebar.segmented_control(
        "Valoda / Language",
        ["LV", "EN"],
        index = 0 if st.session_state["lang"] == "lv" else 1)).lower()

    # If changed, reloads page
    if lang_choice != st.session_state["lang"]:
        st.session_state["lang"] = lang_choice
        st.rerun()

    st.session_state.translations = load_translations(st.session_state["lang"])

# Function
# Sets up the sidebar for user input
def load_sidebar() -> None:
    st.sidebar.header(
        _loc("sidebar_header"))
    mode = st.sidebar.radio( # mode
        _loc("decomp_type"),
        [_loc("rotation_h_rz"), _loc("solovay_kitaev_h_t")])

    # Matrix viewing and editing
    with st.sidebar.expander(_loc("view_unitary")):
        st.write(st.session_state.u_target)

        if st.button(_loc("edit")):
            input_unitary(1)

    # Rotation H+Rz decomposition mode
    if mode == _loc("rotation_h_rz"):
        if st.sidebar.button(_loc("run")):
            launch_rotation_decomp()
    
    # Solovay-Kitaev algorithm mode
    else:
        st.sidebar.subheader(_loc("sk_params"))
        precision_mode = st.sidebar.radio( # target metric
            _loc("precision_mode"),
            [_loc("precision"), _loc("recursion_depth")],
            label_visibility="collapsed")

        # Max error mode
        if precision_mode == _loc("precision"):
            target_epsilon = st.sidebar.number_input(
                _loc("precision"),
                min_value=0.00001, max_value=2.0,
                value=0.01,
                format="%.5f",
                step=0.00001,
                label_visibility="collapsed")
            recursion_depth = None

        # Recursion depth mode
        else:
            recursion_depth = st.sidebar.number_input(
                _loc("recursion_depth"),
                min_value=0, max_value=5,
                value=2,
                label_visibility="collapsed")
            target_epsilon = None

        # Maximum length of base circuits
        max_length = st.sidebar.number_input(
            _loc("max_length"),
            min_value=1, max_value=20,
            value=12)

        if st.sidebar.button(_loc("run")):
            launch_solovay_kitaev(target_epsilon, recursion_depth, max_length)

# Function
# Calls rotation H+Rz decomposition and saves the results
# Streamlit reruns automatically and loads results from the updated session state
def launch_rotation_decomp() -> None:
    qc, precision = qd.rotation_decomposition(st.session_state.U_target)

    st.success(_loc("success_rotation"))
    st.session_state.qc = qc
    st.session_state.precision = precision
    st.session_state.history = []

# Function
# Calls Solovay-Kitaev decomposition and saves the results
# Streamlit reruns automatically and loads results from the updated session state
def launch_solovay_kitaev(epsilon: float|None,
                          recursion_depth: int|None,
                          max_length: int) -> None:

    # Validation
    if epsilon is None and recursion_depth is None: # no target metric specified
        st.error(_loc("specify_precision"))
        return
    
    if epsilon is not None and recursion_depth is not None: # both target metrics specified
        st.error(_loc("specify_precision"))
        return
    
    if epsilon is not None and type(epsilon) != float: # wrong type
        st.error(_loc("invalid_input"))
        return
    
    if recursion_depth is not None and type(recursion_depth) != int: # wrong type
        st.error(_loc("invalid_input"))
        return
    
    if type(max_length) != int: # wrong type
        st.error(_loc("invalid_input"))
        return

    if epsilon is not None and (epsilon < 0.00001 or epsilon > 2): # max error out of bounds
        st.error(_loc("invalid_input"))
        return
    
    if epsilon is None and (recursion_depth < 0 or recursion_depth > 5): # recursion depth out of bounds
        st.error(_loc("invalid_input"))
        return
    
    if max_length < 1 or max_length > 20: # max length out of bounds
        st.error(_loc("invalid_input"))
        return

    # Successfully launches decomposition

    bar = st.progress(
        value=0.0)

    gate_set = qd.ht_gate_set()
    bar.progress(0.0, text=_loc("success_h_t"))

    base_circuits = qd.load_base_circuits(gate_set, max_length)
    bar.progress(0.0, text=f"{_loc("loaded")} {len(base_circuits)} {_loc("short_circuits")}") # "Ielādētas X pamatķēdes"

    # Max error is given
    if epsilon is not None:
        progress_info = [bar, sum([3**i for i in range(7)]), 0] # progress bar + total steps + first step
        qc_0, _ = qd.base_approximation(st.session_state.u_target, base_circuits)
        qc, precision, history = qd.solovay_kitaev_reverse(st.session_state.u_target,
                                                           qc_0, epsilon,
                                                           base_circuits, progress_info)

    # Recursion depth is given
    if recursion_depth is not None:
        progress_info = [bar, sum([3**i for i in range(recursion_depth+1)]), 0] # progress bar + total steps + first step
        qc, precision, history = qd.solovay_kitaev_decomposition(st.session_state.u_target,
                                                                 recursion_depth,
                                                                 base_circuits,
                                                                 progress_info)

    st.success(_loc("success_sk"))
    st.session_state.qc = qc
    st.session_state.precision = precision
    st.session_state.history = history


# Function
# Builds the result page
def load_results() -> None:
    # If decomposition is complete then qc, precision, history are in session state
    if 'qc' in st.session_state:
        history = st.session_state.get('history', [])

        # History slider
        if len(history) > 1:
            idx = st.slider(
                _loc("decomp_step"),
                min_value=0, max_value=len(history)-1,
                value=len(history)-1)
            qc = history[idx]
            precision = qd.compare_su2(st.session_state.u_target, Operator(qc).data)

        else:
            qc = st.session_state.qc
            precision = st.session_state.precision

        _, mid, _ = st.columns([1, 6, 1]) # centers the graphics
        with mid:
            # Quantum circuit
            st.subheader(_loc("circuit_diagram"))
            if qc.size() > 1500:
                st.warning(_loc("circuit_too_large"))
            else:
                fig_circuit = qc.draw(output='mpl')
                st.pyplot(fig_circuit)

            # Precision (=error) plot
            if len(history) > 1:
                st.divider()
                st.subheader(_loc("precision_change"))
                precisions = [qd.compare_su2(st.session_state.u_target, Operator(circ).data) for circ in history]
                fig, ax = plt.subplots()
                ax.plot(range(len(precisions)), precisions, marker='o')
                ax.set_xlabel(_loc("decomp_step"))
                ax.set_ylabel(_loc("precision"))
                ax.set_yscale("log")
                ax.set_xticks(range(len(precisions)))
                st.pyplot(fig)

        st.divider()
        # Circuit metrics
        st.subheader(_loc("performance_metrics"))
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        m_col1.metric(_loc("num_gates"), qc.size())
        m_col2.metric(_loc("circuit_depth"), qc.depth())
        m_col3.metric(_loc("num_qubits"), qc.num_qubits)
        m_col4.metric(_loc("precision_achieved"), f"{precision:.2e}")

        with st.expander(_loc("view_unitary")):
            u = qd.align_phase(Operator(qc).data, st.session_state.u_target)
            u = np.round(u, decimals=6) # rounds for readability
            st.write(u)

    # If decomposition is not complete:
    else:
        st.info(_loc("please_start"))


# Function
# Matrix editing pop-up window
@st.dialog("Ievads / Input", width="medium")
def input_unitary(num_qubits: int) -> None:
    st.subheader(_loc("input_unitary"))

    # Apakšfunkcija, kas nolasa ievadīto matricu
    def read_input_unitary(dim):
        U_temp = np.zeros((dim, dim), dtype=complex)
        # Izveido matricu no ievadītajām vērtībām
        for (i) in range(dim):
            for (j) in range(dim):
                real_part = st.session_state[f"real_{i}_{j}"]
                imag_part = st.session_state[f"imag_{i}_{j}"]
                U_temp[i, j] = complex(real_part, imag_part)
        
        return U_temp
    # ---

    # Apakšfunkcija, kas nolasa ievadīto matricu
    def write_input_unitary(dim, U):
        for (i) in range(dim):
            for (j) in range(dim):
                st.session_state[f"real_{i}_{j}"] = np.real(U[i, j])
                st.session_state[f"imag_{i}_{j}"] = np.imag(U[i, j])
    # ---

    # Apakšfunkcija, kas validē matricu
    def attempt_save(U):
        if qd.is_unitary(U):
            st.session_state.U_target = U
            if 'qc' in st.session_state: # notīra iepriekšējos rezultātus
                del st.session_state['qc']
                del st.session_state['precision']
                del st.session_state['history']
            st.rerun()

        else:
            st.error(_loc("not_unitary"))
    # ---


    U_target = st.session_state.U_target
    dim = 2**num_qubits # pamats paplašināšanai līdz vairākiem kubitiem

    for (i) in range(dim):
        row = st.columns(dim)
        # Definē tabulus ar reālo un imagināro daļu ievadi
        for (j, col) in enumerate(row):
            cell = col.container(border=True, horizontal=True, vertical_alignment="center", horizontal_alignment="distribute")

            real_part = cell.number_input(label=f"{_loc("real")} U[{i},{j}]", min_value=-1., max_value=1., key=f"real_{i}_{j}",
                                          value=float(U_target[i, j].real), format="%.6f", width=90, label_visibility="collapsed")
            cell.markdown("**+**", width="content")
            imag_part = cell.number_input(label=f"{_loc("imaginary")} U[{i},{j}]", min_value=-1., max_value=1., key=f"imag_{i}_{j}",
                                          value=float(U_target[i, j].imag), format="%.6f", width=90, label_visibility="collapsed")
            cell.markdown("**i**", width="content")

    # Rotācijas pogas
    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])

    angle = col1.number_input(label="angle", value=0.0, format="%.6f", label_visibility="collapsed")

    if col2.button("X"):
        U_temp = read_input_unitary(dim)
        R = qd.rotation_matrix(np.array([1,0,0]), angle)
        write_input_unitary(dim, R @ U_temp)

    if col3.button("Y"):
        U_temp = read_input_unitary(dim)
        R = qd.rotation_matrix(np.array([0,1,0]), angle)
        write_input_unitary(dim, R @ U_temp)

    if col4.button("Z"):
        U_temp = read_input_unitary(dim)
        R = qd.rotation_matrix(np.array([0,0,1]), angle)
        write_input_unitary(dim, R @ U_temp)

    # Validācija
    if st.button(_loc("apply")):
        U_temp = read_input_unitary(dim)

        attempt_save(U_temp)


# Entry point
if __name__ == "__main__":
    main()