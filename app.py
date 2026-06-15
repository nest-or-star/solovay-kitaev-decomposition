from typing import Any

import streamlit as st
import numpy as np
from qiskit.quantum_info import Operator
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from modules import solovay_kitaev as qd, parser as mp, decomposition, utils
import json
import os

from modules.parser import MatrixElementParseError
from modules.utils import str_to_circuit


def _loc(key: str) -> str:
    """
    Fetches a translated string from the locale dictionary in session state.
    """
    return st.session_state.translations.get(key, f"NO TRANSLATION: {key}")


@st.cache_data
def load_translations(language_code: str) -> Any:
    """
    Loads JSON translation file and stores it in cache memory.

    Translations are stored in '/locales'.
    """
    file_path = os.path.join("locales", f"{language_code}.json")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except FileNotFoundError:
        st.error(f"No locales found: {file_path}")
        return {}


def main() -> None:
    """
    Initialises the app.
    """
    load_target_unitary()
    load_language()

    st.set_page_config(
        page_title=_loc("page_title"),
        layout="wide")
    st.title(_loc("main_title"))

    load_sidebar()
    load_results()


def load_target_unitary() -> None:
    """
    Makes sure the target unitary is loaded.
    """
    if "u_target" not in st.session_state:
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
        default = st.session_state["lang"].upper())).lower()

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
    qc, precision = decomposition.rotation_decomposition(st.session_state.u_target)

    st.success(_loc("success_rotation"))
    st.session_state.qc = qc
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

    dim = st.session_state.u_target.ndim

    bar = st.progress(value=0.0)
    bar.progress(0.0, text=_loc("success_h_t"))

    base = qd.load_basic_circuits(dim, max_length, "H", "T", "Tdg")
    bar.progress(0.0, text=f"{_loc("loaded")} {len(base)} {_loc("short_circuits")}") # "Loaded X basic circuits"

    vectors = np.array([utils.vectorize_unitary(item[0]) for item in base]) # Feed this to sklearn.neighbors.BallTree
    tree = BallTree(vectors, leaf_size=40)

    # Max error is given WIP
    if epsilon is not None:
        # progress_info = [bar, sum([3**i for i in range(7)]), 0] # progress bar + total steps + first step
        # qc_0, _ = qd.base_approximation(st.session_state.u_target, base_circuits)
        # qc, precision, history = qd.solovay_kitaev_reverse(st.session_state.u_target,
        #                                                    qc_0, epsilon,
        #                                                    base_circuits, progress_info)
        return

    # Recursion depth is given
    if recursion_depth is not None:
        progress_info = [bar, sum([3**i for i in range(recursion_depth+1)]), 0] # progress bar + total steps + first step
        u_approx, seq, history_str = qd.solovay_kitaev_decomposition(st.session_state.u_target,
                                                                 recursion_depth,
                                                                 base,
                                                                 tree,
                                                                 progress_info)

    qc = str_to_circuit(seq, dim)
    history = [str_to_circuit(x, dim) for x in history_str]

    st.success(_loc("success_sk"))
    st.session_state.qc = qc
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

        else:
            qc = st.session_state.qc

        precision = utils.compare_su2(st.session_state.u_target, Operator(qc).data)

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
                precisions = [utils.compare_su2(st.session_state.u_target, Operator(circ).data) for circ in history]
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
            u = utils.align_phase(Operator(qc).data, st.session_state.u_target)
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

    # Input field
    u_str = st.text_area("Unitary field",
                         value=mp.matrix_to_str(st.session_state.u_target),
                         height="stretch",
                         label_visibility="collapsed")

    # Validation
    if st.button(_loc("apply")):
        rows = u_str.split("\n")
        dim = len(rows)
        u = np.zeros((dim, dim), dtype=complex)

        if np.allclose(np.log2(dim) % 1, 0):
            st.error(_loc("u_dim_error"))

        else:
            faulty = False
            for (i, row) in enumerate(rows):
                cells = row.split(',')

                if len(cells) != dim:
                    st.error(_loc("u_dim_error"))
                    break

                for (j, cell) in enumerate(cells):
                    try:
                        result = mp.parse_matrix_element_numeric(cell)
                        u[i, j] = result
                    except MatrixElementParseError:
                        st.error("u_parse_error")
                        faulty = True
                        break

                if faulty:
                    break

            else:
                if utils.is_unitary(u):
                    st.session_state.u_target = u
                    st.rerun()
                else:
                    st.error(_loc("not_unitary"))


# Entry point
if __name__ == "__main__":
    main()