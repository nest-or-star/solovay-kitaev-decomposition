import re
from sympy import pi, E, I, sin, cos, tan, sqrt, exp
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation,
)
from modules import utils

PARSE_NAMES = {
    "pi": pi,
    "e": E,
    "E": E,
    "i": I,
    "I": I,
    "sin": sin,
    "cos": cos,
    "tan": tan,
    "sqrt": sqrt,
    "exp": exp,
}

ALLOWED_IDENTIFIERS = set(PARSE_NAMES.keys())

TRANSFORMATIONS = standard_transformations + (convert_xor,implicit_multiplication_application,function_exponentiation,)

MAX_LEN = 100

IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
VALID_CHARS_RE = re.compile(r"^[0-9A-Za-z+\-*/^().\s]*$")


class MatrixElementParseError(ValueError):
    pass

def parse_matrix_element(text: str):
    text = text.strip()

    if not text:
        raise MatrixElementParseError("Empty input.")

    if len(text) > MAX_LEN:
        raise MatrixElementParseError("Expression is too long.")

    if not VALID_CHARS_RE.fullmatch(text):
        raise MatrixElementParseError("Invalid characters in expression.")

    identifiers = set(IDENT_RE.findall(text))
    bad = identifiers - ALLOWED_IDENTIFIERS
    if bad:
        raise MatrixElementParseError(
            f"Unknown name(s): {', '.join(sorted(bad))}"
        )

    try:
        expr = parse_expr(
            text,
            local_dict=PARSE_NAMES,
            transformations=TRANSFORMATIONS,
            evaluate=True,
        )
    except Exception as exc:
        raise MatrixElementParseError("Invalid mathematical expression.") from exc

    if expr.free_symbols:
        raise MatrixElementParseError("Variables are not allowed.")

    undefined_funcs = expr.atoms(AppliedUndef)
    if undefined_funcs:
        raise MatrixElementParseError("Unknown function used.")

    return expr

def parse_matrix_element_numeric(text: str) -> complex:
    expr = parse_matrix_element(text)
    return complex(expr.evalf())

import numpy as np

def matrix_to_str(arr: np.ndarray) -> str:

    def format_complex(c: complex) -> str:
        c_re, c_im = c.real, c.imag

        if c_im == 0:
            return f"{c_re:.10f}"
        if c_re == 0:
            return f"{c_im:.10f}*i"

        op = "+" if c_im >= 0 else "-"
        return f"{c_re:.10f}{op}{abs(c_im):.10f}*i"

    lines = []
    for row in arr:
        lines.append(",".join(format_complex(x) for x in row))

    return "\n".join(lines)


# Testing

if __name__ == "__main__":
    s = """exp(-i*pi/6),0
    0,exp(i*pi/6)"""
    rows = s.split("\n")
    dim = len(rows)
    u = np.zeros((dim, dim), dtype=complex)

    if np.allclose(np.log2(dim) % 2, 0):
        print("dim error")

    else:
        faulty = False
        for i, row in enumerate(rows):
            cells = row.split(',')

            if len(cells) != dim:
                print("dim error")
                break

            for j, cell in enumerate(cells):
                try:
                    result = parse_matrix_element_numeric(cell)
                    u[i, j] = result
                except MatrixElementParseError:
                    print("parse error")
                    faulty = True
                    break

            if faulty:
                break

        else:
            if utils.is_unitary(u):
                pass
            else:
                print("not unitary")
    print(matrix_to_str(u))