import re
import numpy as np
from sympy import pi, E, I, sin, cos, tan, sqrt, exp, log
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation,
)

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
        # Extract real and imaginary parts
        c_re, c_im = c.real, c.imag

        if c_im == 0:
            return f"{c_re:.10g}"
        if c_re == 0:
            return f"{c_im:.10g}*i"

        # Use + or - sign appropriately to maintain valid syntax
        op = "+" if c_im >= 0 else "-"
        return f"{re:.10g}{op}{abs(c_im):.10g}*i"

    # Handle 1D and 2D arrays
    if arr.ndim == 1:
        return ",".join(format_complex(x) for x in arr)

    lines = []
    for row in arr:
        lines.append(",".join(format_complex(x) for x in row))

    return "\n".join(lines)