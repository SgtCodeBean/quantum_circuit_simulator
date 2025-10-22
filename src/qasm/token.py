"""
Token definitions for QASM parsing in OpenQASM 2.0;

Used by the tokenizer and consumed by the recursive parser.
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Any


class TokenType(Enum):
    # Keywords
    OPENQASM = auto()
    INCLUDE = auto()
    QREG = auto()
    CREG = auto()
    GATE = auto()
    MEASURE = auto()
    BARRIER = auto()
    RESET = auto()
    IF = auto()
    OPAQUE = auto()

    # Literals
    IDENTIFIER = auto()  # gate names, variable names, etc.
    INTEGER = auto()     # integers like 2, 10
    REAL = auto()        # real numbers like 3.14, 2.5
    STRING = auto()      # string literals like "qelib1.inc"

    # Operators
    PLUS = auto()        # +
    MINUS = auto()       # -
    MULTIPLY = auto()    # *
    DIVIDE = auto()      # /
    POWER = auto()       # ^

    # Delimiters
    SEMICOLON = auto()   # ;
    COMMA = auto()       # ,
    LPAREN = auto()      # (
    RPAREN = auto()      # )
    LBRACKET = auto()    # [
    RBRACKET = auto()    # ]
    LBRACE = auto()      # {
    RBRACE = auto()      # }
    ARROW = auto()       # ->

    PI = auto()    # PI constant
    EOF = auto()   # End of file
    ERROR = auto() # Error token


@dataclass
class Token:
    """
    Represents a single token in the QASM source.
    """
    type: TokenType
    value: Any
    line: int = 1
    column: int = 1

    def __repr__(self):
        return f"Token({self.type.name}, {repr(self.value)}, {self.line}:{self.column})"

    def __str__(self):
        return f"{self.type.name}({self.value})"
