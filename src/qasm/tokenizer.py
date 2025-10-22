from typing import List
from qasm.token import Token


class QASMTokenizer:
    """
    Tokenizes OpenQASM 2.0 source code into a stream of tokens.
    """

    def __init__(self, source: str):
        self.source = source
        self.tokens: List[Token] = []
        self.position = 0
        self.line = 1
        self.column = 1

    def tokenize(self) -> List[Token]:
        ...

    def skip_whitespace(self):
        ...

    def skip_comment(self):
        ...