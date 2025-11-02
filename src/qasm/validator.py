from typing import List, Tuple

class QASMValidator:
    """
    Validates OpenQASM 2.0 source code syntax before parsing.
    """

    def __init__(self):
        self.errors: List[Tuple[int, str]] = []  # (line_number, error_message)
        self.warnings: List[Tuple[int, str]] = []

    def validate(self, source: str) -> bool:
        ...

    def check_version_declaration(self, source: str) -> bool:
        ...
