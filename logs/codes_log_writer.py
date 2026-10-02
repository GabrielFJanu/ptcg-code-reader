"""Gravação das mudanças de códigos das cartas em arquivo."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from code_reader import CardReading


class CodesLogWriter:
    """Acrescenta registros de ID e código ao log e gerencia o arquivo."""

    def __init__(self, log_path):
        self._log_file = open(log_path, "a", buffering=1, encoding="utf-8")

    def write_readings(self, readings: list["CardReading"]) -> None:
        """Grava as leituras recebidas no formato ID,código, uma por linha."""
        for reading in readings:
            self._log_file.write(f"{reading.track_id},{reading.code}\n")

    def close(self):
        """Fecha o arquivo, garantindo a gravação dos dados pendentes."""
        self._log_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
