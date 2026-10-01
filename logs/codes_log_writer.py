"""Gravação das mudanças de códigos das cartas em arquivo."""


class CodesLogWriter:
    """Acrescenta registros de ID e código ao log e gerencia o arquivo."""

    def __init__(self, path):
        self._file = open(path, "a", buffering=1, encoding="utf-8")

    def write_updates(self, updates):
        """Grava as mudanças recebidas no formato ID,código, uma por linha."""
        for card_id, code in updates:
            self._file.write(f"{card_id},{code}\n")

    def close(self):
        """Fecha o arquivo, garantindo a gravação dos dados pendentes."""
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
