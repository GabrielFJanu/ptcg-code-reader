"""Captura de frames da webcam ou da tela."""

import cv2
import mss
import numpy as np


class CaptureDevice:
    """Gerencia a captura e libera seus recursos ao sair de um bloco with."""

    def __init__(self, source: str, webcam_index: int = 0, monitor_index: int = 1):
        if source not in ("webcam", "screen"):
            raise ValueError('CAPTURE_SOURCE deve ser "webcam" ou "screen".')

        self._capture_source = source
        self._is_closed = False

        if source == "webcam":
            self._initialize_webcam(webcam_index)
        else:
            self._initialize_screen_capture(monitor_index)

    def _initialize_webcam(self, webcam_index: int):
        self._capture_device = cv2.VideoCapture(webcam_index)

        try:
            if not self._capture_device.isOpened():
                raise RuntimeError("Erro ao iniciar webcam.")
        except Exception:
            self.close()
            raise

    def _initialize_screen_capture(self, monitor_index: int):
        self._capture_device = mss.mss()

        # Se a seleção do monitor falhar, libera a captura já aberta.
        try:
            available_monitors = self._capture_device.monitors
            if not 0 <= monitor_index < len(available_monitors):
                raise ValueError(f"Índice de monitor inválido: {monitor_index}.")

            # No mss, 0 representa todos os monitores; 1 é o primeiro físico.
            self._capture_monitor = available_monitors[monitor_index]
        except Exception:
            self.close()
            raise

    def get_next_frame(self):
        """Retorna um frame BGR; lança RuntimeError se a leitura da webcam falhar."""
        if self._is_closed:
            raise RuntimeError("O dispositivo de captura já foi fechado.")

        if self._capture_source == "screen":
            screenshot = self._capture_device.grab(self._capture_monitor)
            screen_frame = np.array(screenshot)
            return cv2.cvtColor(screen_frame, cv2.COLOR_BGRA2BGR)

        frame_read_success, frame = self._capture_device.read()
        if not frame_read_success:
            raise RuntimeError("Não foi possível capturar um frame da webcam.")

        return frame

    def close(self):
        """Libera o dispositivo; chamadas repetidas não têm efeito."""
        if self._is_closed:
            return

        if self._capture_source == "webcam":
            self._capture_device.release()
        else:
            self._capture_device.close()

        self._is_closed = True

    def __enter__(self):
        if self._is_closed:
            raise RuntimeError("O dispositivo de captura já foi fechado.")

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
