"""Captura de frames da webcam ou da tela."""

import cv2
import mss
import numpy as np


class WebcamCapture:
    """Captura frames de uma webcam usando OpenCV."""

    def __init__(self, webcam_index):
        self._cv2_webcam = cv2.VideoCapture(webcam_index)

        try:
            if not self._cv2_webcam.isOpened():
                raise RuntimeError("Erro ao iniciar webcam.")
        except Exception:
            self.close()
            raise

    def get_next_frame(self):
        frame_read_success, frame = self._cv2_webcam.read()
        if not frame_read_success:
            raise RuntimeError("Não foi possível capturar um frame da webcam.")

        return frame

    def close(self):
        self._cv2_webcam.release()


class ScreenCapture:
    """Captura frames de um monitor usando MSS."""

    def __init__(self, monitor_index):
        self._mss_screen = mss.mss()

        try:
            monitors = self._mss_screen.monitors
            if not 0 <= monitor_index < len(monitors):
                raise ValueError(f"Índice de monitor inválido: {monitor_index}.")

            # No mss, 0 representa todos os monitores; 1 é o primeiro físico.
            self._monitor = monitors[monitor_index]
        except Exception:
            self.close()
            raise

    def get_next_frame(self):
        screenshot = self._mss_screen.grab(self._monitor)
        screen_frame = np.array(screenshot)
        return cv2.cvtColor(screen_frame, cv2.COLOR_BGRA2BGR)

    def close(self):
        self._mss_screen.close()


class FrameCapture:
    """Expõe uma interface única para captura de webcam ou tela."""

    def __init__(self, source: str, webcam_index: int = 0, monitor_index: int = 1):
        if source == "webcam":
            self._frame_capture_device = WebcamCapture(webcam_index)
        elif source == "screen":
            self._frame_capture_device = ScreenCapture(monitor_index)
        else:
            raise ValueError('CAPTURE_SOURCE deve ser "webcam" ou "screen".')

        self._is_closed = False

    def get_next_frame(self):
        """Retorna o próximo frame BGR da fonte configurada."""
        if self._is_closed:
            raise RuntimeError("O dispositivo de captura já foi fechado.")

        return self._frame_capture_device.get_next_frame()

    def close(self):
        """Libera o dispositivo; chamadas repetidas não têm efeito."""
        if self._is_closed:
            return

        self._frame_capture_device.close()
        self._is_closed = True

    def __enter__(self):
        if self._is_closed:
            raise RuntimeError("O dispositivo de captura já foi fechado.")

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
