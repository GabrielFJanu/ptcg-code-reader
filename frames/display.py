"""Exibição dos frames e controle da janela com OpenCV."""

import cv2


class FrameDisplay:
    """Cuida da exibição, da janela e dos eventos de teclado."""

    def __init__(self, display_size, window_name="ptcg-code-reader"):
        self._display_size = display_size
        self._window_name = window_name
        self._quit_requested = False

    @property
    def quit_requested(self) -> bool:
        """Indica se o usuário solicitou o encerramento."""
        return self._quit_requested

    def show(self, annotated_frame):
        """Exibe o frame e registra o pedido de saída ao pressionar Q."""
        cv2.imshow(
            self._window_name,
            cv2.resize(annotated_frame, self._display_size),
        )
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self._quit_requested = True

    def close(self):
        """Fecha as janelas de visualização do OpenCV."""
        cv2.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
