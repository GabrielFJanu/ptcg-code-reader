"""Desenho das cartas e exibição dos frames com OpenCV."""

import cv2
import numpy as np


_YELLOW_BGR = (0, 255, 255)
_GREEN_BGR = (0, 255, 0)


class FrameVisualizer:
    """Cuida das anotações, da janela e dos eventos de teclado."""

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

    def annotate_frame(self, frame, tracked_cards, best_card_readings_by_track_id):
        """Retorna uma cópia anotada usando as melhores leituras das cartas."""
        annotated_frame = frame.copy()
        for tracked_card in tracked_cards:
            annotated_frame = self._draw_card(
                annotated_frame,
                tracked_card,
                best_card_readings_by_track_id.get(tracked_card.track_id),
            )
        return annotated_frame

    @staticmethod
    def _draw_card(annotated_frame, tracked_card, best_card_reading):
        """Desenha a carta e sua melhor leitura no frame."""
        card_left, card_top, card_right, card_bottom = tracked_card.bounding_box

        best_card_code = best_card_reading.code if best_card_reading is not None else None

        # Amarelo enquanto não há código; verde após uma leitura válida.
        card_color = _YELLOW_BGR if best_card_code is None else _GREEN_BGR
        card_polygon_pixels = np.array(tracked_card.polygon, dtype=np.int32)

        mask_overlay = annotated_frame.copy()
        cv2.fillPoly(mask_overlay, [card_polygon_pixels], card_color)
        annotated_frame = cv2.addWeighted(mask_overlay, 0.25, annotated_frame, 0.75, 0)

        # Contorno do retângulo da carta.
        cv2.rectangle(
            annotated_frame,
            (card_left, card_top),
            (card_right, card_bottom),
            card_color,
            1,
        )

        # ID de rastreamento acima da carta.
        cv2.putText(
            annotated_frame,
            f"ID {tracked_card.track_id}",
            (card_left, max(0, card_top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            _YELLOW_BGR,
            2,
            cv2.LINE_AA,
        )

        # Melhor código e sua confiança, acima do ID.
        if best_card_code is not None:
            best_code_confidence_percent = best_card_reading.confidence * 100
            code_label = f"{best_card_code} ({best_code_confidence_percent:.1f}%)"
            cv2.putText(
                annotated_frame,
                code_label,
                (card_left, max(0, card_top - 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                _GREEN_BGR,
                2,
                cv2.LINE_AA,
            )

        return annotated_frame
