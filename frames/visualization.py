"""Desenho das cartas e exibição dos frames com OpenCV."""

import cv2
import numpy as np


class FrameVisualizer:
    """Cuida das anotações, da janela e dos eventos de teclado."""

    def __init__(self, display_size, window_name="ptcg-code-reader"):
        self._display_size = display_size
        self._window_name = window_name
        self._quit_requested = False

    @property
    def quit_requested(self):
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

    def annotate_frame(self, frame, cards, seen_cards):
        """Retorna uma cópia anotada usando as melhores leituras das cartas."""
        annotated_frame = frame.copy()
        for card in cards:
            annotated_frame = self._draw_card(
                annotated_frame, card, seen_cards[card.track_id]
            )
        return annotated_frame

    @staticmethod
    def _draw_card(annotated_frame, card, seen_card):
        """Desenha a carta e sua melhor leitura no frame."""
        card_left, card_top, card_right, card_bottom = card.bounding_box
        
        best_card_code = seen_card.best_code
        best_code_confidence = seen_card.best_confidence

        # Amarelo enquanto não há código; verde após uma leitura válida.
        mask_color = (0, 255, 255) if best_card_code is None else (0, 255, 0)
        card_polygon_pixels = np.array(card.polygon, dtype=np.int32)

        mask_overlay = annotated_frame.copy()
        cv2.fillPoly(mask_overlay, [card_polygon_pixels], mask_color)
        annotated_frame = cv2.addWeighted(mask_overlay, 0.25, annotated_frame, 0.75, 0)

        # Bounding Box
        cv2.rectangle(
            annotated_frame,
            (card_left, card_top),
            (card_right, card_bottom),
            mask_color,
            1
        )

        # ID da carta
        cv2.putText(
            annotated_frame, f"ID {card.track_id}",
            (card_left, max(0, card_top - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2, cv2.LINE_AA
        )

        # Código
        if best_card_code is not None:
            best_code_confidence_percent = best_code_confidence * 100
            cv2.putText(
                annotated_frame, f"{best_card_code} ({best_code_confidence_percent:.1f}%)",
                (card_left, max(0, card_top - 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA
            )

        return annotated_frame
