"""Anotação de frames com as cartas rastreadas e suas melhores leituras."""

import cv2
import numpy as np


_YELLOW_BGR = (0, 255, 255)
_GREEN_BGR = (0, 255, 0)


def annotate_frame(frame, tracked_cards, best_card_readings_by_track_id):
    """Retorna uma cópia anotada usando as melhores leituras das cartas."""
    annotated_frame = frame.copy()
    for tracked_card in tracked_cards:
        annotated_frame = _draw_card(
            annotated_frame,
            tracked_card,
            best_card_readings_by_track_id.get(tracked_card.track_id),
        )
    return annotated_frame


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
