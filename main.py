import cv2
import numpy as np
import torch
from models.CardOrientationClassifier import CardOrientationClassifier
from models.CardSegmenter import CardSegmenter
from models.CharacterDetector import CharacterDetector
from utils import rotate_card_crop
from capture import FrameCapture

from config import (
    CARD_SEGMENTER_WEIGHTS_PATH,
    CARD_TRACKER_CONFIG,
    CARD_ORIENTATION_CLASSIFIER_WEIGHTS_PATH,
    CHARACTER_DETECTOR_WEIGHTS_PATH,
    CAPTURE_SOURCE,
    WEBCAM_INDEX,
    MONITOR_INDEX,
    CARD_SEGMENTER_CONFIDENCE_THRESHOLD,
    CHARACTER_DETECTOR_CONFIDENCE_THRESHOLD,
    CARD_SEGMENTER_IMAGE_SIZE,
    CHARACTER_DETECTOR_IMAGE_SIZE,
    CHARACTER_DETECTOR_IOU_THRESHOLD,
    FRAME_DISPLAY_SIZE,
    CARD_CODES_LOG_PATH,
)


def read_card_code(character_detector, card_crop):
    """Monta o código da esquerda para a direita e multiplica as confianças."""
    has_detected_characters, detected_characters = character_detector.predict(card_crop)
    if not has_detected_characters:
        return "", None

    detected_characters.sort(key=lambda detection: detection.center_x)
    code = "".join(detection.character for detection in detected_characters)

    confidence_product = 1.0
    for detection in detected_characters:
        confidence_product *= detection.confidence

    return code, confidence_product


def update_best_card_reading(card_id, code, confidence, reading, logged_codes, log_file):
    """Atualiza a melhor leitura válida e registra códigos novos para a carta."""
    if len(code) != 13:
        return
    if confidence <= reading["best_prob"]:
        return

    if code not in logged_codes:
        logged_codes.add(code)
        log_file.write(f"{card_id},{code}\n")

    reading["best_code"] = code
    reading["best_prob"] = confidence


def main():
    """Executa a leitura e o registro dos códigos das cartas."""
    # Dispositivo.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Dispositivo de inferência:", device)

    # Carregamento dos modelos.
    card_orientation_classifier = CardOrientationClassifier(
        weights_path=CARD_ORIENTATION_CLASSIFIER_WEIGHTS_PATH,
        device=device,
    )
    card_segmenter = CardSegmenter(
        weights_path=CARD_SEGMENTER_WEIGHTS_PATH,
        confidence_threshold=CARD_SEGMENTER_CONFIDENCE_THRESHOLD,
        image_size=CARD_SEGMENTER_IMAGE_SIZE,
        tracker_config=CARD_TRACKER_CONFIG,
    )
    character_detector = CharacterDetector(
        weights_path=CHARACTER_DETECTOR_WEIGHTS_PATH,
        confidence_threshold=CHARACTER_DETECTOR_CONFIDENCE_THRESHOLD,
        image_size=CHARACTER_DETECTOR_IMAGE_SIZE,
        iou_threshold=CHARACTER_DETECTOR_IOU_THRESHOLD,
    )

    card_readings_by_id = {}  # ID do ByteTrack -> melhor código e confiança

    # guarda códigos já registrados por ID -> set(códigos)
    logged_codes_by_card_id = {}

    try:
        with FrameCapture(
            source=CAPTURE_SOURCE,
            webcam_index=WEBCAM_INDEX,
            monitor_index=MONITOR_INDEX,
        ) as capture, open(CARD_CODES_LOG_PATH, "a", buffering=1, encoding="utf-8") as card_codes_log_file:
            while True:
                frame = capture.get_next_frame()
                annotated_frame = frame.copy()

                # Segmentação e rastreamento das cartas.
                has_tracked_cards, tracked_cards = card_segmenter.track(frame)

                if has_tracked_cards:
                    for card in tracked_cards:
                        card_polygon_points = card.polygon

                        # bounding box padrão YOLO (eixo alinhado)
                        card_left, card_top, card_right, card_bottom = card.bounding_box

                        card_id = card.track_id
                        if card_id not in card_readings_by_id:
                            card_readings_by_id[card_id] = {
                                "best_code": None,
                                "best_prob": 0.0,
                            }

                        # se ainda não temos set de códigos vistos para esse ID, cria
                        if card_id not in logged_codes_by_card_id:
                            logged_codes_by_card_id[card_id] = set()

                        card_reading = card_readings_by_id[card_id]
                        logged_codes = logged_codes_by_card_id[card_id]

                        # recorte da carta a partir da bounding box normal
                        card_crop = frame[card_top:card_bottom, card_left:card_right]
                        if card_crop.size == 0:
                            continue

                        # Classificação e correção da orientação.
                        orientation_result = card_orientation_classifier.predict(card_crop)
                        oriented_card_crop = rotate_card_crop(
                            card_crop, orientation_result.angle_degrees
                        )

                        detected_code, code_confidence = read_card_code(
                            character_detector, oriented_card_crop
                        )
                        update_best_card_reading(
                            card_id,
                            detected_code,
                            code_confidence,
                            card_reading,
                            logged_codes,
                            card_codes_log_file,
                        )

                        # Amarelo enquanto não há código; verde após uma leitura válida.
                        best_card_code = card_reading["best_code"]
                        best_code_confidence = card_reading["best_prob"]

                        mask_color = (0, 255, 255) if best_card_code is None else (0, 255, 0)
                        card_polygon_pixels = np.array(card_polygon_points, dtype=np.int32)

                        mask_overlay = annotated_frame.copy()
                        cv2.fillPoly(mask_overlay, [card_polygon_pixels], mask_color)
                        annotated_frame = cv2.addWeighted(mask_overlay, 0.25, annotated_frame, 0.75, 0)

                        # BOUNDING BOX PADRÃO
                        cv2.rectangle(
                            annotated_frame,
                            (card_left, card_top),
                            (card_right, card_bottom),
                            mask_color,
                            1
                        )

                        # ID da carta
                        cv2.putText(
                            annotated_frame, f"ID {card_id}",
                            (card_left, max(0, card_top - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 255), 2, cv2.LINE_AA
                        )

                        # Melhor código lido para esta carta.
                        if best_card_code is not None:
                            best_code_confidence_percent = best_code_confidence * 100
                            cv2.putText(
                                annotated_frame, f"{best_card_code} ({best_code_confidence_percent:.1f}%)",
                                (card_left, max(0, card_top - 30)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2, cv2.LINE_AA
                            )

                cv2.imshow(
                    "YOLO11 SEG + CardOrientationClassifier",
                    cv2.resize(annotated_frame, FRAME_DISPLAY_SIZE),
                )

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
