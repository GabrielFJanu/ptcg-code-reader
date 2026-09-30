import time

import cv2
import numpy as np
import torch
from models.CardOrientationClassifier import CardOrientationClassifier
from models.CardSegmenter import CardSegmenter
from models.CharacterDetector import CharacterDetector
from utils import generate_random_colors
from code_reader import CodeReader
from capture import FrameCaptureDevice

from config import (
    CARD_TRACKER_CONFIG,
    CARD_SEGMENTER_WEIGHTS_PATH,
    CARD_ORIENTATION_CLASSIFIER_WEIGHTS_PATH,
    CHARACTER_DETECTOR_WEIGHTS_PATH,
    CAPTURE_SOURCE,
    WEBCAM_INDEX,
    MONITOR_INDEX,
    CARD_SEGMENTER_CONFIDENCE_THRESHOLD,
    CHARACTER_DETECTOR_CONFIDENCE_THRESHOLD,
    CARD_SEGMENTER_IMAGE_SIZE,
    CHARACTER_DETECTOR_IMAGE_SIZE,
    DEBUG_CHARACTER_DETECTOR_IOU_THRESHOLD,
    FRAME_DISPLAY_SIZE,
    DEBUG_CHARACTER_DETECTOR_CLASS_COUNT,
    CARD_CROP_DISPLAY_SIZE,
)


def crop_min_area_rect(source_image, rotated_rectangle):
    """Alinha o retângulo da carta e recorta a região dentro da imagem."""
    rectangle_center, rectangle_size, rectangle_angle_degrees = rotated_rectangle
    rectangle_center_x, rectangle_center_y = rectangle_center
    rectangle_width, rectangle_height = rectangle_size
    rotation_matrix = cv2.getRotationMatrix2D(
        rectangle_center, rectangle_angle_degrees, 1.0
    )
    rotated_image = cv2.warpAffine(
        source_image,
        rotation_matrix,
        (source_image.shape[1], source_image.shape[0]),
        flags=cv2.INTER_LINEAR,
    )
    crop_left = max(0, int(rectangle_center_x - rectangle_width / 2))
    crop_top = max(0, int(rectangle_center_y - rectangle_height / 2))
    crop_right = min(source_image.shape[1], int(rectangle_center_x + rectangle_width / 2))
    crop_bottom = min(source_image.shape[0], int(rectangle_center_y + rectangle_height / 2))
    return rotated_image[crop_top:crop_bottom, crop_left:crop_right]


def main():
    """Executa a visualização de debug com recortes e tempos de inferência."""
    character_class_colors = generate_random_colors(DEBUG_CHARACTER_DETECTOR_CLASS_COUNT)

    inference_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🧠 Dispositivo ativo:", inference_device)

    card_orientation_classifier = CardOrientationClassifier(
        weights_path=CARD_ORIENTATION_CLASSIFIER_WEIGHTS_PATH,
        device=inference_device,
    )

    card_segmenter = CardSegmenter(
        weights_path=CARD_SEGMENTER_WEIGHTS_PATH,
        device=inference_device,
        confidence_threshold=CARD_SEGMENTER_CONFIDENCE_THRESHOLD,
        image_size=CARD_SEGMENTER_IMAGE_SIZE,
        tracker_config=CARD_TRACKER_CONFIG,
    )
    character_detector = CharacterDetector(
        weights_path=CHARACTER_DETECTOR_WEIGHTS_PATH,
        device=inference_device,
        confidence_threshold=CHARACTER_DETECTOR_CONFIDENCE_THRESHOLD,
        image_size=CHARACTER_DETECTOR_IMAGE_SIZE,
        iou_threshold=DEBUG_CHARACTER_DETECTOR_IOU_THRESHOLD,
    )

    try:
        with FrameCaptureDevice(
            source=CAPTURE_SOURCE,
            webcam_index=WEBCAM_INDEX,
            monitor_index=MONITOR_INDEX,
        ) as frame_capture_device:
            while True:
                frame_start_time = time.time()

                frame = frame_capture_device.get_next_frame()

                annotated_frame = frame.copy()
                card_comparison_panels = []

                # YOLO SEGMENTAÇÃO
                segmentation_start_time = time.time()
                has_tracked_cards, tracked_cards = card_segmenter.track(frame)
                segmentation_time_ms = (time.time() - segmentation_start_time) * 1000

                if has_tracked_cards:
                    for card in tracked_cards:
                        card_polygon_points = card.polygon
                        card_rotated_rectangle = cv2.minAreaRect(card_polygon_points)
                        card_rectangle_corners = cv2.boxPoints(card_rotated_rectangle).astype(int)

                        # converte pontos do polígono para int
                        card_polygon_pixels = card_polygon_points.astype(np.int32)

                        # cria overlay para desenhar a máscara semi-transparente
                        mask_overlay = annotated_frame.copy()
                        mask_color = (255, 200, 30)  # Cor da máscara da carta.
                        cv2.fillPoly(mask_overlay, [card_polygon_pixels], mask_color)

                        mask_opacity = 0.35  # transparência da máscara
                        annotated_frame = cv2.addWeighted(mask_overlay, mask_opacity, annotated_frame, 1 - mask_opacity, 0)

                        # borda da carta (retângulo mínimo) por cima da máscara
                        cv2.polylines(annotated_frame, [card_rectangle_corners], True, (255, 0, 0), 2)

                        card_crop = crop_min_area_rect(frame, card_rotated_rectangle)
                        if card_crop.size == 0:
                            continue

                        # CROP ANTES (SEM PROCESSAMENTO)
                        original_crop_display = cv2.resize(card_crop, CARD_CROP_DISPLAY_SIZE)
                        cv2.putText(
                            original_crop_display, "Antes",
                            (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2, cv2.LINE_AA
                        )

                        # Classificação e correção da orientação.
                        orientation_start_time = time.time()
                        orientation_result = card_orientation_classifier.predict(card_crop)
                        orientation_time_ms = (time.time() - orientation_start_time) * 1000

                        oriented_card_crop = CodeReader.rotate_card_crop(
                            card_crop, orientation_result.angle_degrees
                        )

                        # Caracteres detectados no recorte com orientação corrigida.
                        has_detected_characters, detected_characters = character_detector.predict(
                            oriented_card_crop
                        )

                        detected_code = ""  # string final do código lido

                        if has_detected_characters:
                            for detection in detected_characters:
                                character_left, character_top, character_right, character_bottom = detection.bounding_box
                                character_color = character_class_colors[detection.class_id % DEBUG_CHARACTER_DETECTOR_CLASS_COUNT]

                                # desenha bbox no CROP CORRIGIDO (NÃO NO DE CIMA)
                                cv2.rectangle(oriented_card_crop, (character_left, character_top), (character_right, character_bottom), character_color, 2)

                                cv2.putText(
                                    oriented_card_crop, detection.character,
                                    (character_left, max(0, character_top - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, character_color, 1, cv2.LINE_AA
                                )

                            # ordenar da esquerda para a direita e montar string
                            detected_characters.sort(key=lambda detection: detection.center_x)
                            detected_code = "".join(detection.character for detection in detected_characters)

                            # escreve o código no crop corrigido (embaixo)
                            if detected_code:
                                cv2.putText(
                                    oriented_card_crop, detected_code,
                                    (10, oriented_card_crop.shape[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 255, 0), 2, cv2.LINE_AA
                                )

                        print(f"SEG: {segmentation_time_ms:.2f} ms | CLS: {orientation_time_ms:.2f} ms")

                        # MONTAR PAINEL: CIMA = ANTES | BAIXO = DEPOIS
                        oriented_crop_display = cv2.resize(oriented_card_crop, CARD_CROP_DISPLAY_SIZE)
                        cv2.putText(
                            oriented_crop_display, "Depois",
                            (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2, cv2.LINE_AA
                        )

                        card_comparison_panel = np.vstack([original_crop_display, oriented_crop_display])
                        card_comparison_panels.append(card_comparison_panel)

                frame_processing_time_ms = (time.time() - frame_start_time) * 1000
                print(f"FRAME TOTAL: {frame_processing_time_ms:.2f} ms")

                cv2.imshow("YOLO11 SEG + CardOrientationClassifier", cv2.resize(annotated_frame, FRAME_DISPLAY_SIZE))

                # JANELA DOS CROPS (ANTES / DEPOIS)
                if card_comparison_panels:
                    # cada item de crops já é painel 2x mais alto; empilha lado a lado
                    combined_crop_panel = np.hstack(card_comparison_panels)
                    cv2.imshow("Crops Corrigidos", combined_crop_panel)
                else:
                    # janela vazia: altura = 2 * CARD_CROP_DISPLAY_SIZE[1] (duas linhas)
                    empty_crop_panel = np.zeros((2 * CARD_CROP_DISPLAY_SIZE[1], CARD_CROP_DISPLAY_SIZE[0], 3), dtype=np.uint8)
                    cv2.imshow("Crops Corrigidos", empty_crop_panel)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
