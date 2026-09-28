import os
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from models.CardOrientationClassifier import CardOrientationClassifier
from models.CardSegmenter import CardSegmenter
from PIL import Image
from utils import generate_random_colors
from capture import CaptureDevice

from config import (
    CARD_SEGMENTER_WEIGHTS_PATH,
    CARD_TRACKER_CONFIG,
    CARD_ORIENTATION_CLASSIFIER_WEIGHTS_PATH,
    DIGIT_DETECTOR_WEIGHTS_PATH,
    CAPTURE_SOURCE,
    WEBCAM_INDEX,
    MONITOR_INDEX,
    CARD_SEGMENTER_CONFIDENCE_THRESHOLD,
    DIGIT_DETECTOR_CONFIDENCE_THRESHOLD,
    CARD_SEGMENTER_IMAGE_SIZE,
    DIGIT_DETECTOR_IMAGE_SIZE,
    FRAME_DISPLAY_SIZE,
    DIGIT_DETECTOR_CLASS_COUNT,
    CARD_CODES_LOG_PATH,
)

# CORES ALEATÓRIAS PARA CADA DIGITO
digit_class_colors = generate_random_colors(DIGIT_DETECTOR_CLASS_COUNT)

# DISPOSITIVO DE INFÊRENCIA
inference_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo de inferência:", inference_device)

# CARREGAR MODELOS
card_orientation_classifier = CardOrientationClassifier.from_weights(
    CARD_ORIENTATION_CLASSIFIER_WEIGHTS_PATH,
    device=inference_device,
)
card_segmenter = CardSegmenter(
    weights_path=CARD_SEGMENTER_WEIGHTS_PATH,
    confidence_threshold=CARD_SEGMENTER_CONFIDENCE_THRESHOLD,
    image_size=CARD_SEGMENTER_IMAGE_SIZE,
    tracker_config=CARD_TRACKER_CONFIG,
)
digit_detector = YOLO(DIGIT_DETECTOR_WEIGHTS_PATH)

# =====================================================
# === LOOP PRINCIPAL ==================================
# =====================================================

card_readings_by_id = {}  # ID do ByteTrack -> melhor código e confiança

# guarda códigos já registrados por ID -> set(códigos)
logged_codes_by_card_id = {}

try:
    with CaptureDevice(
        source=CAPTURE_SOURCE,
        webcam_index=WEBCAM_INDEX,
        monitor_index=MONITOR_INDEX,
    ) as capture, open(CARD_CODES_LOG_PATH, "a", buffering=1, encoding="utf-8") as card_codes_log_file:
        while True:

            frame = capture.get_next_frame()

            annotated_frame = frame.copy()

            # CARD SEGMENTATION
            card_segmentation_result = card_segmenter.track(frame)

            if card_segmentation_result.has_tracked_cards:
                card_track_ids = card_segmentation_result.boxes.id.int().cpu().tolist()
                card_polygons = card_segmentation_result.masks.xy                        # polígonos
                card_box_coordinates = card_segmentation_result.boxes.xyxy.cpu().numpy()   # bounding boxes padrão [N,4]

                for card_index, card_polygon_points in enumerate(card_polygons):
                    card_polygon_points = np.array(card_polygon_points, dtype=np.float32)

                    # bounding box padrão YOLO (eixo alinhado)
                    card_left, card_top, card_right, card_bottom = card_box_coordinates[card_index].astype(int)

                    card_id = card_track_ids[card_index]
                    if card_id not in card_readings_by_id:
                        card_readings_by_id[card_id] = {
                            "best_code": None,
                            "best_prob": 0.0,
                        }

                    # se ainda não temos set de códigos vistos para esse ID, cria
                    if card_id not in logged_codes_by_card_id:
                        logged_codes_by_card_id[card_id] = set()

                    # recorte da carta a partir da bounding box normal
                    card_crop = frame[card_top:card_bottom, card_left:card_right]
                    if card_crop.size == 0:
                        continue

                    # ORIENTATION NET
                    card_crop_image = Image.fromarray(cv2.cvtColor(card_crop, cv2.COLOR_BGR2RGB))
                    orientation_input_tensor = card_orientation_classifier.input_image_transform(card_crop_image).unsqueeze(0).to(inference_device)

                    with torch.no_grad():
                        orientation_logits = card_orientation_classifier(orientation_input_tensor)
                        predicted_orientation_class = torch.argmax(orientation_logits, dim=1).item()
                        orientation_angle_degrees = card_orientation_classifier.orientation_angle_by_class_id.get(predicted_orientation_class, 0)

                    if orientation_angle_degrees == 90:
                        oriented_card_crop = cv2.rotate(card_crop, cv2.ROTATE_90_CLOCKWISE)
                    elif orientation_angle_degrees == 180:
                        oriented_card_crop = cv2.rotate(card_crop, cv2.ROTATE_180)
                    elif orientation_angle_degrees == 270:
                        oriented_card_crop = cv2.rotate(card_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    else:
                        oriented_card_crop = card_crop

                    # YOLO DIGITS
                    digit_detection_results = digit_detector(
                        oriented_card_crop,
                        conf=DIGIT_DETECTOR_CONFIDENCE_THRESHOLD,
                        imgsz=DIGIT_DETECTOR_IMAGE_SIZE,
                        iou=0.8,
                        verbose=False
                    )
                    digit_detection_result = digit_detection_results[0]

                    detected_code = ""
                    code_confidence_product = None

                    if digit_detection_result.boxes is not None and len(digit_detection_result.boxes) > 0:
                        digit_boxes = digit_detection_result.boxes
                        digit_box_coordinates = digit_boxes.xyxy.cpu().numpy().astype(int)
                        digit_class_ids = digit_boxes.cls.cpu().numpy().astype(int)
                        digit_confidences = digit_boxes.conf.cpu().numpy()

                        digit_class_names = digit_detector.names
                        detected_digits = []

                        for (digit_left, digit_top, digit_right, digit_bottom), digit_class_id, digit_confidence in zip(digit_box_coordinates, digit_class_ids, digit_confidences):
                            digit_label = digit_class_names[digit_class_id] if digit_class_id < len(digit_class_names) else str(digit_class_id)
                            digit_center_x = (digit_left + digit_right) / 2
                            clamped_digit_confidence = float(digit_confidence)
                            clamped_digit_confidence = max(1e-6, min(1.0, clamped_digit_confidence))
                            detected_digits.append((digit_center_x, str(digit_label), clamped_digit_confidence))

                        detected_digits.sort(key=lambda digit_item: digit_item[0])
                        detected_code = "".join(digit_character for _, digit_character, _ in detected_digits)

                        code_confidence_product = 1.0
                        for _, _, digit_confidence_factor in detected_digits:
                            code_confidence_product *= digit_confidence_factor

                        if len(detected_code) == 13:
                            # se for um novo melhor código E ainda não foi registrado, atualiza e loga
                            if code_confidence_product > card_readings_by_id[card_id]["best_prob"]:
                                # só registra se o código ainda não está no set daquele ID
                                if detected_code not in logged_codes_by_card_id[card_id]:
                                    logged_codes_by_card_id[card_id].add(detected_code)
                                    card_codes_log_file.write(f"{card_id},{detected_code}\n")

                                card_readings_by_id[card_id]["best_code"] = detected_code
                                card_readings_by_id[card_id]["best_prob"] = code_confidence_product

                    # DESENHAR MÁSCARA (AMARELA/VERDE)
                    best_card_code = card_readings_by_id[card_id]["best_code"]
                    best_code_confidence = card_readings_by_id[card_id]["best_prob"]

                    mask_color = (0,255,255) if best_card_code is None else (0,255,0)
                    card_polygon_pixels = np.array(card_polygon_points, dtype=np.int32)

                    mask_overlay = annotated_frame.copy()
                    cv2.fillPoly(mask_overlay, [card_polygon_pixels], mask_color)
                    annotated_frame = cv2.addWeighted(mask_overlay, 0.25, annotated_frame, 0.75, 0)

                    # BOUNDING BOX PADRÃO
                    cv2.rectangle(
                        annotated_frame,
                        (card_left, card_top),
                        (card_right, card_bottom),
                        (0,255,255) if best_card_code is None else (0,255,0),
                        1
                    )

                    # ID da carta
                    cv2.putText(
                        annotated_frame, f"ID {card_id}",
                        (card_left, max(0, card_top - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2, cv2.LINE_AA
                    )

                    # Melhor código (se quiser manter na tela)
                    if best_card_code is not None:
                        best_code_confidence_percent = best_code_confidence * 100
                        cv2.putText(
                            annotated_frame, f"{best_card_code} ({best_code_confidence_percent:.1f}%)",
                            (card_left, max(0, card_top - 30)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 255, 0), 2, cv2.LINE_AA
                        )

            cv2.imshow("YOLO11 SEG + CardOrientationClassifier", cv2.resize(annotated_frame, FRAME_DISPLAY_SIZE))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    cv2.destroyAllWindows()
