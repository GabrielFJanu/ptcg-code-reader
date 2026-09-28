import cv2
import numpy as np
from ultralytics import YOLO
import mss
import torch
from models import CardOrientationClassifier
from torchvision import transforms
from PIL import Image
import time
from utils import generate_random_colors

from config import (
    CARD_SEGMENTER_WEIGHTS_PATH,
    CARD_ORIENTATION_CLASSIFIER_WEIGHTS_PATH,
    DIGIT_DETECTOR_WEIGHTS_PATH,
    CAPTURE_SOURCE,
    CARD_SEGMENTER_CONFIDENCE_THRESHOLD,
    DIGIT_DETECTOR_CONFIDENCE_THRESHOLD,
    CARD_SEGMENTER_IMAGE_SIZE,
    DIGIT_DETECTOR_IMAGE_SIZE,
    FRAME_DISPLAY_SIZE,
    DIGIT_DETECTOR_CLASS_COUNT,
    CARD_CODES_LOG_PATH,
)

# =====================================================
# === CORES ALEATÓRIAS PARA 24 CLASSES ===============
# =====================================================
digit_class_colors = generate_random_colors(DIGIT_DETECTOR_CLASS_COUNT)

# =====================================================
# === CARREGAR MODELOS ================================
# =====================================================
inference_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🧠 Dispositivo ativo:", inference_device)

card_orientation_classifier = CardOrientationClassifier().to(inference_device)
card_orientation_classifier.load_state_dict(torch.load(CARD_ORIENTATION_CLASSIFIER_WEIGHTS_PATH, map_location=inference_device))
card_orientation_classifier.eval()

orientation_preprocessing = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

orientation_class_to_angle = {0: 0, 2: 90, 1: 180, 3: 270}

card_segmenter = YOLO(CARD_SEGMENTER_WEIGHTS_PATH)
digit_detector = YOLO(DIGIT_DETECTOR_WEIGHTS_PATH)

# =====================================================
# === CAPTURA =========================================
# =====================================================
if CAPTURE_SOURCE == "screen":
    screen_capture = mss.mss()
    capture_monitor = screen_capture.monitors[1]
else:
    video_capture = cv2.VideoCapture(CAPTURE_SOURCE)
    if not video_capture.isOpened():
        raise RuntimeError("❌ Erro ao iniciar webcam.")

# =====================================================
# === TRACKING SIMPLES POR CENTROIDE (COM TTL) =======
# =====================================================
def get_or_assign_card_id(card_center_x, card_center_y, card_tracks, frame_index, next_card_id,
                          max_tracking_distance=100, max_missing_frames=15):
    # Remover tracks muito antigos
    expired_card_ids = []
    for tracked_card_id, card_track in card_tracks.items():
        if frame_index - card_track["last_seen"] > max_missing_frames:
            expired_card_ids.append(tracked_card_id)
    for tracked_card_id in expired_card_ids:
        del card_tracks[tracked_card_id]

    closest_card_id = None
    closest_distance_squared = max_tracking_distance * max_tracking_distance

    for tracked_card_id, card_track in card_tracks.items():
        center_offset_x = card_center_x - card_track["cx"]
        center_offset_y = card_center_y - card_track["cy"]
        distance_squared = center_offset_x * center_offset_x + center_offset_y * center_offset_y
        if distance_squared < closest_distance_squared:
            closest_distance_squared = distance_squared
            closest_card_id = tracked_card_id

    if closest_card_id is not None:
        card_tracks[closest_card_id]["cx"] = card_center_x
        card_tracks[closest_card_id]["cy"] = card_center_y
        card_tracks[closest_card_id]["last_seen"] = frame_index
        return closest_card_id, next_card_id

    # Cria novo ID
    tracked_card_id = next_card_id
    card_tracks[tracked_card_id] = {
        "cx": card_center_x,
        "cy": card_center_y,
        "best_code": None,
        "best_prob": 0.0,
        "last_seen": frame_index,
    }
    return tracked_card_id, next_card_id + 1

# =====================================================
# === LOOP PRINCIPAL ==================================
# =====================================================

card_tracks = {}         # ID -> {cx, cy, best_code, best_prob, last_seen}
next_card_id = 0    # próximo ID disponível
frame_index = 0       # contador de frames

# guarda códigos já registrados por ID -> set(códigos)
logged_codes_by_card_id = {}

# abre CSV em append; sem cabeçalho, só "id,codigo"
card_codes_log_file = open(CARD_CODES_LOG_PATH, "a", buffering=1, encoding="utf-8")

while True:
    frame_start_time = time.time()
    frame_index += 1

    if CAPTURE_SOURCE == "screen":
        frame = np.array(screen_capture.grab(capture_monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        frame_read_success, frame = video_capture.read()
        if not frame_read_success:
            break

    annotated_frame = frame.copy()

    # =================================================
    # YOLO SEGMENTAÇÃO
    # =================================================
    card_segmentation_results = card_segmenter(frame, conf=CARD_SEGMENTER_CONFIDENCE_THRESHOLD, imgsz=CARD_SEGMENTER_IMAGE_SIZE, verbose=False)
    card_segmentation_result = card_segmentation_results[0]

    if card_segmentation_result.masks is not None:
        card_mask_data = card_segmentation_result.masks.data.cpu().numpy()   # [N, h, w]
        card_polygons = card_segmentation_result.masks.xy                        # polígonos
        card_box_coordinates = card_segmentation_result.boxes.xyxy.cpu().numpy()   # bounding boxes padrão [N,4]

        for card_index, card_polygon_points in enumerate(card_polygons):
            card_polygon_points = np.array(card_polygon_points, dtype=np.float32)

            # bounding box padrão YOLO (eixo alinhado)
            card_left, card_top, card_right, card_bottom = card_box_coordinates[card_index].astype(int)

            # centro aproximado para tracking
            card_center_x = (card_left + card_right) / 2
            card_center_y = (card_top + card_bottom) / 2

            card_id, next_card_id = get_or_assign_card_id(
                card_center_x, card_center_y, card_tracks, frame_index, next_card_id
            )

            # se ainda não temos set de códigos vistos para esse ID, cria
            if card_id not in logged_codes_by_card_id:
                logged_codes_by_card_id[card_id] = set()

            # recorte da carta a partir da bounding box normal
            card_crop = frame[card_top:card_bottom, card_left:card_right]
            if card_crop.size == 0:
                continue

            # ORIENTATION NET
            card_crop_image = Image.fromarray(cv2.cvtColor(card_crop, cv2.COLOR_BGR2RGB))
            orientation_input_tensor = orientation_preprocessing(card_crop_image).unsqueeze(0).to(inference_device)

            with torch.no_grad():
                orientation_logits = card_orientation_classifier(orientation_input_tensor)
                predicted_orientation_class = torch.argmax(orientation_logits, dim=1).item()
                orientation_angle_degrees = orientation_class_to_angle.get(predicted_orientation_class, 0)

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
                    if code_confidence_product > card_tracks[card_id]["best_prob"]:
                        # só registra se o código ainda não está no set daquele ID
                        if detected_code not in logged_codes_by_card_id[card_id]:
                            logged_codes_by_card_id[card_id].add(detected_code)
                            card_codes_log_file.write(f"{card_id},{detected_code}\n")

                        card_tracks[card_id]["best_code"] = detected_code
                        card_tracks[card_id]["best_prob"] = code_confidence_product

            # DESENHAR MÁSCARA (AMARELA/VERDE)
            best_card_code = card_tracks[card_id]["best_code"]
            best_code_confidence = card_tracks[card_id]["best_prob"]

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

    # =================================================
    # TEMPO TOTAL DO FRAME (única info no console)
    # =================================================
    frame_processing_time_ms = (time.time() - frame_start_time) * 1000
    print(f"FRAME TOTAL: {frame_processing_time_ms:.2f} ms")

    cv2.imshow("YOLO11 SEG + CardOrientationClassifier", cv2.resize(annotated_frame, FRAME_DISPLAY_SIZE))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if CAPTURE_SOURCE != "screen":
    video_capture.release()

card_codes_log_file.close()
cv2.destroyAllWindows()
