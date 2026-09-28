import cv2
import numpy as np
from ultralytics import YOLO
import torch
from models.CardOrientationClassifier import CardOrientationClassifier
from models.CardSegmenter import CardSegmenter
from PIL import Image
import time
from utils import generate_random_colors
from capture import CaptureDevice

from config import (
    CARD_SEGMENTER_WEIGHTS_PATH,
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
    DEBUG_DIGIT_DETECTOR_CLASS_COUNT,
    CARD_CROP_DISPLAY_SIZE,
)

# =====================================================
# === CORES ALEATÓRIAS PARA 25 CLASSES ===============
# =====================================================
digit_class_colors = generate_random_colors(DEBUG_DIGIT_DETECTOR_CLASS_COUNT)

# =====================================================
# === CARREGAR MODELOS ================================
# =====================================================
inference_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🧠 Dispositivo ativo:", inference_device)

card_orientation_classifier = CardOrientationClassifier.from_weights(
    CARD_ORIENTATION_CLASSIFIER_WEIGHTS_PATH,
    device=inference_device,
)

card_segmenter = CardSegmenter(
    weights_path=CARD_SEGMENTER_WEIGHTS_PATH,
    confidence_threshold=CARD_SEGMENTER_CONFIDENCE_THRESHOLD,
    image_size=CARD_SEGMENTER_IMAGE_SIZE,
)
digit_detector = YOLO(DIGIT_DETECTOR_WEIGHTS_PATH)

# =====================================================
# === FUNÇÃO DE CROP ROTACIONADO ======================
# =====================================================
def crop_min_area_rect(source_image, rotated_rectangle):
    (rectangle_center_x, rectangle_center_y), (rectangle_width, rectangle_height), rectangle_angle_degrees = rotated_rectangle
    rotation_matrix = cv2.getRotationMatrix2D((rectangle_center_x, rectangle_center_y), rectangle_angle_degrees, 1.0)
    rotated_image = cv2.warpAffine(source_image, rotation_matrix, (source_image.shape[1], source_image.shape[0]), flags=cv2.INTER_LINEAR)
    crop_left = max(0, int(rectangle_center_x - rectangle_width / 2))
    crop_top = max(0, int(rectangle_center_y - rectangle_height / 2))
    crop_right = min(source_image.shape[1], int(rectangle_center_x + rectangle_width / 2))
    crop_bottom = min(source_image.shape[0], int(rectangle_center_y + rectangle_height / 2))
    return rotated_image[crop_top:crop_bottom, crop_left:crop_right]

# =====================================================
# === LOOP PRINCIPAL ==================================
# =====================================================
try:
    with CaptureDevice(
        source=CAPTURE_SOURCE,
        webcam_index=WEBCAM_INDEX,
        monitor_index=MONITOR_INDEX,
    ) as capture:
        while True:
            frame_start_time = time.time()

            frame = capture.get_next_frame()

            annotated_frame = frame.copy()
            card_comparison_panels = []

            # =================================================
            # YOLO SEGMENTAÇÃO
            # =================================================
            segmentation_start_time = time.time()
            card_segmentation_result = card_segmenter.predict(frame)
            segmentation_time_ms = (time.time() - segmentation_start_time) * 1000

            if card_segmentation_result.masks is not None:
                for card_polygon in card_segmentation_result.masks.xy:

                    card_polygon_points = np.array(card_polygon, dtype=np.float32)
                    card_rotated_rectangle = cv2.minAreaRect(card_polygon_points)
                    card_rectangle_corners = cv2.boxPoints(card_rotated_rectangle).astype(int)

                    # ================== MÁSCARA DA CARTA NO FRAME ==================
                    # converte pontos do polígono para int
                    card_polygon_pixels = card_polygon_points.astype(np.int32)

                    # cria overlay para desenhar a máscara semi-transparente
                    mask_overlay = annotated_frame.copy()
                    mask_color = (255, 200, 30)  # verde, pode trocar se quiser
                    cv2.fillPoly(mask_overlay, [card_polygon_pixels], mask_color)

                    mask_opacity = 0.35  # transparência da máscara
                    annotated_frame = cv2.addWeighted(mask_overlay, mask_opacity, annotated_frame, 1 - mask_opacity, 0)

                    # borda da carta (retângulo mínimo) por cima da máscara
                    cv2.polylines(annotated_frame, [card_rectangle_corners], True, (255, 0, 0), 2)

                    # ================== CROP DA CARTA ==================
                    card_crop = crop_min_area_rect(frame, card_rotated_rectangle)
                    if card_crop.size == 0:
                        continue

                    # =================================================
                    # CROP ANTES (SEM PROCESSAMENTO)
                    # =================================================
                    original_crop_display = cv2.resize(card_crop, CARD_CROP_DISPLAY_SIZE)
                    cv2.putText(
                        original_crop_display, "Antes",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA
                    )

                    # =================================================
                    # ORIENTATION NET
                    # =================================================
                    card_crop_image = Image.fromarray(cv2.cvtColor(card_crop, cv2.COLOR_BGR2RGB))
                    orientation_input_tensor = card_orientation_classifier.input_image_transform(card_crop_image).unsqueeze(0).to(inference_device)

                    orientation_start_time = time.time()
                    with torch.no_grad():
                        orientation_logits = card_orientation_classifier(orientation_input_tensor)
                        predicted_orientation_class = torch.argmax(orientation_logits, dim=1).item()
                        orientation_angle_degrees = card_orientation_classifier.orientation_angle_by_class_id.get(predicted_orientation_class, 0)
                    orientation_time_ms = (time.time() - orientation_start_time) * 1000

                    if orientation_angle_degrees == 90:
                        oriented_card_crop = cv2.rotate(card_crop, cv2.ROTATE_90_CLOCKWISE)
                    elif orientation_angle_degrees == 180:
                        oriented_card_crop = cv2.rotate(card_crop, cv2.ROTATE_180)
                    elif orientation_angle_degrees == 270:
                        oriented_card_crop = cv2.rotate(card_crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    else:
                        oriented_card_crop = card_crop

                    # =================================================
                    # YOLO DÍGITOS NO CROP CORRIGIDO (APENAS AQUI)
                    # =================================================
                    digit_detection_results = digit_detector(
                        oriented_card_crop, conf=DIGIT_DETECTOR_CONFIDENCE_THRESHOLD, imgsz=DIGIT_DETECTOR_IMAGE_SIZE, iou=0.7, verbose=False
                    )
                    digit_detection_result = digit_detection_results[0]

                    detected_code = ""  # string final do código lido

                    if digit_detection_result.boxes is not None and len(digit_detection_result.boxes) > 0:
                        digit_boxes = digit_detection_result.boxes
                        digit_box_coordinates = digit_boxes.xyxy.cpu().numpy().astype(int)
                        digit_class_ids = digit_boxes.cls.cpu().numpy().astype(int)

                        digit_class_names = digit_detector.names

                        # lista para ordenar os dígitos pelo eixo x (esquerda -> direita)
                        detected_digits = []

                        for (digit_left, digit_top, digit_right, digit_bottom), digit_class_id in zip(digit_box_coordinates, digit_class_ids):
                            digit_color = digit_class_colors[digit_class_id % DEBUG_DIGIT_DETECTOR_CLASS_COUNT]

                            # desenha bbox no CROP CORRIGIDO (NÃO NO DE CIMA)
                            cv2.rectangle(oriented_card_crop, (digit_left, digit_top), (digit_right, digit_bottom), digit_color, 2)

                            if isinstance(digit_class_names, dict):
                                digit_label = digit_class_names.get(digit_class_id, str(digit_class_id))
                            else:
                                digit_label = digit_class_names[digit_class_id] if digit_class_id < len(digit_class_names) else str(digit_class_id)

                            digit_label = str(digit_label)

                            cv2.putText(
                                oriented_card_crop, digit_label,
                                (digit_left, max(0, digit_top - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, digit_color, 1, cv2.LINE_AA
                            )

                            # centro em x pra ordenar, mais o caractere
                            digit_center_x = (digit_left + digit_right) / 2.0
                            detected_digits.append((digit_center_x, digit_label))

                        # ordenar da esquerda para a direita e montar string
                        detected_digits.sort(key=lambda digit_item: digit_item[0])
                        detected_code = "".join(digit_character for _, digit_character in detected_digits)

                        # escreve o código no crop corrigido (embaixo)
                        if detected_code:
                            cv2.putText(
                                oriented_card_crop, detected_code,
                                (10, oriented_card_crop.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2, cv2.LINE_AA
                            )

                    print(f"SEG: {segmentation_time_ms:.2f} ms | CLS: {orientation_time_ms:.2f} ms")

                    # =================================================
                    # MONTAR PAINEL: CIMA = ANTES | BAIXO = DEPOIS
                    # =================================================
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

            # =================================================
            # JANELA DOS CROPS (ANTES / DEPOIS)
            # =================================================
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
