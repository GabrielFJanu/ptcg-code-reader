"""Leitura, rastreamento e registro dos códigos das cartas."""

from dataclasses import dataclass

import cv2
import torch
from models.card_orientation_classifier import CardOrientationClassifier
from models.card_segmenter import CardSegmenter
from models.character_detector import CharacterDetector
from capture import FrameCapture
from visualization import FrameVisualizer

from config.config import (
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


@dataclass
class CardReadingHistory:
    """Histórico de leitura de uma carta identificada pelo tracking."""

    track_id: int
    best_code: str | None = None
    best_confidence: float = 0.0


class CodeReader:
    """Coordena os modelos, a captura e o histórico de leitura de cada carta."""

    def __init__(self):
        """Carrega os modelos e inicializa o histórico usando as configurações."""
        # Dispositivo.
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Dispositivo de inferência:", self._device)

        # Carregamento dos modelos.
        self._card_orientation_classifier = CardOrientationClassifier(
            weights_path=CARD_ORIENTATION_CLASSIFIER_WEIGHTS_PATH,
            device=self._device,
        )
        self._card_segmenter = CardSegmenter(
            weights_path=CARD_SEGMENTER_WEIGHTS_PATH,
            device=self._device,
            confidence_threshold=CARD_SEGMENTER_CONFIDENCE_THRESHOLD,
            image_size=CARD_SEGMENTER_IMAGE_SIZE,
            tracker_config=CARD_TRACKER_CONFIG,
        )
        self._character_detector = CharacterDetector(
            weights_path=CHARACTER_DETECTOR_WEIGHTS_PATH,
            device=self._device,
            confidence_threshold=CHARACTER_DETECTOR_CONFIDENCE_THRESHOLD,
            iou_threshold=CHARACTER_DETECTOR_IOU_THRESHOLD,
            image_size=CHARACTER_DETECTOR_IMAGE_SIZE,
        )

        self._seen_cards: dict[int, CardReadingHistory] = {}

    def run(self):
        """Captura e exibe frames até pressionar Q, liberando os recursos ao sair."""
        with (
            FrameCapture(source=CAPTURE_SOURCE, webcam_index=WEBCAM_INDEX, monitor_index=MONITOR_INDEX) as frame_capture,
            FrameVisualizer(display_size=FRAME_DISPLAY_SIZE) as visualizer,
            open(CARD_CODES_LOG_PATH, "a", buffering=1, encoding="utf-8") as card_codes_log_file
        ):
            while True:
                frame = frame_capture.get_next_frame()
                annotated_frame = self._process_frame(
                    frame, card_codes_log_file, visualizer
                )
                visualizer.show(annotated_frame)
                if visualizer.quit_requested:
                    break

    def _process_frame(self, frame, log_file, visualizer):
        """Rastreia as cartas, atualiza suas leituras e retorna o frame anotado."""

        annotated_frame = frame.copy()

        tracked_cards = self._card_segmenter.track(frame)

        if not tracked_cards:
            return annotated_frame

        self._update_seen_cards(tracked_cards)

        for card in tracked_cards:
            card_id = card.track_id

            card_crop = self._crop_card(frame, card.polygon)
            if card_crop is None:
                continue

            oriented_card_crop = self._correct_card_orientation(card_crop)

            code, code_confidence = self._read_card_code(oriented_card_crop)
            self._update_best_card_reading_history(
                card_id, code, code_confidence, log_file
            )
            annotated_frame = visualizer.draw_card(
                annotated_frame, card, self._seen_cards[card_id]
            )

        return annotated_frame

    def _update_seen_cards(self, tracked_cards):
        """Inicializa o histórico das cartas novas e preserva as já vistas."""
        for card in tracked_cards:
            card_id = card.track_id
            if card_id not in self._seen_cards:
                self._seen_cards[card_id] = CardReadingHistory(track_id=card_id)

    @staticmethod
    def _crop_card(frame, polygon):
        """Alinha o retângulo mínimo da carta e retorna None para recortes vazios."""
        if len(polygon) < 3:
            return None

        rectangle_center, rectangle_size, angle_degrees = cv2.minAreaRect(polygon)
        center_x, center_y = rectangle_center
        width, height = rectangle_size
        if width <= 0 or height <= 0:
            return None

        rotation_matrix = cv2.getRotationMatrix2D(
            rectangle_center, angle_degrees, 1.0
        )
        rotated_frame = cv2.warpAffine(
            frame,
            rotation_matrix,
            (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_LINEAR,
        )
        left = max(0, int(center_x - width / 2))
        top = max(0, int(center_y - height / 2))
        right = min(frame.shape[1], int(center_x + width / 2))
        bottom = min(frame.shape[0], int(center_y + height / 2))
        card_crop = rotated_frame[top:bottom, left:right]
        return card_crop if card_crop.size > 0 else None

    def _correct_card_orientation(self, card_crop):
        """Identifica a orientação da carta e retorna o recorte corrigido."""
        orientation = self._card_orientation_classifier.predict(card_crop)
        return self._rotate_card_crop(card_crop, orientation.angle_degrees)

    @staticmethod
    def _rotate_card_crop(card_crop, angle_degrees):
        """Aplica a rotação prevista pelo classificador ao recorte da carta."""
        rotation_by_angle = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }
        rotation = rotation_by_angle.get(angle_degrees)
        if rotation is None:
            return card_crop
        return cv2.rotate(card_crop, rotation)

    def _read_card_code(self, card_crop):
        """Monta o código da esquerda para a direita e multiplica as confianças."""

        characters = self._character_detector.predict(card_crop)

        if not characters:
            return "", None

        characters.sort(key=lambda detection: detection.center_x)
        code = "".join(detection.character for detection in characters)

        confidence_product = 1.0
        for detection in characters:
            confidence_product *= detection.confidence

        return code, confidence_product

    def _update_best_card_reading_history(self, card_id, code, confidence, log_file):
        """Atualiza a melhor leitura válida e registra mudanças no melhor código."""
        card = self._seen_cards[card_id]

        if len(code) != 13:
            return
        if confidence <= card.best_confidence:
            return

        if code != card.best_code:
            log_file.write(f"{card_id},{code}\n")

        card.best_code = code
        card.best_confidence = confidence
