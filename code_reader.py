"""Leitura, rastreamento e registro dos códigos das cartas."""

from dataclasses import dataclass

import torch

from config.config import (
    CAPTURE_SOURCE,
    CARD_CODES_LOG_PATH,
    CARD_ORIENTATION_CLASSIFIER_WEIGHTS_PATH,
    CARD_SEGMENTER_CONFIDENCE_THRESHOLD,
    CARD_SEGMENTER_INFERENCE_IMAGE_SIZE,
    CARD_SEGMENTER_WEIGHTS_PATH,
    CARD_TRACKER_CONFIG,
    CHARACTER_DETECTOR_CONFIDENCE_THRESHOLD,
    CHARACTER_DETECTOR_INFERENCE_IMAGE_SIZE,
    CHARACTER_DETECTOR_IOU_THRESHOLD,
    CHARACTER_DETECTOR_WEIGHTS_PATH,
    FRAME_DISPLAY_SIZE,
    MONITOR_INDEX,
    WEBCAM_INDEX,
)
from frames.annotation import annotate_frame
from frames.capture import FrameCapture
from frames.processing import crop_card, rotate_card_crop
from frames.display import FrameDisplay
from logs.codes_log_writer import CodesLogWriter
from models.card_orientation_classifier import CardOrientationClassifier
from models.card_segmenter import CardSegmenter, TrackedCard
from models.character_detector import CharacterDetector


_EXPECTED_CODE_LENGTH = 13


@dataclass
class CardReading:
    """Leitura de código de uma carta identificada pelo tracking."""

    track_id: int
    code: str
    confidence: float


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
            inference_image_size=CARD_SEGMENTER_INFERENCE_IMAGE_SIZE,
            tracker_config=CARD_TRACKER_CONFIG,
        )
        self._character_detector = CharacterDetector(
            weights_path=CHARACTER_DETECTOR_WEIGHTS_PATH,
            device=self._device,
            confidence_threshold=CHARACTER_DETECTOR_CONFIDENCE_THRESHOLD,
            iou_threshold=CHARACTER_DETECTOR_IOU_THRESHOLD,
            inference_image_size=CHARACTER_DETECTOR_INFERENCE_IMAGE_SIZE,
        )

        self._best_card_readings_by_track_id: dict[int, CardReading] = {}

    def run(self):
        """Captura e exibe frames até pressionar Q, liberando os recursos ao sair."""
        with (
            FrameCapture(
                source=CAPTURE_SOURCE,
                webcam_index=WEBCAM_INDEX,
                monitor_index=MONITOR_INDEX,
            ) as frame_capture,
            CodesLogWriter(CARD_CODES_LOG_PATH) as codes_log_writer,
            FrameDisplay(display_size=FRAME_DISPLAY_SIZE) as frame_display,
        ):
            while True:
                frame = frame_capture.get_next_frame()
                tracked_cards = self._card_segmenter.track(frame)
                card_readings = self._read_card_codes(frame, tracked_cards)
                changed_code_readings = self._update_best_card_readings(card_readings)
                codes_log_writer.write_readings(changed_code_readings)
                annotated_frame = annotate_frame(
                    frame, tracked_cards, self._best_card_readings_by_track_id
                )
                frame_display.show(annotated_frame)
                if frame_display.quit_requested:
                    break

    def _read_card_codes(
        self, frame, tracked_cards: list[TrackedCard]
    ) -> list[CardReading]:
        """Reúne as leituras obtidas para as cartas do frame."""
        card_readings = []
        for tracked_card in tracked_cards:
            card_reading = self._read_card_code(frame, tracked_card)
            if card_reading is not None:
                card_readings.append(card_reading)
        return card_readings

    def _update_best_card_readings(
        self, card_readings: list[CardReading]
    ) -> list[CardReading]:
        """Atualiza as melhores leituras e retorna aquelas cujo código mudou."""
        changed_code_readings = []
        for card_reading in card_readings:
            changed_code_reading = self._update_best_card_reading(card_reading)
            if changed_code_reading is not None:
                changed_code_readings.append(changed_code_reading)
        return changed_code_readings

    def _correct_card_orientation(self, card_crop):
        """Identifica a orientação da carta e retorna o recorte corrigido."""
        predicted_orientation = self._card_orientation_classifier.predict(card_crop)
        return rotate_card_crop(card_crop, predicted_orientation.angle_degrees)

    def _read_card_code(self, frame, tracked_card: TrackedCard) -> CardReading | None:
        """Recorta, orienta e lê a carta; sem recorte ou caracteres, retorna None."""
        card_crop = crop_card(frame, tracked_card.polygon)
        if card_crop is None:
            return None

        oriented_card_crop = self._correct_card_orientation(card_crop)
        character_detections = self._character_detector.predict(oriented_card_crop)

        if not character_detections:
            return None

        character_detections.sort(key=lambda detection: detection.center_x)
        code = "".join(detection.character for detection in character_detections)

        confidence_product = 1.0
        for detection in character_detections:
            confidence_product *= detection.confidence

        return CardReading(
            track_id=tracked_card.track_id, code=code, confidence=confidence_product
        )

    def _update_best_card_reading(self, reading: CardReading) -> CardReading | None:
        """Atualiza a melhor leitura válida e a retorna apenas se o código mudou."""
        if len(reading.code) != _EXPECTED_CODE_LENGTH:
            return None

        best_card_reading = self._best_card_readings_by_track_id.get(reading.track_id)
        if (
            best_card_reading is not None
            and reading.confidence <= best_card_reading.confidence
        ):
            return None

        code_changed = best_card_reading is None or reading.code != best_card_reading.code
        self._best_card_readings_by_track_id[reading.track_id] = reading
        if code_changed:
            return reading
        return None
