"""Leitura, rastreamento e registro dos códigos das cartas."""

from dataclasses import dataclass

import torch
from models.card_orientation_classifier import CardOrientationClassifier
from models.card_segmenter import CardSegmenter
from models.character_detector import CharacterDetector
from frames.capture import FrameCapture
from frames.processing import crop_card, rotate_card_crop
from frames.visualization import FrameVisualizer

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
class SeenCard:
    """Carta identificada pelo tracking e sua melhor leitura."""

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

        self._seen_cards: dict[int, SeenCard] = {}

    def run(self):
        """Captura e exibe frames até pressionar Q, liberando os recursos ao sair."""
        with (
            FrameCapture(source=CAPTURE_SOURCE, webcam_index=WEBCAM_INDEX, monitor_index=MONITOR_INDEX) as frame_capture,
            FrameVisualizer(display_size=FRAME_DISPLAY_SIZE) as visualizer,
            open(CARD_CODES_LOG_PATH, "a", buffering=1, encoding="utf-8") as card_codes_log_file
        ):
            while True:
                frame = frame_capture.get_next_frame()
                cards = self._detect_cards(frame)
                updates = self._read_card_codes(frame, cards)
                self._write_card_codes_log(card_codes_log_file, updates)
                annotated_frame = visualizer.annotate_frame(frame, cards, self._seen_cards)
                visualizer.show(annotated_frame)
                if visualizer.quit_requested:
                    break

    def _detect_cards(self, frame):
        """Rastreia as cartas e inicializa o histórico das novas detecções."""
        cards = self._card_segmenter.track(frame)
        self._register_seen_cards(cards)
        return cards

    def _read_card_codes(self, frame, cards):
        """Atualiza as melhores leituras e retorna os códigos que mudaram."""
        updates = []
        for card in cards:
            card_id = card.track_id

            card_crop = crop_card(frame, card.polygon)
            if card_crop is None:
                continue

            oriented_card_crop = self._correct_card_orientation(card_crop)

            code, code_confidence = self._read_card_code(oriented_card_crop)
            update = self._update_seen_card(
                card_id, code, code_confidence
            )
            if update is not None:
                updates.append(update)
        return updates

    @staticmethod
    def _write_card_codes_log(log_file, updates):
        """Registra somente as mudanças de código aceitas neste frame."""
        for card_id, code in updates:
            log_file.write(f"{card_id},{code}\n")

    def _register_seen_cards(self, tracked_cards):
        """Inicializa o histórico das cartas novas e preserva as já vistas."""
        for card in tracked_cards:
            card_id = card.track_id
            if card_id not in self._seen_cards:
                self._seen_cards[card_id] = SeenCard(track_id=card_id)

    def _correct_card_orientation(self, card_crop):
        """Identifica a orientação da carta e retorna o recorte corrigido."""
        orientation = self._card_orientation_classifier.predict(card_crop)
        return rotate_card_crop(card_crop, orientation.angle_degrees)

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

    def _update_seen_card(self, card_id, code, confidence):
        """Atualiza a melhor leitura válida e retorna uma mudança de código."""
        card = self._seen_cards[card_id]

        if len(code) != 13:
            return
        if confidence <= card.best_confidence:
            return

        code_changed = code != card.best_code

        card.best_code = code
        card.best_confidence = confidence
        if code_changed:
            return card_id, code
