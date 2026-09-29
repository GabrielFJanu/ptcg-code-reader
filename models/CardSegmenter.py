"""Segmentação e rastreamento de cartas com Ultralytics YOLO."""

from ultralytics import YOLO


class CardSegmenter:
    """Carrega o modelo e mantém as configurações de segmentação e tracking."""

    def __init__(self, weights_path, confidence_threshold, image_size, tracker_config="bytetrack.yaml"):
        self._model = YOLO(weights_path)
        self._confidence_threshold = confidence_threshold
        self._image_size = image_size
        self._tracker_config = tracker_config

    def predict(self, frame):
        """Segmenta um frame sem rastrear as cartas."""
        results = self._model.predict(
            frame,
            conf=self._confidence_threshold,
            imgsz=self._image_size,
            verbose=False,
        )
        result = results[0]
        has_predicted_cards = (
            result.masks is not None
            and result.boxes is not None
            and len(result.boxes) > 0
        )
        return has_predicted_cards, result

    def track(self, frame):
        """Segmenta um frame e mantém o tracking entre chamadas consecutivas."""
        results = self._model.track(
            frame,
            persist=True,
            tracker=self._tracker_config,
            conf=self._confidence_threshold,
            imgsz=self._image_size,
            verbose=False,
        )

        result = results[0]
        has_tracked_cards = (
            result.masks is not None
            and result.boxes is not None
            and result.boxes.id is not None
            and len(result.boxes) > 0
        )
        return has_tracked_cards, result
