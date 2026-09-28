"""Segmentação e rastreamento de cartas com Ultralytics YOLO."""

from ultralytics import YOLO


class CardSegmentationResult:
    """Resultado de um frame, com acesso às máscaras, caixas e tracking."""

    def __init__(self, yolo_result):
        self._yolo_result = yolo_result

    @property
    def masks(self):
        return self._yolo_result.masks

    @property
    def boxes(self):
        return self._yolo_result.boxes

    @property
    def has_tracked_cards(self):
        """Indica se há máscaras e IDs de rastreamento disponíveis."""
        return (
            self.masks is not None
            and self.boxes is not None
            and self.boxes.id is not None
            and len(self.boxes) > 0
        )


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
        return CardSegmentationResult(results[0])

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
        return CardSegmentationResult(results[0])
