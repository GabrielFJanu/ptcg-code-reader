"""Segmentação e rastreamento de cartas com Ultralytics YOLO."""

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class CardSegmentationResult:
    """Carta segmentada, com coordenadas no frame e ID de tracking."""

    polygon: np.ndarray
    bounding_box: tuple[int, int, int, int]
    class_id: int
    confidence: float
    track_id: int


class CardSegmenter:
    """Carrega o modelo e mantém as configurações de segmentação e tracking."""

    def __init__(self, weights_path, confidence_threshold, image_size, tracker_config="bytetrack.yaml"):
        self._yolo_model = YOLO(weights_path)
        self._confidence_threshold = confidence_threshold
        self._image_size = image_size
        self._tracker_config = tracker_config

    def track(self, frame):
        """Retorna cartas com IDs, mantendo o tracking entre chamadas."""
        results = self._yolo_model.track(
            frame,
            persist=True,
            tracker=self._tracker_config,
            conf=self._confidence_threshold,
            imgsz=self._image_size,
            verbose=False,
        )

        result = results[0]
        if (
            result.masks is None
            or result.boxes is None
            or len(result.boxes) == 0
            or result.boxes.id is None
        ):
            return False, []

        coordinates = result.boxes.xyxy.cpu().numpy().astype(int)
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()
        cards = [
            CardSegmentationResult(
                polygon=np.array(polygon, dtype=np.float32),
                bounding_box=tuple(int(coordinate) for coordinate in bounding_box),
                class_id=int(class_id),
                confidence=float(confidence),
                track_id=track_id,
            )
            for bounding_box, polygon, class_id, confidence, track_id in zip(
                coordinates, result.masks.xy, class_ids, confidences, track_ids
            )
        ]
        return True, cards
