"""Segmentação e rastreamento de cartas com Ultralytics YOLO."""

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class SegmentedCard:
    """Carta segmentada, com coordenadas no frame e ID de tracking."""

    polygon: np.ndarray
    bounding_box: tuple[int, int, int, int]
    class_id: int
    track_id: int
    confidence: float


class CardSegmenter:
    """Carrega o modelo e mantém as configurações de segmentação e tracking."""

    def __init__(
        self,
        weights_path,
        device,
        confidence_threshold,
        inference_image_size,
        tracker_config="bytetrack.yaml",
    ):
        self._yolo_model = YOLO(weights_path).to(device)
        self._confidence_threshold = confidence_threshold
        self._inference_image_size = inference_image_size
        self._tracker_config = tracker_config

    def track(self, frame) -> list[SegmentedCard]:
        """Retorna cartas com IDs, mantendo o tracking entre chamadas."""
        frame_results = self._yolo_model.track(
            frame,
            persist=True,
            tracker=self._tracker_config,
            conf=self._confidence_threshold,
            imgsz=self._inference_image_size,
            verbose=False,
        )

        frame_result = frame_results[0]
        if (
            frame_result.masks is None
            or frame_result.boxes is None
            or len(frame_result.boxes) == 0
            or not frame_result.boxes.is_track
        ):
            return []

        bounding_boxes = frame_result.boxes.xyxy.cpu().numpy().astype(int)
        class_ids = frame_result.boxes.cls.cpu().numpy().astype(int)
        confidences = frame_result.boxes.conf.cpu().numpy()
        track_ids = frame_result.boxes.id.int().cpu().tolist()
        segmented_cards = [
            SegmentedCard(
                polygon=np.array(polygon, dtype=np.float32),
                bounding_box=tuple(int(coordinate) for coordinate in bounding_box),
                class_id=int(class_id),
                track_id=track_id,
                confidence=float(confidence),
            )
            for bounding_box, polygon, class_id, confidence, track_id in zip(
                bounding_boxes, frame_result.masks.xy, class_ids, confidences, track_ids
            )
        ]
        return segmented_cards
