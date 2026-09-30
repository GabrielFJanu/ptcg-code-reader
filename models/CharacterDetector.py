"""Detecção dos caracteres que formam o código de uma carta."""

from dataclasses import dataclass

from ultralytics import YOLO


@dataclass(frozen=True)
class CharacterDetectorResult:
    """Caractere detectado, com rótulo e coordenadas no recorte da carta."""

    character: str
    bounding_box: tuple[int, int, int, int]
    class_id: int
    confidence: float

    @property
    def center_x(self):
        left, _, right, _ = self.bounding_box
        return (left + right) / 2


class CharacterDetector:
    """Carrega o modelo e mantém as configurações de detecção de caracteres."""

    def __init__(self, weights_path, device, confidence_threshold, image_size, iou_threshold):
        self._yolo_model = YOLO(weights_path).to(device)
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._image_size = image_size

    def predict(self, card_crop) -> list[CharacterDetectorResult]:
        """Retorna a lista de caracteres detectados com rótulos resolvidos."""
        results = self._yolo_model.predict(
            card_crop,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            imgsz=self._image_size,
            verbose=False,
        )

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        coordinates = result.boxes.xyxy.cpu().numpy().astype(int)
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        detections = [
            CharacterDetectorResult(
                character=str(self._yolo_model.names[int(class_id)]),
                bounding_box=tuple(int(coordinate) for coordinate in bounding_box),
                class_id=int(class_id),
                confidence=float(confidence),
            )
            for bounding_box, class_id, confidence in zip(
                coordinates, class_ids, confidences
            )
        ]
        return detections
