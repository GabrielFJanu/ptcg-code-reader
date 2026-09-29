"""Detecção dos caracteres que formam o código de uma carta."""

from ultralytics import YOLO


class DigitDetector:
    """Carrega o modelo e mantém as configurações de detecção de caracteres."""

    def __init__(self, weights_path, confidence_threshold, image_size, iou_threshold):
        self._model = YOLO(weights_path)
        self._confidence_threshold = confidence_threshold
        self._image_size = image_size
        self._iou_threshold = iou_threshold

    def predict(self, card_crop):
        """Detecta caracteres e retorna a disponibilidade junto do resultado."""
        results = self._model.predict(
            card_crop,
            conf=self._confidence_threshold,
            imgsz=self._image_size,
            iou=self._iou_threshold,
            verbose=False,
        )

        result = results[0]
        has_detected_digits = result.boxes is not None and len(result.boxes) > 0
        return has_detected_digits, result

    def get_class_label(self, class_id):
        """Retorna o rótulo textual de uma classe detectada."""
        return str(self._model.names[class_id])
