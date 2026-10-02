"""Configurações da leitura de códigos."""

# Pesos dos modelos.
CARD_SEGMENTER_WEIGHTS_PATH = "weights/card_segmenter.pt"
CARD_ORIENTATION_CLASSIFIER_WEIGHTS_PATH = "weights/card_orientation_classifier.pth"
CHARACTER_DETECTOR_WEIGHTS_PATH = "weights/character_detector.pt"

# Captura e inferência.
CAPTURE_SOURCE = "webcam"  # "webcam" ou "screen".
WEBCAM_INDEX = 0  # Índice da câmera usada quando a fonte é "webcam".
MONITOR_INDEX = 1  # 1 = primeiro monitor; 0 = área de todos os monitores.
CARD_SEGMENTER_CONFIDENCE_THRESHOLD = 0.5
CHARACTER_DETECTOR_CONFIDENCE_THRESHOLD = 0.5
CARD_SEGMENTER_INFERENCE_IMAGE_SIZE = 448
CHARACTER_DETECTOR_INFERENCE_IMAGE_SIZE = 640
CHARACTER_DETECTOR_IOU_THRESHOLD = 0.8

# Tracker integrado ao Ultralytics (configuração incluída na biblioteca).
CARD_TRACKER_CONFIG = "bytetrack.yaml"

# Visualização (largura, altura).
FRAME_DISPLAY_SIZE = (840, 560)

# Arquivo de log (id,codigo).
CARD_CODES_LOG_PATH = "card_codes_log.csv"
