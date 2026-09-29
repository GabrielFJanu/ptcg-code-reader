"""Configurações dos pipelines de leitura de códigos."""

# Pesos compartilhados pelos pipelines principal e de debug.
CARD_SEGMENTER_WEIGHTS_PATH = "weights/card_segmenter.pt"
CARD_ORIENTATION_CLASSIFIER_WEIGHTS_PATH = "weights/card_orientation_classifier.pth"
DIGIT_DETECTOR_WEIGHTS_PATH = "weights/digit_detector.pt"

# Captura e inferência compartilhadas.
CAPTURE_SOURCE = "webcam"  # "webcam" ou "screen".
WEBCAM_INDEX = 0  # Índice da câmera usada quando a fonte é "webcam".
MONITOR_INDEX = 1  # 1 = primeiro monitor; 0 = área de todos os monitores.
CARD_SEGMENTER_CONFIDENCE_THRESHOLD = 0.5
DIGIT_DETECTOR_CONFIDENCE_THRESHOLD = 0.5
CARD_SEGMENTER_IMAGE_SIZE = 448
DIGIT_DETECTOR_IMAGE_SIZE = 640
DIGIT_DETECTOR_IOU_THRESHOLD = 0.8
DEBUG_DIGIT_DETECTOR_IOU_THRESHOLD = 0.7

# Tracker integrado ao Ultralytics (configuração incluída na biblioteca).
CARD_TRACKER_CONFIG = "bytetrack.yaml"

# Visualização (largura, altura).
FRAME_DISPLAY_SIZE = (840, 560)
CARD_CROP_DISPLAY_SIZE = (400, 250)
DIGIT_DETECTOR_CLASS_COUNT = 24
DEBUG_DIGIT_DETECTOR_CLASS_COUNT = 25

# Arquivo de log (id,codigo).
CARD_CODES_LOG_PATH = "card_codes_log.csv"
