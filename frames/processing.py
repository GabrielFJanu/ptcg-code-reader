"""Recorte e rotação das imagens de cartas para leitura pelos modelos."""

from math import ceil

import cv2


_CV2_ROTATION_BY_ANGLE_DEGREES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def crop_card(frame, polygon):
    """Alinha a carta diretamente no recorte, sem rotacionar o frame inteiro."""
    if len(polygon) < 3:
        return None

    rectangle_center, rectangle_size, angle_degrees = cv2.minAreaRect(polygon)
    center_x, center_y = rectangle_center
    width, height = rectangle_size
    if width <= 0 or height <= 0:
        return None

    crop_width, crop_height = ceil(width), ceil(height)
    rotation_matrix = cv2.getRotationMatrix2D(
        rectangle_center, angle_degrees, 1.0
    )
    # Leva o centro da carta ao centro do destino, inclusive nas bordas do frame.
    rotation_matrix[0, 2] += (crop_width - 1) / 2 - center_x
    rotation_matrix[1, 2] += (crop_height - 1) / 2 - center_y
    return cv2.warpAffine(
        frame,
        rotation_matrix,
        (crop_width, crop_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def rotate_card_crop(card_crop, angle_degrees):
    """Aplica a rotação prevista pelo classificador ao recorte da carta."""
    rotation_code = _CV2_ROTATION_BY_ANGLE_DEGREES.get(angle_degrees)
    if rotation_code is None:
        return card_crop
    return cv2.rotate(card_crop, rotation_code)
