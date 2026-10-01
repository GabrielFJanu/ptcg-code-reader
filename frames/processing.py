"""Recorte e rotação das imagens de cartas para leitura pelos modelos."""

import cv2


def crop_card(frame, polygon):
    """Alinha o retângulo mínimo da carta e retorna None para recortes vazios."""
    if len(polygon) < 3:
        return None

    rectangle_center, rectangle_size, angle_degrees = cv2.minAreaRect(polygon)
    center_x, center_y = rectangle_center
    width, height = rectangle_size
    if width <= 0 or height <= 0:
        return None

    rotation_matrix = cv2.getRotationMatrix2D(
        rectangle_center, angle_degrees, 1.0
    )
    rotated_frame = cv2.warpAffine(
        frame,
        rotation_matrix,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR,
    )
    left = max(0, int(center_x - width / 2))
    top = max(0, int(center_y - height / 2))
    right = min(frame.shape[1], int(center_x + width / 2))
    bottom = min(frame.shape[0], int(center_y + height / 2))
    card_crop = rotated_frame[top:bottom, left:right]
    return card_crop if card_crop.size > 0 else None


def rotate_card_crop(card_crop, angle_degrees):
    """Aplica a rotação prevista pelo classificador ao recorte da carta."""
    rotation_by_angle = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    rotation = rotation_by_angle.get(angle_degrees)
    if rotation is None:
        return card_crop
    return cv2.rotate(card_crop, rotation)
