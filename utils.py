"""Funções utilitárias compartilhadas pelos pipelines."""

import random

import cv2


def generate_random_colors(color_count: int) -> list[tuple[int, int, int]]:
    """Gera cores BGR com canais entre 0 e 255 para uso no OpenCV."""
    return [
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for _ in range(color_count)
    ]


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
