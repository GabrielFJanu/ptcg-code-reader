"""Funções utilitárias compartilhadas pelos pipelines."""

import random


def generate_random_colors(color_count: int) -> list[tuple[int, int, int]]:
    """Gera cores BGR com canais entre 0 e 255 para uso no OpenCV."""
    return [
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for _ in range(color_count)
    ]
