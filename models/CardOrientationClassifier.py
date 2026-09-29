"""Classificação da orientação de cartas."""

from dataclasses import dataclass

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


@dataclass(frozen=True)
class CardOrientationClassifierResult:
    """Orientação prevista e confiança softmax da classe selecionada."""

    angle_degrees: int
    class_id: int
    confidence: float


class CardOrientationPyTorchModel(nn.Module):
    """Rede neural PyTorch que classifica a orientação de uma carta."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, input_tensor):
        features = self.pool(F.relu(self.conv1(input_tensor)))
        features = self.pool(F.relu(self.conv2(features)))
        features = self.pool(F.relu(self.conv3(features)))
        features = features.view(features.size(0), -1)
        features = F.relu(self.fc1(features))
        return self.fc2(features)


class CardOrientationClassifier:
    """Carrega o modelo e encapsula a classificação da orientação de cartas."""

    def __init__(self, weights_path, device):
        self._device = device
        self._input_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        self._angle_degrees_by_class_id = {0: 0, 2: 90, 1: 180, 3: 270}

        self._pytorch_model = CardOrientationPyTorchModel().to(device)
        model_state = torch.load(weights_path, map_location=device)
        self._pytorch_model.load_state_dict(model_state)
        self._pytorch_model.eval()

    def predict(self, card_crop) -> CardOrientationClassifierResult:
        """Classifica um recorte BGR e retorna a classe, o ângulo e a confiança."""
        rgb_card_crop = cv2.cvtColor(card_crop, cv2.COLOR_BGR2RGB)
        card_image = Image.fromarray(rgb_card_crop)
        input_tensor = self._input_transform(card_image).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._pytorch_model(input_tensor)
            predicted_class_id = torch.argmax(logits, dim=1).item()
            class_probabilities = torch.softmax(logits, dim=1)
            confidence = class_probabilities[0, predicted_class_id].item()

        return CardOrientationClassifierResult(
            angle_degrees=self._angle_degrees_by_class_id[predicted_class_id],
            class_id=predicted_class_id,
            confidence=confidence,
        )
