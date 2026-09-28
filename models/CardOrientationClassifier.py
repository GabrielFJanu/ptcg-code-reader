import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class CardOrientationClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 4)

        self.input_image_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        
        self.orientation_angle_by_class_id = {0: 0, 2: 90, 1: 180, 3: 270}

    @classmethod
    def from_weights(cls, weights_path, device):
        """Cria o classificador com pesos locais, pronto para inferência."""
        classifier = cls().to(device)
        model_state = torch.load(weights_path, map_location=device)
        classifier.load_state_dict(model_state)
        classifier.eval()
        return classifier

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
