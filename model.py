import torch
import torch.nn as nn
import torchvision.models as models

# Config parameters
class Config:
    DATA_PATH = "./dataset"  # Update as needed
    IMG_SIZE = 224  # Standard size for most pretrained models
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_SAVE_PATH = "best_model.pth"
    CLASS_NAMES = [
        "Dental benign tumors",
        "Dental Caries",
        "Dental Malignant tumors",
        "Gingivitis",
        "Hypodontia",
        "Mouth Ulcer",
        "Tooth Discoloration"
    ]
    NUM_CLASSES = len(CLASS_NAMES)

# Model Definition
class DentalModel(nn.Module):
    def __init__(self, num_classes=7, model_name="efficientnet_b2"):
        super(DentalModel, self).__init__()
        self.model_name = model_name

        # Choose model based on name
        if model_name.startswith("efficientnet"):
            # EfficientNet has better performance and efficiency for medical imaging tasks
            self.model = getattr(models, model_name)(weights="DEFAULT")
            if model_name == "efficientnet_b0":
                num_ftrs = 1280
            elif model_name == "efficientnet_b1":
                num_ftrs = 1280
            elif model_name == "efficientnet_b2":
                num_ftrs = 1408
            else:
                num_ftrs = self.model.classifier[1].in_features

            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.3, inplace=True),
                nn.Linear(num_ftrs, num_classes)
            )
        elif model_name.startswith("resnet"):
            self.model = getattr(models, model_name)(weights="DEFAULT")
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def freeze_layers(self, freeze_percentage=0.7):
        """Freeze a percentage of the early layers"""
        if self.model_name.startswith("efficientnet"):
            # Get total number of layers in features
            total_layers = len(list(self.model.features))
            freeze_layers = int(total_layers * freeze_percentage)

            # Freeze the first X% layers
            for i, param in enumerate(self.model.features.parameters()):
                if i < freeze_layers:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

        elif self.model_name.startswith("resnet"):
            # Freeze early layers (layer1, layer2) but keep layer3, layer4 and fc trainable
            for name, param in self.model.named_parameters():
                if "layer3" not in name and "layer4" not in name and "fc" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True

    def forward(self, x):
        return self.model(x)