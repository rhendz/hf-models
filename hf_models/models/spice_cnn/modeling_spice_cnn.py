import torch.nn as nn
# from torchsummary import summary

from transformers import PreTrainedModel

from hf_models.models.spice_cnn.configuration_spice_cnn import SpiceCNNConfig

class SpiceCNNModelForImageClassification(PreTrainedModel):
    config_class = SpiceCNNConfig

    def __init__(self, config: SpiceCNNConfig):
        super().__init__(config)
        layers = [
            nn.Conv2d(config.in_channels, 16, kernel_size=config.kernel_size, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=config.pooling_size),
            
            nn.Conv2d(16, 32, kernel_size=config.kernel_size, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=config.pooling_size),

            nn.Conv2d(32, 64, kernel_size=config.kernel_size, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=config.pooling_size),

            nn.Flatten(),
            nn.Linear(64*3*3, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, config.num_classes)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, tensor, labels=None):
        logits = self.model(tensor)
        if labels is not None:
            loss_fnc = nn.CrossEntropyLoss()
            loss = loss_fnc(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
    
# config = SpiceCNNConfig(in_channels=1)
# cnn = SpiceCNNModelForImageClassification(config)
# summary(cnn, (1,28,28))
