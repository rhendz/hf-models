import torch.nn as nn

from transformers import PreTrainedModel

from .configuration_spice_cnn import SpiceCNNConfig


class SpiceCNNModelForImageClassification(PreTrainedModel):
    config_class = SpiceCNNConfig

    def __init__(self, config: SpiceCNNConfig):
        super().__init__(config)
        layers = [
            nn.Conv2d(
                3,
                16,
                kernel_size=config.kernel_size,
                stride=config.stride,
                padding=config.padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=config.pooling_size),
            nn.Conv2d(
                16,
                32,
                kernel_size=config.kernel_size,
                stride=config.stride,
                padding=config.padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=config.pooling_size),
            nn.Flatten(),
            nn.Linear(7 * 7 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, config.num_classes),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, tensor, labels=None):
        logits = self.model(tensor)
        if labels is not None:
            loss_fnc = nn.CrossEntropyLoss()
            loss = loss_fnc(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
