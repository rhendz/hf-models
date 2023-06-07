from transformers import PretrainedConfig

"""Spice CNN model configuration"""

SPICE_CNN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "spicecloud/spice-cnn-base": "https://huggingface.co/spice-cnn-base/resolve/main/config.json"
}


# Define custom convnet configuration
class SpiceCNNConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`SpiceCNNModel`].
    It is used to instantiate an SpiceCNN model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults
    will yield a similar configuration to that of the SpiceCNN
    [spicecloud/spice-cnn-base](https://huggingface.co/spicecloud/spice-cnn-base)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control
    the model outputs. Read the documentation from [`PretrainedConfig`] for more
    information.
    """

    model_type = "spicecnn"

    def __init__(
        self,
        num_classes: int = 10,
        dropout_rate: float = 0.2,
        hidden_size: int = 128,
        num_filters: int = 16,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        pooling_size: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pooling_size = pooling_size
