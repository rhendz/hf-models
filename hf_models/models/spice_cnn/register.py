from hf_models.models.spice_cnn.configuration_spice_cnn import SpiceCNNConfig
from hf_models.models.spice_cnn.image_processing_spice_cnn import SpiceCNNImageProcessor
from hf_models.models.spice_cnn.modeling_spice_cnn import (
    SpiceCNNModelForImageClassification,
)

MODEL_DIRECTORY = "spice-cnn-base"
MODEL_REPO_ID = f"spicecloud/{MODEL_DIRECTORY}"


def main():
    SpiceCNNConfig.register_for_auto_class()
    SpiceCNNImageProcessor.register_for_auto_class("AutoImageProcessor")
    SpiceCNNModelForImageClassification.register_for_auto_class(
        "AutoModelForImageClassification"
    )

    spice_cnn_base_image_processor = SpiceCNNImageProcessor(
        do_resize=False, do_rescale=False, do_normalize=False
    )

    spice_cnn_base_config = SpiceCNNConfig()
    spice_cnn_base = SpiceCNNModelForImageClassification(spice_cnn_base_config)

    spice_cnn_base_image_processor.push_to_hub(repo_id=MODEL_REPO_ID)
    spice_cnn_base.push_to_hub(repo_id=MODEL_REPO_ID)


if __name__ == "__main__":
    main()
