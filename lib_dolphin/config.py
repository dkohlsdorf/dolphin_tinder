class Config:
    """Configuration for the WDP Whisper embedding model."""

    # Model architecture
    embedding_dimension = 128
    num_classes = 5

    # Training parameters
    triplet_margin = 1.0
    triplet_weight = 0.5
    classification_weight = 1.0

    # LoRA configuration
    lora_rank = 16
    lora_alpha = 64
    lora_dropout = 0.0
    lora_bias = "none"
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

    # Default model settings
    base_model_id = "openai/whisper-small"
    checkpoint_path = "model/wdp_whisper_emb_model_small.pt"

    # Model size configurations
    _model_configs = {
        "small": {
            "base_model_id": "openai/whisper-small",
            "checkpoint_path": "model/wdp_whisper_emb_model_small.pt",
        },
        "base": {
            "base_model_id": "openai/whisper-base",
            "checkpoint_path": "model/wdp_whisper_emb_model_L2_base.pt",
        },
        "small_augment_L1": {
            "base_model_id": "openai/whisper-small",
            "checkpoint_path": "model/wdp_whisper_emb_model_L1_small_augment.pt",
        },
        "small_augment_L2": {
            "base_model_id": "openai/whisper-small",
            "checkpoint_path": "model/wdp_whisper_emb_model_L2_small_augment.pt",
        },
    }

    def get_model_config(self, model_size=None):
        """Get configuration for a specific model size."""
        if model_size is None:
            model_size = "small"
        return self._model_configs.get(model_size, self._model_configs["small"])


_config = None

def get_config():
    """Get the configuration singleton."""
    global _config
    if _config is None:
        _config = Config()
    return _config
