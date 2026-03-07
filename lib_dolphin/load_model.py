import torch
import torch.nn as nn
import torch.nn.functional as F


from transformers import WhisperModel, WhisperProcessor
from peft import get_peft_model, LoraConfig, TaskType
from .config import get_config

# Load configuration singleton
config = get_config()


class WDPWhisperEmbModel(nn.Module):

    def __init__(self, lora_encoder, output_dim_embedding=None, n_classes=None):
        super().__init__()
        self.encoder = lora_encoder
        self.output_dim_embedding = output_dim_embedding if output_dim_embedding is not None else config.embedding_dimension
        self.n_classes = n_classes if n_classes is not None else config.num_classes

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head_embedding = nn.Linear(self.encoder.config.d_model, self.output_dim_embedding)
        self.classifier = nn.Linear(self.encoder.config.d_model, self.n_classes)

    @property
    def device(self):
        """Return the device the model is on."""
        return next(self.parameters()).device

    def embed(self, input_features, input_mask):
        out = self.encoder(input_features=input_features)
        emb = out.last_hidden_state
        seq_len = emb.shape[1]
        
        pooled_mask = F.adaptive_avg_pool2d(input_mask.unsqueeze(1), (1, seq_len)).squeeze(1).squeeze(1)  # [B, seq_len]
        mask_expanded = pooled_mask.unsqueeze(-1).expand_as(emb)  # [B, seq_len, d_model]
        masked_emb = emb * mask_expanded  # Zero out masked positions
        sum_emb = masked_emb.sum(dim=1)  # [B, d_model]
        seq_lengths = pooled_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1], avoid div by 0
        emb_pooled = sum_emb / seq_lengths  # True average over non-masked tokens
        embeddings = self.head_embedding(emb_pooled)  # [B, output_dim_embedding]
        return emb, embeddings

    def forward(self, input_features, sequence_labels=None, sequence_positive=None, sequence_negative=None, sequence_mask=None, input_mask=None):
        emb, embeddings = self.embed(input_features, input_mask)

        classifications = self.classifier(emb)

        if sequence_labels is not None:
            active_positions = sequence_mask.view(-1) == 1
            active_logits = classifications.view(-1, self.n_classes)[active_positions]
            active_labels = sequence_labels.view(-1)[active_positions]
            classification_loss = F.cross_entropy(active_logits, active_labels)

            triplet_loss = 0
            if sequence_positive is not None and sequence_negative is not None:
                _, pos_embeddings = self.embed(sequence_positive, input_mask)
                _, neg_embeddings = self.embed(sequence_negative, input_mask)

                triplet_loss = F.triplet_margin_loss(
                    embeddings,
                    pos_embeddings,
                    neg_embeddings,
                    margin=config.triplet_margin
                )

            total_loss = config.triplet_weight * triplet_loss + config.classification_weight * classification_loss

            return {
                "loss": total_loss,
                "embeddings": embeddings,
                "classifications": classifications,
            }
        else:
            return {
                "embeddings": embeddings,
                "classifications": classifications
            }
        

def processor(model_size=None, model_id=None):
    if model_id is None:
        model_config = config.get_model_config(model_size)
        model_id = model_config.get('base_model_id', config.base_model_id)
    print(f"Loading processor from: {model_id}")
    return WhisperProcessor.from_pretrained(model_id)


def base_model(model_size=None, model_id=None):
    if model_id is None:
        model_config = config.get_model_config(model_size)
        model_id = model_config.get('base_model_id', config.base_model_id)
    print(f"Loading base model: {model_id}")
    return WhisperModel.from_pretrained(model_id)


def lora_config():
    return LoraConfig(
        inference_mode=False,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        target_modules=config.lora_target_modules,
    )


def lora_encoder(model_size=None):
    return get_peft_model(base_model(model_size).encoder, lora_config())


def model(model_size=None, model_file=None, cuda=None):
    """
    Load model with automatic CUDA detection.

    Args:
        model_size: Model size configuration
        model_file: Path to checkpoint file
        cuda: Force CUDA on/off. If None (default), auto-detect.
    """
    if model_file is None:
        model_config = config.get_model_config(model_size)
        model_file = model_config.get('checkpoint_path', config.checkpoint_path)

    # Auto-detect CUDA if not explicitly specified
    if cuda is None:
        cuda = torch.cuda.is_available()

    device = torch.device('cuda' if cuda else 'cpu')
    print(f"Loading model on device: {device}")
    print(f"Loading checkpoint from: {model_file}")

    model = WDPWhisperEmbModel(lora_encoder(model_size))
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)

    return model
