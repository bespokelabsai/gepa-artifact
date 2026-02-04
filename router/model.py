import torch
import torch.nn as nn
from transformers import AutoModel


class PromptRouterCrossEncoder(nn.Module):
    """
    Prompt Router (Option B)

    Input:
      - tokenized (query, system_prompt_candidate) pairs

    Output:
      - scalar compatibility score for each pair

    Usage:
      tokenizer(query, prompt, return_tensors="pt")
      score = model(**tokens)
    """

    def __init__(
        self,
        backbone_name: str,
        *,
        dropout: float = 0.1,
        use_pooler_if_available: bool = True,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)
        self.use_pooler_if_available = use_pooler_if_available

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
          input_ids:        [B, T]
          attention_mask:  [B, T]
          token_type_ids:  [B, T] (optional)

        Returns:
          scores: [B]   (one score per (query, prompt) pair)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        if (
            self.use_pooler_if_available
            and hasattr(outputs, "pooler_output")
            and outputs.pooler_output is not None
        ):
            x = outputs.pooler_output            # [B, H]
        else:
            x = outputs.last_hidden_state[:, 0]  # CLS token [B, H]

        x = self.dropout(x)
        scores = self.head(x).squeeze(-1)        # [B]
        return scores
