import torch
import torch.nn as nn


class ArchComplexityTransformer(nn.Module):
    """Predicts one arch-level complexity class (1/2/3) from a per-tooth geometry token
    sequence, so the arch's tooth layout (position, spacing, and - if geom_dim=5 - angle)
    alone drives the prediction. No ResNet features (image embedding or per-tooth FDI
    confidence) are used - complexity is a property of the arch's geometric layout, not of
    what any individual tooth looks like or is classified as.
    geom_dim defaults to 5 ([x1,y1,x2,y2,theta]) but the main pipeline (train.py) uses
    geom_dim=4 ([x1,y1,x2,y2], no theta) as of 2026-08-31 - see train_theta.py for the
    theta-included ablation comparison.
    A learned CLS token pools the variable-length arch into a single prediction.
    """

    def __init__(self, geom_dim=5, d_model=256,
                 nhead=8, num_layers=4, num_classes=3, max_len=12, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(geom_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # +1 slot for the CLS token prepended in forward().
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, geom, key_padding_mask):
        """
        geom: (B, K, geom_dim) padded token features.
        key_padding_mask: (B, K) bool, True at padded positions.
        Returns logits (B, num_classes) - one complexity prediction per arch.
        """
        x = self.input_proj(geom)

        batch_size = x.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed[:, :x.size(1) + 1]

        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=key_padding_mask.device)
        full_mask = torch.cat([cls_mask, key_padding_mask], dim=1)

        x = self.encoder(x, src_key_padding_mask=full_mask)
        return self.head(x[:, 0])
