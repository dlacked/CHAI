import torch
import torch.nn as nn


class ArchToothTransformer(nn.Module):
    """Refines each tooth's ResNet last-digit prediction (1-6) using self-attention over the
    whole dental arch. Each token is [x1,y1,x2,y2,theta] + the ResNet's own image-branch
    embedding + the ResNet's softmax output for that tooth.
    """

    def __init__(self, geom_dim=5, img_dim=512, prob_dim=6, d_model=256,
                 nhead=8, num_layers=4, num_classes=6, max_len=12, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(geom_dim + img_dim + prob_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )
        # Learned position-in-arch-order embedding, complementing the continuous x/y
        # coordinates already present in the geometry features.
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, geom, img_vec, prob_vec, key_padding_mask):
        """
        geom, img_vec, prob_vec: (B, K, *) padded token features.
        key_padding_mask: (B, K) bool, True at padded positions.
        Returns logits (B, K, num_classes).
        """
        tokens = torch.cat([geom, img_vec, prob_vec], dim=-1)
        x = self.input_proj(tokens) + self.pos_embed[:, :tokens.size(1)]
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # Residual refinement: start from the ResNet's own log-probabilities so the
        # Transformer only has to learn a context-driven correction, not the whole
        # distribution from scratch - keeps early training at/above the ResNet baseline.
        prior = torch.log(prob_vec.clamp_min(1e-6))
        return self.head(x) + prior
