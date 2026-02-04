class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h)
        x = x + attn
        x = x + self.ffn(self.norm2(x))
        return x
class MultiScaleTransformer(nn.Module):
    def __init__(self, num_layers=6, d_model=256):
        super().__init__()
        self.level_embed = nn.Parameter(torch.randn(4, d_model))
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model) for _ in range(num_layers)]
        )

    def forward(self, feats):
        tokens = []
        for lvl, f in enumerate(feats):
            B, C, H, W = f.shape
            t = f.flatten(2).permute(2, 0, 1)
            t = t + self.level_embed[lvl].view(1, 1, -1)
            tokens.append(t)

        x = torch.cat(tokens, dim=0)
        for layer in self.layers:
            x = layer(x)

        return x

