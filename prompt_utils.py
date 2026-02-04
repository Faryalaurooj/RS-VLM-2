class VisionLanguageAlign(nn.Module):
    def __init__(self, img_dim=256, text_dim=512):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, text_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def forward(self, img_feat, text_feat):
        img = F.normalize(self.img_proj(img_feat), dim=-1)
        txt = F.normalize(text_feat, dim=-1)
        scale = self.logit_scale.exp()
        return scale * img @ txt.T

