class OrientedHead(nn.Module):
    def __init__(self, d_model=256, num_classes=20):
        super().__init__()
        self.cls = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes)
        )

        self.box = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4)  # x,y,w,h
        )

        self.angle = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2)  # sinθ, cosθ
        )

    def forward(self, x):
        logits = self.cls(x)
        box = self.box(x)
        angle = F.normalize(self.angle(x), dim=-1)
        return logits, box, angle

