class RS_VLM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = FPNBackbone()
        self.transformer = MultiScaleTransformer()
        self.head = OrientedHead(num_classes=num_classes)
        self.vl = VisionLanguageAlign()

    def forward(self, images, text_embeddings=None):
        feats = self.backbone(images)
        tokens = self.transformer(feats)

        cls, box, angle = self.head(tokens)

        out = {
            "pred_logits": cls,
            "pred_boxes": box,
            "pred_angles": angle,
        }

        if text_embeddings is not None:
            out["vl_logits"] = self.vl(tokens.mean(0), text_embeddings)

        return out

