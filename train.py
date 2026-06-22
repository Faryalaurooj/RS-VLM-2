import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from datasets.dota import DOTADataset
from models.detector import SGSMDetector


# -----------------------------
# Collate function
# -----------------------------
def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])

    boxes = [b["boxes"] for b in batch]
    labels = [b["labels"] for b in batch]

    return {
        "images": images,
        "boxes": boxes,
        "labels": labels
    }


# -----------------------------
# Dummy CLIP encoder placeholder
# Replace with real CLIP (openai/clip or open_clip)
# -----------------------------
class DummyCLIP:
    def encode_text(self, texts):
        # returns random embeddings
        return torch.randn(len(texts), 512)


# -----------------------------
# Training loop
# -----------------------------
def train(
    img_dir,
    ann_dir,
    num_classes=15,
    epochs=20,
    batch_size=4,
    lr=1e-4,
    device="cuda"
):

    dataset = DOTADataset(img_dir, ann_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = SGSMDetector(num_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    scaler = torch.amp.GradScaler()

    clip_model = DummyCLIP()

    model.train()

    for epoch in range(epochs):
        total_loss_epoch = 0

        for batch in loader:
            images = batch["images"].to(device)

            # fake text prompts (class names placeholder)
            text_emb = clip_model.encode_text(["object"] * images.size(0)).to(device)

            # dummy rotated augmentation placeholder
            rotated_images = torch.rot90(images, k=1, dims=[2, 3])

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs, losses = model(
                    images,
                    targets=None,
                    rotated_x=rotated_images,
                    text_emb=text_emb
                )

                loss = losses["total_loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss_epoch += loss.item()

        scheduler.step()

        print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss_epoch:.4f}")

    torch.save(model.state_dict(), "sgsm_detector.pth")
    print("Model saved: sgsm_detector.pth")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    train(
        img_dir="datasets/DOTA/images",
        ann_dir="datasets/DOTA/labels",
        epochs=20,
        batch_size=2,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
