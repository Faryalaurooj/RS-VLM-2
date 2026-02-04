from models.rs_vlm import RS_VLM
from utils.misc import set_seed
import torch

def main():
    set_seed()
    model = RS_VLM(num_classes=10).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(50):
        model.train()
        # load batch
        outputs = model(images, text_embeddings)
        loss = ...
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

if __name__ == "__main__":
    main()

