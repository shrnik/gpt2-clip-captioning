import os
import sys

import torch
from torch.utils.data import DataLoader
import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.nn import functional as nnf
from datasets import FlickrDataset
from models import ClipCaptionModel
import argparse


def train(dataset: FlickrDataset, model: ClipCaptionModel, args,
          lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = "", epochs: int = 10):

    device = torch.device('cuda:0')
    batch_size = 128

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs *
        len(train_dataloader)
    )
    # save_config(args)
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = tokens.to(device), mask.to(
                device), prefix.to(device, dtype=torch.float32)
            outputs = model(tokens, prefix, mask)
            logits = outputs.logits[:, dataset.prefix_length - 1: -1]
            loss = nnf.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        progress.close()
        if epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str,
                        default="data/flickr30k_clip_embeddings.pkl")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--prefix_length", type=int, default=6)
    parser.add_argument("--epochs", type=str, default=6)
    args = parser.parse_args()

    dataset = FlickrDataset(args.data_path, args.prefix_length)
    model = ClipCaptionModel(prefix_length=args.prefix_length)
    train(dataset, model, args, output_dir=args.output_dir,
          output_prefix="output", epochs=args.epochs)
