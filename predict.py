import torch
import numpy as np
from transformers.models.gpt2 import GPT2Tokenizer
from cog import BasePredictor, Input, Path
from models import ClipCaptionModel
from prepare_data import load_clip_model, load_clip_processor
import io
import argparse
from PIL import Image
from urllib.request import urlopen


def generate_beam(
    model,
    tokenizer,
    beam_size: int = 5,
    prompt=None,
    embed=None,
    entry_length=67,
    temperature=1.0,
    stop_token: str = ".",
):
    model.eval()
    device = next(model.parameters()).device

    # Handle stop token (assumes single token, for multi-token you'd need a more complex check)
    stop_token_ids = tokenizer.encode(stop_token, add_special_tokens=False)
    assert len(stop_token_ids) == 1, "Only single-token stop tokens are supported."
    stop_token_id = stop_token_ids[0]

    tokens = None
    scores = None
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)

    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            tokens = torch.tensor(tokenizer.encode(
                prompt), device=device).unsqueeze(0)
            generated = model.gpt.transformer.wte(tokens)

        for _ in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits[:, -1, :]

            # Prevent division by zero or negative temperatures
            temp = temperature if temperature > 0 else 1.0
            logits = logits / temp
            log_probs = logits.softmax(dim=-1).log()

            if scores is None:
                # First step: initialize beam with top-k choices
                scores, next_tokens = log_probs.topk(beam_size, dim=-1)
                scores = scores.squeeze(0)  # shape: [beam_size]
                next_tokens = next_tokens.squeeze(
                    0).unsqueeze(1)  # shape: [beam_size, 1]

                generated = generated.expand(beam_size, *generated.shape[1:])
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                # Prevent updating beams that already ended
                log_probs[is_stopped] = -float('inf')
                log_probs[is_stopped, 0] = 0  # Safe score for padding

                # shape: [beam_size, vocab_size]
                scores_sum = scores[:, None] + log_probs
                seq_lengths[~is_stopped] += 1
                scores_avg = scores_sum / seq_lengths[:, None]

                # Flatten and pick top-k
                scores_avg_flat, topk_indices = scores_avg.view(
                    -1).topk(beam_size, dim=-1)
                next_beam_idx = topk_indices // scores_sum.shape[1]
                next_token_id = topk_indices % scores_sum.shape[1]
                next_token_id = next_token_id.unsqueeze(1)

                # Reorder tokens and embeddings based on top beams
                tokens = tokens[next_beam_idx]
                tokens = torch.cat((tokens, next_token_id), dim=1)
                generated = generated[next_beam_idx]
                is_stopped = is_stopped[next_beam_idx]
                seq_lengths = seq_lengths[next_beam_idx]
                scores = scores_avg_flat * seq_lengths  # restore total logprob

            # Append new token embedding
            next_token_embed = model.gpt.transformer.wte(
                next_token_id.squeeze(-1)).unsqueeze(1)
            generated = torch.cat((generated, next_token_embed), dim=1)

            # Update stopped sequences
            is_stopped = is_stopped | next_token_id.squeeze(
                -1).eq(stop_token_id)
            if is_stopped.all():
                break

    # Normalize final scores and decode
    scores = scores / seq_lengths
    token_seqs = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(seq[: int(length)]) for seq, length in zip(token_seqs, seq_lengths)
    ]
    ranked_indices = scores.argsort(descending=True)
    return [output_texts[i] for i in ranked_indices]


class Predictor(BasePredictor):
    def setup(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.image_processor = load_clip_processor()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.clip_model = load_clip_model().to(self.device)
        self.prefix_length = 6
        self.model = ClipCaptionModel(self.prefix_length)
        self.model.load_state_dict(torch.load(
            "output/output-005.pt", map_location=self.device))
        self.model = self.model.eval()
        self.model = self.model.to(self.device)

    def predict(self, image: str):
        print("image", image)
        if image.startswith("http"):
            image = Image.open(urlopen(image)).convert("RGB")
        else:
            image = Image.open(image).convert("RGB")
        processed_images = self.image_processor(
            images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            prefixes = self.clip_model.get_image_features(**processed_images)
            normalized_embeddings = prefixes / \
                prefixes.norm(dim=1, keepdim=True)
            prefix_embed = self.model.projector(
                normalized_embeddings[0]).reshape(1, self.prefix_length, -1)

        return generate_beam(
            self.model,
            self.tokenizer,
            embed=prefix_embed,
            beam_size=self.beam_size,
            prompt="",
            entry_length=60,
            temperature=1,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate captions for images using GPT2-CLIP")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--beam-size", type=int, default=5,
                        help="Beam size for generation")
    parser.add_argument("--temperature", type=float,
                        default=1.0, help="Temperature for generation")
    parser.add_argument("--max-length", type=int, default=60,
                        help="Maximum caption length")

    args = parser.parse_args()

    # Initialize predictor
    predictor = Predictor()
    predictor.setup()
    predictor.beam_size = args.beam_size

    # Generate caption

    captions = predictor.predict(
        image=args.image
    )

    # Print results
    print("\nGenerated Captions:")
    for i, caption in enumerate(captions):
        print(f"{i+1}. {caption}")


if __name__ == "__main__":
    main()
