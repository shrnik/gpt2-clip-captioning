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
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / \
                (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(
                    1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + \
                next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts


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
                normalized_embeddings).reshape(1, self.prefix_length, -1)

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
