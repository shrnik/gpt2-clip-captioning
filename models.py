from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import GPT2LMHeadModel

T = torch.Tensor


class SimpleMappingNetwork(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(SimpleMappingNetwork, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())

        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):
    def __init__(self, prefix_length: int = 6):
        super(ClipCaptionModel, self).__init__()
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.prefix_length = prefix_length
        self.prefix_size = 512
        self.num_layers = 12
        self.projector = SimpleMappingNetwork(
            (self.prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.prefix_length * self.gpt_embedding_size))

    def forward(self, tokens: T, prefix: T, mask: Optional[T] = None):
        caption_embeddings = self.gpt.transformer.wte(tokens)
        prefix_projections = self.projector(
            prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        joined_embeddings = torch.cat(
            prefix_projections, caption_embeddings, dim=1)
        output = self.gpt(inputs_embeds=joined_embeddings, attention_mask=mask)
        return output


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.projector.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self
