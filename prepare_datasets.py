from transformers import GPT2Tokenizer
import pickle
import os
from torch.utils.data import Dataset
import torch
from typing import Tuple


class FlickrDataset(Dataset):
    def __init__(self, data_path: str,  prefix_length: int, gpt2_type: str = "gpt2",
                 normalize_prefix=False):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.imageEmbeddings = data["embeddings"]
        self.captions_raw = data["captions"]
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.max_seq_len = pickle.load(
                    f)
        else:
            self.captions_tokens = []
            max_seq_len = 0
            for captions in self.captions_raw:
                for caption in captions:
                    tokens = torch.tensor(self.tokenizer.encode(
                        caption))
                    self.captions_tokens.append(tokens)
                    max_seq_len = max(
                        max_seq_len, self.captions_tokens[-1].shape[0])
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump(
                    [self.captions_tokens, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i])
                               for i in range(len(self))]).float()
        self.max_seq_len = min(
            int(all_len.mean() + all_len.std() * 10), int(all_len.max()))

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat(
                (tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask),
                         dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask = self.pad_tokens(item)
        # each image has 5 captions
        prefixIndex = item // 5
        prefix = self.imageEmbeddings[prefixIndex]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, prefix


if __name__ == "__main__":
    dataset = FlickrDataset(
        data_path="data/flickr30k_clip_embeddings.pkl",
        prefix_length=6,
        normalize_prefix=True
    )
