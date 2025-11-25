import torch
from torch.utils.data import Dataset
from collections import Counter

class LegalDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_len=100):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        
        # If no vocab is given (Train phase), create it
        if vocab is None:
            self.vocab = self.build_vocab(texts)
        else:
            self.vocab = vocab
            
    def build_vocab(self, texts, max_vocab=10000):
        # Simple word vocabulary builder
        all_words = [word.lower() for text in texts for word in text.split()]
        word_counts = Counter(all_words)
        # Take the most common words, 0=Padding, 1=Unknown
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for word, _ in word_counts.most_common(max_vocab):
            vocab[word] = len(vocab)
        return vocab

    def text_to_indices(self, text):
        indices = []
        for word in text.split():
            indices.append(self.vocab.get(word.lower(), self.vocab["<UNK>"]))
        
        # Padding or Truncating (Make fixed size)
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices)) # Pad
        else:
            indices = indices[:self.max_len] # Truncate
            
        return torch.tensor(indices, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Fix label format: [1] -> 1.0, [] -> 0.0
        is_risky = 1.0 if len(label) > 0 else 0.0
        
        return {
            "input_ids": self.text_to_indices(text),
            "label": torch.tensor(is_risky, dtype=torch.float)
        }