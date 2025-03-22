from .tokenizer import Tokenizer

import torch

class CharacterTokenizer(Tokenizer):
    def __init__(self, verbose: bool = False):
        """
        Initializes the CharacterTokenizer class for French to English translation.
        We ignore capitalization.

        Implement the remaining parts of __init__ by building the vocab.
        Implement the two functions you defined in Tokenizer here. Once you are
        done, you should pass all the tests in test_character_tokenizer.py.
        """
        super().__init__()

        self.vocab = {}

        # Normally, we iterate through the dataset and find all unique characters. To simplify things,
        # we will use a fixed set of characters that we know will be present in the dataset.
        self.characters = "aàâæbcçdeéèêëfghiîïjklmnoôœpqrstuùûüvwxyÿz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "

        self.idx_to_vocab = {}

        for idx, ch in enumerate(self.characters):
            self.vocab[ch] = idx
            self.idx_to_vocab[idx] = ch

    def encode(self, text: str) -> torch.tensor:
        vec = []
        for ch in text:
            vec.append(self.vocab[ch.lower()])
        vec = torch.tensor(vec, dtype=torch.float)
        return vec
    
    def decode(self, vec) -> str:
        string = []
        for e in vec:
            e = int(e)
            string.append(self.idx_to_vocab[e])
        string = ''.join(string)
        return string