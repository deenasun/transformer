from tqdm import tqdm
from pathlib import Path
import os  # Add this import

import torch

from torch.utils.data import DataLoader

from transformer import Transformer, FrenchEnglishDataset
from tokenizer.bpe_tokenizer import BPETokenizer

def main():

    # Ensure the 'ckpts' directory exists
    os.makedirs("ckpts", exist_ok=True)

    tokenizer = BPETokenizer()
    vocab_size = len(tokenizer.vocab)
    train_dataset = FrenchEnglishDataset(Path("smaller.csv"), tokenizer=tokenizer, train=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=FrenchEnglishDataset.collate_fn)

    model = Transformer(vocab_size=vocab_size,
                        num_layers=2,
                        num_heads=2,
                        ffn_hidden_dim=64,
                        embedding_dim=64,
                        qk_length=64,
                        value_length=64,
                        max_length=5000,
                        dropout=0.1)

    for param in model.parameters():
        if param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)

    device = "cpu"
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=0)

    for epoch in range(1000):
        train_tqdm = tqdm(enumerate(train_loader), total=len(train_loader), dynamic_ncols=True)
        for i, (src, tgt) in train_tqdm:
            src = src.to(device)
            tgt = tgt.to(device)

            src = src.to(torch.int64)
            tgt = tgt.to(torch.int64)

            outputs = model(src, tgt)

            outputs = outputs.view(-1, vocab_size)

            tgt = tgt.view(-1)

            optimizer.zero_grad()
            loss = loss_fn(outputs, tgt)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            train_tqdm.set_postfix({"loss": loss.item()})

            if i % 500 == 0:
                torch.save(model.state_dict(), "ckpts/model_last.pt")

        lr_scheduler.step()

        torch.save(model.state_dict(), f"ckpts/model_{epoch}_{i}.pt")

    torch.save(model.state_dict(), "ckpts/model_final.pt")

if __name__ == '__main__':
    main()
