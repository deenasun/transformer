import torch
import torch.nn as nn

from .attention import MultiHeadAttention, FeedForwardNN
from .encoder import PositionalEncoding

class DecoderLayer(nn.Module):
    
    def __init__(self, 
                 num_heads: int, 
                 embedding_dim: int,
                 ffn_hidden_dim: int,
                 qk_length: int, 
                 value_length: int,
                 dropout: float = 0.1):
        """
        Each decoder layer will take in two embeddings of
        shape (B, T, C):

        1. The `target` embedding, which comes from the decoder
        2. The `source` embedding, which comes from the encoder

        and will output a representation
        of the same shape.

        The decoder layer will have three main components:
            1. A Masked Multi-Head Attention layer (you'll need to
               modify the MultiHeadAttention layer to handle this!)
            2. A Multi-Head Attention layer for cross-attention
               between the target and source embeddings.
            3. A Feed-Forward Neural Network layer.

        Remember that for each Multi-Head Attention layer, we
        need create Q, K, and V matrices from the input embedding(s)!
        """
        super().__init__()

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.qk_length = qk_length
        self.value_length = value_length

        # Define any layers you'll need in the forward pass
        self.masked_mha = MultiHeadAttention(num_heads, embedding_dim, qk_length, value_length)
        self.mha = MultiHeadAttention(num_heads, embedding_dim, qk_length, value_length)
        self.ffnn = FeedForwardNN(embedding_dim, ffn_hidden_dim)

        self.self_attn_norm = nn.LayerNorm(embedding_dim)
        self.self_attn_dropout = nn.Dropout(dropout)
        self.cross_attn_norm = nn.LayerNorm(embedding_dim)
        self.cross_attn_dropout = nn.Dropout(dropout)
        self.ffnn_norm = nn.LayerNorm(embedding_dim)
        self.ffnn_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, enc_x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the DecoderLayer.
        """
        # Self attention with masked multi-head attention
        residual = x
        x = self.masked_mha(x, x, x, mask)
        x = x + residual
        x = self.self_attn_dropout(x)
        x = self.self_attn_norm(x)

        # Cross attention using encoder's keys and values
        residual = x
        x = self.mha(x, enc_x, enc_x) # second multi-head attention block receives its keys and values from the encoder
        x = x + residual
        x = self.cross_attn_dropout(x)
        x = self.cross_attn_norm(x)

        # Feed forward
        residual = x
        x = self.ffnn(x)
        x = x + residual
        x = self.ffnn_dropout(x)
        x = self.ffnn_norm(x)

        return x

class Decoder(nn.Module):

    def __init__(self, 
                 vocab_size: int, 
                 num_layers: int, 
                 num_heads: int,
                 embedding_dim: int,
                 ffn_hidden_dim: int,
                 qk_length: int,
                 value_length: int,
                 max_length: int,
                 dropout: float = 0.1):
        """
        Remember that the decoder will take in a sequence
        of tokens AND a source embedding
        and will output an encoded representation
        of shape (B, T, C).

        First, we need to create an embedding from the sequence
        of tokens. For this, we need the vocab size.

        Next, we want to create a series of Decoder layers.
        For this, we need to specify the number of layers 
        and the number of heads.

        Additionally, for every Multi-Head Attention layer, we
        need to know how long each query/key is, and how long
        each value is.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim

        self.qk_length = qk_length
        self.value_length = value_length

        # Define any layers you'll need in the forward pass
        # Hint: You may find `ModuleList`s useful for creating
        # multiple layers in some kind of list comprehension.
        # 
        # Recall that the input is just a sequence of tokens,
        # so we'll have to first create some kind of embedding
        # and then use the other layers we've implemented to
        # build out the Transformer decoder.

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout, max_length)

        self.decoder_layers = nn.ModuleList([DecoderLayer(num_heads, embedding_dim, ffn_hidden_dim, qk_length, value_length, dropout) for _ in range(num_layers)])

        # final output matrix to project output of decoder layers into vocab size
        self.output_matrix = nn.Linear(embedding_dim, vocab_size)


    def make_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create a mask to prevent attention to future tokens.
        """
        B, T, C = x.shape
        mask = torch.ones((T, T))
        mask = torch.triu(mask, 0)
        # all elements on and above diagonal = 1, and should be set to False (not masked)
        # all elements below diagonal = 0, and should be set to True (masked)
        mask = mask == 0
        return mask


    def forward(self, x: torch.Tensor, enc_x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the Decoder.
        """
        # Create embedding of input to Decoder
        x = self.embedding(x)
        x = self.positional_encoding(x)

        # Create mask
        mask = self.make_mask(x)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, enc_x, mask)

        x = self.output_matrix(x)
        return x