from typing import Optional

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):

    def __init__(self, 
                 num_heads: int,
                 embedding_dim: int,
                 qk_length: int,
                 value_length: int
                 ):
        """
        The Multi-Head Attention layer will take in Q, K, and V
        matrices and will output an attention matrix of shape <TODO>.

        First, Q, K, and V should be projected to have
        a shape of (B, T, C) where C = num_heads * qk_length 
        (OR value_length). You are then expected to split 
        the C dimension into num_heads different heads, each 
        with shape (B, T, vec_length).

        Next, you will compute the scaled dot-product attention
        between Q, K, and V.

        Finally, you will concatenate the heads and project the
        output to have a shape of (B, T, C).

        Check out the `masked_fill` method in PyTorch to help
        you implement the masking step!
        """
        super().__init__()

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.qk_length = qk_length
        self.value_length = value_length

        # Define any layers you'll need in the forward pass
        # (hint: number of Linear layers needed != 3)
        self.W_q = nn.Linear(embedding_dim, num_heads * qk_length)
        self.W_k = nn.Linear(embedding_dim, num_heads * qk_length)
        self.W_v = nn.Linear(embedding_dim, num_heads * value_length)
        self.W_o = nn.Linear(num_heads * value_length, embedding_dim)

    def split_heads(self, x: torch.Tensor, vec_length: int) -> torch.Tensor:
        """
        Split the C dimension of the input tensor into num_heads
        different heads, each with shape (B, T, vec_length).

        Args:
            x: torch.Tensor of shape (B, T, C), where C = num_heads * vec_length
            vec_length: int, the length of the query/key/value vectors

        Returns:
            torch.Tensor of shape (B, num_heads, T, vec_length)
        """
        B, T, C = x.shape
        assert C / self.num_heads == vec_length, f"X's last dimension should be of length {self.num_heads * self.qk_length}"

        x_copy = x.view(B, T, self.num_heads, vec_length)
        x_copy = x_copy.permute(0, 2, 1, 3)
        return x_copy        

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine the num_heads different heads into a single tensor.
        Hint: check out the `contiguous` method in PyTorch to help
        you reshape the tensor.

        Args:
            x: torch.Tensor of shape (B, num_heads, T, vec_length)

        Returns:
            torch.Tensor of shape (B, T, num_heads * vec_length)
        """
        B, num_heads, T, vec_length = x.shape
        x = x.permute(0, 2, 1, 3) # swap T and num_heads dimensions
        x = x.contiguous() # the contiguous function ensures that the tensor is stored contiguously in memory which is required for the view method
        x = x.view(B, T, num_heads * vec_length)
        return x

    def scaled_dot_product_attention(self, 
                                     Q: torch.Tensor, 
                                     K: torch.Tensor, 
                                     V: torch.Tensor, 
                                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the scaled dot-product attention given Q, K, and V.

        Args:
            Q: torch.Tensor of shape (B, num_heads, T, qk_length)
            K: torch.Tensor of shape (B, num_heads, T, qk_length)
            V: torch.Tensor of shape (B, num_heads, T, value_length)
            mask: Optional torch.Tensor of shape (B, T, T) or None
        """
        lookup = torch.matmul(Q, torch.transpose(K, -1, -2))
        scaled_lookup = lookup / (self.qk_length ** 0.5)
        
        # Apply mask if provided
        if mask is not None:
            # Ensure mask is broadcastable to the correct shape
            # Since mask is (B, T, T), we need to add a dimension for num_heads
            mask = mask.unsqueeze(1)  # Shape becomes (B, 1, T, T)
            
            # Apply mask to scaled_lookup
            scaled_lookup = scaled_lookup + (mask == 0).int() * -1e9  # Masked values become very negative
        
        attention = nn.functional.softmax(scaled_lookup, dim=-1)
        return torch.matmul(attention, V)

    def forward(self,
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        The forward pass of the Multi-Head Attention layer.

        Args:
            Q: torch.Tensor of shape (B, T, C)
            K: torch.Tensor of shape (B, T, C)
            V: torch.Tensor of shape (B, T, C)
            mask: Optional torch.Tensor of shape (B, T, T) or None

        Returns:
            torch.Tensor of shape (B, T, C)
        """
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        Q = self.split_heads(Q, self.qk_length)
        K = self.split_heads(K, self.qk_length)
        V = self.split_heads(V, self.value_length)

        attention = self.scaled_dot_product_attention(Q, K, V)

        attention = self.combine_heads(attention)

        attention_out = self.W_o(attention)
        
        return attention_out

class FeedForwardNN(nn.Module):

    def __init__(self, 
                 embedding_dim: int,
                 hidden_dim: int):
        """
        The Feed-Forward Neural Network layer will take in
        an input tensor of shape (B, T, C) and will output
        a tensor of the same shape.

        The FFNN will have two linear layers, with a ReLU
        activation function in between.

        Args:
            hidden_dim: int, the size of the hidden layer
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        # Define any layers you'll need in the forward pass
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the FeedForwardNN.
        """
        # linear layers expect a tensor with shape (bath_size, features)
        # reshape x into shape (B*T, C)
        print(x.shape)
        B, T, C = x.shape
        x = x.view(-1, C)

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        x = x.view(B, T, C)
        return x