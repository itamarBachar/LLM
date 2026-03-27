from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F
import math


def create_kqv_matrix(input_vector_dim, n_heads = 1):
    d_head = input_vector_dim // n_heads
    output_dim = 3 * d_head
    return nn.Linear(input_vector_dim, output_dim)

def create_output_projection(input_vector_dim):
    return nn.Linear(input_vector_dim, input_vector_dim)

def kqv(x, linear):
    B, N, D = x.size()
    kqv_output = linear(x)  # Shape: (B, N, 3*d_head)
    k, q, v = torch.split(kqv_output, kqv_output.size(-1) // 3, dim=-1)
    return k, q, v

def attention_scores(a, b):
    B1, N1, D1 = a.size()
    B2, N2, D2 = b.size()
    assert B1 == B2
    assert D1 == D2

    # Compute A (remember: we are computing *scaled* dot product attention. don't forget the scaling.
    # Compute scaled dot product attention: (a @ b^T) / sqrt(d)
    A = torch.matmul(a, b.transpose(-2, -1)) / math.sqrt(D1)
    return A

def create_causal_mask(embed_dim, n_heads, max_context_len):
    # Return a causal mask (a tensor) with zeroes in dimensions we want to zero out.
    # This function receives more arguments than it actually needs. This is just because
    # it is part of an assignment, and I want you to figure out on your own which arguments
    # are relevant.

    # Shape: (1, max_context_len, max_context_len). 1 means this can be broadcast over batch.
    # Lower-triangular ones keep valid positions; zeros above diagonal are masked out.
    mask = torch.tril(torch.ones((1, max_context_len, max_context_len), dtype=torch.float32))
    return mask

def self_attention(v, A, mask = None):
    # TODO compute sa (corresponding to y in the assignemnt text).
    # This should take very few lines of code.
    # As usual, the dimensions of v and of sa are (b x n x d).
    if mask is not None:
        # Derive a view for the current sequence lengths from the precomputed max-size mask.
        current_mask = mask[:, :A.size(-2), :A.size(-1)]
        A = A.masked_fill(current_mask == 0, float("-inf"))
    alpha = F.softmax(A, dim=-1)
    sa = torch.matmul(alpha, v)
    return sa


def self_attention_layer(x, kqv_matrix, attention_mask):
    k, q, v = kqv(x, kqv_matrix)
    att = attention_scores(k, q)
    sa = self_attention(v, att, attention_mask)
    return sa

def multi_head_attention_layer(x, kqv_matrices, mask):
    B, N, D = x.size()
    # TODO implement multi-head attention.
    # This is most easily done using calls to self_attention_layer, each with a different
    # entry in kqv_matrices, and combining the results.
    #
    # There is also a tricker (but more efficient) version of multi-head attention, where we do all the computation
    # using a single multiplication with a single kqv_matrix (or a single kqv_tensor) and re-arranging the results afterwards.
    # If you want a challenge, you can try and implement this. You may need to change additional places in the code accordingly.
    heads = [self_attention_layer(x, kqv_matrix, mask) for kqv_matrix in kqv_matrices]
    sa = torch.cat(heads, dim=-1)
    assert sa.size() == x.size()
    return sa


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, max_context_len):
        super().__init__()
        assert embed_dim % n_heads == 0
        # the linear layers used for k, q, v computations:
        # each linear is for a different head, but for all of k, q and v for this head.
        self.kqv_matrices = nn.ModuleList([create_kqv_matrix(embed_dim, n_heads) for i in range(n_heads)])
        self.output_projection = create_output_projection(embed_dim)
        # For use in the causal part.  "register_buffer" is used to store a tensor which is fixed but is not a parameter of the model.
        # You can then access it with: self.mask
        mask = create_causal_mask(embed_dim, n_heads, max_context_len)
        self.register_buffer("mask", mask)
        self.n_heads = n_heads
        self.embed_dim = embed_dim

    def forward(self, x):
        sa = multi_head_attention_layer(x, self.kqv_matrices, self.mask)
        sa = self.output_projection(sa)
        return sa
