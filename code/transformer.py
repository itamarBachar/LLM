from torch import nn
import torch
import torch.nn.functional as F
import attention
import mlp

class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_heads: int, embed_size: int, mlp_hidden_size: int, max_context_len, with_residuals: bool = False, pre_norm: bool = True):
        super().__init__()
        self.causal_attention = attention.CausalSelfAttention(embed_size, n_heads, max_context_len)
        self.mlp = mlp.MLP(embed_size, mlp_hidden_size)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.with_residuals = with_residuals
        self.pre_norm = pre_norm

    def forward(self, inputs):
        x = inputs
        if self.pre_norm:
            if self.with_residuals:
                x = x + self.causal_attention(self.layer_norm_1(x))
                x = x + self.mlp(self.layer_norm_2(x))
            else:
                x = self.layer_norm_1(x)
                x = self.causal_attention(x)
                x = self.layer_norm_2(x)
                x = self.mlp(x)
        else:
            if self.with_residuals:
                x = self.layer_norm_1(x + self.causal_attention(x))
                x = self.layer_norm_2(x + self.mlp(x))
            else:
                x = self.causal_attention(x)
                x = self.layer_norm_1(x)
                x = self.mlp(x)
                x = self.layer_norm_2(x)
        return x

class Embed(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, max_context_len):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_context_len, embed_size)
        self.max_context_len = max_context_len

    def forward(self, x):
        x = x.long()
        # x has the shape (b x n) where b is batch dimension and n is sequence length.
        # each item is an int, indicating a vocabulary item.
        # The output should be of shape (b x n x d), where d is the embedding dimension.
        _, seq_len = x.shape
        if seq_len > self.max_context_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_context_len {self.max_context_len}")

        tok_embeddings = self.token_embeddings(x)
        position_ids = torch.arange(seq_len, device=x.device)
        pos_embeddings = self.position_embeddings(position_ids).unsqueeze(0)
        return tok_embeddings + pos_embeddings


class TransformerLM(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_heads: int,
            embed_size: int,
            max_context_len: int,
            vocab_size: int,
            mlp_hidden_size: int,
            with_residuals: bool,
            pre_norm: bool = True,
            ):
        super().__init__()
        self.embed = Embed(vocab_size, embed_size, max_context_len)
        self.layers = nn.ModuleList([TransformerDecoderBlock(n_heads, embed_size, mlp_hidden_size, max_context_len, with_residuals, pre_norm) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(embed_size)
        self.word_prediction = nn.Linear(embed_size, vocab_size)
        self.max_context_len = max_context_len

        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print("Parameter count: %.2fM" % (n_params/1e6,))

    def forward(self, inputs):
        x = self.embed(inputs)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        logits = self.word_prediction(x)
        return logits

    def init_weights(self):
        # initialize weights
        # Use module types here: named_parameters() yields Parameters, not modules.
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)
            elif isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def sample_continuation(self, prefix: list[int], max_tokens_to_generate: int) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        model_device = next(self.parameters()).device
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                logits = self(torch.tensor([feed_to_lm], dtype=torch.long, device=model_device))
                logits_for_last_token = logits[0][-1]
                distribution_for_last_token = F.softmax(logits_for_last_token, dim=-1)
                sampled_token = torch.multinomial(distribution_for_last_token, num_samples=1)
                sampled_token_id = int(sampled_token.item())
                generated.append(sampled_token_id)
                feed_to_lm.append(sampled_token_id)
        return generated

    def better_sample_continuation(self, prefix: list[int], max_tokens_to_generate: int, temperature: float, topK: int) -> list[int]:
        feed_to_lm = prefix[:]
        generated = []
        model_device = next(self.parameters()).device
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                logits = self(torch.tensor([feed_to_lm], dtype=torch.long, device=model_device))
                logits_for_last_token = logits[0][-1]
                
                # Apply temperature scaling
                scaled_logits = logits_for_last_token / temperature
                
                # Apply top-k filtering
                topk_logits, topk_indices = torch.topk(scaled_logits, k=topK)
                
                # Create a tensor with -inf for all positions, then fill in top-k
                filtered_logits = torch.full_like(scaled_logits, float('-inf'))
                filtered_logits[topk_indices] = topk_logits
                
                # Compute softmax over the filtered logits
                distribution_for_last_token = F.softmax(filtered_logits, dim=-1)
                
                # Sample from the distribution
                sampled_token = torch.multinomial(distribution_for_last_token, num_samples=1)
                sampled_token_id = int(sampled_token.item())
                generated.append(sampled_token_id)
                feed_to_lm.append(sampled_token_id)
        return generated

