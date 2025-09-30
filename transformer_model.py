"""
Transformer Model Implementation for Debugging Assignment
Based on "Attention Is All You Need" (Vaswani et al., 2017)

This implementation is designed for step-by-step debugging with PyCharm.
Model specifications:
- 2 encoder layers, 2 decoder layers
- 4 attention heads
- Embedding dimension = 128
- Vocabulary size = 1000 (reduced for debugging)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model, num_heads, is_decoder=False):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.is_decoder = is_decoder
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute scaled dot-product attention."""
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights, scores
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear transformations
        Q = self.W_q(query)  # (batch_size, seq_len, d_model)
        K = self.W_k(key)    # (batch_size, seq_len, d_model)
        V = self.W_v(value)  # (batch_size, seq_len, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, key.size(1), self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, value.size(1), self.num_heads, self.d_k).transpose(1, 2)
        # Shape: (batch_size, num_heads, seq_len, d_k)
        
        # Apply attention
        attn_output, attention_weights, attention_scores = self.scaled_dot_product_attention(
            Q, K, V, mask
        )
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear transformation
        output = self.W_o(attn_output)
        
        return output, attention_weights, attention_scores


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """Single encoder layer."""
    
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Self-attention
        attn_output, _, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """Single decoder layer."""
    
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, is_decoder=True)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, encoder_output, mask=None):
        # Masked self-attention
        self_attn_output, _, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-attention
        cross_attn_output, _, _ = self.cross_attn(x, encoder_output, encoder_output)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Transformer(nn.Module):
    """Complete Transformer model."""
    
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_encoder_layers=2, 
                 num_decoder_layers=2, d_ff=512, max_len=100):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff) 
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff) 
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def create_decoder_mask(self, seq_len):
        """Create causal mask for decoder self-attention."""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    def forward(self, src, tgt):
        # Source embeddings
        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb.transpose(0, 1)).transpose(0, 1)
        
        # Target embeddings
        tgt_emb = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)
        
        # Create decoder mask
        tgt_mask = self.create_decoder_mask(tgt.size(1))
        
        # Encoder
        encoder_output = src_emb
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)
        
        # Decoder
        decoder_output = tgt_emb
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, tgt_mask)
        
        # Output projection
        logits = self.output_projection(decoder_output)
        
        return logits, encoder_output, decoder_output


def create_sample_data():
    """Create sample input and target sequences for debugging."""
    # Sample input: "The quick brown fox"
    src_tokens = [1, 45, 123, 67, 89]  # token IDs
    
    # Sample target: "Le renard brun rapide" (French translation)
    tgt_tokens = [2, 156, 234, 78, 145]  # token IDs
    
    # Convert to tensors
    src = torch.tensor([src_tokens], dtype=torch.long)  # (1, 5)
    tgt = torch.tensor([tgt_tokens], dtype=torch.long)  # (1, 5)
    
    return src, tgt


if __name__ == "__main__":
    # Create model
    model = Transformer(
        vocab_size=1000,
        d_model=128,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512
    )
    
    # Create sample data
    src, tgt = create_sample_data()
    
    print("Model created successfully!")
    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Forward pass (for debugging)
    with torch.no_grad():
        logits, encoder_output, decoder_output = model(src, tgt)
        print(f"Logits shape: {logits.shape}")
        print(f"Encoder output shape: {encoder_output.shape}")
        print(f"Decoder output shape: {decoder_output.shape}")
