"""
Debugging Script for Transformer Model
This script is designed to be run in PyCharm with WSL for step-by-step debugging.

Set breakpoints at the marked locations to capture the 43 required snapshots.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from transformer_model import Transformer, create_sample_data


def debug_transformer_forward_pass():
    """
    Main debugging function that steps through the Transformer forward pass.
    Set breakpoints at each marked location to capture the required snapshots.
    """
    
    # Initialize model
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
    
    print("=== TRANSFORMER DEBUGGING SESSION ===")
    print(f"Source tokens: {src}")
    print(f"Target tokens: {tgt}")
    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    
    # ========================================
    # SNAPSHOT 1: Raw input tokens
    # BREAKPOINT HERE - Inspect src tensor
    # ========================================
    raw_input_tokens = src
    print(f"\n--- SNAPSHOT 1: Raw input tokens ---")
    print(f"Shape: {raw_input_tokens.shape}")
    print(f"Values: {raw_input_tokens}")
    
    # ========================================
    # SNAPSHOT 2: Target tokens  
    # BREAKPOINT HERE - Inspect tgt tensor
    # ========================================
    target_tokens = tgt
    print(f"\n--- SNAPSHOT 2: Target tokens ---")
    print(f"Shape: {target_tokens.shape}")
    print(f"Values: {target_tokens}")
    
    # ========================================
    # SNAPSHOT 3: Embedding weight matrix (slice)
    # BREAKPOINT HERE - Inspect model.embedding.weight
    # ========================================
    embedding_weights = model.embedding.weight
    print(f"\n--- SNAPSHOT 3: Embedding weight matrix (slice) ---")
    print(f"Shape: {embedding_weights.shape}")
    print(f"Slice (5x5): {embedding_weights[:5, :5]}")
    
    # ========================================
    # SNAPSHOT 4: Input embeddings after lookup
    # BREAKPOINT HERE - Inspect src_emb
    # ========================================
    src_emb = model.embedding(src) * math.sqrt(model.d_model)
    print(f"\n--- SNAPSHOT 4: Input embeddings after lookup ---")
    print(f"Shape: {src_emb.shape}")
    print(f"Sample values: {src_emb[0, :2, :5]}")
    
    # ========================================
    # SNAPSHOT 5: Embeddings after adding positional encoding
    # BREAKPOINT HERE - Inspect src_emb_with_pos
    # ========================================
    src_emb_with_pos = model.pos_encoding(src_emb.transpose(0, 1)).transpose(0, 1)
    print(f"\n--- SNAPSHOT 5: Embeddings after positional encoding ---")
    print(f"Shape: {src_emb_with_pos.shape}")
    print(f"Sample values: {src_emb_with_pos[0, :2, :5]}")
    
    # Target embeddings
    tgt_emb = model.embedding(tgt) * math.sqrt(model.d_model)
    tgt_emb_with_pos = model.pos_encoding(tgt_emb.transpose(0, 1)).transpose(0, 1)
    
    # Create decoder mask
    tgt_mask = model.create_decoder_mask(tgt.size(1))
    
    # ========================================
    # ENCODER LAYER 1 DEBUGGING
    # ========================================
    encoder_layer = model.encoder_layers[0]
    
    # ========================================
    # SNAPSHOT 6: Encoder block input tensor
    # BREAKPOINT HERE - Inspect encoder_input
    # ========================================
    encoder_input = src_emb_with_pos
    print(f"\n--- SNAPSHOT 6: Encoder block input tensor ---")
    print(f"Shape: {encoder_input.shape}")
    print(f"Sample values: {encoder_input[0, :2, :5]}")
    
    # Self-attention computation
    Q = encoder_layer.self_attn.W_q(encoder_input)
    K = encoder_layer.self_attn.W_k(encoder_input)
    V = encoder_layer.self_attn.W_v(encoder_input)
    
    # ========================================
    # SNAPSHOT 7: Self-attention queries (Q)
    # BREAKPOINT HERE - Inspect Q
    # ========================================
    print(f"\n--- SNAPSHOT 7: Self-attention queries (Q) ---")
    print(f"Shape: {Q.shape}")
    print(f"Sample values: {Q[0, :2, :5]}")
    
    # ========================================
    # SNAPSHOT 8: Self-attention keys (K)
    # BREAKPOINT HERE - Inspect K
    # ========================================
    print(f"\n--- SNAPSHOT 8: Self-attention keys (K) ---")
    print(f"Shape: {K.shape}")
    print(f"Sample values: {K[0, :2, :5]}")
    
    # ========================================
    # SNAPSHOT 9: Self-attention values (V)
    # BREAKPOINT HERE - Inspect V
    # ========================================
    print(f"\n--- SNAPSHOT 9: Self-attention values (V) ---")
    print(f"Shape: {V.shape}")
    print(f"Sample values: {V[0, :2, :5]}")
    
    # Compute attention scores
    d_k = Q.size(-1)
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # ========================================
    # SNAPSHOT 10: Attention score matrix before softmax
    # BREAKPOINT HERE - Inspect attention_scores
    # ========================================
    print(f"\n--- SNAPSHOT 10: Attention scores before softmax ---")
    print(f"Shape: {attention_scores.shape}")
    print(f"Sample values: {attention_scores[0, :2, :2]}")
    
    # Apply softmax
    attention_weights = F.softmax(attention_scores, dim=-1)
    
    # ========================================
    # SNAPSHOT 11: Attention score matrix after softmax
    # BREAKPOINT HERE - Inspect attention_weights
    # ========================================
    print(f"\n--- SNAPSHOT 11: Attention scores after softmax ---")
    print(f"Shape: {attention_weights.shape}")
    print(f"Sample values: {attention_weights[0, :2, :2]}")
    
    # Multi-head split
    batch_size = Q.size(0)
    seq_len = Q.size(1)
    Q_heads = Q.view(batch_size, seq_len, 4, 32).transpose(1, 2)
    K_heads = K.view(batch_size, seq_len, 4, 32).transpose(1, 2)
    V_heads = V.view(batch_size, seq_len, 4, 32).transpose(1, 2)
    
    # ========================================
    # SNAPSHOT 12: Multi-head split (Q/K/V split)
    # BREAKPOINT HERE - Inspect Q_heads
    # ========================================
    print(f"\n--- SNAPSHOT 12: Multi-head split ---")
    print(f"Q_heads shape: {Q_heads.shape}")
    print(f"Sample values: {Q_heads[0, 0, :2, :5]}")
    
    # Multi-head attention output
    attn_output = torch.matmul(attention_weights, V)
    
    # ========================================
    # SNAPSHOT 13: Multi-head attention output after concatenation
    # BREAKPOINT HERE - Inspect attn_output
    # ========================================
    print(f"\n--- SNAPSHOT 13: Multi-head attention output ---")
    print(f"Shape: {attn_output.shape}")
    print(f"Sample values: {attn_output[0, :2, :5]}")
    
    # Residual connection
    residual_input = encoder_input
    attn_output_with_residual = attn_output + residual_input
    
    # ========================================
    # SNAPSHOT 14: Residual connection tensors
    # BREAKPOINT HERE - Inspect residual_input and attn_output
    # ========================================
    print(f"\n--- SNAPSHOT 14: Residual connection tensors ---")
    print(f"Residual input shape: {residual_input.shape}")
    print(f"Attention output shape: {attn_output.shape}")
    print(f"Combined shape: {attn_output_with_residual.shape}")
    
    # Layer normalization
    norm_output = encoder_layer.norm1(attn_output_with_residual)
    
    # ========================================
    # SNAPSHOT 15: Layer normalization output
    # BREAKPOINT HERE - Inspect norm_output
    # ========================================
    print(f"\n--- SNAPSHOT 15: Layer normalization output ---")
    print(f"Shape: {norm_output.shape}")
    print(f"Sample values: {norm_output[0, :2, :5]}")
    
    # Feed-forward network
    ff_input = norm_output
    
    # ========================================
    # SNAPSHOT 16: Feed-forward input
    # BREAKPOINT HERE - Inspect ff_input
    # ========================================
    print(f"\n--- SNAPSHOT 16: Feed-forward input ---")
    print(f"Shape: {ff_input.shape}")
    print(f"Sample values: {ff_input[0, :2, :5]}")
    
    # First linear layer
    ff_hidden = encoder_layer.feed_forward.linear1(ff_input)
    ff_hidden_activated = F.relu(ff_hidden)
    
    # ========================================
    # SNAPSHOT 17: Feed-forward first linear layer output
    # BREAKPOINT HERE - Inspect ff_hidden_activated
    # ========================================
    print(f"\n--- SNAPSHOT 17: Feed-forward first linear layer output ---")
    print(f"Shape: {ff_hidden_activated.shape}")
    print(f"Sample values: {ff_hidden_activated[0, :2, :5]}")
    
    # Second linear layer
    ff_output = encoder_layer.feed_forward.linear2(ff_hidden_activated)
    
    # ========================================
    # SNAPSHOT 18: Feed-forward second linear layer output
    # BREAKPOINT HERE - Inspect ff_output
    # ========================================
    print(f"\n--- SNAPSHOT 18: Feed-forward second linear layer output ---")
    print(f"Shape: {ff_output.shape}")
    print(f"Sample values: {ff_output[0, :2, :5]}")
    
    # Final encoder layer output
    encoder_final = encoder_layer.norm2(ff_output + norm_output)
    
    # ========================================
    # SNAPSHOT 19: Encoder block final output tensor
    # BREAKPOINT HERE - Inspect encoder_final
    # ========================================
    print(f"\n--- SNAPSHOT 19: Encoder block final output tensor ---")
    print(f"Shape: {encoder_final.shape}")
    print(f"Sample values: {encoder_final[0, :2, :5]}")
    
    # ========================================
    # DECODER LAYER 1 DEBUGGING
    # ========================================
    decoder_layer = model.decoder_layers[0]
    
    # ========================================
    # SNAPSHOT 20: Decoder block input tensor
    # BREAKPOINT HERE - Inspect decoder_input
    # ========================================
    decoder_input = tgt_emb_with_pos
    print(f"\n--- SNAPSHOT 20: Decoder block input tensor ---")
    print(f"Shape: {decoder_input.shape}")
    print(f"Sample values: {decoder_input[0, :2, :5]}")
    
    # Masked self-attention
    Q_dec = decoder_layer.self_attn.W_q(decoder_input)
    K_dec = decoder_layer.self_attn.W_k(decoder_input)
    V_dec = decoder_layer.self_attn.W_v(decoder_input)
    
    # ========================================
    # SNAPSHOT 21: Masked self-attention queries (Q)
    # BREAKPOINT HERE - Inspect Q_dec
    # ========================================
    print(f"\n--- SNAPSHOT 21: Masked self-attention queries (Q) ---")
    print(f"Shape: {Q_dec.shape}")
    print(f"Sample values: {Q_dec[0, :2, :5]}")
    
    # ========================================
    # SNAPSHOT 22: Masked self-attention keys (K)
    # BREAKPOINT HERE - Inspect K_dec
    # ========================================
    print(f"\n--- SNAPSHOT 22: Masked self-attention keys (K) ---")
    print(f"Shape: {K_dec.shape}")
    print(f"Sample values: {K_dec[0, :2, :5]}")
    
    # ========================================
    # SNAPSHOT 23: Masked self-attention values (V)
    # BREAKPOINT HERE - Inspect V_dec
    # ========================================
    print(f"\n--- SNAPSHOT 23: Masked self-attention values (V) ---")
    print(f"Shape: {V_dec.shape}")
    print(f"Sample values: {V_dec[0, :2, :5]}")
    
    # Compute masked attention scores
    d_k = Q_dec.size(-1)
    masked_attention_scores = torch.matmul(Q_dec, K_dec.transpose(-2, -1)) / math.sqrt(d_k)
    
    # ========================================
    # SNAPSHOT 24: Masked attention scores before mask
    # BREAKPOINT HERE - Inspect masked_attention_scores
    # ========================================
    print(f"\n--- SNAPSHOT 24: Masked attention scores before mask ---")
    print(f"Shape: {masked_attention_scores.shape}")
    print(f"Sample values: {masked_attention_scores[0, :2, :2]}")
    
    # ========================================
    # SNAPSHOT 25: Mask tensor
    # BREAKPOINT HERE - Inspect tgt_mask
    # ========================================
    print(f"\n--- SNAPSHOT 25: Mask tensor ---")
    print(f"Shape: {tgt_mask.shape}")
    print(f"Values: {tgt_mask[0, 0]}")
    
    # Apply mask
    masked_scores = masked_attention_scores.masked_fill(tgt_mask == 0, -1e9)
    masked_attention_weights = F.softmax(masked_scores, dim=-1)
    
    # ========================================
    # SNAPSHOT 26: Masked attention scores after mask + softmax
    # BREAKPOINT HERE - Inspect masked_attention_weights
    # ========================================
    print(f"\n--- SNAPSHOT 26: Masked attention scores after mask + softmax ---")
    print(f"Shape: {masked_attention_weights.shape}")
    print(f"Sample values: {masked_attention_weights[0, :2, :2]}")
    
    # Multi-head split for decoder
    Q_dec_heads = Q_dec.view(batch_size, seq_len, 4, 32).transpose(1, 2)
    
    # ========================================
    # SNAPSHOT 27: Masked self-attention multi-head split
    # BREAKPOINT HERE - Inspect Q_dec_heads
    # ========================================
    print(f"\n--- SNAPSHOT 27: Masked self-attention multi-head split ---")
    print(f"Shape: {Q_dec_heads.shape}")
    print(f"Sample values: {Q_dec_heads[0, 0, :2, :5]}")
    
    # Masked self-attention output
    masked_attn_output = torch.matmul(masked_attention_weights, V_dec)
    
    # ========================================
    # SNAPSHOT 28: Masked self-attention multi-head concatenated output
    # BREAKPOINT HERE - Inspect masked_attn_output
    # ========================================
    print(f"\n--- SNAPSHOT 28: Masked self-attention multi-head concatenated output ---")
    print(f"Shape: {masked_attn_output.shape}")
    print(f"Sample values: {masked_attn_output[0, :2, :5]}")
    
    # Residual + normalization after masked self-attention
    masked_residual = decoder_input + masked_attn_output
    masked_norm = decoder_layer.norm1(masked_residual)
    
    # ========================================
    # SNAPSHOT 29: Residual + normalization after masked self-attention
    # BREAKPOINT HERE - Inspect masked_norm
    # ========================================
    print(f"\n--- SNAPSHOT 29: Residual + normalization after masked self-attention ---")
    print(f"Shape: {masked_norm.shape}")
    print(f"Sample values: {masked_norm[0, :2, :5]}")
    
    # Cross-attention
    Q_cross = decoder_layer.cross_attn.W_q(masked_norm)
    K_cross = decoder_layer.cross_attn.W_k(encoder_final)
    V_cross = decoder_layer.cross_attn.W_v(encoder_final)
    
    # ========================================
    # SNAPSHOT 30: Cross-attention queries (from decoder)
    # BREAKPOINT HERE - Inspect Q_cross
    # ========================================
    print(f"\n--- SNAPSHOT 30: Cross-attention queries (from decoder) ---")
    print(f"Shape: {Q_cross.shape}")
    print(f"Sample values: {Q_cross[0, :2, :5]}")
    
    # ========================================
    # SNAPSHOT 31: Cross-attention keys (from encoder)
    # BREAKPOINT HERE - Inspect K_cross
    # ========================================
    print(f"\n--- SNAPSHOT 31: Cross-attention keys (from encoder) ---")
    print(f"Shape: {K_cross.shape}")
    print(f"Sample values: {K_cross[0, :2, :5]}")
    
    # ========================================
    # SNAPSHOT 32: Cross-attention values (from encoder)
    # BREAKPOINT HERE - Inspect V_cross
    # ========================================
    print(f"\n--- SNAPSHOT 32: Cross-attention values (from encoder) ---")
    print(f"Shape: {V_cross.shape}")
    print(f"Sample values: {V_cross[0, :2, :5]}")
    
    # Cross-attention scores
    cross_attention_scores = torch.matmul(Q_cross, K_cross.transpose(-2, -1)) / math.sqrt(d_k)
    
    # ========================================
    # SNAPSHOT 33: Cross-attention score matrix before softmax
    # BREAKPOINT HERE - Inspect cross_attention_scores
    # ========================================
    print(f"\n--- SNAPSHOT 33: Cross-attention score matrix before softmax ---")
    print(f"Shape: {cross_attention_scores.shape}")
    print(f"Sample values: {cross_attention_scores[0, :2, :2]}")
    
    cross_attention_weights = F.softmax(cross_attention_scores, dim=-1)
    
    # ========================================
    # SNAPSHOT 34: Cross-attention score matrix after softmax
    # BREAKPOINT HERE - Inspect cross_attention_weights
    # ========================================
    print(f"\n--- SNAPSHOT 34: Cross-attention score matrix after softmax ---")
    print(f"Shape: {cross_attention_weights.shape}")
    print(f"Sample values: {cross_attention_weights[0, :2, :2]}")
    
    # Cross-attention output
    cross_attn_output = torch.matmul(cross_attention_weights, V_cross)
    
    # ========================================
    # SNAPSHOT 35: Cross-attention output after concatenation
    # BREAKPOINT HERE - Inspect cross_attn_output
    # ========================================
    print(f"\n--- SNAPSHOT 35: Cross-attention output after concatenation ---")
    print(f"Shape: {cross_attn_output.shape}")
    print(f"Sample values: {cross_attn_output[0, :2, :5]}")
    
    # Residual + normalization after cross-attention
    cross_residual = masked_norm + cross_attn_output
    cross_norm = decoder_layer.norm2(cross_residual)
    
    # ========================================
    # SNAPSHOT 36: Residual + normalization after cross-attention
    # BREAKPOINT HERE - Inspect cross_norm
    # ========================================
    print(f"\n--- SNAPSHOT 36: Residual + normalization after cross-attention ---")
    print(f"Shape: {cross_norm.shape}")
    print(f"Sample values: {cross_norm[0, :2, :5]}")
    
    # Decoder feed-forward
    decoder_ff_input = cross_norm
    
    # ========================================
    # SNAPSHOT 37: Decoder feed-forward input
    # BREAKPOINT HERE - Inspect decoder_ff_input
    # ========================================
    print(f"\n--- SNAPSHOT 37: Decoder feed-forward input ---")
    print(f"Shape: {decoder_ff_input.shape}")
    print(f"Sample values: {decoder_ff_input[0, :2, :5]}")
    
    # Decoder feed-forward layers
    decoder_ff_hidden = decoder_layer.feed_forward.linear1(decoder_ff_input)
    decoder_ff_hidden_activated = F.relu(decoder_ff_hidden)
    
    # ========================================
    # SNAPSHOT 38: Feed-forward first linear layer output
    # BREAKPOINT HERE - Inspect decoder_ff_hidden_activated
    # ========================================
    print(f"\n--- SNAPSHOT 38: Feed-forward first linear layer output ---")
    print(f"Shape: {decoder_ff_hidden_activated.shape}")
    print(f"Sample values: {decoder_ff_hidden_activated[0, :2, :5]}")
    
    decoder_ff_output = decoder_layer.feed_forward.linear2(decoder_ff_hidden_activated)
    
    # ========================================
    # SNAPSHOT 39: Feed-forward second linear layer output
    # BREAKPOINT HERE - Inspect decoder_ff_output
    # ========================================
    print(f"\n--- SNAPSHOT 39: Feed-forward second linear layer output ---")
    print(f"Shape: {decoder_ff_output.shape}")
    print(f"Sample values: {decoder_ff_output[0, :2, :5]}")
    
    # Decoder block final output
    decoder_final = decoder_layer.norm3(decoder_ff_output + cross_norm)
    
    # ========================================
    # SNAPSHOT 40: Decoder block final output tensor
    # BREAKPOINT HERE - Inspect decoder_final
    # ========================================
    print(f"\n--- SNAPSHOT 40: Decoder block final output tensor ---")
    print(f"Shape: {decoder_final.shape}")
    print(f"Sample values: {decoder_final[0, :2, :5]}")
    
    # ========================================
    # FINAL OUTPUT
    # ========================================
    
    # ========================================
    # SNAPSHOT 41: Decoder final sequence output (before projection)
    # BREAKPOINT HERE - Inspect decoder_final
    # ========================================
    print(f"\n--- SNAPSHOT 41: Decoder final sequence output (before projection) ---")
    print(f"Shape: {decoder_final.shape}")
    print(f"Sample values: {decoder_final[0, :2, :5]}")
    
    # Final linear projection
    logits = model.output_projection(decoder_final)
    
    # ========================================
    # SNAPSHOT 42: Logits after final linear projection
    # BREAKPOINT HERE - Inspect logits
    # ========================================
    print(f"\n--- SNAPSHOT 42: Logits after final linear projection ---")
    print(f"Shape: {logits.shape}")
    print(f"Sample values: {logits[0, :2, :5]}")
    
    # ========================================
    # SNAPSHOT 43: Logits slice (first few values for one token)
    # BREAKPOINT HERE - Inspect logits_slice
    # ========================================
    logits_slice = logits[0, 0, :10]  # First token, first 10 values
    print(f"\n--- SNAPSHOT 43: Logits slice (first token) ---")
    print(f"Shape: {logits_slice.shape}")
    print(f"Values: {logits_slice}")
    
    print("\n=== DEBUGGING SESSION COMPLETE ===")
    print("All 43 snapshots have been captured!")
    
    return logits, encoder_final, decoder_final


if __name__ == "__main__":
    # Run the debugging session
    debug_transformer_forward_pass()
