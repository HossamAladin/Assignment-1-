# Transformer Model Debugging Assignment - Clear Guide

## Assignment Overview

This assignment tests your ability to configure and use PyCharm with WSL for debugging deep learning models. You will analyze how data flows through a Transformer model by inspecting and explaining tensor dimensions at every processing stage.

## Quick Setup Instructions

### 1. Environment Setup
```bash
# Open WSL terminal and navigate to project
cd /mnt/d/IT_level4/Assignment

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib tqdm

# Test setup
python3 test_setup.py
```

### 2. Configure PyCharm
1. Open PyCharm → Create New Project
2. Go to `File` → `Settings` → `Project` → `Python Interpreter`
3. Click gear icon → `Add...` → `WSL`
4. Select your WSL distribution and `/usr/bin/python3`
5. Create Run Configuration for `debug_transformer.py`

### 3. Start Debugging
1. Open `debug_transformer.py` in PyCharm
2. Set breakpoints at all 43 marked locations
3. Run in debug mode (`Shift + F9`)
4. Capture screenshots at each breakpoint

## Model Architecture

- **Type**: Standard encoder-decoder Transformer (Vaswani et al., 2017)
- **Encoder Layers**: 2
- **Decoder Layers**: 2
- **Attention Heads**: 4
- **Embedding Dimension**: 128
- **Vocabulary Size**: 1000 (reduced for debugging)
- **Feed-forward Dimension**: 512

---

# 📋 ASSIGNMENT REPORT TEMPLATE

## Setup Verification

### Screenshot 1: PyCharm with WSL Interpreter
**Instructions**: Take a screenshot showing PyCharm with WSL configured as the Python interpreter.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show File → Settings → Project → Python Interpreter
- Show WSL interpreter selected
- Show interpreter path: /usr/bin/python3
```

### Screenshot 2: Successful Model Run
**Instructions**: Take a screenshot showing the model running successfully in PyCharm.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debug_transformer.py running
- Show console output with "All 43 snapshots have been captured!"
- Show no errors in the output
```

---

## Snapshot Analysis (43 Screenshots Required)

### Snapshot #1 – Raw input tokens
**Expected Shape**: (1, 5)  
**Expected Values**: [1, 45, 123, 67, 89]  
**Explanation**: These are the token IDs representing the input sequence "The quick brown fox" before any processing.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'raw_input_tokens' in Variables panel
- Show tensor shape and values
- Label clearly as "Snapshot #1"
```

### Snapshot #2 – Target tokens
**Expected Shape**: (1, 5)  
**Expected Values**: [2, 156, 234, 78, 145]  
**Explanation**: These are the token IDs representing the target sequence "Le renard brun rapide" for translation.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'target_tokens' in Variables panel
- Show tensor shape and values
- Label clearly as "Snapshot #2"
```

### Snapshot #3 – Embedding weight matrix (slice)
**Expected Shape**: (1000, 128)  
**Expected Values**: 5x5 slice of embedding weights  
**Explanation**: This is the learned embedding matrix that converts token IDs to dense vectors. The slice shows the first 5 tokens' embeddings.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'embedding_weights' in Variables panel
- Show tensor shape and 5x5 slice values
- Label clearly as "Snapshot #3"
```

### Snapshot #4 – Input embeddings after lookup
**Expected Shape**: (1, 5, 128)  
**Expected Values**: Dense vectors for each token  
**Explanation**: After looking up token IDs in the embedding matrix, we get dense vector representations for each token.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'src_emb' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #4"
```

### Snapshot #5 – Embeddings after positional encoding
**Expected Shape**: (1, 5, 128)  
**Expected Values**: Embeddings with position information added  
**Explanation**: Positional encoding is added to help the model understand the order of tokens in the sequence.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'src_emb_with_pos' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #5"
```

### Snapshot #6 – Encoder block input tensor
**Expected Shape**: (1, 5, 128)  
**Expected Values**: Input to the first encoder layer  
**Explanation**: This is the input tensor that will be processed by the encoder's self-attention mechanism.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'encoder_input' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #6"
```

### Snapshot #7 – Self-attention queries (Q)
**Expected Shape**: (1, 5, 128)  
**Expected Values**: Query vectors for self-attention  
**Explanation**: Queries represent "what information am I looking for?" - they determine what the model focuses on.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'Q' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #7"
```

### Snapshot #8 – Self-attention keys (K)
**Expected Shape**: (1, 5, 128)  
**Expected Values**: Key vectors for self-attention  
**Explanation**: Keys represent "what information do I have?" - they determine what information is available to attend to.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'K' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #8"
```

### Snapshot #9 – Self-attention values (V)
**Expected Shape**: (1, 5, 128)  
**Expected Values**: Value vectors for self-attention  
**Explanation**: Values represent "what information do I provide?" - they contain the actual information that gets weighted by attention.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'V' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #9"
```

### Snapshot #10 – Attention score matrix before softmax
**Expected Shape**: (1, 5, 5)  
**Expected Values**: Raw attention scores between all token pairs  
**Explanation**: These are the raw similarity scores between queries and keys, before normalization.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'attention_scores' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #10"
```

### Snapshot #11 – Attention score matrix after softmax
**Expected Shape**: (1, 5, 5)  
**Expected Values**: Normalized attention weights (sum to 1)  
**Explanation**: Softmax converts raw scores to probabilities, ensuring all attention weights sum to 1.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'attention_weights' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #11"
```

### Snapshot #12 – Multi-head split (Q/K/V split)
**Expected Shape**: (1, 4, 5, 32)  
**Expected Values**: Q/K/V split into 4 attention heads  
**Explanation**: Multi-head attention splits the 128-dimensional vectors into 4 heads of 32 dimensions each.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'Q_heads' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #12"
```

### Snapshot #13 – Multi-head attention output after concatenation
**Expected Shape**: (1, 5, 128)  
**Expected Values**: Concatenated output from all attention heads  
**Explanation**: After processing through multiple heads, the outputs are concatenated back to 128 dimensions.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'attn_output' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #13"
```

### Snapshot #14 – Residual connection tensors
**Expected Shape**: (1, 5, 128) for both  
**Expected Values**: Input and output for residual connection  
**Explanation**: Residual connections add the input to the output, helping with gradient flow during training.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variables 'residual_input' and 'attn_output' in Variables panel
- Show tensor shapes and sample values
- Label clearly as "Snapshot #14"
```

### Snapshot #15 – Layer normalization output
**Expected Shape**: (1, 5, 128)  
**Expected Values**: Normalized tensor after residual connection  
**Explanation**: Layer normalization stabilizes the training by normalizing the activations.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'norm_output' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #15"
```

### Snapshot #16 – Feed-forward input
**Expected Shape**: (1, 5, 128)  
**Expected Values**: Input to the feed-forward network  
**Explanation**: This is the input to the position-wise feed-forward network.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'ff_input' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #16"
```

### Snapshot #17 – Feed-forward first linear layer output
**Expected Shape**: (1, 5, 512)  
**Expected Values**: Output after first linear layer + ReLU  
**Explanation**: The first linear layer expands from 128 to 512 dimensions, followed by ReLU activation.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'ff_hidden_activated' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #17"
```

### Snapshot #18 – Feed-forward second linear layer output
**Expected Shape**: (1, 5, 128)  
**Expected Values**: Output after second linear layer  
**Explanation**: The second linear layer compresses back from 512 to 128 dimensions.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'ff_output' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #18"
```

### Snapshot #19 – Encoder block final output tensor
**Expected Shape**: (1, 5, 128)  
**Expected Values**: Final output of the encoder layer  
**Explanation**: This is the final output after all processing in the encoder layer.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'encoder_final' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #19"
```

### Snapshot #20 – Decoder block input tensor
**Expected Shape**: (1, 5, 128)  
**Expected Values**: Input to the first decoder layer  
**Explanation**: This is the input tensor for the decoder, which will process the target sequence.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'decoder_input' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #20"
```

### Snapshot #21 – Masked self-attention queries (Q)
**Expected Shape**: (1, 5, 128)  
**Expected Values**: Query vectors for masked self-attention  
**Explanation**: Queries for the decoder's self-attention mechanism.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'Q_dec' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #21"
```

### Snapshot #22 – Masked self-attention keys (K)
**Expected Shape**: (1, 5, 128)  
**Expected Values**: Key vectors for masked self-attention  
**Explanation**: Keys for the decoder's self-attention mechanism.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'K_dec' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #22"
```

### Snapshot #23 – Masked self-attention values (V)
**Expected Shape**: (1, 5, 128)  
**Expected Values**: Value vectors for masked self-attention  
**Explanation**: Values for the decoder's self-attention mechanism.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'V_dec' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #23"
```

### Snapshot #24 – Masked attention scores before mask
**Expected Shape**: (1, 5, 5)  
**Expected Values**: Raw attention scores before masking  
**Explanation**: Raw attention scores before applying the causal mask.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'masked_attention_scores' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #24"
```

### Snapshot #25 – Mask tensor
**Expected Shape**: (1, 1, 5, 5)  
**Expected Values**: Causal mask (lower triangular matrix)  
**Explanation**: The causal mask prevents the model from attending to future tokens during training.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'tgt_mask' in Variables panel
- Show tensor shape and values
- Label clearly as "Snapshot #25"
```

### Snapshot #26 – Masked attention scores after mask + softmax
**Expected Shape**: (1, 1, 5, 5)  
**Expected Values**: Masked and normalized attention weights  
**Explanation**: After applying the mask and softmax, we get valid attention weights.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'masked_attention_weights' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #26"
```

### Snapshot #27 – Masked self-attention multi-head split
**Expected Shape**: (1, 4, 5, 32)  
**Expected Values**: Q/K/V split into multiple attention heads  
**Explanation**: Multi-head attention splits the vectors into 4 heads of 32 dimensions each.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'Q_dec_heads' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #27"
```

### Snapshot #28 – Masked self-attention multi-head concatenated output
**Expected Shape**: (1, 1, 5, 128)  
**Expected Values**: Concatenated output from all attention heads  
**Explanation**: After processing through multiple heads, the outputs are concatenated.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'masked_attn_output' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #28"
```

### Snapshot #29 – Residual + normalization after masked self-attention
**Expected Shape**: (1, 1, 5, 128)  
**Expected Values**: Output after residual connection and normalization  
**Explanation**: After the masked self-attention, we apply residual connection and layer normalization.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'masked_norm' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #29"
```

### Snapshot #30 – Cross-attention queries (from decoder)
**Expected Shape**: (1, 1, 5, 128)  
**Expected Values**: Query vectors for cross-attention  
**Explanation**: Queries from the decoder for attending to encoder outputs.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'Q_cross' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #30"
```

### Snapshot #31 – Cross-attention keys (from encoder)
**Expected Shape**: (1, 5, 128)  
**Expected Values**: Key vectors from encoder output  
**Explanation**: Keys from the encoder output for cross-attention.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'K_cross' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #31"
```

### Snapshot #32 – Cross-attention values (from encoder)
**Expected Shape**: (1, 5, 128)  
**Expected Values**: Value vectors from encoder output  
**Explanation**: Values from the encoder output for cross-attention.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'V_cross' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #32"
```

### Snapshot #33 – Cross-attention score matrix before softmax
**Expected Shape**: (1, 1, 5, 5)  
**Expected Values**: Raw cross-attention scores  
**Explanation**: Raw similarity scores between decoder queries and encoder keys.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'cross_attention_scores' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #33"
```

### Snapshot #34 – Cross-attention score matrix after softmax
**Expected Shape**: (1, 1, 5, 5)  
**Expected Values**: Normalized cross-attention weights  
**Explanation**: Softmax converts raw scores to probabilities for cross-attention.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'cross_attention_weights' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #34"
```

### Snapshot #35 – Cross-attention output after concatenation
**Expected Shape**: (1, 1, 5, 128)  
**Expected Values**: Cross-attention output  
**Explanation**: The weighted combination of encoder values based on cross-attention weights.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'cross_attn_output' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #35"
```

### Snapshot #36 – Residual + normalization after cross-attention
**Expected Shape**: (1, 1, 5, 128)  
**Expected Values**: Output after residual connection and normalization  
**Explanation**: After cross-attention, we apply residual connection and layer normalization.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'cross_norm' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #36"
```

### Snapshot #37 – Decoder feed-forward input
**Expected Shape**: (1, 1, 5, 128)  
**Expected Values**: Input to decoder feed-forward network  
**Explanation**: This is the input to the decoder's feed-forward network.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'decoder_ff_input' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #37"
```

### Snapshot #38 – Feed-forward first linear layer output
**Expected Shape**: (1, 1, 5, 512)  
**Expected Values**: Output after first linear layer + ReLU  
**Explanation**: The first linear layer expands from 128 to 512 dimensions, followed by ReLU activation.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'decoder_ff_hidden_activated' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #38"
```

### Snapshot #39 – Feed-forward second linear layer output
**Expected Shape**: (1, 1, 5, 128)  
**Expected Values**: Output after second linear layer  
**Explanation**: The second linear layer compresses back from 512 to 128 dimensions.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'decoder_ff_output' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #39"
```

### Snapshot #40 – Decoder block final output tensor
**Expected Shape**: (1, 1, 5, 128)  
**Expected Values**: Final output of the decoder layer  
**Explanation**: This is the final output after all processing in the decoder layer.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'decoder_final' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #40"
```

### Snapshot #41 – Decoder final sequence output (before projection)
**Expected Shape**: (1, 1, 5, 128)  
**Expected Values**: Final decoder output before projection  
**Explanation**: This is the final decoder output before the linear projection to vocabulary logits.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'decoder_final' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #41"
```

### Snapshot #42 – Logits after final linear projection
**Expected Shape**: (1, 1, 5, 1000)  
**Expected Values**: Final vocabulary logits  
**Explanation**: The final linear projection converts decoder output to vocabulary logits for each token.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'logits' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #42"
```

### Snapshot #43 – Logits slice (first few values for one token)
**Expected Shape**: (5, 1000)  
**Expected Values**: First few values for each token  
**Explanation**: A slice showing the first few logit values for each token in the sequence.

**Screenshot Space**:
```
[INSERT SCREENSHOT HERE]
- Show debugger stopped at breakpoint
- Show variable 'logits_slice' in Variables panel
- Show tensor shape and sample values
- Label clearly as "Snapshot #43"
```

---

## Answers to Guiding Questions

### Question 1: What do each of the dimensions represent at embedding, attention, feed-forward, and output stages?

**Answer**: 
- **Batch dimension (1)**: Represents the number of sequences processed simultaneously
- **Sequence length (5)**: Represents the number of tokens in the input/target sequence
- **Embedding dimension (128)**: Represents the size of the dense vector representation for each token
- **Attention heads (4)**: Represents the number of parallel attention mechanisms
- **Head dimension (32)**: Represents the size of each attention head (128/4 = 32)
- **Feed-forward dimension (512)**: Represents the expanded dimension in the feed-forward network
- **Vocabulary size (1000)**: Represents the number of possible tokens in the vocabulary

### Question 2: Why do Q, K, V tensors have the same shape, and why are they split into heads?

**Answer**: 
- **Same shape**: Q, K, V all have shape (1, 5, 128) because they are all derived from the same input tensor through different linear transformations. They represent different aspects of the same information.
- **Split into heads**: Multi-head attention allows the model to attend to different types of relationships simultaneously. Each head can focus on different aspects (syntax, semantics, etc.), and splitting into heads increases the model's representational capacity.

### Question 3: What do the attention score matrices represent, and why must they be square?

**Answer**: 
- **Attention scores**: Represent the similarity/affinity between each query token and each key token. Higher scores indicate stronger attention.
- **Square matrices**: Must be square because each token in the sequence can attend to every other token in the sequence (including itself). For a sequence of length 5, we need a 5×5 matrix where each row represents one query token and each column represents one key token.

### Question 4: Why is masking necessary in the decoder, and how does the mask tensor enforce it?

**Answer**: 
- **Why masking**: During training, the decoder must not see future tokens to prevent cheating. It should only use information from previous tokens to predict the next token.
- **How mask works**: The causal mask is a lower triangular matrix where:
  - 1s allow attention to previous and current tokens
  - 0s (or -∞) prevent attention to future tokens
  - This ensures each token can only attend to itself and previous tokens

### Question 5: How do residual connections and layer normalization ensure consistency of shapes across blocks?

**Answer**: 
- **Residual connections**: Add the input tensor to the output tensor, requiring both to have the same shape. This ensures the output shape matches the input shape.
- **Layer normalization**: Normalizes across the feature dimension (last dimension) without changing the tensor shape, maintaining consistency.
- **Shape preservation**: Both operations preserve the tensor shape, allowing the same architecture to be stacked multiple times.

### Question 6: Why must the embedding dimension remain constant through all layers?

**Answer**: 
- **Consistency**: All layers must work with the same embedding dimension to maintain compatibility
- **Residual connections**: Require input and output to have the same shape
- **Multi-head attention**: Splits the embedding dimension into heads, so the total must remain constant
- **Architecture design**: Allows the same layer to be repeated multiple times in the stack

### Question 7: How does the final projection connect decoder output to vocabulary logits?

**Answer**: 
- **Linear transformation**: The final projection is a linear layer that maps from embedding dimension (128) to vocabulary size (1000)
- **Purpose**: Converts the continuous vector representation back to discrete token probabilities
- **Process**: Each position in the sequence gets a vector of 1000 logits, representing the probability of each vocabulary token
- **Output**: The logits can be converted to probabilities using softmax for token prediction

---

## Conclusion

**Instructions**: Write a brief summary of your key insights and learnings from this debugging exercise.

**Space for Conclusion**:
```
[WRITE YOUR CONCLUSION HERE]
- What did you learn about Transformer architecture?
- What insights did you gain about attention mechanisms?
- How did debugging help your understanding?
- What was most surprising or interesting?
```

---

## Submission Checklist

- [ ] All 43 screenshots captured and clearly labeled
- [ ] Screenshots show variable values, shapes, and sample data
- [ ] All guiding questions answered thoroughly
- [ ] Setup verification screenshots included
- [ ] Conclusion written
- [ ] Code files submitted (`transformer_model.py` and `debug_transformer.py`)

---

**Good luck with your assignment! The debugging process will give you deep insights into how Transformer models work.**
