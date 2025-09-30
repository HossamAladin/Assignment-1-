# Transformer Model Debugging Assignment

This repository contains the complete implementation and debugging setup for the Transformer model debugging assignment using PyCharm and WSL.

## Project Structure

```
Assignment/
├── transformer_model.py      # Complete Transformer implementation
├── debug_transformer.py     # Debugging script with 43 snapshot points
├── requirements.txt         # Python dependencies
├── setup_wsl_pycharm.md    # Detailed setup instructions
└── README.md               # This file
```

## Quick Start

### 1. Environment Setup
```bash
# Install dependencies in WSL
pip3 install -r requirements.txt
```

### 2. Configure PyCharm
1. Follow the detailed instructions in `setup_wsl_pycharm.md`
2. Set up WSL as your Python interpreter
3. Configure debugging settings

### 3. Run Debugging Session
1. Open `debug_transformer.py` in PyCharm
2. Set breakpoints at the 43 marked locations
3. Run in debug mode (`Shift + F9`)
4. Capture screenshots at each breakpoint

## Model Specifications

- **Architecture**: Standard encoder-decoder Transformer (Vaswani et al., 2017)
- **Encoder Layers**: 2
- **Decoder Layers**: 2
- **Attention Heads**: 4
- **Embedding Dimension**: 128
- **Vocabulary Size**: 1000 (reduced for debugging)
- **Feed-forward Dimension**: 512

## Required Snapshots

The debugging script captures 43 specific snapshots:

### Input & Embedding (Snapshots 1-5)
1. Raw input tokens
2. Target tokens
3. Embedding weight matrix (slice)
4. Input embeddings after lookup
5. Embeddings after positional encoding

### Encoder Layer (Snapshots 6-19)
6. Encoder block input tensor
7. Self-attention queries (Q)
8. Self-attention keys (K)
9. Self-attention values (V)
10. Attention score matrix before softmax
11. Attention score matrix after softmax
12. Multi-head split (Q/K/V split)
13. Multi-head attention output after concatenation
14. Residual connection tensors
15. Layer normalization output
16. Feed-forward input
17. Feed-forward first linear layer output
18. Feed-forward second linear layer output
19. Encoder block final output tensor

### Decoder Layer (Snapshots 20-40)
20. Decoder block input tensor
21. Masked self-attention queries (Q)
22. Masked self-attention keys (K)
23. Masked self-attention values (V)
24. Masked attention scores before mask
25. Mask tensor
26. Masked attention scores after mask + softmax
27. Masked self-attention multi-head split
28. Masked self-attention multi-head concatenated output
29. Residual + normalization after masked self-attention
30. Cross-attention queries (from decoder)
31. Cross-attention keys (from encoder)
32. Cross-attention values (from encoder)
33. Cross-attention score matrix before softmax
34. Cross-attention score matrix after softmax
35. Cross-attention output after concatenation
36. Residual + normalization after cross-attention
37. Decoder feed-forward input
38. Feed-forward first linear layer output
39. Feed-forward second linear layer output
40. Decoder block final output tensor

### Final Output (Snapshots 41-43)
41. Decoder final sequence output (before projection)
42. Logits after final linear projection
43. Logits slice (first few values for one token)

## Sample Data

The model uses sample input and target sequences:
- **Input**: "The quick brown fox" (token IDs: [1, 45, 123, 67, 89])
- **Target**: "Le renard brun rapide" (token IDs: [2, 156, 234, 78, 145])

## Debugging Instructions

1. **Set Breakpoints**: Place breakpoints at each of the 43 marked locations in `debug_transformer.py`
2. **Run Debugger**: Use `Shift + F9` to start debugging
3. **Capture Snapshots**: At each breakpoint, capture:
   - Variable name and value
   - Tensor shape
   - Sample tensor values
   - Clear labeling with snapshot number
4. **Document**: Record explanations for each snapshot

## Guiding Questions

Answer these questions alongside your snapshots:

1. What do each of the dimensions represent at embedding, attention, feed-forward, and output stages?
2. Why do Q, K, V tensors have the same shape, and why are they split into heads?
3. What do the attention score matrices represent, and why must they be square?
4. Why is masking necessary in the decoder, and how does the mask tensor enforce it?
5. How do residual connections and layer normalization ensure consistency of shapes across blocks?
6. Why must the embedding dimension remain constant through all layers?
7. How does the final projection connect decoder output to vocabulary logits?

## Troubleshooting

### Common Issues

1. **WSL Connection Issues**: Ensure WSL2 is running and PyCharm is configured correctly
2. **Import Errors**: Install all dependencies in WSL using `pip3 install -r requirements.txt`
3. **Debugger Not Stopping**: Verify breakpoints are enabled and running in debug mode
4. **Tensor Display Issues**: Use `torch.tensor.detach().numpy()` for better inspection

### Getting Help

1. Check the detailed setup guide in `setup_wsl_pycharm.md`
2. Verify your WSL installation: `wsl --status`
3. Test Python in WSL: `wsl python3 --version`
4. Check PyTorch installation: `wsl python3 -c "import torch; print(torch.__version__)"`

## Assignment Submission

Submit the following:
1. **43 Screenshots**: One for each snapshot with clear labeling
2. **Written Report**: Answers to all guiding questions
3. **Code Files**: `transformer_model.py` and `debug_transformer.py`

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [PyCharm WSL Documentation](https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
# Assignment_1_
