# Transformer Model Debugging Assignment - Complete Guide

## Assignment Overview

This assignment tests your ability to configure and use PyCharm with WSL for debugging deep learning models. You will analyze how data flows through a Transformer model by inspecting and explaining tensor dimensions at every processing stage.

## Learning Objectives

- ✅ Set up and use PyCharm with WSL correctly
- ✅ Debug a Transformer model step by step using PyCharm debugger
- ✅ Understand how tensors move through all Transformer layers
- ✅ Capture 43 numbered, labeled snapshots with values and shapes
- ✅ Answer guiding questions about Transformer architecture

## Project Structure

```
Assignment/
├── transformer_model.py      # Complete Transformer implementation
├── debug_transformer.py     # Debugging script with 43 snapshot points
├── test_setup.py           # Setup verification script
├── requirements.txt        # Python dependencies
├── setup_wsl_pycharm.md   # Detailed setup instructions
├── README.md              # Project overview
└── ASSIGNMENT_GUIDE.md    # This comprehensive guide
```

## Step 1: Environment Setup

### 1.1 Install WSL2 (if not already installed)
```bash
# In PowerShell as Administrator
wsl --install
# Restart computer after installation
```

### 1.2 Install Python and Dependencies in WSL
```bash
# Update package list
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip python3-venv

# Install PyTorch (CPU version for debugging)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip3 install numpy matplotlib tqdm
```

### 1.3 Test Your Setup
```bash
# Run the test script to verify everything works
python3 test_setup.py
```

## Step 2: Configure PyCharm

### 2.1 Open PyCharm and Create New Project
1. Open PyCharm
2. Create a new project in your Windows file system
3. Navigate to the project folder

### 2.2 Configure Python Interpreter
1. Go to `File` → `Settings` (or `PyCharm` → `Preferences` on Mac)
2. Navigate to `Project` → `Python Interpreter`
3. Click the gear icon → `Add...`
4. Select `WSL` from the left panel
5. Choose your WSL distribution (usually Ubuntu)
6. Select the Python interpreter path (usually `/usr/bin/python3`)
7. Click `OK`

### 2.3 Create Run Configuration
1. Go to `Run` → `Edit Configurations...`
2. Click `+` → `Python`
3. Set the following:
   - **Name**: `Debug Transformer`
   - **Script path**: `debug_transformer.py`
   - **Python interpreter**: WSL Python interpreter
   - **Working directory**: Your project directory

## Step 3: Understanding the Transformer Model

### 3.1 Model Architecture
- **Type**: Standard encoder-decoder Transformer (Vaswani et al., 2017)
- **Encoder Layers**: 2
- **Decoder Layers**: 2
- **Attention Heads**: 4
- **Embedding Dimension**: 128
- **Vocabulary Size**: 1000 (reduced for debugging)
- **Feed-forward Dimension**: 512

### 3.2 Key Components
1. **Embeddings**: Convert token IDs to dense vectors
2. **Positional Encoding**: Add position information to embeddings
3. **Encoder**: Processes input sequence with self-attention
4. **Decoder**: Generates output sequence with masked self-attention and cross-attention
5. **Attention**: Computes relationships between tokens
6. **Feed-forward**: Applies non-linear transformations
7. **Residual Connections**: Helps with gradient flow
8. **Layer Normalization**: Stabilizes training

## Step 4: Debugging Process

### 4.1 Set Breakpoints
Open `debug_transformer.py` and set breakpoints at each of the 43 marked locations:

```python
# Example breakpoint locations:
# Line 45:  SNAPSHOT 1: Raw input tokens
# Line 52:  SNAPSHOT 2: Target tokens
# Line 59:  SNAPSHOT 3: Embedding weight matrix
# ... and so on for all 43 snapshots
```

### 4.2 Run Debugging Session
1. Set breakpoints at all 43 snapshot locations
2. Run in debug mode (`Shift + F9`)
3. At each breakpoint, capture:
   - Variable name and value
   - Tensor shape
   - Sample tensor values
   - Clear labeling with snapshot number

### 4.3 Capture Screenshots
For each snapshot, take a screenshot showing:
- The debugger stopped at the breakpoint
- Variable values in the Variables panel
- Tensor shapes and sample values
- Clear labeling with snapshot number

## Step 5: Required Snapshots (43 Total)

### Input & Embedding (Snapshots 1-5)
1. **Raw input tokens** - Token IDs before embedding lookup
2. **Target tokens** - Target sequence token IDs
3. **Embedding weight matrix** - Learned embedding weights (slice)
4. **Input embeddings after lookup** - Dense vectors from embedding layer
5. **Embeddings after positional encoding** - Embeddings with position information

### Encoder Layer (Snapshots 6-19)
6. **Encoder block input tensor** - Input to first encoder layer
7. **Self-attention queries (Q)** - Query vectors for self-attention
8. **Self-attention keys (K)** - Key vectors for self-attention
9. **Self-attention values (V)** - Value vectors for self-attention
10. **Attention score matrix before softmax** - Raw attention scores
11. **Attention score matrix after softmax** - Normalized attention weights
12. **Multi-head split** - Q/K/V split into multiple attention heads
13. **Multi-head attention output** - Concatenated attention output
14. **Residual connection tensors** - Input and output for residual connection
15. **Layer normalization output** - After first layer normalization
16. **Feed-forward input** - Input to feed-forward network
17. **Feed-forward first linear layer output** - After first linear layer + ReLU
18. **Feed-forward second linear layer output** - After second linear layer
19. **Encoder block final output** - Final output of encoder layer

### Decoder Layer (Snapshots 20-40)
20. **Decoder block input tensor** - Input to first decoder layer
21. **Masked self-attention queries (Q)** - Query vectors for masked self-attention
22. **Masked self-attention keys (K)** - Key vectors for masked self-attention
23. **Masked self-attention values (V)** - Value vectors for masked self-attention
24. **Masked attention scores before mask** - Raw scores before masking
25. **Mask tensor** - Causal mask for preventing future token attention
26. **Masked attention scores after mask + softmax** - Masked and normalized scores
27. **Masked self-attention multi-head split** - Q/K/V split for masked attention
28. **Masked self-attention multi-head concatenated output** - Output after concatenation
29. **Residual + normalization after masked self-attention** - After first residual + norm
30. **Cross-attention queries (from decoder)** - Query vectors for cross-attention
31. **Cross-attention keys (from encoder)** - Key vectors from encoder output
32. **Cross-attention values (from encoder)** - Value vectors from encoder output
33. **Cross-attention score matrix before softmax** - Raw cross-attention scores
34. **Cross-attention score matrix after softmax** - Normalized cross-attention weights
35. **Cross-attention output after concatenation** - Cross-attention output
36. **Residual + normalization after cross-attention** - After second residual + norm
37. **Decoder feed-forward input** - Input to decoder feed-forward network
38. **Feed-forward first linear layer output** - After first linear layer + ReLU
39. **Feed-forward second linear layer output** - After second linear layer
40. **Decoder block final output** - Final output of decoder layer

### Final Output (Snapshots 41-43)
41. **Decoder final sequence output** - Before final projection layer
42. **Logits after final linear projection** - Final vocabulary logits
43. **Logits slice** - First few values for one token

## Step 6: Guiding Questions (Must Answer)

### 6.1 Tensor Dimensions
- What do each of the dimensions represent at embedding, attention, feed-forward, and output stages?
- Why do Q, K, V tensors have the same shape, and why are they split into heads?
- Why must the embedding dimension remain constant through all layers?

### 6.2 Attention Mechanism
- What do the attention score matrices represent, and why must they be square?
- Why is masking necessary in the decoder, and how does the mask tensor enforce it?
- How does cross-attention differ from self-attention?

### 6.3 Architecture Design
- How do residual connections and layer normalization ensure consistency of shapes across blocks?
- How does the final projection connect decoder output to vocabulary logits?
- Why is positional encoding necessary, and how does it work?

## Step 7: Report Format

### 7.1 Screenshot Requirements
Each screenshot must show:
- Clear snapshot number and name
- Variable name and value in debugger
- Tensor shape
- Sample tensor values
- Brief explanation (1-2 sentences)

### 7.2 Written Report Structure
```
# Transformer Model Debugging Report

## Setup Verification
- [Screenshot of PyCharm with WSL interpreter]
- [Screenshot of successful model run]

## Snapshot Analysis (43 snapshots)
### Snapshot #1 – Raw input tokens
**Shape**: (1, 5)
**Values**: [1, 45, 123, 67, 89]
**Explanation**: These are the token IDs representing the input sequence "The quick brown fox" before any processing.

### Snapshot #2 – Target tokens
**Shape**: (1, 5)
**Values**: [2, 156, 234, 78, 145]
**Explanation**: These are the token IDs representing the target sequence "Le renard brun rapide" for translation.

[... continue for all 43 snapshots ...]

## Answers to Guiding Questions
### Question 1: Tensor Dimensions
[Your detailed answer]

### Question 2: Attention Mechanism
[Your detailed answer]

[... continue for all questions ...]

## Conclusion
[Summary of key insights and learnings]
```

## Step 8: Troubleshooting

### Common Issues

#### Issue: PyCharm can't find WSL Python
**Solution**: 
1. Verify WSL is running: `wsl --list --verbose`
2. Check Python path in WSL: `which python3`
3. Update PyCharm interpreter path

#### Issue: Import errors in WSL
**Solution**:
1. Install packages in WSL: `pip3 install -r requirements.txt`
2. Verify installation: `python3 -c "import torch; print(torch.__version__)"`

#### Issue: Debugger not stopping at breakpoints
**Solution**:
1. Ensure you're running in debug mode (`Shift + F9`)
2. Check that breakpoints are enabled (red circles)
3. Verify the script is running from the correct location

#### Issue: Tensor values not displaying properly
**Solution**:
1. Use `torch.tensor.detach().numpy()` to convert to numpy for inspection
2. Use `torch.tensor.shape` to check dimensions
3. Use `torch.tensor[:5]` to inspect first few values

## Step 9: Submission Checklist

### Required Files
- [ ] `transformer_model.py` - Complete model implementation
- [ ] `debug_transformer.py` - Debugging script
- [ ] 43 numbered screenshots - One for each snapshot
- [ ] Written report - Answers to all guiding questions
- [ ] Setup verification - Screenshots showing PyCharm + WSL working

### Quality Checklist
- [ ] All 43 snapshots captured and clearly labeled
- [ ] Screenshots show variable values, shapes, and sample data
- [ ] Written explanations for each snapshot
- [ ] All guiding questions answered thoroughly
- [ ] Code runs without errors
- [ ] Setup is properly documented

## Step 10: Grading Rubric

### Technical Setup (20 points)
- PyCharm + WSL configuration working
- All dependencies installed correctly
- Debugger functioning properly

### Snapshot Capture (40 points)
- All 43 snapshots captured (1 point each)
- Clear labeling and numbering
- Proper tensor shape and value display
- Meaningful explanations

### Understanding (30 points)
- Accurate answers to guiding questions
- Clear understanding of tensor flow
- Proper explanation of attention mechanisms
- Understanding of residual connections and normalization

### Documentation (10 points)
- Clear, well-organized report
- Professional presentation
- Complete setup documentation

## Additional Resources

- [PyCharm WSL Documentation](https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Transformer Paper](https://arxiv.org/abs/1706.03762)
- [Attention Mechanism Visualization](https://jalammar.github.io/illustrated-transformer/)

## Support

If you encounter issues:
1. Check the PyCharm logs: `Help` → `Show Log in Explorer`
2. Verify WSL is working: `wsl --status`
3. Test Python in WSL: `wsl python3 --version`
4. Check PyTorch installation: `wsl python3 -c "import torch; print(torch.__version__)"`
5. Run the test script: `python3 test_setup.py`

---

**Good luck with your assignment! Remember to take your time with each snapshot and really understand what's happening at each step. The debugging process will give you deep insights into how Transformer models work.**
