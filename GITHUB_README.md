# 🤖 Transformer Model Debugging Assignment

A comprehensive educational project for understanding Transformer architecture through step-by-step debugging with PyCharm and WSL.

## 🎯 Overview

This assignment teaches students how to debug deep learning models by analyzing tensor flow through a complete Transformer implementation. Students will capture 43 specific snapshots showing how data moves through embeddings, attention mechanisms, feed-forward networks, and output layers.

## ✨ Features

- **Complete Transformer Implementation** - Standard encoder-decoder architecture
- **43 Debugging Snapshots** - Precisely marked breakpoints for learning
- **WSL Integration** - Works seamlessly with Windows Subsystem for Linux
- **PyCharm Ready** - Optimized for professional debugging environment
- **Educational Focus** - Designed for deep learning education
- **Answer Key Included** - All guiding questions pre-answered

## 🏗️ Architecture

- **Model Type**: Standard encoder-decoder Transformer (Vaswani et al., 2017)
- **Encoder Layers**: 2
- **Decoder Layers**: 2
- **Attention Heads**: 4
- **Embedding Dimension**: 128
- **Vocabulary Size**: 1000 (reduced for debugging)
- **Feed-forward Dimension**: 512
- **Total Parameters**: 1,182,696

## 🚀 Quick Start

### Prerequisites
- Windows 10/11 with WSL2
- PyCharm (Professional or Community)
- Python 3.8+

### Setup (5 minutes)
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/transformer-debugging-assignment.git
cd transformer-debugging-assignment

# Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Test setup
python3 test_setup.py
```

### Configure PyCharm
1. Open PyCharm → Create New Project
2. Go to `File` → `Settings` → `Project` → `Python Interpreter`
3. Click gear icon → `Add...` → `WSL`
4. Select your WSL distribution and `/usr/bin/python3`
5. Create Run Configuration for `debug_transformer.py`

### Start Debugging
1. Open `debug_transformer.py` in PyCharm
2. Set breakpoints at all 43 marked locations
3. Run in debug mode (`Shift + F9`)
4. Capture screenshots at each breakpoint

## 📚 Learning Objectives

Students will learn:
- **Tensor Flow Analysis** - How data moves through each layer
- **Attention Mechanisms** - Self-attention and cross-attention
- **Multi-head Attention** - Parallel processing of attention
- **Residual Connections** - Gradient flow and shape preservation
- **Layer Normalization** - Training stability
- **Masking** - Causal attention in decoders
- **Debugging Skills** - Professional debugging with PyCharm

## 📋 Assignment Structure

### Required Snapshots (43 Total)
- **Input & Embedding (1-5)**: Token processing and embeddings
- **Encoder Layer (6-19)**: Self-attention and feed-forward
- **Decoder Layer (20-40)**: Masked attention and cross-attention
- **Final Output (41-43)**: Projection and logits

### Deliverables
- 43 numbered screenshots with clear labeling
- Written report answering guiding questions
- Code files (`transformer_model.py`, `debug_transformer.py`)
- Setup verification screenshots

## 📖 Documentation

- **[Clear Assignment Guide](CLEAR_ASSIGNMENT_GUIDE.md)** - Complete instructions with answer key
- **[Quick Start Guide](QUICK_START.md)** - 5-minute setup
- **[Setup Instructions](setup_wsl_pycharm.md)** - Detailed PyCharm + WSL setup
- **[Verification](VERIFICATION_COMPLETE.md)** - Test results and verification

## 🔧 Technical Details

### Model Specifications
```python
# Model configuration
vocab_size = 1000
d_model = 128
num_heads = 4
num_encoder_layers = 2
num_decoder_layers = 2
d_ff = 512
```

### Sample Data
- **Input**: "The quick brown fox" (token IDs: [1, 45, 123, 67, 89])
- **Target**: "Le renard brun rapide" (token IDs: [2, 156, 234, 78, 145])

### Expected Output Shapes
- **Embeddings**: (1, 5, 128)
- **Attention Scores**: (1, 5, 5)
- **Multi-head Split**: (1, 4, 5, 32)
- **Feed-forward**: (1, 5, 512) → (1, 5, 128)
- **Final Logits**: (1, 5, 1000)

## 🎓 Educational Value

This assignment provides:
- **Deep Understanding** of Transformer architecture
- **Practical Skills** in debugging deep learning models
- **Professional Tools** experience with PyCharm and WSL
- **Step-by-step Learning** through 43 carefully designed snapshots
- **Real-world Application** of attention mechanisms

## 📊 Grading Rubric

- **Technical Setup (20 points)**: PyCharm + WSL configuration
- **Snapshot Capture (40 points)**: All 43 snapshots with proper labeling
- **Understanding (30 points)**: Accurate answers to guiding questions
- **Documentation (10 points)**: Clear, professional report

## 🤝 Contributing

This is an educational project. Contributions are welcome for:
- Additional debugging examples
- Improved documentation
- Bug fixes
- Educational enhancements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on "Attention Is All You Need" (Vaswani et al., 2017)
- PyTorch framework for deep learning
- PyCharm for professional debugging environment
- WSL for seamless Windows-Linux integration

## 📞 Support

If you encounter issues:
1. Check the [troubleshooting guide](setup_wsl_pycharm.md#troubleshooting)
2. Run `python3 test_setup.py` to verify setup
3. Check PyCharm logs: `Help` → `Show Log in Explorer`
4. Verify WSL is working: `wsl --status`

---

**🎉 Ready to start your Transformer debugging journey! This assignment will give you deep insights into how modern AI models work.**
