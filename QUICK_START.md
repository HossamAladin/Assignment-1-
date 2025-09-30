# Quick Start Guide - Transformer Debugging Assignment

## 🚀 Get Started in 5 Minutes

### 1. Open WSL Terminal
```bash
# Open WSL terminal (Windows Terminal or WSL directly)
wsl
```

### 2. Navigate to Project Directory
```bash
# Navigate to your assignment folder
cd /mnt/d/IT_level4/Assignment
```

### 3. Install Dependencies
```bash
# Install required packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install numpy matplotlib tqdm
```

### 4. Test Your Setup
```bash
# Run the test script to verify everything works
python3 test_setup.py
```

**Expected Output:**
```
TRANSFORMER DEBUGGING SETUP TEST
==================================================
=== ENVIRONMENT TEST ===
PyTorch version: 2.0.0
PyTorch device: cpu
NumPy version: 1.21.0
CUDA available: False
Environment test passed!

=== MODEL CREATION TEST ===
✓ Model created successfully
✓ Model parameters: 1,234,567
✓ Model device: cpu

=== FORWARD PASS TEST ===
✓ Sample data created
  Source shape: torch.Size([1, 5])
  Target shape: torch.Size([1, 5])
✓ Forward pass completed
  Logits shape: torch.Size([1, 5, 1000])
  Encoder output shape: torch.Size([1, 5, 128])
  Decoder output shape: torch.Size([1, 5, 128])
✓ All shape checks passed

=== DEBUGGING SCRIPT TEST ===
✓ Debugging script imported successfully
✓ Debugging function is callable

=== TENSOR OPERATIONS TEST ===
✓ Tensor creation: torch.Size([1, 5, 128])
✓ Tensor slicing: torch.Size([2, 5])
✓ Matrix multiplication: torch.Size([1, 5, 5])
✓ Softmax: torch.Size([1, 5, 5])
✓ Masking: torch.Size([1, 5, 5])
✓ All tensor operations passed

==================================================
✅ ALL TESTS PASSED!
✅ Your setup is ready for debugging!
```

### 5. Configure PyCharm

1. **Open PyCharm**
2. **Create New Project** in your Windows file system
3. **Configure WSL Interpreter:**
   - Go to `File` → `Settings` → `Project` → `Python Interpreter`
   - Click gear icon → `Add...` → `WSL`
   - Select your WSL distribution
   - Choose `/usr/bin/python3` as interpreter
4. **Create Run Configuration:**
   - Go to `Run` → `Edit Configurations...`
   - Click `+` → `Python`
   - Set script path to `debug_transformer.py`
   - Set interpreter to WSL Python

### 6. Start Debugging

1. **Open `debug_transformer.py`** in PyCharm
2. **Set Breakpoints** at all 43 marked locations
3. **Run in Debug Mode** (`Shift + F9`)
4. **Capture Screenshots** at each breakpoint

## 📋 Assignment Checklist

### Required Files
- [ ] `transformer_model.py` - Complete model implementation
- [ ] `debug_transformer.py` - Debugging script with 43 snapshots
- [ ] `test_setup.py` - Setup verification script
- [ ] `requirements.txt` - Python dependencies
- [ ] `setup_wsl_pycharm.md` - Detailed setup instructions
- [ ] `ASSIGNMENT_GUIDE.md` - Complete assignment guide

### 43 Required Snapshots
- [ ] **Snapshots 1-5**: Input & Embedding
- [ ] **Snapshots 6-19**: Encoder Layer
- [ ] **Snapshots 20-40**: Decoder Layer  
- [ ] **Snapshots 41-43**: Final Output

### Guiding Questions (Must Answer)
- [ ] What do each of the dimensions represent?
- [ ] Why do Q, K, V tensors have the same shape?
- [ ] What do attention score matrices represent?
- [ ] Why is masking necessary in the decoder?
- [ ] How do residual connections work?
- [ ] Why must embedding dimension remain constant?
- [ ] How does final projection work?

## 🔧 Troubleshooting

### Issue: Python not found
```bash
# Install Python in WSL
sudo apt update
sudo apt install python3 python3-pip
```

### Issue: PyTorch installation fails
```bash
# Try alternative installation
pip3 install torch torchvision torchaudio
```

### Issue: PyCharm can't connect to WSL
1. Verify WSL is running: `wsl --status`
2. Check Python path: `which python3`
3. Update PyCharm interpreter path

### Issue: Debugger not stopping
1. Ensure breakpoints are enabled (red circles)
2. Run in debug mode (`Shift + F9`)
3. Check script path in run configuration

## 📚 Next Steps

1. **Read the complete guide**: `ASSIGNMENT_GUIDE.md`
2. **Follow setup instructions**: `setup_wsl_pycharm.md`
3. **Start debugging**: Open `debug_transformer.py` in PyCharm
4. **Capture snapshots**: Take screenshots at all 43 breakpoints
5. **Write report**: Answer all guiding questions

## 🎯 Success Criteria

- ✅ All 43 snapshots captured with clear labeling
- ✅ Screenshots show variable values, shapes, and sample data
- ✅ Written explanations for each snapshot
- ✅ All guiding questions answered thoroughly
- ✅ Code runs without errors
- ✅ Setup properly documented

---

**Ready to start? Run `python3 test_setup.py` in WSL to verify your setup!**
