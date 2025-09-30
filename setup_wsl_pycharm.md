# PyCharm + WSL Setup Guide for Transformer Debugging Assignment

## Prerequisites
- Windows 10/11 with WSL2 installed
- PyCharm Professional (recommended) or Community Edition
- Python 3.8+ in WSL

## Step 1: Install WSL2 (if not already installed)

```bash
# In PowerShell as Administrator
wsl --install
# Restart computer after installation
```

## Step 2: Install Python and Dependencies in WSL

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

## Step 3: Configure PyCharm for WSL

### 3.1 Open PyCharm and Create New Project
1. Open PyCharm
2. Create a new project in your Windows file system
3. Navigate to the project folder

### 3.2 Configure Python Interpreter
1. Go to `File` → `Settings` (or `PyCharm` → `Preferences` on Mac)
2. Navigate to `Project` → `Python Interpreter`
3. Click the gear icon → `Add...`
4. Select `WSL` from the left panel
5. Choose your WSL distribution (usually Ubuntu)
6. Select the Python interpreter path (usually `/usr/bin/python3`)
7. Click `OK`

### 3.3 Configure Project Structure
1. In PyCharm settings, go to `Project` → `Project Structure`
2. Add your project folder as a content root
3. Mark the `src` folder as a source root if you have one

## Step 4: Set Up Debugging Configuration

### 4.1 Create Run Configuration
1. Go to `Run` → `Edit Configurations...`
2. Click `+` → `Python`
3. Set the following:
   - **Name**: `Debug Transformer`
   - **Script path**: `debug_transformer.py`
   - **Python interpreter**: WSL Python interpreter
   - **Working directory**: Your project directory

### 4.2 Configure Debugger Settings
1. In PyCharm settings, go to `Build, Execution, Deployment` → `Python Debugger`
2. Enable `Gevent compatible` if you encounter issues
3. Set `Attach to subprocess automatically while debugging` to `Always`

## Step 5: Test the Setup

### 5.1 Run the Model
1. Open `debug_transformer.py` in PyCharm
2. Set a breakpoint on line 1 of the `debug_transformer_forward_pass()` function
3. Run the script in debug mode (`Shift + F9`)
4. Verify that the debugger stops at your breakpoint

### 5.2 Verify WSL Integration
1. In the debugger, check that the Python interpreter path shows WSL path
2. Verify that you can inspect variables and step through code
3. Test the variable inspection panel

## Step 6: Debugging the Transformer Model

### 6.1 Set Breakpoints
Set breakpoints at each of the 43 snapshot locations marked in `debug_transformer.py`:

1. **Snapshots 1-5**: Input and embedding processing
2. **Snapshots 6-19**: Encoder layer processing
3. **Snapshots 20-40**: Decoder layer processing
4. **Snapshots 41-43**: Final output processing

### 6.2 Capture Snapshots
For each breakpoint:
1. Step through the code (`F8` for step over, `F7` for step into)
2. Inspect the variable values in the debugger panel
3. Take screenshots of the debugger showing:
   - Variable name and value
   - Tensor shape
   - Sample tensor values
4. Record the information for your report

### 6.3 Debugging Tips
- Use the **Variables** panel to inspect tensor shapes and values
- Use the **Watches** panel to monitor specific variables
- Use the **Console** to execute Python commands during debugging
- Use `torch.tensor.shape` to check tensor dimensions
- Use `torch.tensor[:5, :5]` to inspect tensor slices

## Step 7: Troubleshooting

### Common Issues and Solutions

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

## Step 8: Assignment Submission

### Required Deliverables
1. **43 Screenshots**: One for each snapshot showing:
   - Variable name and value
   - Tensor shape
   - Sample tensor values
   - Clear labeling with snapshot number

2. **Written Report**: Answer the guiding questions:
   - What do each of the dimensions represent?
   - Why do Q, K, V tensors have the same shape?
   - What do attention score matrices represent?
   - Why is masking necessary in the decoder?
   - How do residual connections work?
   - Why must embedding dimension remain constant?
   - How does final projection work?

3. **Code**: Submit your `transformer_model.py` and `debug_transformer.py` files

### Report Format
```
Snapshot #X – [Name]
Shape: (batch_size, seq_len, d_model)
Values: [sample values]
Explanation: [1-2 sentences explaining the tensor's purpose and shape]
```

## Additional Resources

- [PyCharm WSL Documentation](https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Transformer Paper](https://arxiv.org/abs/1706.03762)

## Support

If you encounter issues:
1. Check the PyCharm logs: `Help` → `Show Log in Explorer`
2. Verify WSL is working: `wsl --status`
3. Test Python in WSL: `wsl python3 --version`
4. Check PyTorch installation: `wsl python3 -c "import torch; print(torch.__version__)"`
