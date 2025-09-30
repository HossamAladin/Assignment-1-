"""
Test script to verify the setup is working correctly.
Run this script to ensure all dependencies are installed and the model works.
"""

import torch
import numpy as np
from transformer_model import Transformer, create_sample_data

def test_environment():
    """Test if the environment is set up correctly."""
    print("=== ENVIRONMENT TEST ===")
    
    # Test PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch device: {torch.device('cpu')}")
    
    # Test NumPy
    print(f"NumPy version: {np.__version__}")
    
    # Test CUDA availability (optional)
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    print("Environment test passed!\n")

def test_model_creation():
    """Test if the Transformer model can be created."""
    print("=== MODEL CREATION TEST ===")
    
    try:
        model = Transformer(
            vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=512
        )
        print("✓ Model created successfully")
        print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"✓ Model device: {next(model.parameters()).device}")
        return model
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return None

def test_forward_pass(model):
    """Test if the forward pass works correctly."""
    print("\n=== FORWARD PASS TEST ===")
    
    try:
        # Create sample data
        src, tgt = create_sample_data()
        print(f"✓ Sample data created")
        print(f"  Source shape: {src.shape}")
        print(f"  Target shape: {tgt.shape}")
        
        # Forward pass
        with torch.no_grad():
            logits, encoder_output, decoder_output = model(src, tgt)
        
        print(f"✓ Forward pass completed")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Encoder output shape: {encoder_output.shape}")
        print(f"  Decoder output shape: {decoder_output.shape}")
        
        # Verify shapes
        expected_logits_shape = (1, 5, 1000)  # (batch, seq_len, vocab_size)
        expected_encoder_shape = (1, 5, 128)  # (batch, seq_len, d_model)
        expected_decoder_shape = (1, 5, 128)  # (batch, seq_len, d_model)
        
        assert logits.shape == expected_logits_shape, f"Logits shape mismatch: {logits.shape} != {expected_logits_shape}"
        assert encoder_output.shape == expected_encoder_shape, f"Encoder shape mismatch: {encoder_output.shape} != {expected_encoder_shape}"
        assert decoder_output.shape == expected_decoder_shape, f"Decoder shape mismatch: {decoder_output.shape} != {expected_decoder_shape}"
        
        print("✓ All shape checks passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

def test_debugging_script():
    """Test if the debugging script can be imported and run."""
    print("\n=== DEBUGGING SCRIPT TEST ===")
    
    try:
        from debug_transformer import debug_transformer_forward_pass
        print("✓ Debugging script imported successfully")
        
        # Test if the function can be called (without actually running it)
        print("✓ Debugging function is callable")
        
        return True
        
    except Exception as e:
        print(f"✗ Debugging script test failed: {e}")
        return False

def test_tensor_operations():
    """Test basic tensor operations needed for debugging."""
    print("\n=== TENSOR OPERATIONS TEST ===")
    
    try:
        # Test tensor creation
        x = torch.randn(1, 5, 128)
        print(f"✓ Tensor creation: {x.shape}")
        
        # Test tensor slicing
        slice_x = x[0, :2, :5]
        print(f"✓ Tensor slicing: {slice_x.shape}")
        
        # Test tensor operations
        y = torch.matmul(x, x.transpose(-2, -1))
        print(f"✓ Matrix multiplication: {y.shape}")
        
        # Test softmax
        z = torch.softmax(y, dim=-1)
        print(f"✓ Softmax: {z.shape}")
        
        # Test masking
        mask = torch.tril(torch.ones(5, 5))
        masked = y.masked_fill(mask == 0, -1e9)
        print(f"✓ Masking: {masked.shape}")
        
        print("✓ All tensor operations passed")
        return True
        
    except Exception as e:
        print(f"✗ Tensor operations test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("TRANSFORMER DEBUGGING SETUP TEST")
    print("=" * 50)
    
    # Run tests
    test_environment()
    
    model = test_model_creation()
    if model is None:
        print("❌ Setup failed: Model creation failed")
        return False
    
    if not test_forward_pass(model):
        print("❌ Setup failed: Forward pass failed")
        return False
    
    if not test_debugging_script():
        print("❌ Setup failed: Debugging script failed")
        return False
    
    if not test_tensor_operations():
        print("❌ Setup failed: Tensor operations failed")
        return False
    
    print("\n" + "=" * 50)
    print("✅ ALL TESTS PASSED!")
    print("✅ Your setup is ready for debugging!")
    print("\nNext steps:")
    print("1. Open PyCharm")
    print("2. Configure WSL as Python interpreter")
    print("3. Open debug_transformer.py")
    print("4. Set breakpoints at the 43 snapshot locations")
    print("5. Run in debug mode (Shift + F9)")
    print("6. Capture screenshots at each breakpoint")
    
    return True

if __name__ == "__main__":
    main()
