"""
Safe PyTorch installation for tfmetal environment
Preserves TensorFlow and numpy compatibility
"""

import subprocess
import sys

print("="*60)
print("SAFE PYTORCH INSTALLATION FOR TFMETAL")
print("="*60)

# Check current versions
print("\n1. Checking current environment...")
try:
    import numpy as np
    import tensorflow as tf
    print(f"   ✅ NumPy: {np.__version__}")
    print(f"   ✅ TensorFlow: {tf.__version__}")
    current_numpy = np.__version__
except ImportError as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Check if PyTorch already installed
try:
    import torch
    print(f"   ℹ️  PyTorch already installed: {torch.__version__}")
    print("\n   No installation needed!")
    sys.exit(0)
except ImportError:
    print("   ℹ️  PyTorch not installed - proceeding with installation...")

# Install PyTorch with specific numpy constraint
print("\n2. Installing PyTorch (preserving numpy version)...")
print("   This may take a few minutes...")

# Use pip with constraints to prevent numpy upgrade
result = subprocess.run([
    sys.executable, "-m", "pip", "install",
    "torch", "torchvision", "torchaudio",
    "--index-url", "https://download.pytorch.org/whl/cpu",
    "--no-deps"  # Don't install dependencies automatically
], capture_output=True, text=True)

if result.returncode != 0:
    print(f"   ❌ Installation failed: {result.stderr}")
    sys.exit(1)

print("   ✅ PyTorch installed")

# Install PyTorch dependencies manually (excluding numpy)
print("\n3. Installing PyTorch dependencies (excluding numpy)...")
dependencies = ["typing-extensions", "sympy", "networkx", "jinja2", "fsspec", "filelock"]

for dep in dependencies:
    subprocess.run([sys.executable, "-m", "pip", "install", dep, "-q"], check=True)

print("   ✅ Dependencies installed")

# Verify numpy wasn't changed
print("\n4. Verifying numpy version...")
import numpy as np
new_numpy = np.__version__

if new_numpy != current_numpy:
    print(f"   ⚠️  WARNING: NumPy changed from {current_numpy} to {new_numpy}")
    print("   Attempting to fix...")
    subprocess.run([sys.executable, "-m", "pip", "install", f"numpy=={current_numpy}"], check=True)
    print(f"   ✅ Reverted to numpy {current_numpy}")
else:
    print(f"   ✅ NumPy unchanged: {new_numpy}")

# Final verification
print("\n5. Final verification...")
try:
    import tensorflow as tf
    import torch
    import numpy as np

    print(f"   ✅ TensorFlow: {tf.__version__}")
    print(f"   ✅ PyTorch: {torch.__version__}")
    print(f"   ✅ NumPy: {np.__version__}")

    # Test basic operations
    print("\n6. Testing basic operations...")

    # TensorFlow test
    tf_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    print(f"   ✅ TensorFlow tensor: {tf_tensor.shape}")

    # PyTorch test
    torch_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    print(f"   ✅ PyTorch tensor: {torch_tensor.shape}")

except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ INSTALLATION SUCCESSFUL!")
print("="*60)
print("\n⚠️  IMPORTANT NOTES:")
print("   - PyTorch on Mac uses CPU (no CUDA)")
print("   - TensorFlow uses Metal GPU acceleration")
print("   - This is normal - different backends on Mac")
print("   - Both libraries can coexist safely")
print("\n   To use PyTorch GPU on Mac, use MPS backend:")
print("   >>> device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')")
