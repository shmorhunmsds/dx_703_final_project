#!/bin/bash
# Safe PyTorch installation for tfmetal environment
# This preserves TensorFlow compatibility

echo "Installing PyTorch while preserving TensorFlow compatibility..."
echo ""
echo "Current environment: tfmetal"
echo "TensorFlow version: 2.16.1 (requires numpy < 2.0)"
echo "NumPy version: 1.26.4"
echo ""

# Activate tfmetal environment
source /opt/anaconda3/bin/activate tfmetal

# Install PyTorch with CPU support (Mac doesn't need CUDA)
# Use --no-deps to avoid automatic numpy upgrade
echo "Step 1: Installing PyTorch (CPU version for Mac)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-deps

# Now install PyTorch dependencies manually, excluding numpy
echo ""
echo "Step 2: Installing PyTorch dependencies (excluding numpy)..."
pip install typing-extensions sympy networkx jinja2 fsspec filelock

# Verify numpy version wasn't changed
echo ""
echo "Step 3: Verifying numpy version..."
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Test both imports
echo ""
echo "Step 4: Testing imports..."
python -c "import tensorflow as tf; import torch; print(f'✅ TensorFlow: {tf.__version__}'); print(f'✅ PyTorch: {torch.__version__}')"

echo ""
echo "✅ Installation complete!"
echo ""
echo "⚠️  IMPORTANT: PyTorch on Mac uses CPU only (MPS backend)"
echo "   TensorFlow will use Metal GPU acceleration"
echo "   This is normal - they use different backends on Mac"
