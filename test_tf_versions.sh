#!/bin/bash
# Test different TensorFlow versions for GPU compatibility

ENV_NAME="dx_703_project"

echo "======================================================================"
echo "TensorFlow Version Test Script"
echo "======================================================================"

# Function to test a TensorFlow version
test_tf_version() {
    local version=$1
    echo ""
    echo "----------------------------------------------------------------------"
    echo "Testing TensorFlow $version"
    echo "----------------------------------------------------------------------"

    # Install version
    echo "Installing tensorflow==$version..."
    conda run -n $ENV_NAME pip install -q tensorflow==$version

    # Run test
    echo "Running environment test..."
    conda run -n $ENV_NAME python test_environment.py

    local result=$?

    if [ $result -eq 0 ]; then
        echo "✓ TensorFlow $version: PASSED"
        return 0
    else
        echo "✗ TensorFlow $version: FAILED"
        return 1
    fi
}

# Test versions
VERSIONS=("2.17.0" "2.18.0")

for version in "${VERSIONS[@]}"; do
    test_tf_version $version

    if [ $? -eq 0 ]; then
        echo ""
        echo "======================================================================"
        echo "✓ FOUND WORKING VERSION: TensorFlow $version"
        echo "======================================================================"
        echo ""
        echo "Keep this version installed and restart your Jupyter kernel."
        exit 0
    fi
done

# If we get here, none worked
echo ""
echo "======================================================================"
echo "⚠️  No GPU-compatible TensorFlow version found"
echo "======================================================================"
echo ""
echo "Reverting to TensorFlow 2.16.1 (CPU mode)"
conda run -n $ENV_NAME pip install -q tensorflow==2.16.1

echo ""
echo "Options:"
echo "1. Use CPU mode (add os.environ['CUDA_VISIBLE_DEVICES'] = '-1')"
echo "2. Use your other TF environment"
echo "3. Install system CUDA libraries"
echo ""

exit 1
