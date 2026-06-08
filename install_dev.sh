#!/bin/bash
# Fast-BEV environment setup script for WSL (conda dev environment)
# RTX 5090 + CUDA 12.8

set -e

echo "=========================================="
echo "Fast-BEV Environment Setup (dev)"
echo "=========================================="

# Activate conda dev environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dev

echo "[1/3] Verifying environment..."
python -c "
import torch
print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
import mmcv; print('MMCV:', mmcv.__version__)
import mmdet; print('MMDet:', mmdet.__version__)
import mmseg; print('MMSeg:', mmseg.__version__)
import numpy; print('NumPy:', numpy.__version__)
"

echo ""
echo "[2/3] Installing Fast-BEV in editable mode (compiling CUDA extensions)..."
cd /mnt/e/others/code_demo/Fast-BEV-dev
pip install -v -e . 2>&1

echo ""
echo "[3/3] Verifying Fast-BEV installation..."
python -c "
import mmdet3d
print('mmdet3d imported successfully:', mmdet3d.__version__)
import torch
print('CUDA available:', torch.cuda.is_available())
print('Setup complete!')
"

echo ""
echo "=========================================="
echo "Fast-BEV dev environment setup complete!"
echo "=========================================="
