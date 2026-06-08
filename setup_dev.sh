#!/bin/bash
# ============================================================
# Fast-BEV Training Environment Setup Script
# Target: WSL Ubuntu 24.04, RTX 5090, CUDA 12.8
# Conda environment: dev (Python 3.11)
# ============================================================

set -e

echo "============================================"
echo " Fast-BEV Environment Setup (dev)"
echo "============================================"

# Step 1: Activate conda
source ~/anaconda3/etc/profile.d/conda.sh
conda activate dev
echo "[OK] Conda dev activated: $(python --version)"

# Step 2: Set CUDA architecture for RTX 5090 (Blackwell, sm_120)
export TORCH_CUDA_ARCH_LIST="12.0"
export FORCE_CUDA=1
echo "[OK] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

# Step 3: Verify pre-installed packages
echo ""
echo "--------------------------------------------"
echo " Pre-installed packages"
echo "--------------------------------------------"
python -c "
import torch; print(f'  PyTorch:    {torch.__version__}  CUDA: {torch.version.cuda}')
import mmcv;  print(f'  MMCV-full:  {mmcv.__version__}')
import mmdet; print(f'  MMDet:      {mmdet.__version__}')
import mmseg; print(f'  MMSeg:      {mmseg.__version__}')
import numpy; print(f'  NumPy:      {numpy.__version__}')
print(f'  GPU:        {torch.cuda.get_device_name(0)}')
"

# Step 4: Install Fast-BEV in editable mode (compiles 11 CUDA extensions)
echo ""
echo "--------------------------------------------"
echo " Compiling CUDA extensions (this takes a while)..."
echo "--------------------------------------------"
cd /mnt/e/others/code_demo/Fast-BEV-dev
pip install -v -e . 2>&1 | tee install_log.txt

# Step 5: Verify installation
echo ""
echo "--------------------------------------------"
echo " Verifying installation..."
echo "--------------------------------------------"
python -c "
import mmdet3d
print(f'  mmdet3d: {mmdet3d.__version__}')

# Test CUDA extensions
from mmdet3d.ops.iou3d import iou3d_cuda
print('  iou3d_cuda: OK')

from mmdet3d.ops.voxel import voxel_layer
print('  voxel_layer: OK')

import torch
x = torch.randn(1, 3, 224, 224).cuda()
print(f'  CUDA tensor test: OK (shape={x.shape})')

print()
print('============================================')
print(' Fast-BEV dev environment ready!')
print('============================================')
print()
print(' To train:')
print('   conda activate dev')
print('   cd /mnt/e/others/code_demo/Fast-BEV-dev')
print('   python tools/train.py configs/fastbev/exp/paper/fastbev-m0.sh')
print()
"
