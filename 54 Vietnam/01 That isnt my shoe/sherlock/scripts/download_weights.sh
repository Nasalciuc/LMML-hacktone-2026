#!/bin/sh
# ImageNet checkpoint used for fine-tuning (only needed to RE-TRAIN; exporting/verifying does not need it).
# timm's legacy GitHub release asset: MobileNetV3-Large, ImageNet-21k pre-trained -> ImageNet-1k fine-tuned (MIIL), 78.0% top-1.
set -e
cd "$(dirname "$0")/.."
mkdir -p weights
curl -L -o weights/mobilenetv3_large_100_1k_miil_78_0-66471c13.pth \
  https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/mobilenetv3_large_100_1k_miil_78_0-66471c13.pth
# other backbones that were only used in the frozen-probe comparison are listed in src/extract_feats.py (CANDS)
