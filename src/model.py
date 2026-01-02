"""
Construction du modèle (à implémenter par l'étudiant·e).

Signature imposée :
build_model(config: dict) -> torch.nn.Module
"""

import torch
import torch.nn as nn


class DilatedCNN(nn.Module):
    """3-stage dilated CNN for Tiny ImageNet classification."""
    
    def __init__(self, num_classes=200, blocks_per_stage=[2, 2, 2], 
                 dilations=[1, 2, 2], channels=[64, 128, 256]):
        super(DilatedCNN, self).__init__()
        
        self.stage1 = self._make_stage(3, channels[0], blocks_per_stage[0], dilations[0])
        self.stage2 = self._make_stage(channels[0], channels[1], blocks_per_stage[1], dilations[1])
        self.stage3 = self._make_stage(channels[1], channels[2], blocks_per_stage[2], dilations[2])
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels[2], num_classes)
    
    def _make_stage(self, in_channels, out_channels, num_blocks, dilation):
        """Create a stage with num_blocks of Conv-BN-ReLU blocks."""
        layers = []
        
        # First block handles channel dimension change
        padding = dilation  # padding = dilation for 3x3 kernel to maintain spatial dims
        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        
        # Remaining blocks
        for _ in range(num_blocks - 1):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Stage 1: Conv blocks + MaxPool
        x = self.stage1(x)  # [B, 64, 64, 64]
        x = self.maxpool(x)  # [B, 64, 32, 32]
        
        # Stage 2: Conv blocks + MaxPool
        x = self.stage2(x)  # [B, 128, 32, 32]
        x = self.maxpool(x)  # [B, 128, 16, 16]
        
        # Stage 3: Conv blocks + Global Average Pooling
        x = self.stage3(x)  # [B, 256, 16, 16]
        x = self.global_avgpool(x)  # [B, 256, 1, 1]
        
        # Classifier
        x = torch.flatten(x, 1)  # [B, 256]
        x = self.classifier(x)  # [B, 200]
        
        return x


def build_model(config: dict):
    """Construit et retourne un nn.Module selon la config."""
    model_config = config.get('model', {})
    
    # Extract model parameters
    num_classes = model_config.get('num_classes', 200)
    blocks_per_stage = model_config.get('blocks_per_stage', [2, 2, 2])
    dilations = model_config.get('dilations', [1, 2, 2])
    channels = model_config.get('channels', [64, 128, 256])
    
    # Validate parameters
    if len(blocks_per_stage) != 3:
        raise ValueError("blocks_per_stage must have exactly 3 values")
    if len(dilations) != 3:
        raise ValueError("dilations must have exactly 3 values")
    if len(channels) != 3:
        raise ValueError("channels must have exactly 3 values")
    
    model = DilatedCNN(
        num_classes=num_classes,
        blocks_per_stage=blocks_per_stage,
        dilations=dilations,
        channels=channels
    )
    
    return model