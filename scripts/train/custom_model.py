"""
Custom 3D CNN Modeli - MRI Sınıflandırması
Sıfırdan eğitim için optimize edilmiş mimari.

Temel Özellikler:
- GroupNorm (küçük batch size için stabil)
- Residual bağlantılar (gradyan akışı)
- SE Blokları (kanal dikkat mekanizması)
- Son katman sıfır initialization (dengeli başlangıç)
- Input normalizasyonu (seyrek veri için kritik)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """3D Residual Blok (GroupNorm ile)"""
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride, bias=False)
        self.gn1 = nn.GroupNorm(min(8, out_ch), out_ch)
        
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(8, out_ch), out_ch)
        
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout3d(dropout)
        
        # Boyut eşitleme
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.gn2(self.conv2(out))
        
        out += residual
        out = self.relu(out)
        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1, 1)
        return x * y


class Custom3DCNN(nn.Module):
    """
    Custom 3D CNN - MRI Sınıflandırması
    
    Mimari:
    - Input Normalization (seyrek MRI verisi için kritik)
    - Giriş: (B, 1, 128, 128, 128)
    - 4 Residual blok (32->64->128->256)
    - SE blokları (son 2 katmanda)
    - Global Average Pooling
    - FC sınıflandırıcı (son katman sıfır init)
    - Çıkış: (B, num_classes)
    """
    def __init__(self, num_classes=3, dropout=0.2):
        super(Custom3DCNN, self).__init__()
        
        # Input Normalizasyonu: Instance Normalization
        # Her görüntüyü kendi içinde normalize eder (batch'ten bağımsız)
        self.input_norm = nn.InstanceNorm3d(1, affine=True)
        
        # Giriş katmanı: 1 -> 32, 128->64
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=False)
        )
        
        # Blok 1: 32 -> 64, 64->32
        self.layer1 = ResidualBlock(32, 64, stride=2, dropout=0.05)
        
        # Blok 2: 64 -> 128, 32->16
        self.layer2 = ResidualBlock(64, 128, stride=2, dropout=0.1)
        self.se2 = SEBlock(128, reduction=8)
        
        # Blok 3: 128 -> 256, 16->8
        self.layer3 = ResidualBlock(128, 256, stride=2, dropout=0.15)
        self.se3 = SEBlock(256, reduction=16)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool3d(1)
        
        # Sınıflandırıcı
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.GroupNorm, nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Son FC katmanını küçük değerlerle başlat
        # Sıfır init gradyanları öldürür (dL/dW_prev = dL/dout * W_last, W_last=0 → grad=0)
        # Küçük normal dağılım: dengeli başlangıç + gradyan akışı sağlar
        last_linear = self.classifier[-1]
        nn.init.normal_(last_linear.weight, mean=0, std=0.01)
        nn.init.zeros_(last_linear.bias)
    
    def forward(self, x):
        # Input normalizasyonu (seyrek MRI verisi için kritik)
        x = self.input_norm(x)
        
        x = self.stem(x)      # (B,32,64,64,64)
        x = self.layer1(x)    # (B,64,32,32,32)
        
        x = self.layer2(x)    # (B,128,16,16,16)
        x = self.se2(x)
        
        x = self.layer3(x)    # (B,256,8,8,8)
        x = self.se3(x)
        
        x = self.gap(x)       # (B,256,1,1,1)
        x = self.classifier(x)
        return x


class Custom3DCNN_Large(nn.Module):
    """
    Enlarged Custom 3D CNN - ~8M parameters
    
    Architecture:
    - Input Normalization (InstanceNorm3d)
    - Input: (B, 1, 128, 128, 128)
    - 5 Residual blocks (32->64->128->256->512)
    - SE blocks (last 3 layers)
    - Global Average Pooling
    - FC classifier (last layer small init)
    - Output: (B, num_classes)
    """
    def __init__(self, num_classes=3, dropout=0.3):
        super(Custom3DCNN_Large, self).__init__()
        
        self.input_norm = nn.InstanceNorm3d(1, affine=True)
        
        # Stem: 1 -> 32, 128->64
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1, stride=2, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=False)
        )
        
        # Block 1: 32 -> 64, 64->32
        self.layer1 = ResidualBlock(32, 64, stride=2, dropout=0.05)
        
        # Block 2: 64 -> 128, 32->16
        self.layer2 = ResidualBlock(64, 128, stride=2, dropout=0.1)
        self.se2 = SEBlock(128, reduction=8)
        
        # Block 3: 128 -> 256, 16->8
        self.layer3 = ResidualBlock(128, 256, stride=2, dropout=0.15)
        self.se3 = SEBlock(256, reduction=16)
        
        # Block 4: 256 -> 512, 8->4  (NEW — parameter increase)
        self.layer4 = ResidualBlock(256, 512, stride=2, dropout=0.2)
        self.se4 = SEBlock(512, reduction=16)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool3d(1)
        
        # Classifier (512 -> 256 -> num_classes)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.GroupNorm, nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        last_linear = self.classifier[-1]
        nn.init.normal_(last_linear.weight, mean=0, std=0.01)
        nn.init.zeros_(last_linear.bias)
    
    def forward(self, x):
        x = self.input_norm(x)
        
        x = self.stem(x)      # (B,32,64,64,64)
        x = self.layer1(x)    # (B,64,32,32,32)
        
        x = self.layer2(x)    # (B,128,16,16,16)
        x = self.se2(x)
        
        x = self.layer3(x)    # (B,256,8,8,8)
        x = self.se3(x)
        
        x = self.layer4(x)    # (B,512,4,4,4)
        x = self.se4(x)
        
        x = self.gap(x)       # (B,512,1,1,1)
        x = self.classifier(x)
        return x
