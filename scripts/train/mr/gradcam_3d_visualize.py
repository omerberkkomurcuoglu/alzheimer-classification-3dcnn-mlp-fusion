"""
3D GradCAM Görselleştirme - ResNet50 Modeli İçin
Hangi beyin bölgelerinin sınıflandırma kararında önemli olduğunu gösterir.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import os
import sys

# Parent dizini ekle (mri_pipeline.py scripts/train altında bulunuyor)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mri_pipeline import load_preprocessed_mri_volume

# ResNet50 (BrainClassifier) Tanımlaması
class BrainClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float) -> None:
        super().__init__()
        from monai.networks.nets import resnet50 as monai_r50
        backbone = monai_r50(spatial_dims=3, n_input_channels=1, num_classes=1, shortcut_type="B", bias_downsample=False)
        self.conv1, self.bn1, self.act, self.maxpool = backbone.conv1, backbone.bn1, backbone.act, backbone.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.act(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.head(self.pool(x))

# ============================================================
# 1. 3D GRADCAM
# ============================================================
class GradCAM3D:
    """3D MRI için GradCAM implementasyonu"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook'ları kaydet
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, target_class=None):
        """GradCAM haritası oluştur"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # GradCAM hesapla
        gradients = self.gradients[0]    # (C, D, H, W)
        activations = self.activations[0]  # (C, D, H, W)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2, 3))  # (C,)
        
        # Weighted combination
        cam = torch.zeros(activations.shape[1:], device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU ve normalize
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.cpu().numpy(), target_class, output.softmax(dim=1)[0].detach().cpu().numpy()

# ============================================================
# 2. GÖRSELLEŞTIRME
# ============================================================
def visualize_gradcam_3d(mri_data, cam, title="GradCAM", slice_indices=None):
    """3 düzlemde GradCAM görselleştir (axial, sagittal, coronal)"""
    
    # CAM'i MRI boyutuna upsample et
    cam_resized = zoom(cam, (mri_data.shape[0]/cam.shape[0], 
                             mri_data.shape[1]/cam.shape[1], 
                             mri_data.shape[2]/cam.shape[2]), order=1)
    
    if slice_indices is None:
        slice_indices = [s // 2 for s in mri_data.shape]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    views = [
        ('Axial', mri_data[:, :, slice_indices[2]], cam_resized[:, :, slice_indices[2]]),
        ('Coronal', mri_data[:, slice_indices[1], :], cam_resized[:, slice_indices[1], :]),
        ('Sagittal', mri_data[slice_indices[0], :, :], cam_resized[slice_indices[0], :, :])
    ]
    
    for idx, (view_name, mri_slice, cam_slice) in enumerate(views):
        axes[0, idx].imshow(mri_slice.T, cmap='gray', origin='lower')
        axes[0, idx].set_title(f'{view_name} - MRI')
        axes[0, idx].axis('off')
        
        axes[1, idx].imshow(mri_slice.T, cmap='gray', origin='lower')
        axes[1, idx].imshow(cam_slice.T, cmap='jet', alpha=0.5, origin='lower')
        axes[1, idx].set_title(f'{view_name} - GradCAM')
        axes[1, idx].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig

def visualize_gradcam_slices(mri_data, cam, num_slices=8, axis=2):
    """Belirli bir eksen boyunca birden fazla kesit göster"""
    
    cam_resized = zoom(cam, (mri_data.shape[0]/cam.shape[0], 
                             mri_data.shape[1]/cam.shape[1], 
                             mri_data.shape[2]/cam.shape[2]), order=1)
    
    total_slices = mri_data.shape[axis]
    indices = np.linspace(total_slices*0.2, total_slices*0.8, num_slices).astype(int)
    
    fig, axes = plt.subplots(2, num_slices, figsize=(20, 6))
    
    for i, idx in enumerate(indices):
        if axis == 0:
            mri_slice = mri_data[idx, :, :]
            cam_slice = cam_resized[idx, :, :]
        elif axis == 1:
            mri_slice = mri_data[:, idx, :]
            cam_slice = cam_resized[:, idx, :]
        else:
            mri_slice = mri_data[:, :, idx]
            cam_slice = cam_resized[:, :, idx]
        
        axes[0, i].imshow(mri_slice.T, cmap='gray', origin='lower')
        axes[0, i].set_title(f'Slice {idx}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(mri_slice.T, cmap='gray', origin='lower')
        axes[1, i].imshow(cam_slice.T, cmap='jet', alpha=0.5, origin='lower')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    return fig

# ============================================================
# 3. MAIN
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Model yükle - ResNet50
    model = BrainClassifier(num_classes=3, dropout=0.30)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, "weights", "resnet.pth")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"[OK] Model loaded: {weights_path}")
    else:
        print(f"[WARN] Model not found: {weights_path}")
        exit()
    
    model.to(device)
    model.eval()
    
    # ResNet50 eğitiminde layer3 daha stabil özellikler çıkardığı için GradCAM'i ona göre ayarlıyoruz.
    target_layer = model.layer3
    gradcam = GradCAM3D(model, target_layer)
    
    # Test verisi yükle
    processed_dir = "C:\\Users\\berkk\\Desktop\\Bitirme_Projesi\\data\\processed_dataset_128"
    data_csv = pd.read_csv(os.path.join(script_dir, "0-data_leakage_mri_clean_180.csv"))
    
    # Her sınıftan örnek seç
    class_names = ['Normal', 'Mild', 'Moderate_Severe']
    class_names_display = ['Normal', 'Mild', 'Moderate/Severe']
    label_map = {0: 0, 0.5: 1, 1: 2, 2: 2}
    
    print("\n" + "="*60)
    print("GRADCAM ANALYSIS (ResNet50)")
    print("="*60)
    
    samples_per_class = 2
    
    print("\n[Search] Mining confident predictions (Optimized - Fast search)...")
    class_candidates = {0: [], 1: [], 2: []}
    target_count = 5 # Her sınıftan 5 doğru örnek bulduğumuzda aramayı durduracağız
    
    import sys
    
    for idx, row in data_csv.iterrows():
        if all(len(c) >= target_count for c in class_candidates.values()):
            break # Tüm sınıflardan yeterince bulduk, erken çıkış yap
            
        label_val = row['memory']
        try:
            label_val = float(label_val)
        except:
            continue
        
        target_class = label_map.get(label_val, 0)
        
        # Bu sınıftan zaten yeterince bulduysak, bu örneği boşuna işlemeyelim (Ciddi hızlandırma)
        if len(class_candidates[target_class]) >= target_count:
            continue
            
        npy_path = os.path.join(processed_dir, f"{row['path']}.npy")
        if not os.path.exists(npy_path):
            continue
            
        # Terminale dinamik olarak ilerleme durumunu bas
        sys.stdout.write(f"\rScanning... [Normal: {len(class_candidates[0])}/{target_count}] [Mild: {len(class_candidates[1])}/{target_count}] [Severe: {len(class_candidates[2])}/{target_count}]")
        sys.stdout.flush()
            
        mri_data = load_preprocessed_mri_volume(npy_path)
        input_tensor = torch.tensor(mri_data).unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            probs = output.softmax(dim=1)[0].cpu().numpy()
            pred_class = output.argmax(dim=1).item()
            
        # Sadece modelin DOĞRU tahmin ettiği örnekleri al
        if pred_class == target_class:
            class_candidates[target_class].append({
                'path': npy_path,
                'session_id': row['path'],
                'prob': probs[target_class],
                'all_probs': probs
            })
            
    print("\n[Search] Search completed!")
            
    for target_class in range(3):
        print(f"\n[Search] Generating GradCAM for Class {target_class} ({class_names_display[target_class]})...")
        
        # Olasılığa göre azalan şekilde sırala ve en iyi N tanesini al
        candidates = sorted(class_candidates[target_class], key=lambda x: x['prob'], reverse=True)
        top_candidates = candidates[:samples_per_class]
        
        if not top_candidates:
            print(f"[WARN] No correct predictions found for class {target_class}!")
            continue
            
        for i, cand in enumerate(top_candidates):
            mri_data = load_preprocessed_mri_volume(cand['path'])
            input_tensor = torch.tensor(mri_data).unsqueeze(0).unsqueeze(0).to(device)
            
            cam, pred_class, probs = gradcam.generate_cam(input_tensor, target_class=target_class)
            
            title = (f"GradCAM (ResNet50) - {cand['session_id']}\n"
                     f"Actual: {class_names_display[target_class]} | "
                     f"Predicted: {class_names_display[pred_class]} | "
                     f"Probs: N:{probs[0]:.2f} M:{probs[1]:.2f} S:{probs[2]:.2f}")
            
            fig = visualize_gradcam_3d(mri_data, cam, title=title)
            save_path = f"gradcam_{class_names[target_class]}_{i+1}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[OK] Saved {save_path} (Confidence: {cand['prob']:.2f})")
