"""
3D GradCAM Görselleştirme - Custom3DCNN Modeli İçin
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

# Custom3DCNN import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from custom_model import Custom3DCNN

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
    
    # Model yükle - Custom3DCNN
    model = Custom3DCNN(num_classes=3, dropout=0.2)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, "weights", "Custom3DCNN_best_balanced.pth")
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"✅ Model loaded: {weights_path}")
    else:
        print(f"⚠️ Model not found: {weights_path}")
        print("Please train the model first with '14.1-train_simplecnn.py'!")
        exit()
    
    model.to(device)
    model.eval()
    
    # Custom3DCNN son residual blok: layer3 (256 kanal, 8x8x8 feature map)
    target_layer = model.layer3
    gradcam = GradCAM3D(model, target_layer)
    
    # Test verisi yükle
    processed_dir = "C:\\Users\\berkk\\Desktop\\Bitirme_Projesi\\processed_dataset_128"
    data_csv = pd.read_csv(os.path.join(script_dir, "12-data_leakage.csv"))
    
    # Her sınıftan örnek seç
    class_names = ['Normal', 'Mild', 'Moderate_Severe']
    class_names_display = ['Normal', 'Mild', 'Moderate/Severe']
    label_map = {0: 0, 0.5: 1, 1: 2, 2: 2}
    
    print("\n" + "="*60)
    print("GRADCAM ANALYSIS (Custom3DCNN)")
    print("="*60)
    
    samples_per_class = 2
    
    for target_class in range(3):
        print(f"\n🔍 Processing samples for Class {target_class} ({class_names_display[target_class]})...")
        
        class_samples = []
        for idx, row in data_csv.iterrows():
            label_val = row['memory']
            try:
                label_val = float(label_val)
            except:
                continue
            if label_map.get(label_val, 0) == target_class:
                npy_path = os.path.join(processed_dir, f"{row['path']}.npy")
                if os.path.exists(npy_path):
                    class_samples.append((npy_path, row['path']))
            if len(class_samples) >= samples_per_class:
                break
        
        for i, (npy_path, session_id) in enumerate(class_samples):
            # Veri yükle (0-1 arası normalize edilmiş)
            mri_data = np.load(npy_path).astype(np.float32)
            mri_data = np.clip(mri_data, 0, 1)
            
            # Tensor'a çevir - Custom3DCNN kendi InstanceNorm'unu kullanıyor
            input_tensor = torch.tensor(mri_data).unsqueeze(0).unsqueeze(0).to(device)
            
            # GradCAM hesapla
            cam, pred_class, probs = gradcam.generate_cam(input_tensor, target_class=target_class)
            
            title = (f"GradCAM (Custom3DCNN) - {session_id}\n"
                     f"Actual: {class_names_display[target_class]} | "
                     f"Predicted: {class_names_display[pred_class]} | "
                     f"Probs: N:{probs[0]:.2f} M:{probs[1]:.2f} S:{probs[2]:.2f}")
            
            # Ana görselleştirme
            fig = visualize_gradcam_3d(mri_data, cam, title=title)
            save_path = f"gradcam_{class_names[target_class]}_{i+1}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ Saved: {save_path}")
            plt.close()
            
            # Axial kesitler
            fig = visualize_gradcam_slices(mri_data, cam, num_slices=8, axis=2)
            save_path = f"gradcam_{class_names[target_class]}_{i+1}_slices.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    print("\n✅ GradCAM analysis complete!")
    print("Generated files:")
    print("  - gradcam_Normal_*.png")
    print("  - gradcam_Mild_*.png")
    print("  - gradcam_Moderate_Severe_*.png")
