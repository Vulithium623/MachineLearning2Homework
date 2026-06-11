import os
import csv
import glob
import argparse
import logging
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from peft import LoraConfig, get_peft_model

# ==========================================
# Utils & Logging Setup
# ==========================================
def get_run_dir(base_dir="runs/pseudo", name=None):
    os.makedirs(base_dir, exist_ok=True)
    if name is None:
        i = 1
        while os.path.exists(os.path.join(base_dir, str(i))):
            i += 1
        run_dir = os.path.join(base_dir, str(i))
        os.makedirs(run_dir)
        return run_dir
    else:
        target_dir = os.path.join(base_dir, name)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            return target_dir
        i = 1
        while True:
            new_dir = f"{target_dir}_{i}"
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
                return new_dir
            i += 1

def setup_logger(run_dir, log_filename='generate.log'):
    logger = logging.getLogger(f"Pseudo_{run_dir}")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler(os.path.join(run_dir, log_filename))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

# ==========================================
# Model Definition
# ==========================================
class CustomViTModel(nn.Module):
    def __init__(self, backbone_name, weight_path, num_classes):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=False, 
            num_classes=0, 
            checkpoint_path=weight_path
        )
        self.head = nn.Linear(384, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# ==========================================
# Dataset (CPU: Only basic transforms)
# ==========================================
class UnlabeledDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(valid_extensions):
                    self.image_paths.append(os.path.join(root, f))
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, filename

# ==========================================
# Main Generation Logic
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Multi-Model GPU-TTA Pseudo Label Generator")
    parser.add_argument('--data_path', type=str, default='~/autodl-tmp/data/test_shuffled/', help='Unlabeled data path')
    parser.add_argument('--weights_dir', type=str, default='./weights/eval/', help='Directory containing k fold .pth models')
    parser.add_argument('--base_weight_path', type=str, default='~/autodl-tmp/weights/dinov2_small.bin', help='Base offline weights')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size. For RTX 5090 32GB, 64 or 128 is safe.')
    parser.add_argument('--num_workers', type=int, default=8, help='Dataloader workers')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    
    args = parser.parse_args()
    args.data_path = os.path.expanduser(args.data_path)
    args.weights_dir = os.path.expanduser(args.weights_dir)
    args.base_weight_path = os.path.expanduser(args.base_weight_path)
    
    run_dir = get_run_dir(base_dir="runs/pseudo", name=args.name)
    logger = setup_logger(run_dir, 'generate.log')

    logger.info("="*50)
    logger.info("Pseudo Label Generation Configuration:")
    for arg_name, arg_value in sorted(vars(args).items()):
        logger.info(f"  {arg_name:<18}: {arg_value}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    weight_files = sorted(glob.glob(os.path.join(args.weights_dir, "*.pth")))
    if not weight_files:
        logger.error(f"No .pth files found in {args.weights_dir}. Exiting.")
        return
    logger.info(f"Found {len(weight_files)} models for ensembling.")

    models = []
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["qkv"],
        lora_dropout=args.lora_dropout,
        bias="none",
        modules_to_save=["head"]
    )

    logger.info("Loading models into memory...")
    for idx, w_path in enumerate(weight_files, 1):
        logger.info(f"Loading Model {idx}/{len(weight_files)}: {os.path.basename(w_path)}")
        base_model = CustomViTModel(
            backbone_name='vit_small_patch14_dinov2.lvd142m', 
            weight_path=args.base_weight_path,
            num_classes=args.num_classes
        )
        model = get_peft_model(base_model, lora_config)
        model.load_state_dict(torch.load(w_path, map_location='cpu'))
        model.to(device)
        model.eval()
        models.append(model)
        
        # Optional: PyTorch 2.0 compile for extra speed
        if hasattr(torch, 'compile'):
            models[idx-1] = torch.compile(models[idx-1])

    logger.info("Models loaded successfully.")

    # CPU base transform
    base_transform = transforms.Compose([
        transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    dataset = UnlabeledDataset(args.data_path, transform=base_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    logger.info(f"Total unlabeled images to process: {len(dataset)}")
    logger.info("Using fast GPU-TTA (8 variations: 2 flips x 4 rotations) with Mixed Precision (bfloat16).")

    csv_path = os.path.join(run_dir, 'pseudo_labels.csv')
    
    headers = ['Image_Name']
    for i in range(1, len(models) + 1):
        headers.extend([f'M{i}_Class', f'M{i}_Prob'])
    headers.extend(['Ensemble_Class', 'Ensemble_Prob'])

    logger.info("Starting inference loop...")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        with torch.no_grad():
            for batch_idx, (images, filenames) in enumerate(dataloader):
                B = images.size(0)
                images = images.to(device, non_blocking=True)
                
                # GPU-accelerated TTA Generation
                # Shape logic: 2 flips * 4 rotations = 8 variations
                tta_list = []
                for flip in [False, True]:
                    img_f = torch.flip(images, dims=[3]) if flip else images
                    for k in range(4):
                        img_r = torch.rot90(img_f, k=k, dims=[2, 3])
                        tta_list.append(img_r)
                
                # tta_images shape: [B, 8, C, H, W]
                tta_images = torch.stack(tta_list, dim=1)
                _, num_tta, C, H, W = tta_images.shape
                flat_images = tta_images.view(-1, C, H, W)
                
                batch_model_logits = []
                
                # Execute inference with bfloat16 mixed precision for extreme speed
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    for model in models:
                        flat_logits = model(flat_images) 
                        reshaped_logits = flat_logits.view(B, num_tta, args.num_classes)
                        avg_logits = reshaped_logits.mean(dim=1)
                        batch_model_logits.append(avg_logits)
                
                # Convert back to float32 for stable softmax and ensemble calculations
                stacked_logits = torch.stack(batch_model_logits).float()
                
                batch_model_probs = F.softmax(stacked_logits, dim=-1)
                
                ensemble_logits = stacked_logits.mean(dim=0)
                ensemble_probs = F.softmax(ensemble_logits, dim=-1)
                
                for i in range(B):
                    row = [filenames[i]]
                    
                    for m_idx in range(len(models)):
                        probs_m = batch_model_probs[m_idx, i]
                        max_prob_m, class_m = torch.max(probs_m, dim=0)
                        row.extend([class_m.item(), f"{max_prob_m.item():.4f}"])
                        
                    max_prob_ens, class_ens = torch.max(ensemble_probs[i], dim=0)
                    row.extend([class_ens.item(), f"{max_prob_ens.item():.4f}"])
                    
                    writer.writerow(row)
                
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
                    logger.info(f"Processed Batch [{batch_idx + 1}/{len(dataloader)}] ...")

    logger.info("="*50)
    logger.info("Inference completed successfully!")
    logger.info(f"Detailed pseudo-labels saved to: {csv_path}")

if __name__ == "__main__":
    main()