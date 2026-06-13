import os
import argparse
import logging
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from peft import LoraConfig, get_peft_model

def get_run_dir(base_dir="runs/eval", name="final_submission"):
    os.makedirs(base_dir, exist_ok=True)
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

def setup_logger(run_dir, log_filename='eval.log'):
    logger = logging.getLogger(f"Eval_{run_dir}")
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
# Datasets & Transforms
# ==========================================
class InferenceDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.img_names = [f for f in sorted(os.listdir(img_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, img_name

# ==========================================
# Models
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

def init_lora_model(weight_path, num_classes, r, alpha, dropout, device):
    base_model = CustomViTModel(
        backbone_name='vit_small_patch14_dinov2.lvd142m', 
        weight_path=weight_path,
        num_classes=num_classes
    )
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["qkv"],
        lora_dropout=dropout,
        bias="none",
        modules_to_save=["head"]
    )
    model = get_peft_model(base_model, lora_config).to(device)
    return model

# ==========================================
# Main Execution
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Final Inference Script for Unlabeled Data")
    
    parser.add_argument('--unlabeled_data_path', type=str, default='~/autodl-tmp/data/test_shuffled/')
    parser.add_argument('--base_weight_path', type=str, default='~/autodl-tmp/weights/dinov2_small.bin')
    parser.add_argument('--lora_weight_path', type=str, default='./weights/best.pth')
    
    parser.add_argument('--name', type=str, default='submission')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_classes', type=int, default=5)
    
    # Must match the student model configuration
    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=64)
    
    args = parser.parse_args()
    args.unlabeled_data_path = os.path.expanduser(args.unlabeled_data_path)
    args.base_weight_path = os.path.expanduser(args.base_weight_path)
    args.lora_weight_path = os.path.expanduser(args.lora_weight_path)
    
    run_dir = get_run_dir(base_dir="runs/eval", name=args.name)
    logger = setup_logger(run_dir, 'eval.log')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("="*50)
    logger.info("Final Evaluation Started (Pure Single Model Inference)")
    logger.info("="*50)
    
    if not os.path.exists(args.lora_weight_path):
        logger.error(f"Cannot find LoRA weights at {args.lora_weight_path}!")
        return

    transform_test = transforms.Compose([
        transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    test_dataset = InferenceDataset(args.unlabeled_data_path, transform_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    logger.info(f"Loaded {len(test_dataset)} images for inference.")
    
    logger.info(f"Initializing Model (r={args.lora_r}, alpha={args.lora_alpha})")
    model = init_lora_model(
        args.base_weight_path, args.num_classes, 
        r=args.lora_r, alpha=args.lora_alpha, dropout=0.0, device=device
    )
    
    model.load_state_dict(torch.load(args.lora_weight_path, map_location=device))
    model.eval()
    
    all_filenames = []
    all_predictions = []
    
    logger.info("Starting inference...")
    
    with torch.no_grad():
        for inputs, img_names in tqdm(test_loader, desc="Inferring"):
            inputs = inputs.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
            
            preds = torch.argmax(outputs, dim=1)
            preds_np = preds.cpu().numpy()
            
            all_filenames.extend(img_names)
            all_predictions.extend([f"Class_{p}" for p in preds_np])
            
    # Create DataFrame and save
    df_submission = pd.DataFrame({
        'filename': all_filenames,
        'label': all_predictions
    })
    
    output_csv = os.path.join(run_dir, "submission.csv")
    df_submission.to_csv(output_csv, index=False)
    
    logger.info("="*50)
    logger.info(f"Inference Completed. Saved {len(df_submission)} predictions.")
    logger.info(f"Submission file located at: {output_csv}")
    logger.info("="*50)

if __name__ == "__main__":
    main()