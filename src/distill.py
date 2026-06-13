import os
import argparse
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as F_t
import timm
from sklearn.metrics import f1_score, balanced_accuracy_score
from peft import LoraConfig, get_peft_model

def get_run_dir(base_dir="runs/distill", name=None):
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

def setup_logger(run_dir, log_filename='distill.log'):
    logger = logging.getLogger(f"Experiment_{run_dir}")
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
class ExtractDataset(Dataset):
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

class StudentKDDataset(Dataset):
    def __init__(self, img_dir, soft_labels_dict, transform):
        self.img_dir = img_dir
        self.img_names = list(soft_labels_dict.keys())
        self.soft_labels_dict = soft_labels_dict
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        soft_logits = self.soft_labels_dict[img_name]
        return img, soft_logits

class SafeRandomAffine:
    def __init__(self, enable_rotate=True, translate_frac=0.1, mode='reflect'):
        self.enable_rotate = enable_rotate
        self.translate_frac = translate_frac
        self.mode = mode
        self.pad_mode = 'constant' if mode == 'zeros' else mode

    def __call__(self, img):
        if not self.enable_rotate and self.translate_frac <= 0:
            return img
            
        degrees = [-180.0, 180.0] if self.enable_rotate else [0.0, 0.0]
        w, h = img.size
        angle = transforms.RandomRotation.get_params(degrees)
        
        if self.translate_frac > 0:
            max_dx = float(self.translate_frac * w)
            max_dy = float(self.translate_frac * h)
            tx = int(np.round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(np.round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if self.pad_mode == 'constant':
            return F_t.affine(img, angle=angle, translate=translations, scale=1.0, shear=0.0, fill=0)
        else:
            pad_w, pad_h = w // 2, h // 2
            img_padded = F_t.pad(img, (pad_w, pad_h, pad_w, pad_h), padding_mode=self.pad_mode)
            img_transformed = F_t.affine(img_padded, angle=angle, translate=translations, scale=1.0, shear=0.0)
            return F_t.center_crop(img_transformed, (h, w))

class SafeElasticTransform:
    def __init__(self, alpha=25.0, sigma=4.0, mode='reflect'):
        self.transform = transforms.ElasticTransform(alpha=alpha, sigma=sigma)
        self.mode = mode
        self.pad_mode = 'constant' if mode == 'zeros' else mode
        self.pad_size = int(alpha)

    def __call__(self, img):
        if self.pad_mode == 'constant':
            return self.transform(img)
            
        w, h = img.size
        pad_p = self.pad_size
        img_padded = F_t.pad(img, (pad_p, pad_p, pad_p, pad_p), padding_mode=self.pad_mode)
        img_transformed = self.transform(img_padded)
        return F_t.center_crop(img_transformed, (h, w))

# ==========================================
# Models & Loss
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

def distillation_loss(student_logits, teacher_logits, temperature=3.0):
    log_student_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    
    loss_kd = F.kl_div(log_student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    return loss_kd

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
    parser = argparse.ArgumentParser(description="Offline Fast Knowledge Distillation")
    
    # Data Args
    parser.add_argument('--true_data_path', type=str, default='~/autodl-tmp/data/train_few_shot/')
    parser.add_argument('--unlabeled_data_path', type=str, default='~/autodl-tmp/data/test_shuffled/')
    parser.add_argument('--weight_path', type=str, default='~/autodl-tmp/weights/dinov2_small.bin')
    parser.add_argument('--teacher_run_dir', type=str, required=True)
    parser.add_argument('--cache_file', type=str, default='teacher_soft_labels_cache.pt')
    
    # Training Args
    parser.add_argument('--name', type=str, default='distill_student')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Distillation Args
    parser.add_argument('--temperature', type=float, default=3.0)
    
    # Teacher Architecture Args
    parser.add_argument('--k_folds', type=int, default=5)
    parser.add_argument('--teacher_lora_r', type=int, default=8)
    parser.add_argument('--teacher_lora_alpha', type=int, default=16)
    
    # Student Architecture Args
    parser.add_argument('--student_lora_r', type=int, default=32)
    parser.add_argument('--student_lora_alpha', type=int, default=64)
    parser.add_argument('--student_lora_dropout', type=float, default=0.1)
    
    # Augmentation Args
    parser.add_argument('--translate', type=float, default=0.1)
    parser.add_argument('--rotate_pad_mode', type=str, default='reflect')
    parser.add_argument('--color_jitter', type=float, default=0.3)
    parser.add_argument('--elastic_prob', type=float, default=0.5)
    parser.add_argument('--elastic_alpha', type=float, default=25.0)
    parser.add_argument('--elastic_sigma', type=float, default=4.0)
    
    args = parser.parse_args()
    args.true_data_path = os.path.expanduser(args.true_data_path)
    args.unlabeled_data_path = os.path.expanduser(args.unlabeled_data_path)
    args.weight_path = os.path.expanduser(args.weight_path)
    
    run_dir = get_run_dir(base_dir="runs/distill", name=args.name)
    logger = setup_logger(run_dir, 'distill.log')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    temp_val = datasets.ImageFolder(root=args.true_data_path, transform=None)
    num_classes = len(temp_val.classes)

    logger.info("="*50)
    logger.info("Offline Knowledge Distillation Started")
    logger.info("="*50)

    # ---------------------------------------------------------
    # PHASE 1: PRE-COMPUTE SOFT LABELS (OFFLINE EXTRACTION)
    # ---------------------------------------------------------
    cache_path = os.path.join(args.teacher_run_dir, args.cache_file)
    
    if os.path.exists(cache_path):
        logger.info(f"Found existing Soft Labels cache at {cache_path}. Skipping extraction.")
        soft_labels_dict = torch.load(cache_path)
    else:
        logger.info(f"No cache found. Starting offline extraction of unlabeled images...")
        
        teachers = []
        for f in range(1, args.k_folds + 1):
            teacher = init_lora_model(
                args.weight_path, num_classes, 
                r=args.teacher_lora_r, alpha=args.teacher_lora_alpha, dropout=0.0, device=device
            )
            ckpt_path = os.path.join(args.teacher_run_dir, f"fold_{f}", "best.pth")
            teacher.load_state_dict(torch.load(ckpt_path, map_location=device))
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False
            teachers.append(teacher)
            
        transform_weak = transforms.Compose([
            transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        extract_dataset = ExtractDataset(args.unlabeled_data_path, transform_weak)
        extract_loader = DataLoader(extract_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        soft_labels_dict = {}
        
        logger.info("Extracting features. This will take a few minutes but only runs ONCE.")
        with torch.no_grad():
            for inputs, img_names in tqdm(extract_loader, desc="Pre-computing Soft Labels"):
                inputs = inputs.to(device, non_blocking=True)
                outputs = [t(inputs) for t in teachers]
                ensemble_logits = torch.stack(outputs).mean(dim=0).cpu()
                
                for i, name in enumerate(img_names):
                    soft_labels_dict[name] = ensemble_logits[i].clone()
                    
        torch.save(soft_labels_dict, cache_path)
        logger.info(f"Successfully saved {len(soft_labels_dict)} soft labels to {cache_path}")
        
        del teachers
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # PHASE 2: STUDENT TRAINING
    # ---------------------------------------------------------
    transform_val = transforms.Compose([
        transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    val_dataset = datasets.ImageFolder(root=args.true_data_path, transform=transform_val)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    transform_strong = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        SafeRandomAffine(enable_rotate=True, translate_frac=args.translate, mode=args.rotate_pad_mode),
        transforms.RandomApply([SafeElasticTransform(alpha=args.elastic_alpha, sigma=args.elastic_sigma, mode=args.rotate_pad_mode)], p=args.elastic_prob),
        transforms.ColorJitter(brightness=args.color_jitter, contrast=args.color_jitter),
        transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    student_dataset = StudentKDDataset(args.unlabeled_data_path, soft_labels_dict, transform_strong)
    student_loader = DataLoader(student_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    logger.info(f"Initializing Student Model (r={args.student_lora_r}, alpha={args.student_lora_alpha})")
    student = init_lora_model(
        args.weight_path, num_classes, 
        r=args.student_lora_r, alpha=args.student_lora_alpha, dropout=args.student_lora_dropout, device=device
    )
    
    optimizer = optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_f1 = 0.0
    scaler = torch.cuda.amp.GradScaler()

    logger.info("="*50)
    logger.info("Beginning Fast Offline Knowledge Distillation (AMP Enabled)")
    
    for epoch in range(1, args.epochs + 1):
        student.train()
        train_loss = 0.0
        
        train_pbar = tqdm(student_loader, desc=f"Epoch [{epoch:02d}/{args.epochs}] Training", leave=False)
        for inputs_strong, target_logits in train_pbar:
            inputs_strong = inputs_strong.to(device, non_blocking=True)
            target_logits = target_logits.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                student_logits = student(inputs_strong)
                loss = distillation_loss(student_logits, target_logits, temperature=args.temperature)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * inputs_strong.size(0)
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        train_loss /= len(student_loader.dataset)
        current_lr = scheduler.get_last_lr()[0]
        
        student.eval()
        all_preds, all_labels = [], []
        
        val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch:02d}/{args.epochs}] Validating", leave=False)
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    outputs = student(inputs)
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        
        logger.info(f"Epoch [{epoch:02d}/{args.epochs}] | LR: {current_lr:.2e} | KD Loss: {train_loss:.4f} | Real Val F1: {macro_f1:.4f} | Real Val B-Acc: {balanced_acc:.4f}")
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_path = os.path.join(run_dir, 'best_student.pth')
            torch.save(student.state_dict(), best_path)
            logger.info(f"  --> Saved new best Student (F1: {best_f1:.4f})")
            
        scheduler.step()
        
    last_path = os.path.join(run_dir, 'last_student.pth')
    torch.save(student.state_dict(), last_path)
    
    logger.info("="*50)
    logger.info(f"Distillation Completed. Best Student F1: {best_f1:.4f}")
    logger.info(f"Final Student Weights saved at: {os.path.join(run_dir, 'best_student.pth')}")

if __name__ == "__main__":
    main()