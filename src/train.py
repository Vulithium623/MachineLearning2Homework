import os
import math
import argparse
import logging
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as F_t
import torchvision
import matplotlib.pyplot as plt
import timm
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from peft import LoraConfig, get_peft_model

def get_run_dir(base_dir="runs/train_pseudo", name=None):
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

def setup_logger(run_dir, log_filename='run.log'):
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
class DatasetWrapper(Dataset):
    def __init__(self, subset_dataset, indices, transform=None):
        self.subset_dataset = subset_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.subset_dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label

class PseudoDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

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

def mixup_data(x, y, alpha=0.8):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def save_augmentation_preview(loader, run_dir, logger, num_images=16):
    logger.info("Generating data augmentation preview...")
    batch_x, batch_y = next(iter(loader))
    batch_x = batch_x[:num_images]
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    batch_x = batch_x * std + mean
    batch_x = torch.clamp(batch_x, 0, 1)
    
    grid = torchvision.utils.make_grid(batch_x, nrow=4, padding=2, normalize=False)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.tight_layout()
    preview_path = os.path.join(run_dir, 'augmentation_preview.png')
    plt.savefig(preview_path, dpi=150, bbox_inches='tight')
    plt.close()

# ==========================================
# Models & Loss
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        log_pt = -self.ce_loss(inputs, targets)
        pt = torch.exp(log_pt)
        focal_loss = -((1 - pt) ** self.gamma) * log_pt

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

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
# Core Training Function
# ==========================================
def train_fold(fold, mixed_train_loader, pure_train_loader, val_loader, args, fold_dir, num_classes, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    base_model = CustomViTModel(
        backbone_name='vit_small_patch14_dinov2.lvd142m', 
        weight_path=args.weight_path,
        num_classes=num_classes
    )
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["qkv"],
        lora_dropout=args.lora_dropout,
        bias="none",
        modules_to_save=["head"]
    )
    model = get_peft_model(base_model, lora_config).to(device)
    
    criterion = FocalLoss(gamma=args.focal_gamma)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_f1 = 0.0
    washout_start_epoch = args.epochs - args.washout_epochs
    
    logger.info(f"=== Starting Training for Fold {fold} ===")
    
    for epoch in range(1, args.epochs + 1):
        in_washout = (args.washout_epochs > 0) and (epoch > washout_start_epoch)
        
        if in_washout:
            if epoch == washout_start_epoch + 1:
                logger.info(f"[Fold {fold}] Epoch {epoch}: Entering Wash-out Phase! Using PURE true data. MixUp remains active.")
            current_loader = pure_train_loader
        else:
            current_loader = mixed_train_loader
        
        # --- Train ---
        model.train()
        train_loss = 0.0
        
        for inputs, labels in current_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            mixup_active = (args.mixup_alpha > 0)
            
            if mixup_active:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, args.mixup_alpha)
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(current_loader.dataset)
        current_lr = scheduler.get_last_lr()[0]
        
        # --- Eval ---
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        
        logger.info(f"[Fold {fold}] Epoch [{epoch:02d}/{args.epochs}] | LR: {current_lr:.2e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | F1: {macro_f1:.4f} | B-Acc: {balanced_acc:.4f}")
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), os.path.join(fold_dir, 'best.pth'))
            logger.info(f"[Fold {fold}] --> Saved new best model (F1: {best_f1:.4f})")
            
        scheduler.step()
            
    torch.save(model.state_dict(), os.path.join(fold_dir, 'last.pth'))
    return best_f1

def main():
    parser = argparse.ArgumentParser(description="K-Fold LoRA Fine-tuning with Pseudo Labels")
    
    # Base Data Args
    parser.add_argument('--data_path', type=str, default='~/autodl-tmp/data/train_few_shot/', help='Path to true dataset')
    parser.add_argument('--weight_path', type=str, default='~/autodl-tmp/weights/dinov2_small.bin', help='Path to offline weights')
    
    # Pseudo Label Args
    parser.add_argument('--unlabeled_data_path', type=str, default='~/autodl-tmp/data/test_shuffled/', help='Path to unlabeled images')
    parser.add_argument('--pseudo_csv_path', type=str, default='./runs/pseudo/2/pseudo_labels.csv', help='Path to pseudo labels CSV')
    parser.add_argument('--pseudo_prob_thresh', type=float, default=0.85, help='Threshold for pseudo label confidence (legacy)')
    parser.add_argument('--washout_epochs', type=int, default=5, help='Number of epochs at the end to train only on true labels')
    
    # Training Args
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs per fold')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate (Max LR for Cosine)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers')
    parser.add_argument('--k_folds', type=int, default=7, help='Number of folds for CV')
    
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    
    # Augmentation Args
    parser.add_argument('--no_hflip', action='store_true')
    parser.add_argument('--no_vflip', action='store_true')
    parser.add_argument('--no_rotate', action='store_true')
    parser.add_argument('--translate', type=float, default=0.1)
    parser.add_argument('--rotate_pad_mode', type=str, default='reflect')
    parser.add_argument('--color_jitter', type=float, default=0.2)
    parser.add_argument('--mixup_alpha', type=float, default=0.8)
    
    parser.add_argument('--elastic_prob', type=float, default=0.5)
    parser.add_argument('--elastic_alpha', type=float, default=25.0)
    parser.add_argument('--elastic_sigma', type=float, default=4.0)
    
    args = parser.parse_args()
    args.data_path = os.path.expanduser(args.data_path)
    args.weight_path = os.path.expanduser(args.weight_path)
    args.unlabeled_data_path = os.path.expanduser(args.unlabeled_data_path)
    args.pseudo_csv_path = os.path.expanduser(args.pseudo_csv_path)
    
    run_dir = get_run_dir(base_dir="runs/train_pseudo", name=args.name)
    main_logger = setup_logger(run_dir, 'main_run.log')

    main_logger.info("="*50)
    main_logger.info("Experiment Arguments (Hyperparameters):")
    for arg_name, arg_value in sorted(vars(args).items()):
        main_logger.info(f"  {arg_name:<18}: {arg_value}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    main_logger.info("="*50)
    main_logger.info(f"Phase 4: Aggressive Semi-Supervised Training Started.")
    main_logger.info(f"Logs & Models saving to: {run_dir}")
    main_logger.info("="*50)

    # Load Full True Dataset
    full_dataset = datasets.ImageFolder(root=args.data_path, transform=None)
    num_classes = len(full_dataset.classes)
    targets = full_dataset.targets
    
    # Parse and Filter Pseudo Labels (Aggressive Version)
    main_logger.info(f"Parsing pseudo-labels from: {args.pseudo_csv_path}")
    df_pseudo = pd.read_csv(args.pseudo_csv_path)
    model_class_cols = [c for c in df_pseudo.columns if c.endswith('_Class') and c != 'Ensemble_Class']
    
    # Condition: At least 4 models must match the Ensemble_Class prediction
    agreements = df_pseudo[model_class_cols].eq(df_pseudo['Ensemble_Class'], axis=0).sum(axis=1)
    cond_agree = agreements >= 4
    df_agree = df_pseudo[cond_agree].copy()
    
    target_hard = 700  # 0.70 <= prob < 0.85
    target_easy = 700  # 0.85 <= prob <= 1.00
    
    class_dfs = []
    for c in range(num_classes):
        df_c = df_agree[df_agree['Ensemble_Class'] == c]
        
        df_hard = df_c[(df_c['Ensemble_Prob'] >= 0.80) & (df_c['Ensemble_Prob'] < 0.9)]
        df_easy = df_c[(df_c['Ensemble_Prob'] >= 0.9)]
        
        sampled_hard = df_hard.sample(n=min(len(df_hard), target_hard), random_state=args.seed)
        sampled_easy = df_easy.sample(n=min(len(df_easy), target_easy), random_state=args.seed)
        
        # Fill missing quotas from the other pool if available
        short_hard = target_hard - len(sampled_hard)
        if short_hard > 0 and len(df_easy) > target_easy:
            extra_easy = df_easy.drop(sampled_easy.index).sample(n=min(len(df_easy) - len(sampled_easy), short_hard), random_state=args.seed)
            sampled_easy = pd.concat([sampled_easy, extra_easy])
            
        short_easy = target_easy - len(sampled_easy)
        if short_easy > 0 and len(df_hard) > target_hard:
            extra_hard = df_hard.drop(sampled_hard.index).sample(n=min(len(df_hard) - len(sampled_hard), short_easy), random_state=args.seed)
            sampled_hard = pd.concat([sampled_hard, extra_hard])
            
        class_dfs.append(pd.concat([sampled_hard, sampled_easy]))
        
    n_min = min([len(df) for df in class_dfs])
    pseudo_data_list = []
    
    if n_min == 0:
        main_logger.warning("One or more classes have 0 pseudo-labels under the current criteria! No pseudo-labels will be used.")
    else:
        selected_rows = []
        for df_c in class_dfs:
            if len(df_c) > n_min:
                selected_rows.append(df_c.sample(n=n_min, random_state=args.seed))
            else:
                selected_rows.append(df_c)
                
        final_pseudo_df = pd.concat(selected_rows)
        for _, row in final_pseudo_df.iterrows():
            img_name = row['Image_Name']
            label = int(row['Ensemble_Class'])
            full_path = os.path.join(args.unlabeled_data_path, img_name)
            pseudo_data_list.append((full_path, label))
            
        main_logger.info(f"Aggressive filtering completed. Class-balanced top-K: {n_min}. Total pseudo images: {len(pseudo_data_list)}")

    # Setup Transforms
    train_transform_list = []
    if not args.no_hflip: train_transform_list.append(transforms.RandomHorizontalFlip())
    if not args.no_vflip: train_transform_list.append(transforms.RandomVerticalFlip())
    if not args.no_rotate or args.translate > 0:
        train_transform_list.append(SafeRandomAffine(enable_rotate=not args.no_rotate, translate_frac=args.translate, mode=args.rotate_pad_mode))
    if args.elastic_prob > 0:
        train_transform_list.append(transforms.RandomApply([SafeElasticTransform(alpha=args.elastic_alpha, sigma=args.elastic_sigma, mode=args.rotate_pad_mode)], p=args.elastic_prob))
    if args.color_jitter > 0:
        train_transform_list.append(transforms.ColorJitter(brightness=args.color_jitter, contrast=args.color_jitter))
        
    train_transform_list.extend([
        transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    train_transform = transforms.Compose(train_transform_list)
    val_transform = transforms.Compose([
        transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets), 1):
        fold_dir = os.path.join(run_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        fold_logger = setup_logger(fold_dir, f'fold_{fold}.log')
        
        fold_logger.info(f"\n--- Preparing Fold {fold}/{args.k_folds} ---")
        
        true_train_dataset = DatasetWrapper(full_dataset, train_idx, transform=train_transform)
        true_val_dataset = DatasetWrapper(full_dataset, val_idx, transform=val_transform)
        pseudo_dataset = PseudoDataset(pseudo_data_list, transform=train_transform)
        
        # Calculate Oversampling Multiplier for True Data
        num_true_train = len(true_train_dataset)
        num_pseudo = len(pseudo_dataset)
        multiplier = 5

        fold_logger.info(f"Base True Samples: {num_true_train} | Pseudo Samples: {num_pseudo}")
        if num_pseudo > 0:
            fold_logger.info(f"Oversampling True Data by factor of {multiplier} to balance Mixed Batch.")
            
        mixed_train_dataset = torch.utils.data.ConcatDataset([true_train_dataset] * multiplier + [pseudo_dataset])
        
        mixed_train_loader = DataLoader(mixed_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        pure_train_loader = DataLoader(true_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(true_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        if fold == 1: 
            save_augmentation_preview(mixed_train_loader, run_dir, main_logger, num_images=16)

        best_fold_f1 = train_fold(fold, mixed_train_loader, pure_train_loader, val_loader, args, fold_dir, num_classes, fold_logger)
        fold_results.append(best_fold_f1)
        
        main_logger.info(f"Fold {fold} Finished. Best F1: {best_fold_f1:.4f}")

    main_logger.info("="*50)
    main_logger.info("FINAL K-FOLD CROSS VALIDATION RESULTS:")
    for f, score in enumerate(fold_results, 1):
        main_logger.info(f"  Fold {f}: {score:.4f}")
    
    avg_f1 = np.mean(fold_results)
    std_f1 = np.std(fold_results)
    main_logger.info("-" * 20)
    main_logger.info(f"  Average Macro-F1 : {avg_f1:.4f} ± {std_f1:.4f}")
    main_logger.info("="*50)

if __name__ == "__main__":
    main()