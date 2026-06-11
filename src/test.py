import os
import argparse
import logging
import numpy as np
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
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model

def get_run_dir(base_dir="runs/train", name=None):
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
    logger.handlers = [] # Clear existing handlers
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler(os.path.join(run_dir, log_filename))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

# ==========================================
# Focal Loss Implementation
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

def generate_analysis_plots(epoch_results, run_dir, logger):
    logger.info("Generating analysis plots...")
    epochs = [r['epoch'] for r in epoch_results]
    val_f1s = [r['val_f1'] for r in epoch_results]
    test_f1s = [r['test_f1'] for r in epoch_results]
    
    plt.figure(figsize=(14, 6))
    
    # Plot 1: F1 over Epochs
    plt.subplot(1, 2, 1)
    plt.plot(epochs, val_f1s, label='Validation F1', marker='o', alpha=0.7)
    plt.plot(epochs, test_f1s, label='Test F1', marker='s', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1 Score')
    plt.title('Validation and Test F1 over Epochs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: Scatter Val vs Test F1
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(val_f1s, test_f1s, c=epochs, cmap='viridis', s=50, alpha=0.8)
    cbar = plt.colorbar(scatter)
    cbar.set_label('Epoch Number')
    
    # Identity line for reference
    min_val = min(min(val_f1s), min(test_f1s))
    max_val = max(max(val_f1s), max(test_f1s))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x (Perfect Match)')
    
    plt.xlabel('Validation F1')
    plt.ylabel('Test F1')
    plt.title('Correlation: Validation F1 vs Test F1')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_path = os.path.join(run_dir, 'validation_test_correlation.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Plots saved to {plot_path}")

# ==========================================
# Core Training Function
# ==========================================
def train_single_split(train_loader, val_loader, test_loader, args, run_dir, num_classes, logger):
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

    mixup_disabled_flag = False
    cutoff_epoch = args.epochs - int(args.epochs * args.mixup_cutoff) if args.mixup_cutoff > 0 else args.epochs + 1
    
    logger.info("=== Starting Training ===")
    
    epoch_results = []
    best_val_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        if args.mixup_alpha > 0 and args.mixup_cutoff > 0 and epoch > cutoff_epoch and not mixup_disabled_flag:
            logger.info(f"Epoch {epoch}: MixUp is now DISABLED!")
            mixup_disabled_flag = True

        # --- Train ---
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            mixup_active = (args.mixup_alpha > 0) and (not mixup_disabled_flag)
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
            
        train_loss /= len(train_loader.dataset)
        current_lr = scheduler.get_last_lr()[0]
        
        # --- Eval on Validation Set ---
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        
        # --- Eval on Test Set ---
        test_preds, test_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
                
        test_f1 = f1_score(test_labels, test_preds, average='macro', zero_division=0)
        
        logger.info(f"Epoch [{epoch:02d}/{args.epochs}] | LR: {current_lr:.2e} | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")
        
        epoch_results.append({
            'epoch': epoch,
            'val_f1': val_f1,
            'test_f1': test_f1
        })
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(run_dir, 'best_val.pth'))
            
        # Step the Cosine Scheduler
        scheduler.step()
            
    torch.save(model.state_dict(), os.path.join(run_dir, 'last.pth'))
    return epoch_results

def main():
    parser = argparse.ArgumentParser(description="Hypothesis Verification: LoRA Fine-tuning with DINOv2")
    parser.add_argument('--data_path', type=str, default='~/autodl-tmp/data/train_few_shot/', help='Path to dataset')
    parser.add_argument('--weight_path', type=str, default='~/autodl-tmp/weights/dinov2_small.bin', help='Path to offline weights')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate (Max LR for Cosine)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers')
    
    # Dataset Splits
    parser.add_argument('--test_size', type=float, default=0.2, help='Ratio of data reserved for testing')
    parser.add_argument('--val_size', type=float, default=0.15, help='Ratio of the remaining data used for validation')
    
    # Optimizer & Loss args
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for AdamW')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma value for Focal Loss')
    
    # LoRA args
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    
    # Augmentation args
    parser.add_argument('--no_hflip', action='store_true')
    parser.add_argument('--no_vflip', action='store_true')
    parser.add_argument('--no_rotate', action='store_true')
    parser.add_argument('--translate', type=float, default=0.1)
    parser.add_argument('--rotate_pad_mode', type=str, default='reflect')
    parser.add_argument('--color_jitter', type=float, default=0.2)
    parser.add_argument('--mixup_alpha', type=float, default=0.8)
    parser.add_argument('--mixup_cutoff', type=float, default=0.2)
    
    # Medical Elastic Transform
    parser.add_argument('--elastic_prob', type=float, default=0.5, help='Probability to apply Elastic Transform')
    parser.add_argument('--elastic_alpha', type=float, default=25.0, help='Elastic Transform Alpha')
    parser.add_argument('--elastic_sigma', type=float, default=4.0, help='Elastic Transform Sigma')
    
    args = parser.parse_args()
    args.data_path = os.path.expanduser(args.data_path)
    args.weight_path = os.path.expanduser(args.weight_path)
    
    run_dir = get_run_dir(base_dir="runs/train", name=args.name)
    logger = setup_logger(run_dir, 'main_run.log')

    logger.info("="*50)
    logger.info("Experiment Arguments (Hyperparameters):")
    for arg_name, arg_value in sorted(vars(args).items()):
        logger.info(f"  {arg_name:<18}: {arg_value}")
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info("="*50)
    logger.info(f"Phase 2.5: Hypothesis Verification Experiment Started.")
    logger.info(f"Logs & Models saving to: {run_dir}")
    logger.info(f"Splitting strategy: {args.test_size*100}% Test. From the remaining, {args.val_size*100}% Validation.")
    logger.info("="*50)

    # Setup Transforms
    train_transform_list = []
    if not args.no_hflip: train_transform_list.append(transforms.RandomHorizontalFlip())
    if not args.no_vflip: train_transform_list.append(transforms.RandomVerticalFlip())
    if not args.no_rotate or args.translate > 0:
        train_transform_list.append(SafeRandomAffine(enable_rotate=not args.no_rotate, translate_frac=args.translate, mode=args.rotate_pad_mode))
    if args.elastic_prob > 0:
        train_transform_list.append(
            transforms.RandomApply(
                [SafeElasticTransform(alpha=args.elastic_alpha, sigma=args.elastic_sigma, mode=args.rotate_pad_mode)],
                p=args.elastic_prob
            )
        )
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

    # Load Full Dataset
    full_dataset = datasets.ImageFolder(root=args.data_path, transform=None)
    num_classes = len(full_dataset.classes)
    targets = full_dataset.targets
    
    # 1. Train/Test Splitting (Holdout 20%)
    indices = np.arange(len(targets))
    train_val_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, stratify=targets, random_state=args.seed
    )
    
    # 2. Train/Val Splitting (15% of the remaining 80%)
    train_val_targets = [targets[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=args.val_size, stratify=train_val_targets, random_state=args.seed
    )
    
    logger.info(f"Dataset Details:")
    logger.info(f"  Total Samples : {len(targets)}")
    logger.info(f"  Train Samples : {len(train_idx)}")
    logger.info(f"  Val Samples   : {len(val_idx)}")
    logger.info(f"  Test Samples  : {len(test_idx)}")
    
    train_dataset = DatasetWrapper(full_dataset, train_idx, transform=train_transform)
    val_dataset = DatasetWrapper(full_dataset, val_idx, transform=val_transform)
    test_dataset = DatasetWrapper(full_dataset, test_idx, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    save_augmentation_preview(train_loader, run_dir, logger, num_images=16)

    # Execute Training and get metrics for all epochs
    epoch_results = train_single_split(train_loader, val_loader, test_loader, args, run_dir, num_classes, logger)

    # Sort results by Validation F1
    sorted_results = sorted(epoch_results, key=lambda x: x['val_f1'], reverse=True)

    logger.info("="*50)
    logger.info("FINAL MODELS RANKED BY VALIDATION F1 SCORE:")
    logger.info("Rank | Epoch | Validation F1 | Test F1")
    logger.info("-" * 45)
    for rank, res in enumerate(sorted_results, 1):
        logger.info(f" {rank:02d}  |  {res['epoch']:02d}   |    {res['val_f1']:.4f}     |  {res['test_f1']:.4f}")
    logger.info("="*50)

    # Generate visual correlation plots
    generate_analysis_plots(epoch_results, run_dir, logger)

if __name__ == "__main__":
    main()