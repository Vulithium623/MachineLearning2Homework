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
import timm
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
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
            new_dir = f"{target_dir}{i}"
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
                return new_dir
            i += 1

def setup_logger(run_dir):
    logger = logging.getLogger("Experiment")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler(os.path.join(run_dir, 'run.log'))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

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

class SafeRandomRotation:
    """
    Custom Rotation that supports reflection/edge padding to avoid black borders.
    """
    def __init__(self, degrees, mode='reflect'):
        self.degrees = degrees
        self.mode = mode
        if self.mode == 'zeros':
            self.pad_mode = 'constant'
        else:
            self.pad_mode = self.mode

    def __call__(self, img):
        angle = transforms.RandomRotation.get_params([-self.degrees, self.degrees])
        
        if self.pad_mode == 'constant':
            return F_t.rotate(img, angle, fill=0)
        else:
            w, h = img.size
            pad_w, pad_h = w // 2, h // 2
            # 1. Pad using the chosen mode
            img_padded = F_t.pad(img, (pad_w, pad_h, pad_w, pad_h), padding_mode=self.pad_mode)
            # 2. Rotate
            img_rotated = F_t.rotate(img_padded, angle)
            # 3. Center crop back to original size
            return F_t.center_crop(img_rotated, (h, w))

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

def main():
    parser = argparse.ArgumentParser(description="LoRA Fine-tuning with DINOv2 (Fixed Transforms)")
    parser.add_argument('--data_path', type=str, default='~/autodl-tmp/data/train_few_shot/', help='Path to dataset')
    parser.add_argument('--weight_path', type=str, default='~/autodl-tmp/weights/dinov2_small.bin', help='Path to weights')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--val_split', type=float, default=0.15, help='Validation set split ratio')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers')
    
    # LoRA specific arguments
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    
    # Data Augmentation Arguments
    parser.add_argument('--no_hflip', action='store_true', help="Disable Horizontal Flip")
    parser.add_argument('--no_vflip', action='store_true', help="Disable Vertical Flip")
    parser.add_argument('--rotate_deg', type=float, default=15.0, help="Rotation degrees (default: 15)")
    parser.add_argument('--rotate_pad_mode', type=str, default='reflect', choices=['zeros', 'reflect', 'edge', 'symmetric'], help="Padding mode for rotation")
    parser.add_argument('--color_jitter', type=float, default=0.2, help="Color Jitter factor for brightness/contrast (default: 0.2, 0 to disable)")
    parser.add_argument('--mixup_alpha', type=float, default=0.8, help="MixUp alpha (default: 0.8, 0 to disable)")
    parser.add_argument('--mixup_cutoff', type=float, default=0.2, help="Disable MixUp in the last X proportion of epochs (default: 0.2. 0 to never disable)")
    
    args = parser.parse_args()
    
    args.data_path = os.path.expanduser(args.data_path)
    args.weight_path = os.path.expanduser(args.weight_path)
    
    run_dir = get_run_dir(base_dir="runs/train", name=args.name)
    logger = setup_logger(run_dir)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info("="*50)
    logger.info(f"Experiment Started. Results saved to {run_dir}")
    logger.info("="*50)
    logger.info("HYPERPARAMETERS:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    use_hflip = not args.no_hflip
    use_vflip = not args.no_vflip
    
    cutoff_epoch = args.epochs - int(args.epochs * args.mixup_cutoff) if args.mixup_cutoff > 0 else args.epochs + 1
    
    logger.info("="*50)
    logger.info("DATA AUGMENTATION SETTINGS:")
    logger.info(f"  Horizontal Flip : {'Enabled' if use_hflip else 'Disabled'}")
    logger.info(f"  Vertical Flip   : {'Enabled' if use_vflip else 'Disabled'}")
    logger.info(f"  Rotation        : Random [-{args.rotate_deg}, +{args.rotate_deg}] degrees (Pad Mode: {args.rotate_pad_mode})")
    logger.info(f"  Color Jitter    : {'Enabled (Brightness/Contrast: ' + str(args.color_jitter) + ', Saturation: Disabled)' if args.color_jitter > 0 else 'Disabled'}")
    logger.info(f"  MixUp           : {'Enabled (Alpha: ' + str(args.mixup_alpha) + ')' if args.mixup_alpha > 0 else 'Disabled'}")
    if args.mixup_alpha > 0 and args.mixup_cutoff > 0:
        logger.info(f"  MixUp Cutoff    : Enabled (Will disable after epoch {cutoff_epoch})")
    logger.info("="*50)

    # Transforms Pipeline: Geometrics & Color first, Resize LAST.
    train_transform_list = []
    
    if use_hflip:
        train_transform_list.append(transforms.RandomHorizontalFlip())
    if use_vflip:
        train_transform_list.append(transforms.RandomVerticalFlip())
    if args.rotate_deg > 0:
        train_transform_list.append(SafeRandomRotation(args.rotate_deg, mode=args.rotate_pad_mode))
    if args.color_jitter > 0:
        # Note: saturation is omitted explicitly
        train_transform_list.append(transforms.ColorJitter(brightness=args.color_jitter, contrast=args.color_jitter))
        
    # Resize must happen right before ToTensor
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

    logger.info("Loading dataset and applying Stratified Split...")
    full_dataset = datasets.ImageFolder(root=args.data_path, transform=None)
    num_classes = len(full_dataset.classes)
    
    targets = full_dataset.targets
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_split, random_state=args.seed)
    train_indices, val_indices = next(sss.split(np.zeros(len(targets)), targets))
    
    train_dataset = DatasetWrapper(full_dataset, train_indices, transform=train_transform)
    val_dataset = DatasetWrapper(full_dataset, val_indices, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    logger.info(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

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
    
    model = get_peft_model(base_model, lora_config)
    model = model.to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable Parameters: {trainable_params} / {all_params} ({100 * trainable_params / all_params:.4f}%)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = 0.0
    mixup_disabled_flag = False
    
    for epoch in range(1, args.epochs + 1):
        if args.mixup_alpha > 0 and args.mixup_cutoff > 0 and epoch > cutoff_epoch and not mixup_disabled_flag:
            logger.info(f"--- Epoch {epoch}: MixUp is now DISABLED for the remaining epochs to sharpen predictions! ---")
            mixup_disabled_flag = True

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
            
        train_loss /= len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_loss /= len(val_dataset)
        
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        
        logger.info(f"Epoch [{epoch:02d}/{args.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Macro-F1: {macro_f1:.4f} | Balanced Acc: {balanced_acc:.4f}")
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), os.path.join(run_dir, 'best.pth'))
            logger.info(f"--> Saved new best model with Macro-F1: {best_f1:.4f}")
            
    torch.save(model.state_dict(), os.path.join(run_dir, 'last.pth'))
    logger.info("Training completed. Last model saved.")

if __name__ == "__main__":
    main()