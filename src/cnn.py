import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, balanced_accuracy_score
import numpy as np
import timm

def get_next_run_dir(base_dir="runs/train", name="experiment"):
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

def setup_logger(log_dir):
    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

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
    parser = argparse.ArgumentParser(description="ConvNeXt-V2 Linear Probe with Max Augmentation")
    parser.add_argument('--data_dir', type=str, default=os.path.expanduser('~/autodl-tmp/data/train_few_shot/'))
    parser.add_argument('--weight_path', type=str, default=os.path.expanduser('~/autodl-tmp/weights/convnextv2_nano.bin'))
    parser.add_argument('--name', type=str, default='convnext_max_aug')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val_split', type=float, default=0.15)
    
    # Data Augmentation Arguments (Defaults are set to MAX)
    parser.add_argument('--no_hflip', action='store_true', help="Disable Horizontal Flip")
    parser.add_argument('--no_vflip', action='store_true', help="Disable Vertical Flip")
    parser.add_argument('--rotate_deg', type=float, default=90.0, help="Rotation degrees (default: 90)")
    parser.add_argument('--color_jitter', type=float, default=0.2, help="Color Jitter factor (default: 0.2, 0 to disable)")
    parser.add_argument('--mixup_alpha', type=float, default=0.8, help="MixUp alpha (default: 0.8, 0 to disable)")
    
    args = parser.parse_args()
    
    run_dir = get_next_run_dir(base_dir="runs/train", name=args.name)
    logger = setup_logger(run_dir)
    
    logger.info("="*50)
    logger.info(f"Experiment Started. Results saved to {run_dir}")
    logger.info("="*50)
    logger.info("HYPERPARAMETERS:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    
    # Explicitly log augmentation settings
    use_hflip = not args.no_hflip
    use_vflip = not args.no_vflip
    logger.info("="*50)
    logger.info("DATA AUGMENTATION SETTINGS:")
    logger.info(f"  Horizontal Flip : {'Enabled' if use_hflip else 'Disabled'}")
    logger.info(f"  Vertical Flip   : {'Enabled' if use_vflip else 'Disabled'}")
    logger.info(f"  Rotation        : Random [-{args.rotate_deg}, +{args.rotate_deg}] degrees")
    logger.info(f"  Color Jitter    : {'Enabled (Factor: ' + str(args.color_jitter) + ')' if args.color_jitter > 0 else 'Disabled'}")
    logger.info(f"  MixUp           : {'Enabled (Alpha: ' + str(args.mixup_alpha) + ')' if args.mixup_alpha > 0 else 'Disabled'}")
    logger.info("="*50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Transforms
    train_transform_list = [transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)]
    if use_hflip:
        train_transform_list.append(transforms.RandomHorizontalFlip())
    if use_vflip:
        train_transform_list.append(transforms.RandomVerticalFlip())
    if args.rotate_deg > 0:
        train_transform_list.append(transforms.RandomRotation(args.rotate_deg))
    if args.color_jitter > 0:
        train_transform_list.append(transforms.ColorJitter(brightness=args.color_jitter, 
                                                           contrast=args.color_jitter, 
                                                           saturation=args.color_jitter))
    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    train_transform = transforms.Compose(train_transform_list)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # Dataset loading
    base_dataset = ImageFolder(root=args.data_dir)
    labels = [y for _, y in base_dataset.samples]
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_split, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
    
    train_subset = Subset(base_dataset, train_idx)
    val_subset = Subset(base_dataset, val_idx)
    
    train_dataset = DatasetWrapper(train_subset, transform=train_transform)
    val_dataset = DatasetWrapper(val_subset, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    logger.info(f"Dataset splits: {len(train_dataset)} Train | {len(val_dataset)} Val")

    # Load ConvNeXt-V2 Model
    logger.info("Loading pre-trained ConvNeXt-V2...")
    model = timm.create_model('convnextv2_nano', pretrained=False, num_classes=1000)
    
    if os.path.exists(args.weight_path):
        state_dict = torch.load(args.weight_path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Loaded weights from {args.weight_path}")
    else:
        logger.error(f"Weight file not found at {args.weight_path}!")
        return

    # Reset classification head for 5 classes
    model.reset_classifier(num_classes=5)
    
    # Freeze backbone, train only the head
    for name, param in model.named_parameters():
        if 'head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params} / {total_params} ({100 * trainable_params / total_params:.4f}%)")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)

    best_macro_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Apply MixUp if enabled
            if args.mixup_alpha > 0:
                images, targets_a, targets_b, lam = mixup_data(images, labels, args.mixup_alpha)
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_loss = val_loss / len(val_loader.dataset)
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        bal_acc = balanced_accuracy_score(all_labels, all_preds)
        
        logger.info(f"Epoch [{epoch:02d}/{args.epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Macro-F1: {macro_f1:.4f} | Balanced Acc: {bal_acc:.4f}")
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), os.path.join(run_dir, 'best.pth'))
            logger.info(f"--> Saved new best model with Macro-F1: {best_macro_f1:.4f}")

    torch.save(model.state_dict(), os.path.join(run_dir, 'last.pth'))
    logger.info("Training completed. Last model saved.")

if __name__ == '__main__':
    main()