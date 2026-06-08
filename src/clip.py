import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import open_clip
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

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

def get_run_dir(base_dir="runs/train", name=None):
    os.makedirs(base_dir, exist_ok=True)
    if name is None:
        i = 1
        while os.path.exists(os.path.join(base_dir, str(i))):
            i += 1
        run_dir = os.path.join(base_dir, str(i))
    else:
        run_dir = os.path.join(base_dir, name)
        if os.path.exists(run_dir):
            i = 1
            while os.path.exists(f"{run_dir}{i}"):
                i += 1
            run_dir = f"{run_dir}{i}"
    os.makedirs(run_dir)
    return run_dir

class BiomedCLIPLinearProbe(nn.Module):
    def __init__(self, weight_path, num_classes):
        super().__init__()
        # Instantiate standard CLIP ViT-B-16 (Matches BiomedCLIP vision architecture)
        base_model = open_clip.create_model('ViT-B-16')
        
        # Load the offline BiomedCLIP weights gracefully
        # strict=False allows us to ignore the PubMedBERT text encoder weights completely
        checkpoint = torch.load(weight_path, map_location='cpu')
        base_model.load_state_dict(checkpoint, strict=False)
        
        # Extract only the visual encoder
        self.backbone = base_model.visual
        
        # Freeze the entire visual backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # BiomedCLIP visual output projection dimension is 512
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        with torch.no_grad():
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

def main():
    parser = argparse.ArgumentParser(description="Linear Probe with BiomedCLIP")
    parser.add_argument('--data_path', type=str, default='~/autodl-tmp/data/train_few_shot/', help='Path to dataset')
    parser.add_argument('--weight_path', type=str, default='~/autodl-tmp/weights/biomedclip/open_clip_pytorch_model.bin', help='Path to weights')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (higher for linear probe)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--val_split', type=float, default=0.15, help='Validation set split ratio')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers')
    
    args = parser.parse_args()
    
    args.data_path = os.path.expanduser(args.data_path)
    args.weight_path = os.path.expanduser(args.weight_path)
    
    run_dir = get_run_dir(base_dir="runs/train", name=args.name)
    logger = setup_logger(run_dir)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info("=== Experiment Configurations ===")
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")
    logger.info(f"Run directory: {run_dir}")

    # Standard CLIP normalization stats
    clip_mean = (0.48145466, 0.4578275, 0.40821073)
    clip_std = (0.26862954, 0.26130258, 0.27577711)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_mean, std=clip_std)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=clip_mean, std=clip_std)
    ])

    logger.info("Loading dataset and applying Stratified Split...")
    full_dataset = datasets.ImageFolder(root=args.data_path, transform=None)
    num_classes = len(full_dataset.classes)
    logger.info(f"Classes found: {full_dataset.classes}")
    
    targets = full_dataset.targets
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_split, random_state=args.seed)
    train_indices, val_indices = next(sss.split(np.zeros(len(targets)), targets))
    
    train_dataset = DatasetWrapper(full_dataset, train_indices, transform=train_transform)
    val_dataset = DatasetWrapper(full_dataset, val_indices, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    logger.info(f"Train size: {len(train_dataset)} | Val size: {len(val_dataset)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = BiomedCLIPLinearProbe(weight_path=args.weight_path, num_classes=num_classes)
    model = model.to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable Parameters: {trainable_params} || All Parameters: {all_params} || Trainable %: {100 * trainable_params / all_params:.4f}%")

    criterion = nn.CrossEntropyLoss()
    # Only pass the trainable head parameters to the optimizer
    optimizer = optim.Adam(model.head.parameters(), lr=args.lr)

    best_f1 = 0.0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
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
        
        logger.info(f"Epoch [{epoch:02d}/{args.epochs}] | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Macro-F1: {macro_f1:.4f} | "
                    f"Balanced Acc: {balanced_acc:.4f}")
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save(model.state_dict(), os.path.join(run_dir, 'best.pth'))
            logger.info(f"--> Saved new best model with Macro-F1: {best_f1:.4f}")
            
    torch.save(model.state_dict(), os.path.join(run_dir, 'last.pth'))
    logger.info("Training completed. Last model saved.")

if __name__ == "__main__":
    main()