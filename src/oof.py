import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import timm
from peft import LoraConfig, get_peft_model
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# ==========================================
# 1. 搬运自 train.py 的模型结构和 Dataset
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

# ==========================================
# 2. OOF 主函数
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Calculate Out-Of-Fold (OOF) Metrics")
    
    # 你需要指定上一次训练结束后的 run 目录，例如 runs/train_pseudo/1
    parser.add_argument('--run_dir', type=str, required=True, help='Path to the experiment run directory containing fold_x folders')
    
    # 路径参数 (默认值对齐 train.py)
    parser.add_argument('--data_path', type=str, default='~/autodl-tmp/data/train_few_shot/', help='Path to true dataset')
    parser.add_argument('--weight_path', type=str, default='~/autodl-tmp/weights/dinov2_small.bin', help='Path to offline weights')
    
    # 超参数 (必须与 train.py 完全对齐，否则无法加载权重或复现分折)
    parser.add_argument('--k_folds', type=int, default=7, help='Number of folds for CV (must match train.py)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (must match train.py)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    
    args = parser.parse_args()
    args.data_path = os.path.expanduser(args.data_path)
    args.weight_path = os.path.expanduser(args.weight_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载完整有标签数据集
    print(f"Loading full dataset from {args.data_path}")
    full_dataset = datasets.ImageFolder(root=args.data_path, transform=None)
    num_classes = len(full_dataset.classes)
    targets = np.array(full_dataset.targets)
    image_paths = [s[0] for s in full_dataset.samples] # 获取图片绝对路径
    
    # 对齐验证集 Transform
    val_transform = transforms.Compose([
        transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # 准备存储 OOF 结果的数组
    # 我们要记录这 250 张图“未见过”该图的模型对它的预测
    oof_preds = np.zeros(len(targets), dtype=int)
    oof_probs = np.zeros((len(targets), num_classes), dtype=float)
    
    # 2. 完全对齐的分折逻辑
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    
    print("="*50)
    print("Starting OOF Evaluation...")
    print("="*50)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets), 1):
        print(f"Evaluating Fold {fold}/{args.k_folds} ...", end=" ")
        
        # 找到该折的 Best 模型权重
        weight_file = os.path.join(args.run_dir, f"fold_{fold}", "best.pth")
        if not os.path.exists(weight_file):
            raise FileNotFoundError(f"Missing weight file: {weight_file}")

        # 重新初始化模型
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
        
        # 加载这折的权重
        model.load_state_dict(torch.load(weight_file, map_location=device))
        model.to(device)
        model.eval()

        # 仅对 val_idx 所在的子集进行预测
        val_dataset = DatasetWrapper(full_dataset, val_idx, transform=val_transform)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        fold_preds = []
        fold_probs = []
        
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs) # logits
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                fold_preds.extend(preds.cpu().numpy())
                fold_probs.extend(probs.cpu().numpy())
                
        # 填入全局 OOF 数组
        oof_preds[val_idx] = fold_preds
        oof_probs[val_idx] = fold_probs
        
        print("Done.")

    # ==========================================
    # 3. 计算全局 OOF 指标 (挤干水分的真实指标)
    # ==========================================
    print("\n" + "="*50)
    print("FINAL OOF METRICS (Calculated on combined 100% data)")
    print("="*50)
    
    oof_f1 = f1_score(targets, oof_preds, average='macro')
    oof_bacc = balanced_accuracy_score(targets, oof_preds)
    
    print(f"OOF Macro F1         : {oof_f1:.4f}")
    print(f"OOF Balanced Acc     : {oof_bacc:.4f}\n")
    
    print("OOF Classification Report:")
    print(classification_report(targets, oof_preds, target_names=full_dataset.classes, digits=4))
    
    # ==========================================
    # 4. 保存 OOF 预测结果 (方便你溯源哪些图始终判断错)
    # ==========================================
    df_oof = pd.DataFrame({
        'Image_Path': image_paths,
        'True_Label': targets,
        'OOF_Pred': oof_preds,
    })
    # 加入各类的概率列
    for c in range(num_classes):
        df_oof[f'Prob_Class_{c}'] = oof_probs[:, c]
        
    df_oof['Is_Correct'] = df_oof['True_Label'] == df_oof['OOF_Pred']
    
    out_csv = os.path.join(args.run_dir, "oof_predictions.csv")
    df_oof.to_csv(out_csv, index=False)
    print(f"\nSaved detailed OOF predictions to {out_csv}")
    print(f"You can check this CSV to find the hardest samples (Is_Correct == False).")

if __name__ == "__main__":
    main()