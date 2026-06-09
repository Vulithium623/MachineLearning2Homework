import os
import io
import re
import time
import base64
import argparse
import logging
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, balanced_accuracy_score

import timm
from peft import get_peft_model, LoraConfig
from dotenv import load_dotenv
from openai import OpenAI

# ==========================================
# 0. Logging & Directory Setup
# ==========================================

def get_run_dir(base_dir="runs/eval", name=None):
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

def setup_logger(run_dir):
    logger = logging.getLogger("RAG_Eval")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler(os.path.join(run_dir, 'run.log'))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def save_debug_markdown(messages, raw_response, run_dir, filename="prompt_debug.md"):
    """Saves the multimodal prompt and the model's raw response to a markdown file."""
    md_path = os.path.join(run_dir, filename)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# VLM Prompt Debug Log\n\n")
        f.write("This file contains the exact payload sent to the API and the model's raw response. ")
        f.write("Click 'Preview Markdown' in your IDE to see the images.\n\n")
        f.write("---\n\n")
        
        # 1. Log the Input Messages
        for msg in messages:
            f.write(f"### Role: {msg['role'].upper()}\n\n")
            if isinstance(msg['content'], list):
                for item in msg['content']:
                    if item['type'] == 'text':
                        f.write(f"{item['text']}\n\n")
                    elif item['type'] == 'image_url':
                        img_url = item['image_url']['url']
                        f.write(f"![embedded_image]({img_url})\n\n")
            else:
                f.write(f"{msg['content']}\n\n")
                
        f.write("---\n\n")
        
        # 2. Log the Raw Output from the Model
        f.write("### Model Raw Response\n\n")
        if raw_response:
            f.write(f"```text\n{raw_response}\n```\n\n")
        else:
            f.write("*(Empty response or API failed)*\n\n")

# ==========================================
# 1. Custom Model for Offline Weights
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
# 2. Image Encoding & Prompts
# ==========================================

def pil_to_base64(img: Image.Image, format="JPEG", size=(224, 224)):
    img_resized = img.resize(size, Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    img_resized.convert("RGB").save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def build_multimodal_messages(reference_pairs, query_img):
    content = [
        {
            "type": "text", 
            "text": (
                "You are an expert pathologist. I am providing you with reference images "
                "for EACH of the possible cell classes. Strictly based on the visual "
                "morphological similarity (texture, cell shape, staining color) compared "
                "to these multi-class references, classify the final query image.\n\n"
            )
        }
    ]
    
    # Iterate through class-balanced references
    for i, (ref_img, label) in enumerate(reference_pairs):
        content.append({"type": "text", "text": f"Reference Image {i+1} (Class: {label}):"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{pil_to_base64(ref_img)}"}
        })
        
    content.append({"type": "text", "text": "\nNow, carefully compare the Query Image below to all the references above and classify it:"})
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{pil_to_base64(query_img)}"}
    })
    content.append({
        "type": "text", 
        "text": "\nOutput ONLY the exact class label (e.g., Class_0) and nothing else."
    })
    
    return [{"role": "user", "content": content}]

# ==========================================
# 3. Core RAG Logic
# ==========================================

def main(args):
    run_dir = get_run_dir(base_dir="runs/eval", name=args.name)
    logger = setup_logger(run_dir)
    
    logger.info("="*50)
    logger.info(f"RAG Evaluation Started. Logs saved to {run_dir}")
    logger.info(f"Retrieval Strategy: {args.k_per_class} images PER CLASS")
    logger.info("="*50)
    
    load_dotenv()
    api_key = os.getenv("VLM_API_KEY")
    base_url = os.getenv("VLM_BASE_URL")
    
    if not api_key or not base_url:
        logger.error("VLM_API_KEY or VLM_BASE_URL not found in .env file.")
        return

    client = OpenAI(api_key=api_key, base_url=base_url)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device} | Model: {args.model}")

    # Offline Weights Loading
    logger.info("Loading Phase 2 DINOv2-LoRA model offline...")
    base_model = CustomViTModel(
        backbone_name='vit_small_patch14_dinov2.lvd142m', 
        weight_path=args.base_weight_path,
        num_classes=5
    )
    peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=["qkv"], lora_dropout=0.1)
    model = get_peft_model(base_model, peft_config)
    
    state_dict = torch.load(args.lora_weight_path, map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.base_model.model.head = nn.Identity()
    model.to(device)
    model.eval()

    # Dataset Split (Exact match with train.py)
    dino_transform = transforms.Compose([
        transforms.Resize((518, 518), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    full_dataset = datasets.ImageFolder(args.data_path)
    class_names = full_dataset.classes
    labels = [label for _, label in full_dataset.samples]
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(sss.split(full_dataset.samples, labels))
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    
    # Build Database
    logger.info("Extracting feature vectors for Knowledge Base...")
    db_features, db_labels, db_images = [], [], []
    with torch.no_grad():
        for i in range(len(train_subset)):
            img, label_idx = train_subset[i]
            tensor_img = dino_transform(img).unsqueeze(0).to(device)
            db_features.append(model(tensor_img).cpu())
            db_labels.append(class_names[label_idx])
            db_images.append(img)
    db_features = torch.cat(db_features, dim=0) 
    
    logger.info("Starting Multi-modal Evaluation...")
    true_labels, pred_labels = [], []
    
    for i in range(len(val_subset)):
        val_img, val_label_idx = val_subset[i]
        true_class = class_names[val_label_idx]
        
        # Retrieval
        with torch.no_grad():
            val_tensor = dino_transform(val_img).unsqueeze(0).to(device)
            query_feat = model(val_tensor).cpu()
            
        sims = F.cosine_similarity(query_feat, db_features, dim=1) # Shape: [num_train_samples]
        
        # CLASS-BALANCED RETRIEVAL (Stratified Top-K)
        reference_pairs = []
        for cls_name in class_names:
            # Find indices for the current class
            cls_indices = [idx for idx, label in enumerate(db_labels) if label == cls_name]
            if not cls_indices:
                continue
                
            cls_indices_tensor = torch.tensor(cls_indices, dtype=torch.long)
            cls_sims = sims[cls_indices_tensor]
            
            # Get top K for this specific class
            k = min(args.k_per_class, len(cls_indices))
            topk_local_idx = torch.topk(cls_sims, k=k).indices.tolist()
            
            # Map back to global db_images and append
            for local_idx in topk_local_idx:
                global_idx = cls_indices[local_idx]
                reference_pairs.append((db_images[global_idx], db_labels[global_idx]))
        
        # Build Prompt
        messages = build_multimodal_messages(reference_pairs, val_img)
            
        # API Call
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_tokens=1024,  # Give it enough room to think
                temperature=0.0,
            )
            
            # Extract Token Usage!
            usage = response.usage
            if usage:
                logger.info(f"Tokens: Prompt={usage.prompt_tokens} (Cached={getattr(usage.prompt_tokens_details, 'cached_tokens', 0)}) | Completion={usage.completion_tokens} | Total={usage.total_tokens}")

            content = response.choices[0].message.content
            if content is not None:
                raw_response = content.strip()
            else:
                raw_response = ""
                finish_reason = getattr(response.choices[0], 'finish_reason', 'unknown')
                logger.warning(f"API returned None! Finish reason: {finish_reason}")
                
        except Exception as e:
            logger.error(f"API Request Failed: {e}")
            raw_response = ""
            time.sleep(3)
            
        # Save Markdown for the very first API request (INCLUDING raw_response)
        if i == 0:
            save_debug_markdown(messages, raw_response, run_dir)
            logger.info(f"Visual Debug Log saved to {os.path.join(run_dir, 'prompt_debug.md')}")
            
        # Regex parsing
        pred_class = "unknown"
        if raw_response:
            # 仅仅寻找回答中的第一个 0 到 4 之间的数字
            match = re.search(r'[0-4]', raw_response)
            if match:
                digit = match.group(0)
                # 将提取到的数字 (例如 "4") 与实际的 dataset class_names 自动对齐
                for c_name in class_names:
                    if c_name.endswith(digit):
                        pred_class = c_name
                        break
                
                # 如果因为某种原因没在 dataset 里找到，给一个 fallback
                if pred_class == "unknown":
                    pred_class = f"class_{digit}"
            else:
                logger.warning(f"Failed to parse label. Raw output: {raw_response}")

        true_labels.append(true_class)
        pred_labels.append(pred_class)
        
        logger.info(f"Sample {i+1:02d}/{len(val_subset)} | True: {true_class} | Pred: {pred_class}")
        
        if args.dry_run:
            logger.info("--- DRY RUN MODE ENABLED ---")
            logger.info(f"Successfully executed 1 API call to '{args.model}'. Check '{run_dir}/prompt_debug.md'.")
            logger.info("Exiting early.")
            break
            
        time.sleep(1) # Protect API rate limit

    if not args.dry_run:
        macro_f1 = f1_score(true_labels, pred_labels, labels=class_names, average='macro', zero_division=0)
        b_acc = balanced_accuracy_score(true_labels, pred_labels)
        logger.info("=" * 40)
        logger.info(f"FINAL RAG RESULTS ({args.model}):")
        logger.info(f"Retrieval Setup:  {args.k_per_class} per class ({args.k_per_class * len(class_names)} total)")
        logger.info(f"Macro-F1:         {macro_f1:.4f}")
        logger.info(f"Balanced Acc:     {b_acc:.4f}")
        logger.info("=" * 40)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 3: Multi-modal Visual RAG Evaluation')
    parser.add_argument('--name', type=str, default=None, help='Experiment log folder name')
    parser.add_argument('--data_path', type=str, default=os.path.expanduser('~/autodl-tmp/data/train_few_shot/'), help='Path to dataset')
    parser.add_argument('--base_weight_path', type=str, default=os.path.expanduser('~/autodl-tmp/weights/dinov2_small.bin'), help='Path to offline DINOv2 weights')
    parser.add_argument('--lora_weight_path', type=str, default='./weights/best.pth', help='Path to Phase 2 best LoRA weights (e.g., ./runs/train/xxx/best.pth)')
    parser.add_argument('--k_per_class', type=int, default=3, help='Number of reference images to retrieve PER CLASS (default 3, total 15)')
    parser.add_argument('--model', type=str, default='gemini-3.1-pro-preview', help='Target VLM model name')
    parser.add_argument('--dry_run', action='store_true', help='Execute strictly 1 sample to verify API and logs')
    
    args = parser.parse_args()
    main(args)