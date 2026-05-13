"""
This script trains and evaluates a baseline multi-modal classifier using 
ResNet50 visual embeddings and multiple text encoders. It tests different 
dropout configurations to identify the best model for fake news detection.
"""

import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# Directory setup relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
emb_dir = os.path.join(project_root, "data", "embeddings")
checkpoint_dir = os.path.join(project_root, "outputs", "checkpoints")

os.makedirs(checkpoint_dir, exist_ok=True)

class MultimodalDataset(Dataset):
    """Dataset class to store and provide aligned text and image embeddings."""
    def __init__(self, text_data, img_data, labels):
        self.text_data = torch.tensor(text_data, dtype=torch.float32)
        self.img_data = torch.tensor(img_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.text_data[idx], self.img_data[idx], self.labels[idx]

class FusionClassifier(nn.Module):
    """Neural network that fuses text and image features for binary classification."""
    def __init__(self, input_dim, dropout_rate):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )

    def forward(self, text, img):
        # Concatenate embeddings into a single feature vector
        combined = torch.cat([text, img], dim=1)
        return self.network(combined).squeeze(1)

def get_data(split, text_model, img_model):
    """Loads and combines embeddings and labels for a specific data split."""
    all_text, all_img, all_labels = [], [], []
    
    for label_name, val in [('fake', 1.0), ('real', 0.0)]:
        t_path = os.path.join(emb_dir, text_model, f"{split}_{label_name}_text.npy")
        i_path = os.path.join(emb_dir, img_model, f"{split}_{label_name}_image.npy")
        
        if os.path.exists(t_path) and os.path.exists(i_path):
            all_text.append(np.load(t_path))
            all_img.append(np.load(i_path))
            all_labels.append(np.full(len(all_text[-1]), val, dtype=np.float32))

    return (np.concatenate(all_text, axis=0), 
            np.concatenate(all_img, axis=0), 
            np.concatenate(all_labels, axis=0))

def evaluate_model(model, loader, criterion, device):
    """Calculates evaluation metrics over the provided data loader."""
    model.eval()
    loss_sum = 0
    probs, preds, targets = [], [], []

    with torch.no_grad():
        for t, i, l in loader:
            t, i, l = t.to(device), i.to(device), l.to(device)
            logits = model(t, i)
            loss_sum += criterion(logits, l).item()

            p = torch.sigmoid(logits).cpu().numpy()
            probs.extend(p)
            preds.extend((p > 0.5).astype(int))
            targets.extend(l.long().cpu().numpy())

    targets = np.array(targets)
    accuracy = np.mean(np.array(preds) == targets)
    f1 = f1_score(targets, preds, average='macro')
    auc = roc_auc_score(targets, probs)

    return loss_sum / len(loader), accuracy, f1, auc

def run_training():
    """Iterates through model and dropout combinations to find the best configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training configuration
    epochs = 20
    batch_size = 128
    lr = 2e-4
    patience = 5
    img_encoder = "resnet50"
    input_dim = 768 + 2048 # Text features (768) + ResNet features (2048)
    
    text_encoders = ["muril", "xlmr", "mbert"]
    dropout_list = [0.3, 0.4, 0.5, 0.6, 0.7]

    for text_encoder in text_encoders:
        # Construct datasets once per text encoder
        train_set = MultimodalDataset(*get_data('train', text_encoder, img_encoder))
        val_set = MultimodalDataset(*get_data('val', text_encoder, img_encoder))
        test_set = MultimodalDataset(*get_data('test', text_encoder, img_encoder))

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        for drop in dropout_list:
            model = FusionClassifier(input_dim, drop).to(device)
            loss_func = nn.BCEWithLogitsLoss()
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

            best_val_loss = float('inf')
            early_stop_count = 0
            temp_path = os.path.join(checkpoint_dir, f"resnet_{text_encoder}_{drop}.pt")

            for epoch in range(1, epochs + 1):
                model.train()
                for t, i, l in train_loader:
                    t, i, l = t.to(device), i.to(device), l.to(device)
                    optimizer.zero_grad()
                    loss_func(model(t, i), l).backward()
                    optimizer.step()
                
                val_loss, _, _, _ = evaluate_model(model, val_loader, loss_func, device)
                scheduler.step()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_count = 0
                    torch.save(model.state_dict(), temp_path)
                else:
                    early_stop_count += 1
                    if early_stop_count >= patience:
                        break

            # Run final test with the best model found during validation
            model.load_state_dict(torch.load(temp_path))
            _, acc, f1, auc = evaluate_model(model, test_loader, loss_func, device)

            print(f"[{text_encoder.upper()} + RESNET50 | Dropout: {drop}] "
                  f"Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

            # Remove checkpoint after test evaluation
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    run_training()
