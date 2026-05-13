"""
This script trains and evaluates a gated multi-modal classifier for fake news 
detection. It integrates ResNet50 image features, transformer-based text 
features, and TransE knowledge graph embeddings. The model uses attention 
pooling and a gating mechanism to fuse these modalities effectively.
"""

import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, roc_auc_score

# Establish paths relative to the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
emb_dir = os.path.join(project_root, "data", "embeddings")
checkpoint_dir = os.path.join(project_root, "outputs", "checkpoints")

os.makedirs(checkpoint_dir, exist_ok=True)

class GatedDataset(Dataset):
    """Dataset class for managing aligned text, image, and KG entity embeddings."""
    def __init__(self, text_features, img_features, kg_embedding_lists, labels):
        self.text_features = torch.tensor(text_features, dtype=torch.float32)
        self.img_features = torch.tensor(img_features, dtype=torch.float32)
        self.kg_embedding_lists = kg_embedding_lists   
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.text_features[idx], self.img_features[idx], self.kg_embedding_lists[idx], self.labels[idx]

def collate_gated_samples(batch):
    """Pads entity embedding lists to the maximum length within a batch."""
    texts, imgs, kg_lists, labels = zip(*batch)
    texts = torch.stack(texts)
    imgs = torch.stack(imgs)
    labels = torch.stack(labels)
    batch_size = len(batch)

    max_ents = max([len(q) for q in kg_lists]) if kg_lists else 1
    if max_ents == 0: max_ents = 1 

    kg_tensor = torch.zeros((batch_size, max_ents, 100), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_ents), dtype=torch.int32)

    for i, q_list in enumerate(kg_lists):
        length = len(q_list)
        if length > 0:
            kg_tensor[i, :length, :] = torch.tensor(np.array(q_list), dtype=torch.float32)
            mask[i, :length] = 1 

    return texts, imgs, kg_tensor, mask, labels

class GatedAttentionClassifier(nn.Module):
    """Fusion model combining text, image, and KG data with attention and gating."""
    def __init__(self, text_dim=768, img_dim=2048, kg_dim=100, proj_dim=256, dropout=0.5):
        super().__init__()

        self.text_proj = nn.Sequential(nn.Linear(text_dim, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU())
        self.img_proj = nn.Sequential(nn.Linear(img_dim, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU())
        self.kg_proj = nn.Sequential(nn.Linear(kg_dim, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU())

        self.attn_pool = nn.Linear(kg_dim, 1)
        self.gate = nn.Sequential(nn.Linear(kg_dim, 1), nn.Sigmoid())

        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, text, img, kg_tensor, mask):
        # Pool entities using an attention mechanism
        attn_scores = self.attn_pool(kg_tensor).squeeze(-1)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(attn_scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        kg_pooled = (weights.unsqueeze(-1) * kg_tensor).sum(dim=1)

        # Apply learnable gate to control KG feature influence
        kg_gate = self.gate(kg_pooled)
        kg_gated_features = kg_gate * kg_pooled

        # Project and concatenate all modalities
        fused_features = torch.cat([
            self.text_proj(text), 
            self.img_proj(img), 
            self.kg_proj(kg_gated_features)
        ], dim=1)
        
        return self.classifier(fused_features).squeeze(1)

def load_data_split(split, text_model, img_model, label_name):
    """Loads feature files and generates labels for a specific split/class."""
    text_path = os.path.join(emb_dir, text_model, f"{split}_{label_name}_text.npy")
    img_path = os.path.join(emb_dir, img_model, f"{split}_{label_name}_image.npy")
    kg_path = os.path.join(emb_dir, "transe", f"{split}_{label_name}_qid_transe_embeddings.pkl")

    t_emb = np.load(text_path).astype(np.float32)
    i_emb = np.load(img_path).astype(np.float32)
    kg_embs = pd.read_pickle(kg_path)
    labels = np.full(len(t_emb), 1.0 if label_name == 'fake' else 0.0, dtype=np.float32)

    return t_emb, i_emb, kg_embs, labels

def create_split_dataset(split, text_model, img_model):
    """Combines fake and real data split into a single dataset."""
    t_f, i_f, q_f, l_f = load_data_split(split, text_model, img_model, 'fake')
    t_r, i_r, q_r, l_r = load_data_split(split, text_model, img_model, 'real')

    return GatedDataset(
        np.concatenate([t_f, t_r], axis=0),
        np.concatenate([i_f, i_r], axis=0),
        q_f + q_r,
        np.concatenate([l_f, l_r], axis=0)
    )

def evaluate_model(model, loader, criterion, device):
    """Executes evaluation and returns accuracy, F1, and AUC metrics."""
    model.eval()
    total_loss = 0
    probabilities, predictions, targets = [], [], []

    with torch.no_grad():
        for t, i, kg, m, l in loader:
            t, i, kg, m, l = t.to(device), i.to(device), kg.to(device), m.to(device), l.to(device)
            logits = model(t, i, kg, m)
            total_loss += criterion(logits, l).item()
            probs = torch.sigmoid(logits).cpu().numpy()
            probabilities.extend(probs)
            predictions.extend((probs > 0.5).astype(int))
            targets.extend(l.cpu().numpy())

    targets = np.array(targets)
    acc = np.mean(np.array(predictions) == targets)
    f1 = f1_score(targets, predictions, average='macro')
    auc = roc_auc_score(targets, probabilities)
    return total_loss / len(loader), acc, f1, auc

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    epochs = 20
    batch_size = 64
    learning_rate = 1e-4
    patience = 5
    image_encoder = "resnet50"
    
    text_encoders = ["muril", "xlmr", "mbert"]
    dropout_rates = [0.3, 0.4, 0.5, 0.6, 0.7]

    for text_enc in text_encoders:
        train_loader = DataLoader(create_split_dataset('train', text_enc, image_encoder), 
                                  batch_size=batch_size, shuffle=True, collate_fn=collate_gated_samples)
        val_loader = DataLoader(create_split_dataset('val', text_enc, image_encoder), 
                                batch_size=batch_size, shuffle=False, collate_fn=collate_gated_samples)
        test_loader = DataLoader(create_split_dataset('test', text_enc, image_encoder), 
                                 batch_size=batch_size, shuffle=False, collate_fn=collate_gated_samples)

    for drop in dropout_rates:
        model = GatedAttentionClassifier(img_dim=2048, dropout=drop).to(device)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float('inf')
        early_stop_count = 0
        ckpt_path = os.path.join(checkpoint_dir, f"gated_resnet_{text_enc}_{drop}.pt")

        for epoch in range(1, epochs + 1):
            model.train()
            for t, i, kg, m, l in train_loader:
                t, i, kg, m, l = t.to(device), i.to(device), kg.to(device), m.to(device), l.to(device)
                optimizer.zero_grad()
                loss_fn(model(t, i, kg, m), l).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            v_loss, _, _, _ = evaluate_model(model, val_loader, loss_fn, device)
            scheduler.step()

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                early_stop_count = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                early_stop_count += 1
                if early_stop_count >= patience: break

        # Final evaluation on the test set using the best model
        model.load_state_dict(torch.load(ckpt_path))
        _, acc, f1, auc = evaluate_model(model, test_loader, loss_fn, device)
        print(f"[{text_enc.upper()} + RESNET50 + KG | Dropout: {drop}] Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        if os.path.exists(ckpt_path): os.remove(ckpt_path)

if __name__ == "__main__":
    run_training()
