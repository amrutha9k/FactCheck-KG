"""
This script trains and evaluates a gated multi-modal classifier for fake news detection.
It fuses representations from text encoders, CLIP visual features, and TransE 
knowledge graph embeddings using an attention-based pooling and gating mechanism.
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

# Directory setup relative to project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
data_dir = os.path.join(project_root, "data")
emb_dir = os.path.join(data_dir, "embeddings")
checkpoint_dir = os.path.join(project_root, "outputs", "checkpoints")

os.makedirs(checkpoint_dir, exist_ok=True)

class GatedMultimodalDataset(Dataset):
    """Dataset for handling aligned text, image, and variable-length KG embeddings."""
    def __init__(self, text_emb, img_emb, kg_emb_lists, labels):
        self.text_emb = torch.tensor(text_emb, dtype=torch.float32)
        self.img_emb = torch.tensor(img_emb, dtype=torch.float32)
        self.kg_emb_lists = kg_emb_lists   
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.text_emb[idx], self.img_emb[idx], self.kg_emb_lists[idx], self.labels[idx]

def collate_batch(batch):
    """Custom collation to handle padding of variable-length knowledge graph entity lists."""
    texts, imgs, kg_lists, labels = zip(*batch)

    texts = torch.stack(texts)
    imgs = torch.stack(imgs)
    labels = torch.stack(labels)
    batch_size = len(batch)

    max_entities = max([len(q) for q in kg_lists]) if kg_lists else 1
    if max_entities == 0: max_entities = 1 

    kg_tensor = torch.zeros((batch_size, max_entities, 100), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_entities), dtype=torch.int32)

    for i, q_list in enumerate(kg_lists):
        length = len(q_list)
        if length > 0:
            kg_tensor[i, :length, :] = torch.tensor(np.array(q_list), dtype=torch.float32)
            mask[i, :length] = 1 

    return texts, imgs, kg_tensor, mask, labels

class GatedAttentionClassifier(nn.Module):
    """Classifier utilizing attention pooling and gating for multi-modal fusion."""
    def __init__(self, text_dim=768, img_dim=768, kg_dim=100, proj_dim=256, dropout=0.5):
        super().__init__()

        self.text_proj = nn.Sequential(nn.Linear(text_dim, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU())
        self.img_proj = nn.Sequential(nn.Linear(img_dim, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU())
        self.kg_proj = nn.Sequential(nn.Linear(kg_dim, proj_dim), nn.LayerNorm(proj_dim), nn.ReLU())

        self.attn_pool = nn.Linear(kg_dim, 1)
        self.gate = nn.Sequential(nn.Linear(kg_dim, 1), nn.Sigmoid())

        combined_dim = proj_dim * 3  

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, text, img, kg_tensor, mask):
        # Attention-based pooling of KG entities
        scores = self.attn_pool(kg_tensor).squeeze(-1)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        kg_pooled = (weights.unsqueeze(-1) * kg_tensor).sum(dim=1)

        # Apply gating mechanism to KG features
        g = self.gate(kg_pooled)
        kg_gated = g * kg_pooled

        # Project features to shared latent space
        text_f = self.text_proj(text)      
        img_f = self.img_proj(img)        
        kg_f = self.kg_proj(kg_gated)    

        fused = torch.cat([text_f, img_f, kg_f], dim=1)  
        return self.classifier(fused).squeeze(1)

def load_split(split, text_model, img_model, label_val):
    """Loads text, image, and TransE embeddings for a given split and label."""
    t_path = os.path.join(emb_dir, text_model, f"{split}_{label_val}_text.npy")
    i_path = os.path.join(emb_dir, img_model, f"{split}_{label_val}_image.npy")
    kg_path = os.path.join(emb_dir, "transe", f"{split}_{label_val}_qid_transe_embeddings.pkl")

    text_emb = np.load(t_path).astype(np.float32)
    img_emb = np.load(i_path).astype(np.float32)
    kg_embs = pd.read_pickle(kg_path)
    labels = np.full(len(text_emb), 1.0 if label_val == 'fake' else 0.0, dtype=np.float32)

    return text_emb, img_emb, kg_embs, labels

def build_split_dataset(split, text_model, img_model):
    """Combines fake and real data into a single dataset for the split."""
    t_f, i_f, q_f, l_f = load_split(split, text_model, img_model, 'fake')
    t_r, i_r, q_r, l_r = load_split(split, text_model, img_model, 'real')

    return GatedMultimodalDataset(
        np.concatenate([t_f, t_r], axis=0),
        np.concatenate([i_f, i_r], axis=0),
        q_f + q_r,
        np.concatenate([l_f, l_r], axis=0)
    )

def evaluate_performance(model, loader, criterion, device):
    """Computes evaluation metrics across the dataset loader."""
    model.eval()
    loss_sum = 0
    probabilities, predictions, targets = [], [], []

    with torch.no_grad():
        for t, i, kg, m, l in loader:
            t, i, kg, m, l = t.to(device), i.to(device), kg.to(device), m.to(device), l.to(device)
            logits = model(t, i, kg, m)
            loss_sum += criterion(logits, l).item()

            probs = torch.sigmoid(logits).cpu().numpy()
            probabilities.extend(probs)
            predictions.extend((probs > 0.5).astype(int))
            targets.extend(l.long().cpu().numpy())

    targets = np.array(targets)
    acc = np.mean(np.array(predictions) == targets)
    f1 = f1_score(targets, predictions, average='macro')
    auc = roc_auc_score(targets, probabilities)

    return loss_sum / len(loader), acc, f1, auc

def run_training_process():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuration constants
    EPOCHS = 20
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-5
    PATIENCE = 5
    IMAGE_MODEL = "clip"
    
    TEXT_ENCODERS = ["muril", "xlmr", "mbert"]
    DROPOUTS = [0.3, 0.4, 0.5, 0.6, 0.7]

    for txt_encoder in TEXT_ENCODERS:
        train_loader = DataLoader(build_split_dataset('train', txt_encoder, IMAGE_MODEL), 
                                  batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
        val_loader = DataLoader(build_split_dataset('val', txt_encoder, IMAGE_MODEL), 
                                batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
        test_loader = DataLoader(build_split_dataset('test', txt_encoder, IMAGE_MODEL), 
                                 batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

        for dr in DROPOUTS:
            model = GatedAttentionClassifier(dropout=dr).to(device)
            loss_fn = nn.BCEWithLogitsLoss()
            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

            min_val_loss = float('inf')
            early_stop_count = 0
            ckpt_path = os.path.join(checkpoint_dir, f"gated_clip_{txt_encoder}_{dr}.pt")

            for epoch in range(1, EPOCHS + 1):
                model.train()
                for t, i, kg, m, l in train_loader:
                    t, i, kg, m, l = t.to(device), i.to(device), kg.to(device), m.to(device), l.to(device)
                    optimizer.zero_grad()
                    loss_fn(model(t, i, kg, m), l).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                v_loss, _, _, _ = evaluate_performance(model, val_loader, loss_fn, device)
                scheduler.step()

                if v_loss < min_val_loss:
                    min_val_loss = v_loss
                    early_stop_count = 0
                    torch.save(model.state_dict(), ckpt_path)
                else:
                    early_stop_count += 1
                    if early_stop_count >= PATIENCE:
                        break

            # Perform final assessment on test split
            model.load_state_dict(torch.load(ckpt_path))
            _, acc, f1, auc = evaluate_performance(model, test_loader, loss_fn, device)

            print(f"[{txt_encoder.upper()} + CLIP + KG | Dropout: {dr}] "
                  f"Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)

if __name__ == "__main__":
    run_training_process()
