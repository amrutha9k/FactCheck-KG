"""
This script performs training and evaluation for a multi-modal fake news classifier
using CLIP visual embeddings paired with multiple text encoders. It iterates through
various dropout rates and text models to establish baseline performance metrics.
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

# Directory setup relative to project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../../"))
data_dir = os.path.join(project_root, "data")
emb_dir = os.path.join(data_dir, "embeddings")
output_dir = os.path.join(project_root, "outputs")
checkpoint_dir = os.path.join(output_dir, "checkpoints")

os.makedirs(checkpoint_dir, exist_ok=True)

class NewsDataset(Dataset):
    """Encapsulates pre-computed text and image embeddings for the classifier."""
    def __init__(self, text_features, image_features, target_labels):
        self.text_features = torch.tensor(text_features, dtype=torch.float32)
        self.image_features = torch.tensor(image_features, dtype=torch.float32)
        self.labels = torch.tensor(target_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.text_features[idx], self.image_features[idx], self.labels[idx]

class MultiModalClassifier(nn.Module):
    """Linear architecture to fuse and classify text and image representations."""
    def __init__(self, input_size, dropout_rate):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 1024),
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

    def forward(self, text, image):
        # Feature fusion via concatenation
        combined_features = torch.cat([text, image], dim=1)
        return self.classifier(combined_features).squeeze(1)

def prepare_split_data(split, text_model, image_model):
    """Loads and combines embeddings for both fake and real classes for a given split."""
    all_text, all_image, all_labels = [], [], []
    
    for label_type, label_val in [('fake', 1.0), ('real', 0.0)]:
        t_path = os.path.join(emb_dir, text_model, f"{split}_{label_type}_text.npy")
        i_path = os.path.join(emb_dir, image_model, f"{split}_{label_type}_image.npy")
        
        if os.path.exists(t_path) and os.path.exists(i_path):
            text_data = np.load(t_path).astype(np.float32)
            image_data = np.load(i_path).astype(np.float32)
            all_text.append(text_data)
            all_image.append(image_data)
            all_labels.append(np.full(len(text_data), label_val, dtype=np.float32))

    return (np.concatenate(all_text, axis=0), 
            np.concatenate(all_image, axis=0), 
            np.concatenate(all_labels, axis=0))

def run_evaluation(model, loader, criterion, device):
    """Computes standard classification metrics over the provided data loader."""
    model.eval()
    cumulative_loss = 0
    predictions, probabilities, true_labels = [], [], []

    with torch.no_grad():
        for text, img, label in loader:
            text, img, label = text.to(device), img.to(device), label.to(device)
            logits = model(text, img)
            cumulative_loss += criterion(logits, label).item()

            probs = torch.sigmoid(logits).cpu().numpy()
            probabilities.extend(probs)
            predictions.extend((probs > 0.5).astype(int))
            true_labels.extend(label.long().cpu().numpy())

    true_labels = np.array(true_labels)
    accuracy = np.mean(np.array(predictions) == true_labels)
    macro_f1 = f1_score(true_labels, predictions, average='macro')
    auc_score = roc_auc_score(true_labels, probabilities)

    return cumulative_loss / len(loader), accuracy, macro_f1, auc_score

def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Experiment parameters
    EPOCHS = 20
    BATCH_SIZE = 64
    LEARNING_RATE = 2e-4
    EARLY_STOPPING_PATIENCE = 5
    IMAGE_MODEL = "clip"
    INPUT_DIM = 768 + 768 # Text CLS(768) + CLIP ViT(768)
    
    TEXT_MODELS = ["muril", "xlmr", "mbert"]
    DROPOUT_RATES = [0.3, 0.4, 0.5, 0.6, 0.7]

    for txt_encoder in TEXT_MODELS:
        # Load pre-processed datasets
        train_data = NewsDataset(*prepare_split_data('train', txt_encoder, IMAGE_MODEL))
        val_data = NewsDataset(*prepare_split_data('val', txt_encoder, IMAGE_MODEL))
        test_data = NewsDataset(*prepare_split_data('test', txt_encoder, IMAGE_MODEL))

        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

        for drop in DROPOUT_RATES:
            model = MultiModalClassifier(INPUT_DIM, drop).to(device)
            loss_fn = nn.BCEWithLogitsLoss()
            optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

            min_val_loss = float('inf')
            stagnation_count = 0
            temp_ckpt = os.path.join(checkpoint_dir, f"baseline_clip_{txt_encoder}_{drop}.pt")

            for epoch in range(1, EPOCHS + 1):
                model.train()
                for text, img, label in train_loader:
                    text, img, label = text.to(device), img.to(device), label.to(device)
                    optimizer.zero_grad()
                    loss_fn(model(text, img), label).backward()
                    optimizer.step()
                
                v_loss, _, _, _ = run_evaluation(model, val_loader, loss_fn, device)
                scheduler.step()

                # Early stopping based on validation loss
                if v_loss < min_val_loss:
                    min_val_loss = v_loss
                    stagnation_count = 0
                    torch.save(model.state_dict(), temp_ckpt)
                else:
                    stagnation_count += 1
                    if stagnation_count >= EARLY_STOPPING_PATIENCE:
                        break

            # Evaluate the best version of the model on the test split
            model.load_state_dict(torch.load(temp_ckpt))
            _, acc, f1, auc = run_evaluation(model, test_loader, loss_fn, device)

            print(f"[{txt_encoder.upper()} + {IMAGE_MODEL.upper()} | Drop: {drop}] "
                  f"Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

            if os.path.exists(temp_ckpt):
                os.remove(temp_ckpt)

if __name__ == "__main__":
    run_experiment()
