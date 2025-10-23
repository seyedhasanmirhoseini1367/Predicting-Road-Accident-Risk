import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os

# ADVANCED AUTOENCODER VARIANTS
print("Training Advanced Autoencoder Variants...")

# Variant 1: Denoising Autoencoder
class DenoisingVAE(nn.Module):
    def __init__(self, numerical_dim, categorical_info, embedding_dim=10):
        super(DenoisingVAE, self).__init__()
        
        self.numerical_dim = numerical_dim
        self.categorical_info = categorical_info
        
        # Embedding layers
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0
        
        for col, info in categorical_info.items():
            embedding_size = min(embedding_dim, (info['unique_count'] + 1) // 2)
            self.embeddings[col] = nn.Embedding(info['unique_count'], embedding_size)
            total_embedding_dim += embedding_size
        
        total_input_dim = numerical_dim + total_embedding_dim
        
        # Encoder with noise robustness
        self.encoder = nn.Sequential(
            nn.Linear(total_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
        )
        
        self.fc_mu = nn.Linear(64, 32)
        self.fc_logvar = nn.Linear(64, 32)
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(32, 24),
            nn.BatchNorm1d(24),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            
            nn.Linear(24, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def add_noise(self, x, noise_factor=0.1):
        if self.training:
            noise = torch.randn_like(x) * noise_factor
            return x + noise
        return x
    
    def embed_categorical(self, categorical_dict):
        embedded = []
        for col, tensor in categorical_dict.items():
            embed = self.embeddings[col](tensor)
            embedded.append(embed)
        return torch.cat(embedded, dim=1)
    
    def forward(self, numerical, categorical_dict):
        # Add noise during training for robustness
        numerical_noisy = self.add_noise(numerical)
        
        cat_embedded = self.embed_categorical(categorical_dict)
        combined = torch.cat([numerical_noisy, cat_embedded], dim=1)
        
        h = self.encoder(combined)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Reparameterize
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        
        risk_pred = self.predictor(z)
        return risk_pred.squeeze(), mu, logvar, combined, combined

# Variant 2: Beta-VAE with controlled capacity
class BetaVAE(nn.Module):
    def __init__(self, numerical_dim, categorical_info, embedding_dim=10, beta=0.001):
        super(BetaVAE, self).__init__()
        
        self.numerical_dim = numerical_dim
        self.categorical_info = categorical_info
        self.beta = beta
        
        # Embedding layers
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0
        
        for col, info in categorical_info.items():
            embedding_size = min(embedding_dim, (info['unique_count'] + 1) // 2)
            self.embeddings[col] = nn.Embedding(info['unique_count'], embedding_size)
            total_embedding_dim += embedding_size
        
        total_input_dim = numerical_dim + total_embedding_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(total_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
        )
        
        self.fc_mu = nn.Linear(64, 32)
        self.fc_logvar = nn.Linear(64, 32)
        
        # Predictor with capacity control
        self.predictor = nn.Sequential(
            nn.Linear(32, 24),
            nn.BatchNorm1d(24),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.08),
            
            nn.Linear(24, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def embed_categorical(self, categorical_dict):
        embedded = []
        for col, tensor in categorical_dict.items():
            embed = self.embeddings[col](tensor)
            embedded.append(embed)
        return torch.cat(embedded, dim=1)
    
    def forward(self, numerical, categorical_dict):
        cat_embedded = self.embed_categorical(categorical_dict)
        combined = torch.cat([numerical, cat_embedded], dim=1)
        
        h = self.encoder(combined)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Reparameterize
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        
        risk_pred = self.predictor(z)
        return risk_pred.squeeze(), mu, logvar, combined, combined

# Train multiple variants
def train_advanced_variants():
    variants = [
        ('DenoisingVAE', DenoisingVAE(numerical_dim, categorical_info)),
        ('BetaVAE', BetaVAE(numerical_dim, categorical_info, beta=0.001)),
    ]
    
    all_predictions = {}
    
    for name, model in variants:
        print(f"\n=== Training {name} ===")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
        
        best_val_loss = float('inf')
        
        for epoch in range(80):
            # Training
            model.train()
            total_loss = 0
            total_pred_loss = 0
            total_kl_loss = 0
            
            for batch_num, batch_cat, batch_y in train_loader:
                optimizer.zero_grad()
                
                categorical_dict = {
                    'road_type': batch_cat[:, 0],
                    'lighting': batch_cat[:, 1], 
                    'weather': batch_cat[:, 2],
                    'time_of_day': batch_cat[:, 3]
                }
                
                risk_pred, mu, logvar, _, _ = model(batch_num, categorical_dict)
                
                pred_loss = F.mse_loss(risk_pred, batch_y)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                
                if name == 'BetaVAE':
                    loss = pred_loss + model.beta * kl_loss
                else:
                    loss = 0.995 * pred_loss + 0.005 * kl_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                
                total_loss += loss.item()
                total_pred_loss += pred_loss.item()
                total_kl_loss += kl_loss.item()
            
            scheduler.step()
            
            # Validation
            model.eval()
            val_loss = 0
            val_pred_loss = 0
            
            with torch.no_grad():
                for batch_num, batch_cat, batch_y in val_loader:
                    categorical_dict = {
                        'road_type': batch_cat[:, 0],
                        'lighting': batch_cat[:, 1],
                        'weather': batch_cat[:, 2],
                        'time_of_day': batch_cat[:, 3]
                    }
                    
                    risk_pred, mu, logvar, _, _ = model(batch_num, categorical_dict)
                    pred_loss = F.mse_loss(risk_pred, batch_y)
                    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    
                    if name == 'BetaVAE':
                        loss = pred_loss + model.beta * kl_loss
                    else:
                        loss = 0.995 * pred_loss + 0.005 * kl_loss
                    
                    val_loss += loss.item()
                    val_pred_loss += pred_loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            avg_val_pred = val_pred_loss / len(val_loader)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f'best_{name}.pth')
                if epoch % 10 == 0:
                    print(f'Epoch {epoch:3d}: *** BEST *** Train={total_loss/len(train_loader):.6f} (Pred={total_pred_loss/len(train_loader):.6f}, KL={total_kl_loss/len(train_loader):.6f}), '
                          f'Val={avg_val_loss:.6f} (Pred={avg_val_pred:.6f})')
        
        # Load best model and predict
        model.load_state_dict(torch.load(f'best_{name}.pth'))
        model.eval()
        
        predictions = []
        with torch.no_grad():
            for batch_num, batch_cat in test_loader:
                categorical_dict = {
                    'road_type': batch_cat[:, 0],
                    'lighting': batch_cat[:, 1],
                    'weather': batch_cat[:, 2],
                    'time_of_day': batch_cat[:, 3]
                }
                preds, _, _, _, _ = model(batch_num, categorical_dict)
                predictions.extend(preds.cpu().numpy())
        
        all_predictions[name] = predictions
        print(f"{name} completed. Best val_loss: {best_val_loss:.6f}")
    
    return all_predictions

# Train advanced variants
variant_predictions = train_advanced_variants()

# Create ensemble of all variants
all_preds = list(variant_predictions.values())
ensemble_preds = np.mean(all_preds, axis=0)

# Apply calibration
ensemble_calibrated = ultra_smart_calibration(ensemble_preds, train_df['accident_risk'].values)

# Create ensemble submission
ensemble_submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': ensemble_calibrated
})

ensemble_submission['accident_risk'] = ensemble_submission['accident_risk'].clip(0.02, 0.98)

ensemble_path = "/content/drive/MyDrive/playground-series-s5e10/autoencoder_ensemble.csv"
ensemble_submission.to_csv(ensemble_path, index=False)
print(f"Autoencoder ensemble saved to {ensemble_path}")
