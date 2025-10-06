import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch.nn.functional as F

from google.colab import drive
import os
drive.mount('/content/drive')

train_path = "/content/drive/MyDrive/playground-series-s5e10/train.csv"
test_path = "/content/drive/MyDrive/playground-series-s5e10/test.csv"
sub_path = "/content/drive/MyDrive/playground-series-s5e10/sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sub_df = pd.read_csv(sub_path)

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch.nn.functional as F


# Data Preprocessing with Embedding Support
def preprocess_data_with_embedding(train_df, test_df):
    # Save IDs
    train_ids = train_df['id'].copy()
    test_ids = test_df['id'].copy()

    # Drop ID columns
    train_processed = train_df.drop('id', axis=1)
    test_processed = test_df.drop('id', axis=1)

    # Convert boolean columns to integers
    bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
    for col in bool_cols:
        train_processed[col] = train_processed[col].astype(int)
        test_processed[col] = test_processed[col].astype(int)

    # Get unique values for each categorical column for embedding
    categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
    categorical_info = {}

    for col in categorical_cols:
        # Label encode for embedding indices
        le = LabelEncoder()
        train_processed[col] = le.fit_transform(train_processed[col])
        test_processed[col] = le.transform(test_processed[col])
        categorical_info[col] = {
            'num_unique': len(le.classes_),
            'values': le.classes_
        }

    return train_processed, test_processed, train_ids, test_ids, categorical_info

# Enhanced Dataset with separate categorical and numerical features
class AccidentDataset(Dataset):
    def __init__(self, df, target_col='accident_risk', is_test=False, scaler=None):
        self.df = df.reset_index(drop=True)
        self.is_test = is_test
        self.target_col = target_col

        # Define feature types
        self.categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
        self.bool_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']
        self.numerical_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']

        if not is_test:
            self.targets = self.df[target_col].values
            self.features_df = self.df.drop(target_col, axis=1)
        else:
            self.features_df = self.df

        # Scale numerical features
        self.numerical_features = self.features_df[self.numerical_cols].values
        if scaler is None and not is_test:
            self.scaler = StandardScaler()
            self.numerical_scaled = self.scaler.fit_transform(self.numerical_features)
        elif scaler is not None:
            self.scaler = scaler
            self.numerical_scaled = self.scaler.transform(self.numerical_features)
        else:
            self.scaler = None
            self.numerical_scaled = self.numerical_features

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Categorical features
        categorical_data = []
        for col in self.categorical_cols:
            categorical_data.append(self.features_df[col].iloc[idx])

        # Numerical features
        numerical_data = self.numerical_scaled[idx]

        # Boolean features
        bool_data = []
        for col in self.bool_cols:
            bool_data.append(self.features_df[col].iloc[idx])

        # Convert to tensors
        categorical_tensor = torch.tensor(categorical_data, dtype=torch.long)
        numerical_tensor = torch.tensor(numerical_data, dtype=torch.float32)
        bool_tensor = torch.tensor(bool_data, dtype=torch.float32)

        if self.is_test:
            return categorical_tensor, numerical_tensor, bool_tensor
        else:
            target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32)
            return categorical_tensor, numerical_tensor, bool_tensor, target_tensor

# Enhanced Model with Embeddings
class VAEPredictor(nn.Module):
    def __init__(self, categorical_info, numerical_dim=4, bool_dim=4, latent_dim=128):
        super(EnhancedVAEPredictor, self).__init__()

        # Embedding layers for categorical variables
        self.embeddings = nn.ModuleDict()
        embedding_dims = {}
        total_embedding_dim = 0

        for col, info in categorical_info.items():
            embedding_dim = min(50, (info['num_unique'] + 1) // 2)  # Adaptive embedding size
            embedding_dim = max(4, embedding_dim)  # At least 4 dimensions
            self.embeddings[col] = nn.Embedding(info['num_unique'], embedding_dim)
            embedding_dims[col] = embedding_dim
            total_embedding_dim += embedding_dim

        # Input dimensions
        self.total_embedding_dim = total_embedding_dim
        self.numerical_dim = numerical_dim
        self.bool_dim = bool_dim
        self.latent_dim = latent_dim

        total_input_dim = total_embedding_dim + numerical_dim + bool_dim

        # Enhanced Encoder
        self.encoder = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )

        # VAE latent space
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Enhanced Predictor
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32), # Corrected input dimension from 64 to 32
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # Decoder for VAE
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, total_input_dim)
        )

    def encode(self, categorical, numerical, boolean):
        # Process embeddings
        embedded_features = []
        for i, (col, embed_layer) in enumerate(self.embeddings.items()):
            embedded = embed_layer(categorical[:, i])
            embedded_features.append(embedded)

        # Concatenate all features
        embedded_cat = torch.cat(embedded_features, dim=1)
        all_features = torch.cat([embedded_cat, numerical, boolean], dim=1)

        # Encode
        encoded = self.encoder(all_features)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return mu, logvar, all_features

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, categorical, numerical, boolean, training=True):
        mu, logvar, input_features = self.encode(categorical, numerical, boolean)

        if training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        # Predict
        risk = self.predictor(z)

        if training:
            # Reconstruction for VAE loss
            reconstructed = self.decoder(z)
            return risk.squeeze(), mu, logvar, input_features, reconstructed
        else:
            return risk.squeeze()

# Enhanced Loss Function
def enhanced_vae_loss(risk_pred, targets, mu, logvar, input_features, reconstructed, alpha=0.8, beta=0.01):
    # Prediction loss
    pred_loss = F.mse_loss(risk_pred, targets)

    # VAE reconstruction loss
    recon_loss = F.mse_loss(reconstructed, input_features)

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / input_features.size(0)

    # Combined loss
    total_loss = alpha * pred_loss + (1 - alpha) * recon_loss + beta * kl_loss

    return total_loss, pred_loss, recon_loss, kl_loss

# Training Function
def train_model(model, train_loader, epochs=30, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    model.train()
    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0
        total_pred_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        for batch in train_loader:
            categorical, numerical, boolean, targets = batch

            optimizer.zero_grad()

            # Forward pass
            risk_pred, mu, logvar, input_features, reconstructed = model(categorical, numerical, boolean, training=True)

            # Calculate loss
            loss, pred_loss, recon_loss, kl_loss = enhanced_vae_loss(
                risk_pred, targets, mu, logvar, input_features, reconstructed
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_pred_loss += pred_loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        if epoch % 5 == 0:
            print(f'Epoch {epoch}: Total={avg_loss:.4f}, Pred={total_pred_loss/len(train_loader):.4f}, '
                  f'Recon={total_recon_loss/len(train_loader):.4f}, KL={total_kl_loss/len(train_loader):.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save best model
            torch.save(model.state_dict(), 'best_enhanced_vae_model.pth')

    return model

# Main  Pipeline
print("Preprocessing data with embeddings...")
train_processed, test_processed, train_ids, test_ids, categorical_info = preprocess_data_with_embedding(train_df, test_df)

print("Categorical info:", {k: v['num_unique'] for k, v in categorical_info.items()})

# Create enhanced datasets
train_dataset = AccidentDataset(train_processed, target_col='accident_risk', is_test=False)
test_dataset = AccidentDataset(test_processed, is_test=True, scaler=train_dataset.scaler)

train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=2)

# Initialize enhanced model
model = VAEPredictor(
    categorical_info,
    numerical_dim=4,  # num_lanes, curvature, speed_limit, num_reported_accidents
    bool_dim=4,       # 4 boolean columns
    latent_dim=64
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Train enhanced model
print("Training VAE Predictor...")
model = train_model(model, train_loader, epochs=30, lr=0.001)

# Load best model for prediction
model.load_state_dict(torch.load('best_enhanced_vae_model.pth'))
model.eval()

# Predict on test set
print("Making predictions...")
test_predictions = []

with torch.no_grad():
    for batch in test_loader:
        categorical, numerical, boolean = batch
        preds = model(categorical, numerical, boolean, training=False)
        test_predictions.extend(preds.cpu().numpy())

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': test_predictions
})

# Ensure predictions are between 0 and 1
submission['accident_risk'] = submission['accident_risk'].clip(0, 1)

# Save submission
submission.to_csv('enhanced_vae_accident_predictions.csv', index=False)
print("Submission file saved as 'enhanced_vae_accident_predictions.csv'")

# Print statistics
print(f"\nPrediction statistics:")
print(f"Min risk: {submission['accident_risk'].min():.4f}")
print(f"Max risk: {submission['accident_risk'].max():.4f}")
print(f"Mean risk: {submission['accident_risk'].mean():.4f}")
print(f"Std risk: {submission['accident_risk'].std():.4f}")



# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': test_predictions
})

# Ensure predictions are between 0 and 1
submission['accident_risk'] = submission['accident_risk'].clip(0, 1)

submission_path = "/content/drive/MyDrive/playground-series-s5e10/enhanced_vae_accident_predictions.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file saved to {submission_path}")

