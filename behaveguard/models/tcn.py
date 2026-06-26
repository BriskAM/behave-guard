import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

class CausalConv1d(nn.Module):
    """Single dilated causal convolution with layer normalization and residual connection."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.res_proj = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]  # remove future padding
        out = out.transpose(1, 2)   # [B, T, C]
        out = self.norm(out)
        out = out.transpose(1, 2)   # [B, C, T]
        out = self.activation(out)
        out = self.dropout(out)
        return out + self.res_proj(x)

class TCNEncoder(nn.Module):
    """Dilated TCN Encoder. Receptive field ~50 timesteps."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 16, kernel_size: int = 3):
        super().__init__()
        dilations = [1, 2, 4, 8]
        layers = []
        in_ch = input_dim
        for d in dilations:
            layers.append(CausalConv1d(in_ch, hidden_dim, kernel_size, d))
            in_ch = hidden_dim
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.project = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F] -> [B, F, T]
        x = x.transpose(1, 2)
        x = self.tcn(x)              # [B, H, T]
        x = self.pool(x).squeeze(-1) # [B, H]
        return self.project(x)       # [B, latent_dim]

class TCNDecoder(nn.Module):
    """TCN Decoder - mirrors encoder and reconstructs sequence from latent."""
    def __init__(self, latent_dim: int, hidden_dim: int = 64, output_dim: int = 10, seq_len: int = 50):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.expand = nn.Linear(latent_dim, hidden_dim * seq_len)
        
        dilations = [8, 4, 2, 1]
        layers = []
        in_ch = hidden_dim
        for d in dilations:
            layers.append(CausalConv1d(in_ch, hidden_dim, 3, d))
            in_ch = hidden_dim
        self.tcn = nn.Sequential(*layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        x = self.expand(z)                        # [B, H * T]
        x = x.view(B, self.hidden_dim, self.seq_len)  # [B, H, T]
        x = self.tcn(x)                           # [B, H, T]
        x = x.transpose(1, 2)                     # [B, T, H]
        return self.output_proj(x)                # [B, T, output_dim]

class TCNAutoencoder(nn.Module):
    """Full TCN Autoencoder model."""
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, latent_dim: int = 16, seq_len: int = 50):
        super().__init__()
        self.encoder = TCNEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = TCNDecoder(latent_dim, hidden_dim, input_dim, seq_len)
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

class BehaveGuardTCN:
    """
    TCN Autoencoder model wrapper for BehaveGuard.
    """
    def __init__(
        self,
        seq_len: int = 50,
        feature_dim: int = 10,
        latent_dim: int = 16,
        epochs: int = 100,
        lr: float = 0.005,
        batch_size: int = 16,
        alpha: float = 0.6
    ):
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.alpha = alpha

        self.model = TCNAutoencoder(feature_dim, 64, latent_dim, seq_len)
        self.scaler = StandardScaler()
        
        self.t_anomaly = 0.0
        self.t_anomaly_raw = 0.0
        self.enrollment_mean = None
        self.enrollment_std = None
        self.latent_centroid = None
        self.latent_radius = 0.0
        self.is_trained = False
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        self.model.to(self.device)

    def fit(self, sequences: List[np.ndarray]) -> Dict[str, Any]:
        """
        Train the TCN autoencoder on genuine sequences.
        """
        X = np.stack(sequences)  # (N, 50, 10)
        N = len(sequences)

        # Preprocessing
        X_flat = X.reshape(-1, self.feature_dim)
        self.scaler.fit(X_flat)
        X_scaled = self.scaler.transform(X_flat).reshape(N, self.seq_len, self.feature_dim)

        self.model.train()
        dataset = torch.tensor(X_scaled, dtype=torch.float32)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        feature_weights = torch.tensor([1.4, 1.0, 1.6, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.8], device=self.device)

        for epoch in range(self.epochs):
            permutation = torch.randperm(dataset.size(0))
            for i in range(0, dataset.size(0), self.batch_size):
                indices = permutation[i : i + self.batch_size]
                batch = dataset[indices].to(self.device)

                optimizer.zero_grad()
                recon, latent = self.model(batch)

                # Custom weighted MSE loss
                diff = (recon - batch) ** 2
                recon_loss = (diff * feature_weights).mean()

                # Compactness loss
                latent_mean = latent.mean(dim=0, keepdim=True)
                compactness = ((latent - latent_mean) ** 2).mean()

                loss = recon_loss + 0.2 * compactness
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        # Calibration
        self.model.eval()
        raw_errors = []
        latents_list = []
        with torch.no_grad():
            for i in range(N):
                seq_scaled = X_scaled[i:i+1]
                batch = torch.tensor(seq_scaled, dtype=torch.float32).to(self.device)
                recon, latent = self.model(batch)

                diff = (recon - batch) ** 2
                err = (diff * feature_weights).mean().item()
                raw_errors.append(err)
                latents_list.append(latent.cpu().numpy()[0])

        self.t_anomaly_raw = float(np.percentile(raw_errors, 95))
        latents = np.stack(latents_list)
        self.latent_centroid = np.mean(latents, axis=0)
        latent_distances = np.linalg.norm(latents - self.latent_centroid, axis=1)
        self.latent_radius = float(np.percentile(latent_distances, 95))

        # Calibrate combined threshold
        combined_scores = []
        for err, lat in zip(raw_errors, latents_list):
            l_dist = np.linalg.norm(lat - self.latent_centroid)
            norm_recon = err / self.t_anomaly_raw if self.t_anomaly_raw > 0 else err
            norm_latent = l_dist / self.latent_radius if self.latent_radius > 0 else l_dist
            combined_scores.append(self.alpha * norm_recon + (1.0 - self.alpha) * norm_latent)
        self.t_anomaly = float(np.percentile(combined_scores, 95))

        self.enrollment_mean = np.mean(X_flat, axis=0)
        self.enrollment_std = np.std(X_flat, axis=0) + 1e-8
        self.is_trained = True

        return {
            "t_anomaly": self.t_anomaly,
            "t_anomaly_raw": self.t_anomaly_raw,
            "latent_radius": self.latent_radius,
            "n_windows": N
        }

    def score_window(self, sequence: np.ndarray) -> Dict[str, Any]:
        """
        Score a single sequence of shape (50, 10).
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained.")

        self.model.eval()
        seq_scaled = self.scaler.transform(sequence)
        
        feature_weights = torch.tensor([1.4, 1.0, 1.6, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.8], device=self.device)
        batch = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            recon, latent = self.model(batch)
            diff = (recon - batch) ** 2
            raw_recon = float((diff * feature_weights).mean().item())

        latent_vec = latent.cpu().numpy()[0]
        latent_dist = float(np.linalg.norm(latent_vec - self.latent_centroid))

        # Normalize components
        norm_recon = raw_recon / self.t_anomaly_raw if self.t_anomaly_raw > 0 else raw_recon
        norm_latent = latent_dist / self.latent_radius if self.latent_radius > 0 else latent_dist

        # Combined score
        combined_score = self.alpha * norm_recon + (1.0 - self.alpha) * norm_latent

        # Map to anomaly_score in [0, 1]
        if self.t_anomaly > 0:
            norm_score = combined_score / (self.t_anomaly * 1.5)
        else:
            norm_score = combined_score / 1.5
        anomaly_score = float(np.clip(norm_score, 0.0, 1.0))

        # Verdict
        lo = self.t_anomaly * 0.6
        hi = self.t_anomaly * 1.4
        if combined_score <= lo:
            verdict = "legitimate"
        elif combined_score <= hi:
            verdict = "uncertain"
        else:
            verdict = "anomaly"

        return {
            "anomaly_score": anomaly_score,
            "raw_decision": combined_score,
            "recon_error": raw_recon,
            "latent_distance": latent_dist,
            "verdict": verdict,
            "threshold": self.t_anomaly
        }

    def score_session(self, window_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate window scores into a session verdict."""
        if not window_scores:
            return {"session_verdict": "unknown", "anomaly_rate": 0.0, "mean_score": 0.0}

        anomaly_rate = sum(1 for w in window_scores if w["verdict"] == "anomaly") / len(window_scores)
        mean_score = float(np.mean([w["anomaly_score"] for w in window_scores]))

        if anomaly_rate < 0.25:
            verdict = "legitimate"
        elif anomaly_rate > 0.60:
            verdict = "impostor"
        else:
            verdict = "uncertain"

        return {
            "session_verdict": verdict,
            "anomaly_rate": anomaly_rate,
            "mean_score": mean_score,
            "n_windows": len(window_scores)
        }

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state_dict_cpu = {k: v.cpu() for k, v in self.model.state_dict().items()}
        state = {
            "model_state_dict": state_dict_cpu,
            "scaler": self.scaler,
            "t_anomaly": self.t_anomaly,
            "t_anomaly_raw": self.t_anomaly_raw,
            "enrollment_mean": self.enrollment_mean,
            "enrollment_std": self.enrollment_std,
            "latent_centroid": self.latent_centroid,
            "latent_radius": self.latent_radius,
            "is_trained": self.is_trained,
            "seq_len": self.seq_len,
            "feature_dim": self.feature_dim,
            "latent_dim": self.latent_dim,
            "alpha": self.alpha
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: str | Path):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.seq_len = state.get("seq_len", 50)
        self.feature_dim = state.get("feature_dim", 10)
        self.latent_dim = state.get("latent_dim", 16)
        self.alpha = state.get("alpha", 0.6)

        self.model = TCNAutoencoder(self.feature_dim, 64, self.latent_dim, self.seq_len)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.scaler = state["scaler"]
        self.t_anomaly = state["t_anomaly"]
        self.t_anomaly_raw = state["t_anomaly_raw"]
        self.enrollment_mean = state["enrollment_mean"]
        self.enrollment_std = state["enrollment_std"]
        self.latent_centroid = state["latent_centroid"]
        self.latent_radius = state["latent_radius"]
        self.is_trained = state["is_trained"]
