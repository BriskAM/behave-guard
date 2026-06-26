import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder following BehaveGuard architecture:
    Inputs: [batch_size, seq_len, feature_dim]
    """
    def __init__(self, seq_len: int = 50, feature_dim: int = 10, latent_dim: int = 16):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_lstm1 = nn.LSTM(input_size=feature_dim, hidden_size=64, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
        self.latent_proj = nn.Linear(32, latent_dim)

        # Decoder initial states projection
        self.dec_h0_proj = nn.Linear(latent_dim, 32)
        self.dec_c0_proj = nn.Linear(latent_dim, 32)

        # Decoder
        self.decoder_lstm1 = nn.LSTM(input_size=latent_dim, hidden_size=32, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(input_size=32, hidden_size=64, batch_first=True)
        self.decoder_dense = nn.Linear(64, feature_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encoder
        out1, _ = self.encoder_lstm1(x)
        out2, (hn, _) = self.encoder_lstm2(out1)
        
        # Take last hidden state of final encoder layer
        latent = self.latent_proj(hn[-1])  # [batch_size, latent_dim]

        # Repeat latent vector seq_len times
        latent_repeated = latent.unsqueeze(1).repeat(1, self.seq_len, 1)

        # Project latent to decoder initial state (h0, c0)
        h0 = self.dec_h0_proj(latent).unsqueeze(0)
        c0 = self.dec_c0_proj(latent).unsqueeze(0)

        # Decoder
        dec_out1, _ = self.decoder_lstm1(latent_repeated, (h0, c0))
        dec_out2, _ = self.decoder_lstm2(dec_out1)
        reconstructed = self.decoder_dense(dec_out2)

        return reconstructed, latent

class BehaveGuardLSTM:
    """
    LSTM Autoencoder model wrapper for BehaveGuard.
    Trains on genuine user sequences (shape: [N, 50, 10]).
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
        self.alpha = alpha  # combined score = alpha * recon_error + (1-alpha) * latent_distance

        self.model = LSTMAutoencoder(seq_len, feature_dim, latent_dim)
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
        Train the autoencoder on genuine sequences.
        sequences: list of shape (50, 10) arrays
        """
        X = np.stack(sequences)  # (N, 50, 10)
        N = len(sequences)

        # Fit StandardScaler on flattened sequence data
        X_flat = X.reshape(-1, self.feature_dim)
        self.scaler.fit(X_flat)
        X_scaled = self.scaler.transform(X_flat).reshape(N, self.seq_len, self.feature_dim)

        self.model.train()
        dataset = torch.tensor(X_scaled, dtype=torch.float32)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.epochs)

        # Feature weights for weighted reconstruction error
        # Order: dwell_ms, flight_ms, digraph_ms, [4x key_cat], time_sin, time_cos, freq_weight
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

                # Compactness loss: pull latents to their mean
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

        # Normalize score components
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
        lo = self.t_anomaly * 0.7
        hi = self.t_anomaly * 1.1
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
        """Aggregate multiple window scores into a session verdict."""
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
        # Move state dict to CPU to prevent GPU dependency on load
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

        self.model = LSTMAutoencoder(self.seq_len, self.feature_dim, self.latent_dim)
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
