"""LLLM (Latent Linguistic Lil Model) — PyTorch Neural Core.

Dual-encoder cross-attention architecture:
  V1 = {LD (Lil Durk), LB (Lil Baby), LW (Lil Wayne)} — High Entropy / Raw Cadence
  V2 = {MD (Mos Def)} — Semantic Bridging / Conscious Logic

Cross-attention: V1 queries attend to V2 keys/values, fusing street grit
with polysyllabic lyricism. Output: logits over vocabulary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class LLLMConfig:
    """Configuration for the LLLM architecture."""
    hidden_dim: int = 512
    vocab_size: int = 50000
    num_heads: int = 8
    num_layers: int = 6
    max_seq_len: int = 2048
    dropout_v1: float = 0.3  # Higher dropout for V1 (raw street semantics)
    dropout_v2: float = 0.1  # Lower dropout for V2 (conscious lyricism)
    activation: str = "gelu"
    layer_norm_eps: float = 1e-5


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[: x.size(0), :]


class V1Encoder(nn.Module):
    """V1 Encoder: High-Entropy layer for raw street cadence.

    Processes LD (Lil Durk), LB (Lil Baby), LW (Lil Wayne) style inputs.
    Higher dropout (0.3) for grittier, more unpredictable outputs.
    """

    def __init__(self, config: LLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.pos_encoding = PositionalEncoding(config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout_v1,
            activation=config.activation,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )
        self.norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.norm(x)


class V2Encoder(nn.Module):
    """V2 Encoder: Semantic Bridging layer (Mos Def logic).

    Lower dropout (0.1) for precise, conscious lyricism.
    Acts as the piano anchor — temporal bridging and metaphor.
    """

    def __init__(self, config: LLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.pos_encoding = PositionalEncoding(config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout_v2,
            activation=config.activation,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )
        self.norm = nn.LayerNorm(config.hidden_dim, eps=config.layer_norm_eps)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        return self.norm(x)


class LLLM_Architecture(nn.Module):
    """Latent Linguistic Lil Model — Dual-encoder cross-attention neural core.

    Architecture:
        V1 Input (street grit: LD, LB, LW)
            -> V1 Encoder (dropout=0.3)
            -> Cross-Attention Query
        V2 Input (conscious logic: MD)
            -> V2 Encoder (dropout=0.1)
            -> Cross-Attention Key/Value
        Cross-Attention (V1_Q @ V2_K, V1_Q @ V2_V)
            -> Fused Features
        Decoder (Linear: hidden_dim -> vocab_size)
            -> Output Logits

    The cross-attention mechanism mathematically fuses rapid-fire trap
    delivery (V1) with golden-era conscious lyricism (V2), producing
    output that balances raw street semantics with polysyllabic depth.
    """

    def __init__(self, config: Optional[LLLMConfig] = None) -> None:
        super().__init__()
        self.config = config or LLLMConfig()
        cfg = self.config

        # Token embedding shared across both encoders
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)

        # V1: High-Entropy street cadence encoder
        self.v1_encoder = V1Encoder(cfg)

        # V2: Semantic bridging conscious encoder
        self.v2_encoder = V2Encoder(cfg)

        # Cross-attention: V1 queries attend to V2 key/value
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout_v2,
            batch_first=True,
        )

        # Feed-forward after cross-attention
        self.ffn = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(cfg.dropout_v2),
            nn.Linear(cfg.hidden_dim * 4, cfg.hidden_dim),
            nn.Dropout(cfg.dropout_v2),
        )

        self.cross_norm = nn.LayerNorm(cfg.hidden_dim, eps=cfg.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(cfg.hidden_dim, eps=cfg.layer_norm_eps)

        # Output decoder: hidden_dim -> vocab_size
        self.decoder = nn.Linear(cfg.hidden_dim, cfg.vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        v1_input: Tensor,
        v2_input: Tensor,
        v1_mask: Optional[Tensor] = None,
        v2_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass through the LLLM architecture.

        Args:
            v1_input: Token IDs for V1 (street grit) [batch, seq_len_v1]
            v2_input: Token IDs for V2 (conscious logic) [batch, seq_len_v2]
            v1_mask: Padding mask for V1 [batch, seq_len_v1]
            v2_mask: Padding mask for V2 [batch, seq_len_v2]

        Returns:
            Output logits [batch, seq_len_v1, vocab_size]
        """
        # Embed tokens
        v1_embedded = self.token_embedding(v1_input)
        v2_embedded = self.token_embedding(v2_input)

        # Encode V1 (high-entropy street cadence)
        v1_features = self.v1_encoder(v1_embedded, mask=v1_mask)

        # Encode V2 (semantic bridging conscious logic)
        v2_features = self.v2_encoder(v2_embedded, mask=v2_mask)

        # Cross-attention: V1 acts as Query, V2 as Key/Value
        # This fuses street grit with conscious lyricism
        fused_features, attention_weights = self.cross_attention(
            query=v1_features,
            key=v2_features,
            value=v2_features,
            key_padding_mask=v2_mask,
        )

        # Residual connection + layer norm
        fused_features = self.cross_norm(v1_features + fused_features)

        # Feed-forward
        ffn_output = self.ffn(fused_features)
        fused_features = self.ffn_norm(fused_features + ffn_output)

        # Decode to vocabulary logits
        output_logits = self.decoder(fused_features)

        return output_logits

    def generate_verse(
        self,
        v1_prompt: Tensor,
        v2_prompt: Tensor,
        max_length: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> Tensor:
        """Generate a verse autoregressively.

        Args:
            v1_prompt: V1 style prompt tokens
            v2_prompt: V2 style prompt tokens
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter

        Returns:
            Generated token IDs
        """
        self.eval()
        generated = v1_prompt.clone()

        with torch.no_grad():
            for _ in range(max_length):
                logits = self.forward(generated, v2_prompt)
                next_token_logits = logits[:, -1, :] / temperature

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(
                        next_token_logits, top_k
                    )[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float("-inf")

                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=-1)

                # Stop on EOS token (assuming token ID 2)
                if next_token.item() == 2:
                    break

        return generated


@dataclass
class TrainingConfig:
    """Training configuration for LLLM."""
    batch_size: int = 16
    learning_rate: float = 3e-4
    num_epochs: int = 100
    warmup_steps: int = 4000
    max_grad_norm: float = 1.0
    save_every: int = 1000
    eval_every: int = 500
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingHistory:
    """Track training metrics."""
    losses: List[float] = field(default_factory=list)
    perplexities: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    epochs: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "losses": self.losses,
            "perplexities": self.perplexities,
            "val_losses": self.val_losses,
            "epochs": self.epochs,
        }


class LLLM_TrainingPipeline:
    """End-to-end training pipeline for LLLM."""

    def __init__(
        self,
        model: LLLM_Architecture,
        config: TrainingConfig,
    ) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.history = TrainingHistory()
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs
        )
        self.device = config.device

    def train_step(
        self, v1_batch: Tensor, v2_batch: Tensor, targets: Tensor
    ) -> float:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        v1_batch = v1_batch.to(self.device)
        v2_batch = v2_batch.to(self.device)
        targets = targets.to(self.device)

        logits = self.model(v1_batch, v2_batch)

        # Cross-entropy loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, train_loader: Any) -> float:
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            v1, v2, targets = batch
            loss = self.train_step(v1, v2, targets)
            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.history.losses.append(avg_loss)
        self.history.epochs.append(len(self.history.epochs) + 1)
        self.history.perplexities.append(math.exp(avg_loss))

        self.scheduler.step()
        return avg_loss

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "history": self.history.to_dict(),
                "config": self.config,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.history = TrainingHistory(**checkpoint["history"])


if __name__ == "__main__":
    # Quick sanity check
    cfg = LLLMConfig(hidden_dim=128, vocab_size=1000)
    model = LLLM_Architecture(cfg)

    batch_size, seq_len = 2, 10
    v1 = torch.randint(0, 1000, (batch_size, seq_len))
    v2 = torch.randint(0, 1000, (batch_size, seq_len))

    logits = model(v1, v2)
    assert logits.shape == (batch_size, seq_len, 1000)
    print(f"LLLM Architecture test passed. Output shape: {logits.shape}")
