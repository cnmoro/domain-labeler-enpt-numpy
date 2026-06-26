"""
Lightweight domain labeler — pure numpy + tokenizers, no PyTorch, no ONNX.

No sparse vector is ever materialized: SPLADE projection uses only the ~80
active dimensions via scatter-free indexing into the weight matrix.

Docker image: ~265 MB (python:3.12-slim + numpy + tokenizers)
"""

import json
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer


class DomainLabeler:
    """Bilingual domain classifier. Pure numpy — no PyTorch, no ONNX.

    Usage:
        model = DomainLabeler(".")
        model.predict("O Google Chrome é um navegador web")
        # → ["computer_science_and_technology"]
    """

    SPLADE_SPECIAL = {0, 100, 101, 102, 103}  # [PAD] [UNK] [CLS] [SEP] [MASK]

    def __init__(self, model_dir="."):
        d = Path(model_dir) / "assets"
        load = lambda name: np.load(str(d / name))

        # Tokenizers
        self.splade_tok = Tokenizer.from_file(str(d / "splade_tokenizer.json"))
        self.nomic_tok = Tokenizer.from_file(str(d / "nomic_tokenizer.json"))

        # SPLADE weights
        self.splade_w = load("splade_weights.npy")  # (30522,)

        # Nomic weights
        self.nomic_emb = load("nomic_embeddings.npy")   # (32000, 384)
        self.nomic_map = load("nomic_mapping.npy")       # (276214,)
        self.nomic_w = load("nomic_weights.npy")         # (276214,)

        # MLP weights  (all saved as mlp_{key}.npy with . replaced by _)
        self.w_splade_proj = load("mlp_splade_proj_weight.npy")   # (512, 30522)
        self.w_norm1 = load("mlp_norm1_weight.npy")               # (512,)
        self.b_norm1 = load("mlp_norm1_bias.npy")                 # (512,)
        self.w_nomic_0 = load("mlp_nomic_proj_0_weight.npy")      # (256, 384)
        self.b_nomic_0 = load("mlp_nomic_proj_0_bias.npy")        # (256,)
        self.w_norm_nomic = load("mlp_nomic_proj_1_weight.npy")   # (256,)
        self.b_norm_nomic = load("mlp_nomic_proj_1_bias.npy")     # (256,)
        self.w_fc1 = load("mlp_fc1_weight.npy")                   # (512, 768)
        self.b_fc1 = load("mlp_fc1_bias.npy")                     # (512,)
        self.w_norm2 = load("mlp_norm2_weight.npy")               # (512,)
        self.b_norm2 = load("mlp_norm2_bias.npy")                 # (512,)
        self.w_fc2 = load("mlp_fc2_weight.npy")                   # (67, 512)
        self.b_fc2 = load("mlp_fc2_bias.npy")                     # (67,)

        # Label mapping
        with open(d / "id2label.json") as f:
            self.id2label = {int(k): v for k, v in json.load(f).items()}

    # ── Encoding helpers ──

    def _encode_splade(self, text):
        tokens = self.splade_tok.encode(text)
        # Build dict of active dims via max-pool (filters special tokens)
        active = {}
        for tid in tokens.ids:
            if tid in self.SPLADE_SPECIAL or tid >= len(self.splade_w):
                continue
            w = self.splade_w[tid]
            if w > active.get(tid, -1e10):
                active[tid] = w
        # ReLU filter
        active = {k: v for k, v in active.items() if v > 0}
        # Keep top 80
        if len(active) > 80:
            threshold = sorted(active.values())[-80]
            active = {k: v for k, v in active.items() if v >= threshold}
        return active  # dict {dim_id: value}

    def _encode_nomic(self, text):
        tokens = self.nomic_tok.encode(text)
        ids = np.array(tokens.ids, dtype=np.int64)
        indices = self.nomic_map[ids]       # (n_tokens,) → which embedding row
        embeds = self.nomic_emb[indices]    # (n_tokens, 384)
        w = self.nomic_w[ids, None]         # (n_tokens, 1)
        pooled = np.sum(embeds * w, axis=0) / (np.sum(w) + 1e-10)
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm
        return pooled  # (384,)

    # ── MLP helpers (numpy) ──

    @staticmethod
    def _layer_norm(x, weight, bias, eps=1e-5):
        mean = x.mean()
        var = x.var(ddof=0)
        return (x - mean) / np.sqrt(var + eps) * weight + bias

    @staticmethod
    def _relu(x):
        return np.maximum(x, 0)

    def _mlp_forward(self, splade_active, dense_vec):
        """Full MLP forward. splade_active is dict {dim: value} (sparse)."""
        # ── SPLADE branch (no 30522 materialization) ──
        h = np.zeros(self.w_splade_proj.shape[0], dtype=np.float32)
        for dim, v in splade_active.items():
            h += v * self.w_splade_proj[:, dim]
        # relu → norm1 (matches PT: norm1(dropout(relu(x))))
        h = self._layer_norm(self._relu(h), self.w_norm1, self.b_norm1)

        # ── Nomic branch (Linear → LayerNorm → ReLU) ──
        n = dense_vec @ self.w_nomic_0.T + self.b_nomic_0         # (256,)
        n = self._layer_norm(n, self.w_norm_nomic, self.b_norm_nomic)
        n = self._relu(n)

        # ── Fusion: concat + fc1 + norm2 + relu + fc2 ──
        x = np.concatenate([h, n])                                 # (768,)
        x = x @ self.w_fc1.T + self.b_fc1                         # (512,)
        x = self._relu(self._layer_norm(x, self.w_norm2, self.b_norm2))
        logits = x @ self.w_fc2.T + self.b_fc2                    # (67,)
        return logits

    # ── Public API ──

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        results = []
        for text in texts:
            s = self._encode_splade(text)
            d = self._encode_nomic(text)
            logits = self._mlp_forward(s, d)
            results.append(self.id2label[int(logits.argmax())])
        return results

    def predict_proba(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        results = []
        for text in texts:
            s = self._encode_splade(text)
            d = self._encode_nomic(text)
            logits = self._mlp_forward(s, d)
            exp = np.exp(logits - logits.max())
            probs = exp / exp.sum()
            results.append({self.id2label[i]: float(p) for i, p in enumerate(probs)})
        return results

    def __call__(self, texts):
        return self.predict(texts)
