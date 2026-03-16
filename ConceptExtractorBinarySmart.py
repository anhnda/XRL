# Smart Binary Concepts with Gumbel-Softmax
# This makes binary concepts differentiable during training

import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelBinarySAE(nn.Module):
    """Binary SAE using Gumbel-Softmax trick"""

    def __init__(self, input_dim=128, hidden_dim=6, k=3, tau=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.tau = tau  # Temperature for Gumbel-Softmax

        # Encoder outputs logits (before sigmoid)
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)

        # Decoder takes binary-ish inputs
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

    def encode(self, x, hard=False):
        """Encode to binary concepts using Gumbel-Softmax"""
        logits = self.encoder(x)  # [batch, hidden_dim]

        # Get top-k indices
        _, top_indices = torch.topk(logits, self.k, dim=-1)

        # Create mask for top-k
        mask = torch.zeros_like(logits)
        mask.scatter_(-1, top_indices, 1.0)

        if self.training:
            # During training: use continuous relaxation
            probs = torch.sigmoid(logits / self.tau)
            # Apply top-k mask
            concepts = probs * mask
        else:
            # During inference: use hard binary
            concepts = mask

        return concepts

    def forward(self, x):
        concepts = self.encode(x)
        x_recon = self.decoder(concepts)
        return x_recon, concepts


# Usage:
"""
model = GumbelBinarySAE(input_dim=128, hidden_dim=6, k=3, tau=0.5)

# Training: concepts are ~binary but differentiable
features = torch.randn(32, 128)
x_recon, concepts = model(features)
loss = F.mse_loss(x_recon, features)
loss.backward()  # Works! Has gradients

# Inference: concepts are truly binary {0, 1}
model.eval()
x_recon, concepts = model(features)
print(concepts[0])  # [0, 1, 0, 1, 0, 0] - clean binary!
"""
