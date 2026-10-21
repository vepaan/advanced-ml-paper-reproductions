"""2D Shapes VAE architecture and β-weighted negative ELBO.

Source: Higgins et al. (ICLR 2017), Appendix A.1, Table 1.
Details not pinned down by paper are documented in README.md.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = 64 * 64
    hidden_dim: int = 1200
    latent_dim: int = 10


class BetaVAE(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
        )
        self.mu_head = nn.Linear(h, config.latent_dim)
        self.logvar_head = nn.Linear(h, config.latent_dim)
        # Appendix A.1: 1200,1200,1200,4096; Tanh; Bernoulli.
        # Hidden decoder activations are Tanh. Final outputs are Bernoulli logits.
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, h), nn.Tanh(),
            nn.Linear(h, config.input_dim),
        )

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x.view(x.shape[0], -1))
        return self.mu_head(h), self.logvar_head(h)

    @staticmethod
    def sample(mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode_logits(self, z: Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x: Tensor, *, deterministic: bool = False):
        mu, logvar = self.encode(x)
        z = mu if deterministic else self.sample(mu, logvar)
        return self.decode_logits(z), mu, logvar


def loss_terms(logits: Tensor, x: Tensor, mu: Tensor, logvar: Tensor, beta: float):
    """Return per-example-averaged terms; pixel and latent sums stay unnormalized.

    Loss = [sum_pixels BCE-with-logits + beta * sum_latents KL] / batch_size.
    The coefficient beta is not accidentally multiplied by 4096.
    """
    if beta < 0:
        raise ValueError('beta must be nonnegative')
    n = x.shape[0]
    x = x.view(n, -1)
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction='sum') / n
    kl = -0.5 * (1 + logvar - mu.square() - logvar.exp()).sum() / n
    return recon + beta * kl, recon, kl


def model_metadata(config: ModelConfig) -> dict:
    return asdict(config)
