import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=128, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_channels=3, hidden_dim=128, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class STSCQQuantizer(nn.Module):
    """Simplified Switchable Token-Specific Codebook Quantization.

    The paper proposes switching between codebook groups and using token-specific
    codebooks. This class implements that core idea in a compact form suitable
    for an educational PyTorch implementation.
    """

    def __init__(self, num_tokens, embedding_dim, num_groups=4, codebook_size=128):
        super().__init__()
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.num_groups = num_groups
        self.codebook_size = codebook_size

        # Shape: [number_of_groups, number_of_tokens, entries_per_codebook, embedding_dim]
        self.codebooks = nn.Parameter(
            torch.randn(num_groups, num_tokens, codebook_size, embedding_dim) * 0.02
        )

        # Image-level router: chooses one codebook group for each image.
        self.router = nn.Sequential(
            nn.Linear(num_tokens * embedding_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_groups),
        )

    def forward(self, z):
        # z: [B, D, H, W]
        batch_size, dim, height, width = z.shape
        tokens = z.permute(0, 2, 3, 1).reshape(batch_size, height * width, dim)

        route_logits = self.router(tokens.reshape(batch_size, -1))
        route_idx = torch.argmax(route_logits, dim=-1)

        selected_codebooks = self.codebooks[route_idx]  # [B, T, K, D]
        tokens_expanded = tokens.unsqueeze(2)            # [B, T, 1, D]

        distances = torch.sum((tokens_expanded - selected_codebooks) ** 2, dim=-1)
        indices = torch.argmin(distances, dim=-1)        # [B, T]

        gathered_indices = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, dim)
        quantized_tokens = torch.gather(selected_codebooks, 2, gathered_indices).squeeze(2)

        # Straight-through estimator: forward uses quantized tokens,
        # backward passes gradients to encoder outputs.
        quantized_tokens_st = tokens + (quantized_tokens - tokens).detach()
        quantized_z = quantized_tokens_st.reshape(batch_size, height, width, dim).permute(0, 3, 1, 2)

        commitment_loss = F.mse_loss(tokens, quantized_tokens.detach())
        codebook_loss = F.mse_loss(quantized_tokens, tokens.detach())
        quantization_loss = commitment_loss + codebook_loss

        return quantized_z, quantization_loss, indices, route_idx

    def calculate_bpp(self, image_size):
        height, width = image_size
        route_bits = math.log2(self.num_groups)
        token_bits = self.num_tokens * math.log2(self.codebook_size)
        return (route_bits + token_bits) / (height * width)


class STSCQAutoencoder(nn.Module):
    def __init__(self, image_size=128, latent_dim=64, hidden_dim=128, num_groups=4, codebook_size=128):
        super().__init__()
        latent_size = image_size // 8
        num_tokens = latent_size * latent_size

        self.encoder = Encoder(3, hidden_dim, latent_dim)
        self.quantizer = STSCQQuantizer(num_tokens, latent_dim, num_groups, codebook_size)
        self.decoder = Decoder(3, hidden_dim, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        z_q, q_loss, indices, route_idx = self.quantizer(z)
        reconstructed = self.decoder(z_q)
        return reconstructed, q_loss, indices, route_idx
