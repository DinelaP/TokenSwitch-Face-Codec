import torch
import torch.nn.functional as F


def calculate_psnr(original, reconstructed):
    mse = F.mse_loss(reconstructed, original)
    if mse.item() == 0:
        return 100.0
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


def reconstruction_mse(original, reconstructed):
    return F.mse_loss(reconstructed, original).item()
