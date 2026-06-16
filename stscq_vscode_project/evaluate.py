import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from configs.config import DATA_DIR, BATCH_SIZE, NUM_WORKERS, CHECKPOINT_DIR, OUTPUT_DIR, CHECKPOINT_NAME
from src.dataset import build_dataloaders
from src.model import STSCQAutoencoder
from src.metrics import calculate_psnr, reconstruction_mse
from src.visualize import save_reconstructions


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run train.py first.")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    image_size = checkpoint["image_size"]
    train_loader, val_loader, _, val_dataset = build_dataloaders(
        DATA_DIR, image_size, BATCH_SIZE, NUM_WORKERS
    )

    model = STSCQAutoencoder(
        image_size=image_size,
        latent_dim=checkpoint["latent_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_groups=checkpoint["num_groups"],
        codebook_size=checkpoint["codebook_size"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_mse = 0.0
    total_psnr = 0.0

    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            reconstructed, _, _, _ = model(images)

            total_mse += reconstruction_mse(images, reconstructed)
            total_psnr += calculate_psnr(images, reconstructed)

    avg_mse = total_mse / len(val_loader)
    avg_psnr = total_psnr / len(val_loader)
    estimated_bpp = checkpoint["estimated_bpp"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_path = save_reconstructions(model, val_loader, device, OUTPUT_DIR)

    print("Final evaluation results")
    print(f"Validation images: {len(val_dataset)}")
    print(f"MSE: {avg_mse:.6f}")
    print(f"PSNR: {avg_psnr:.4f}")
    print(f"Estimated bpp: {estimated_bpp:.6f}")
    print(f"Reconstruction image: {image_path}")


if __name__ == "__main__":
    evaluate()
