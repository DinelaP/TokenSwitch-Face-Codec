import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from configs.config import (
    DATA_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    LAMBDA_Q,
    LATENT_DIM,
    HIDDEN_DIM,
    NUM_GROUPS,
    CODEBOOK_SIZE,
    NUM_WORKERS,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    CHECKPOINT_NAME,
)
from src.dataset import build_dataloaders
from src.model import STSCQAutoencoder
from src.metrics import calculate_psnr
from src.visualize import save_reconstructions


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset path: {DATA_DIR}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_loader, val_loader, train_dataset, val_dataset = build_dataloaders(
        DATA_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS
    )

    print(f"Train images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Classes: {len(train_dataset.classes)}")

    model = STSCQAutoencoder(
        image_size=IMAGE_SIZE,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        num_groups=NUM_GROUPS,
        codebook_size=CODEBOOK_SIZE,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    estimated_bpp = model.quantizer.calculate_bpp((IMAGE_SIZE, IMAGE_SIZE))
    print(f"Estimated bpp: {estimated_bpp:.6f}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_q_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for images, _ in progress:
            images = images.to(device)

            optimizer.zero_grad()
            reconstructed, q_loss, _, _ = model(images)

            recon_loss = F.mse_loss(reconstructed, images)
            loss = recon_loss + LAMBDA_Q * q_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_q_loss += q_loss.item()

            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "q": f"{q_loss.item():.4f}",
            })

        avg_loss = total_loss / len(train_loader)
        avg_recon = total_recon_loss / len(train_loader)
        avg_q = total_q_loss / len(train_loader)

        print(f"Epoch {epoch + 1} finished")
        print(f"Average loss: {avg_loss:.6f}")
        print(f"Average reconstruction loss: {avg_recon:.6f}")
        print(f"Average quantization loss: {avg_q:.6f}")

        validate(model, val_loader, device, estimated_bpp)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "image_size": IMAGE_SIZE,
        "latent_dim": LATENT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_groups": NUM_GROUPS,
        "codebook_size": CODEBOOK_SIZE,
        "estimated_bpp": estimated_bpp,
    }, checkpoint_path)

    print(f"Checkpoint saved to: {checkpoint_path}")
    image_path = save_reconstructions(model, val_loader, device, OUTPUT_DIR)
    print(f"Reconstruction image saved to: {image_path}")


def validate(model, val_loader, device, estimated_bpp):
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0

    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            reconstructed, q_loss, _, _ = model(images)
            recon_loss = F.mse_loss(reconstructed, images)
            loss = recon_loss + LAMBDA_Q * q_loss
            psnr = calculate_psnr(images, reconstructed)

            val_loss += loss.item()
            val_psnr += psnr

    val_loss /= len(val_loader)
    val_psnr /= len(val_loader)

    print("Validation results")
    print(f"Validation loss: {val_loss:.6f}")
    print(f"Validation PSNR: {val_psnr:.4f}")
    print(f"Estimated bpp: {estimated_bpp:.6f}")


if __name__ == "__main__":
    train()
