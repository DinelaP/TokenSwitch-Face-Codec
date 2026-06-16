import os
import torch
import matplotlib.pyplot as plt


def save_reconstructions(model, loader, device, output_dir, num_images=4):
    os.makedirs(output_dir, exist_ok=True)

    model.eval()

    images, _ = next(iter(loader))
    images = images.to(device)

    # Important fix: do not try to show more images than exist in the batch
    num_images = min(num_images, images.size(0))

    with torch.no_grad():
        reconstructed, _, _, _ = model(images)

    images = images[:num_images].cpu()
    reconstructed = reconstructed[:num_images].cpu()

    plt.figure(figsize=(12, 4))

    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        plt.imshow(images[i].permute(1, 2, 0))
        plt.axis("off")

        if i == 0:
            plt.ylabel("Original")

        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(reconstructed[i].permute(1, 2, 0))
        plt.axis("off")

        if i == 0:
            plt.ylabel("Reconstructed")

    save_path = os.path.join(output_dir, "reconstructions.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return save_path