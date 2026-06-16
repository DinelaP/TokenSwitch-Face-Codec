import os

# Default path now points to the included tiny sample dataset.
# Replace this with your ImageNet/ImageNet-subset folder when you want real training.
# Required structure:
# dataset/
#   train/
#     class_1/
#     class_2/
#   val/
#     class_1/
#     class_2/
DATA_DIR = os.getenv("DATA_DIR", "./data/imagenet_subset")

# CPU-friendly defaults so the project runs immediately on a normal laptop.
# For real GPU/ImageNet runs, increase IMAGE_SIZE, BATCH_SIZE and EPOCHS.
IMAGE_SIZE = 64
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-4
LAMBDA_Q = 0.25

LATENT_DIM = 32
HIDDEN_DIM = 64
NUM_GROUPS = 4
CODEBOOK_SIZE = 64

# On Windows/CPU, NUM_WORKERS=0 avoids multiprocessing issues.
NUM_WORKERS = 0
CHECKPOINT_DIR = "./checkpoints"
OUTPUT_DIR = "./outputs"
CHECKPOINT_NAME = "stscq_autoencoder.pth"
