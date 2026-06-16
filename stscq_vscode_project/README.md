# STSCQ Image Compression - VS Code Project

This project is a simplified PyTorch implementation of the main idea from the paper:

**Switchable Token-Specific Codebook Quantization For Face Image Compression**

The original paper focuses on face image compression. For this assignment, the implementation was adapted to work with an ImageNet-style dataset structure.

## Important Note

No official public code repository was found for this paper. Because of that, this project manually implements the core idea of the algorithm based on the paper description.

The implementation includes:

* encoder-decoder image compression model,
* switchable image-level codebook groups,
* token-specific codebooks,
* straight-through quantization,
* evaluation using MSE, PSNR and estimated bpp,
* reconstruction image saving for visual comparison.

## Project Structure

```text
stscq_vscode_project/
  .venv/
  checkpoints/
    stscq_autoencoder.pth
  configs/
    config.py
  data/
    imagenet_subset/
      train/
        class_1/
        class_2/
      val/
        class_1/
        class_2/
  outputs/
    reconstructions.png
  scripts/
  src/
    dataset.py
    metrics.py
    model.py
    visualize.py
  .gitignore
  .gitkeep
  evaluate.py
  README.md
  requirements.txt
  train.py
```

## Dataset Structure

The project uses an ImageNet-style folder structure. Images must be placed inside class folders.

Example:

```text
data/imagenet_subset/
  train/
    class_1/
      image1.jpg
      image2.jpg
    class_2/
      image1.jpg
      image2.jpg
  val/
    class_1/
      image1.jpg
      image2.jpg
    class_2/
      image1.jpg
      image2.jpg
```

The current project contains a small sample dataset so the code can run immediately. This sample dataset is only used to test that the implementation works.

The same code can be used with full ImageNet-1k by replacing the sample dataset with the real ImageNet-1k folder structure.

## How to Run the Project in Visual Studio Code

### 1. Open the Project

Open the `stscq_vscode_project` folder in Visual Studio Code.

### 2. Create a Virtual Environment

In the VS Code terminal, run:

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\activate
```

When the environment is active, the terminal should show:

```text
(.venv)
```

### 3. Install Dependencies

Install the required libraries:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision torchaudio
python -m pip install tqdm matplotlib pillow numpy
```

Alternatively, if the environment works correctly, dependencies can be installed with:

```powershell
python -m pip install -r requirements.txt
```

### 4. Check Dataset Path

The dataset path is configured in:

```text
configs/config.py
```

Current setting:

```python
DATA_DIR = "./data/imagenet_subset"
```

This path already matches the included sample dataset.

### 5. Train the Model

Run:

```powershell
python train.py
```

The training script performs:

* dataset loading,
* model creation,
* training,
* validation,
* checkpoint saving,
* reconstruction image saving.

After training, the checkpoint is saved to:

```text
checkpoints/stscq_autoencoder.pth
```

The reconstruction image is saved to:

```text
outputs/reconstructions.png
```

### 6. Evaluate the Model

Run:

```powershell
python evaluate.py
```

The evaluation script loads the trained checkpoint and calculates final metrics on the validation dataset.

## Current Test Results

The implementation was successfully tested on a small ImageNet-style subset.

```text
Device: CPU
Train images: 16
Validation images: 8
Number of classes: 2
Epochs: 1
Estimated bpp: 0.094238
Validation MSE: 0.039600
Validation PSNR: 14.2050 dB
```

These results confirm that the full implementation pipeline works. However, because the model was trained on a very small sample dataset and only for one epoch on CPU, these results are not directly comparable with the original paper.

## Metrics

The original paper evaluates face compression using identity-based metrics such as MeanAcc and IDS. Since this project is adapted for ImageNet-style data, those face identity metrics are not directly applicable.

This implementation uses:

* **MSE** - Mean Squared Error between original and reconstructed images.
* **PSNR** - Peak Signal-to-Noise Ratio, used to measure reconstruction quality.
* **bpp** - bits per pixel, used to estimate compression rate.

If full ImageNet-1k evaluation is required, Top-1 and Top-5 accuracy can be added by running a pretrained ImageNet classifier on reconstructed images.

## Implementation Explanation

The model follows an encoder-quantizer-decoder architecture.

First, the encoder converts the input image into a smaller latent representation. Then, the STSCQ quantizer performs discrete quantization. The quantizer contains multiple codebook groups. A routing network selects one codebook group for each image. After that, each latent token is quantized using its own token-specific codebook.

The straight-through estimator is used so that gradients can pass through the quantization step during training. Finally, the decoder reconstructs the image from the quantized latent representation.

## Problems Faced

The first problem was that no official public code repository was found for the selected paper. This was solved by manually implementing a simplified version of the main algorithm.

The second problem was that the original paper focuses on face datasets, while the assignment requires ImageNet-1k. This was solved by adapting the code to support the standard ImageNet folder structure.

The third problem was hardware limitation. Full ImageNet-1k requires large storage, GPU memory and longer training time. Because of that, the implementation was tested on a small ImageNet-style subset. The same code can run on the full ImageNet-1k dataset if the dataset path is changed.

## Documentation Text

The implementation was developed in Python using PyTorch and Visual Studio Code. Since the authors did not provide a publicly available official implementation, a simplified implementation was created based on the algorithm described in the paper. The model follows an encoder-quantizer-decoder architecture. The encoder converts images into latent tokens, the STSCQ quantizer selects a codebook group using a routing network and quantizes each token using a token-specific codebook, and the decoder reconstructs the image from the quantized representation.

The original paper evaluates the algorithm on face datasets using identity-preserving metrics. However, the assignment requires ImageNet-1k, so the evaluation was adapted to use reconstruction quality metrics and estimated bitrate. The code supports the standard ImageNet folder structure and can also be tested on a smaller ImageNet-style subset due to hardware limitations.

## Conclusion

This project demonstrates the main idea of Switchable Token-Specific Codebook Quantization in a simplified PyTorch implementation. The system successfully trains, evaluates and saves reconstructed images. Although the current test was performed on a small dataset, the project structure supports larger ImageNet-style datasets by changing the dataset folder.
