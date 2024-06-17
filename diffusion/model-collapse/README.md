# Model Collapse
Optimizing GAN-generated image performance on CIFAR10 classification.

The pipeline is carried out as follows:
1. Train a variational autoencoder (`vqvae`) on CIFAR10 images. This model will be used for feature extraction to compare original and synthetic images.
2. Perform GAN optimization (`gan-optimization`) to generate optimized synthetic images.
3. Evaluate results using classifier evaluation (`classifier-evaluation`).

Please see the `README.md` in each folder for specific instructions on running each pipeline section.

For t-SNE visualization of real vs synthetic images, please run the `tsne.ipynb` notebook.