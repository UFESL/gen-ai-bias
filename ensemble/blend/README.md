# Blend Ensemble Model

1. Ensure the `celebA_class_dataset` folder contains subfolders with subsets of CelebA images. These subfolders serve as the **real dataset**.

2. Place images generated from other models (e.g., DCGAN, StyleGAN) into the `gan_outputs` folder. This folder serves as the **generated dataset**.

3. Open and run the `VAE_ensemble.ipynb` notebook to train a Variational Autoencoder (VAE) model:
   - The number of latent space dimensions can be defined by setting the variable `latent_dim` in the model initialization section.

4. The notebook will:
   - Save reconstructions of both real and generated images.
   - Calculate FID scores for:
     - Reconstructions of real images with respect to the original real images.
     - Reconstructions of GAN-generated images with respect to the original real images.

5. Once executed, the notebook saves the trained model weights. If using the saved weights, the training cells in the notebook can be skipped in subsequent runs.

6. **Note**: The above folders (`celebA_class_dataset` and `gan_outputs`) are used by default by the notebook. However, any real and corresponding generated images can be used to train the model. Be sure to update the appropriate directories in the notebook if using a different dataset.
