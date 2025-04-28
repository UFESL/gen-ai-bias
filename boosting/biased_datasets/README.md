Use the `biased_subsets.ipynb` to create a mix of images from two subclasses of the CelebA dataset saved in the `celebA_class_dataset` folder. It also saves the images, their features, and FID metrics in a convenient manner. The following shows the final structure for a biased subset:

```bash
boosting/
└── biased_datasets/
    ├── biasedset1/
    │   ├── images/
    │   │   └── class/
    │   │       ├── img1.jpg
    │   │       └── ...
    │   ├── features/
    │   │   ├── img1.npy
    │   │   └── ...
    │   └── fid_stats.npz
    └── ...
