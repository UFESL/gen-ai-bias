This folder should contain the features of CelebA images, along with the mean and covariance of these features. Use the `fid_custom.py` file to generate both the features and the statistics, and save them in the structure below:

Final file structure:
```bash
├── boosting/
│   └── celebA_data/
│       ├── features/
│       │   ├── img1.npy
│       │   ├── img2.npy
│       │   └── ...
│       └── celebA_fid_stats.npz
```
