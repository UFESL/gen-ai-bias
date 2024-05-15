This section holds Jupyter notebooks for code supporting various figures in the report.

## Sections
- [butterfly figures](./butterflies.ipynb)

## Usage
1. If you need to generate butterfly images, download [pretrained butterfly diffusion model](https://uflorida-my.sharepoint.com/:f:/g/personal/laurachang_ufl_edu/Ell4PMR4xzVBpGRsSU3AU9YBC9okm4s4uXs0rSYtEFIExw?e=rx52GQ) and place the folder directly in this directory:
    ```bash
    .
    ├── ddpm-butterflies-64                  # pretrained diffusion model
    │   └── ...
    ├── butterflies.ipynb
    └── ...
    ```
    Alternatively, [train your own](https://github.com/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb).

2. Set up the Conda environment
    1. Create the environment from the environment.yml file
    ```bash
    conda env create -f environment.yml
    ```
    2. Activate the new environment: `conda activate expanded_diff_lin`
3. Run desired cells from Jupyter notebooks using the kernel created in step 2
