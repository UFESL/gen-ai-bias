# Classifier Evaluation
This section adapts CIFAR10 classification code from [here](https://github.com/kuangliu/pytorch-cifar) and is used to evaluate performance of CIFAR10 images. Performance can be seen by generated classification reports and confusion matrices on the CIFAR10 testset. Testing uses the original CIFAR10 test split, while training is done on separate train and validation subsets.

## Run Instructions

### 1. Prerequisites
Move the synthetic images generated from [gan-optimization](../gan-optimization/) into `./data/gan-opt`

The folder structure should be as follows:
```bash
.
├── models                  # classifier model architectures
│   └── ...
├── data                    # synthetic image results are saved here
│   ├── cifar-10-batches.py # original CIFAR10 images, will be automatically downloaded when needed
│   └── gan-opt             # optimized synthetic images
│       └── {dataset_size}
│           ├── {optimization iteration}
│           │   ├── bird
│           │   ├── car
│           │   └── ...
│           └── {optimization iteration}
│               ├── bird
│               ├── car
│               └── ...
├── output  # classifier-evaluation results will be saved here
│   └── CIFAR
│       └── {dataset_size}_0_{model_architecture}
│           ├── classification_report_{epoch}
│           ├── confusion_matrix_{epoch}
│           ├── train_dist.png
│           ├── training_progress.txt # commmand line output is rerouted here
│           └── val_dist.png
│   ├── model_config.json
│   ├── model.pt
│   ├── training_config.json
├── debiasing_eval_cifar.py
├── README.md
└── utils.py
```

### 2. Classifier Evaluation
For real CIFAR10 images:
``` bash
python debiasing_eval_cifar.py --train_size={dataset_size} --architecture='VGG'
```

For original synthetic images:
``` bash
python debiasing_eval_cifar.py --train_size={dataset_size} --architecture='VGG' --synthetic
```

For optimized synthetic images:
``` bash
python debiasing_eval_cifar.py --train_size={dataset_size} --architecture='VGG' --optimized
```

Results are stored in `./output/`

**Note** Training progress has been rerouted from command line output to  `training_progress.txt` (shown in folder structure), so the terminal will not show updates during model training.
