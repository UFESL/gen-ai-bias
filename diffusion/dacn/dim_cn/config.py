import argparse


def get_training_args():
    parser = argparse.ArgumentParser(
        description="Arguments for training a Dimensional-Aware ControlNet."
    )

    # ***
    # Pretrained models
    # ***
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.",
    )
    parser.add_argument(
        "--pretrained_controlnet",
        type=str,
        default=None,
        help="Path to pretrained Dimensional-Aware ControlNet.",
    )

    # ***
    # Directories
    # ***
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        required=True,
        help="The output directory where model checkpoints and other outputs will be written. Defaults to output.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="The logging directory where logs are stored, e.g. tensorboard output. Defaults to logs.",
    )

    # ***
    # Misc
    # ***
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for reproducible training."
    )

    # ***
    # Dataset
    # ***
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ade",
        choices=["ade"],
        help="Training dataset to use. Choices: ade. Defaults to ade.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for training/validation images. Will resize all images to this size. Defaults to 512.",
    )

    # ***
    # Training
    # ***
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training dataloader batch size. Defaults to 4.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs. Defaults to 1.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Number of steps to perform before checkpointing the model. Defaults to 500.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to perform before performing a backward/update pass. Defaults to 1.",
    )

    # ***
    # Training: Learning Rate
    # ***
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-6,
        help="Initial learning rate to use. Defaults to 5e-6.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by number of GPUs, gradient accumulation steps, and batch size. Defaults to False.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the LR scheduler. Defaults to 500.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler. Defaults to 1.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler. Defaults to 1.0.",
    )

    # ***
    # Training: Adam
    # ***
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Beta1 parameter for Adam. Defaults to 0.9.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Beta2 parameter for Adam. Defaults to 0.999.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use. Defaults to 1e-2.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for Adam. Defaults to 1e-08.",
    )

    # ***
    # Validation
    # ***
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help="Validation prompt to use for corresponding validation image. Defaults to None.",
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help="Path to validation images. Defaults to None.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each prompt, image pair. Defaults to 4.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help="Number of steps in between validation runs. Defaults to 100.",
    )

    return parser.parse_args()
