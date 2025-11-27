
import argparse
import os
import torch
from datasets import load_dataset
from diffusers import DDPMScheduler, UNet2DModel, DDPMTrainingPipeline
from diffusers.hub_utils import init_git_repo, push_to_hub
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)

def main():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--dataset_name", type=str, default="dataset", help="The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private, dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
    )
    parser.add_argument("--output_dir",type=str,default="fine_tuned_model",help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--overwrite_output_dir",action="store_true",help="Overwrite the content of the output directory. Use this to continue training if output_dir points to a checkpoint directory.",
    )
    parser.add_argument("--resolution", type=int, default=64, help="The resolution for input images, all images in the train/validation dataset will be resized to this resolution")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate",type=float,default=1e-4,help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--lr_scheduler",type=str,default="cosine",help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]',
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--logging_dir", type=str, default="logs", help="Location for logging")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument("--hub_model_id",type=str,default=None,help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument("--hub_private_repo", action="store_true", help="Whether or not to create a private repository.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Initalize the model
    model = UNet2DModel(
        sample_size=args.resolution,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )

    # Initalize the scheduler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Initalize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the dataset
    dataset = load_dataset(args.dataset_name, split="train")

    # Preprocessing the datasets and DataLoaders creation.
    augmentations = Compose(
        [
            Resize(args.resolution, interpolation=InterpolationMode.BILINEAR),
            CenterCrop(args.resolution),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize([0.5], [0.5]),
        ]
    )

    def transforms(examples):
        images = [augmentations(Image.open(file).convert("RGB")) for file in examples["file_name"]]
        return {"input": images}

    dataset.set_transform(transforms)

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True
    )

    # Initalize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_train_epochs),
    )

    # Training loop
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["input"]
            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Get the model prediction
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

            # Calculate the loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1}/{args.num_train_epochs}, Loss: {loss.item()}")

    # Save the model
    pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
    pipeline.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
