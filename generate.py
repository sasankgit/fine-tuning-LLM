
import argparse
import torch
from diffusers import DDPMPipeline

def main():
    parser = argparse.ArgumentParser(description="Simple example of a generation script.")
    parser.add_argument("--model_path",type=str,default="fine_tuned_model",help="The path to the fine-tuned model.",
    )
    parser.add_argument("--output_path", type=str, default="generated_image.png", help="The path to save the generated image.")
    args = parser.parse_args()

    # Load the fine-tuned model
    pipeline = DDPMPipeline.from_pretrained(args.model_path)

    # Generate an image
    image = pipeline().images[0]

    # Save the image
    image.save(args.output_path)

if __name__ == "__main__":
    main()
