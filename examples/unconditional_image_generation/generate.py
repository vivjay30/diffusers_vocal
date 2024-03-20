import argparse

from diffusers import DiffusionPipeline
import torch

def main(args):
    pipeline = DiffusionPipeline.from_pretrained(args.model_path).to("cuda")
    images = pipeline(batch_size=1)
    images[0][0].save("test.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate from a pretrained checkpoint')
    parser.add_argument("model_path", type=str, help="Path to pretrained model")
    main(parser.parse_args())
