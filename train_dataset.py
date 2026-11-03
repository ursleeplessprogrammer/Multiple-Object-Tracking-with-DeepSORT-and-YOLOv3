import argparse
import subprocess
import sys

from dataset import build_dataset_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations")
    parser.add_argument("--yaml", default="dataset.yaml")
    parser.add_argument("--root", default=".")
    parser.add_argument("--train-images", default="images")
    parser.add_argument("--val-images", default="images")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--weights", default="model")
    parser.add_argument("--train-script", default="train.py")
    args = parser.parse_args()

    yaml_path = build_dataset_yaml(
        args.annotations, args.yaml, args.root, args.train_images, args.val_images
    )
    command = [sys.executable, args.train_script, "--epochs", str(args.epochs),
               "--data", str(yaml_path), "--weights", args.weights]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
