import argparse
import json
from .workflow import train, predict

def main():
    parser = argparse.ArgumentParser(description="A handwritten-digit CNN implemented in NumPy")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="Run the quick offline digits experiment")
    training = commands.add_parser("train")
    training.add_argument("--mode", choices=["quick", "full"], default="full")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--output", default="outputs/full")
    inference = commands.add_parser("predict")
    inference.add_argument("--model", default="outputs/full/models/cnn.npz")
    inputs = inference.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--image", help="Grayscale image; white strokes on black background")
    inputs.add_argument("--index", type=int, help="Index in the bundled digits benchmark")
    inference.add_argument("--invert", action="store_true", help="Invert dark strokes on white")
    args = parser.parse_args()
    if args.command == "demo":
        train("quick", 42, "outputs/demo")
    elif args.command == "train":
        train(args.mode, args.seed, args.output)
    else:
        print(json.dumps(predict(args.model, args.image, args.index, args.invert), indent=2))
