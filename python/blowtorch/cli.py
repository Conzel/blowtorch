import argparse
from pathlib import Path
from .export_weights import export
from .generate_models import generate_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('specification', metavar='SPEC', type=Path,
                        help='Specification we should use to create the model file.')

    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export", help="TODO")
    generate = subparsers.add_parser("generate", help="TODO")

    export.add_argument("--out", metavar="OUT",
                        type=Path, default="weights.npz", help="Name of the file the weights are saved to.")
    export.add_argument("checkpoint", metavar="CHECKPOINT",
                        type=Path, help="Path to the checkpoint to export weights from, or pretrained model.")

    args = parser.parse_args()
    if args.command == "generate":
        generate_models(args.specification)
    elif args.command == "export":
        print("TODO")
    else:
        raise ValueError("Unknown command")


if __name__ == "__main__":
    main()
