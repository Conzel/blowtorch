import argparse
from pathlib import Path
from .export_weights import export
from .generate_models import generate_models
import os
import sys


def main():
    """Initializes the argument parser and presents the user with a CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "specification",
        metavar="SPEC",
        type=Path,
        help="Specification we should use to create the model file.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export", help="TODO")
    generate_parser = subparsers.add_parser("generate", help="TODO")

    generate_parser.add_argument(
        "--skip-validation",
        type=bool,
        help="If set to true, does not validate the passed specification with the model jsonscheme.",
    )

    export_parser.add_argument(
        "--out",
        metavar="OUT",
        type=Path,
        default="weights.npz",
        help="Name of the file the weights are saved to.",
    )
    export_parser.add_argument(
        "checkpoint",
        metavar="CHECKPOINT",
        type=Path,
        help="Path to the checkpoint to export weights from, or pretrained model.",
    )

    args = parser.parse_args()
    if args.command == "generate":
        generate_models(args.specification, args.skip_validation)
    elif args.command == "export":
        # caller path
        sys.path.append(os.getcwd())
        export(args.specification, args.checkpoint, args.out)
    else:
        raise ValueError("Unknown command")


if __name__ == "__main__":
    main()
