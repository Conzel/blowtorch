import argparse
from pathlib import Path
from .export_weights import export
from .generate_models import generate_models
import os
import sys

def _make_parser() -> argparse.ArgumentParser:
    """Creates the argument parser. A separate function to create
    an argument parser is required for sphinx autodoc."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "specification",
        metavar="SPEC",
        type=Path,
        help="Model specification (JSON) used to create the model files and export the weights",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export", help="Exports Rust weights for a given Pytorch weight file (.pt ending) that contains a saved model fitting SPEC")
    generate_parser = subparsers.add_parser("generate", help="Generates model files with the given SPEC. Models are saved for Rust (models.rs) and Python (models.py) under the folder ./models")

    generate_parser.add_argument(
        "--skip-validation",
        type=bool,
        help="If set to true, does not validate the passed specification with the model jsonschema",
    )

    export_parser.add_argument(
        "--out",
        metavar="OUT",
        type=Path,
        default="weights.npz",
        help="Name of the file the weights are saved to",
    )
    export_parser.add_argument(
        "checkpoint",
        metavar="CHECKPOINT",
        type=Path,
        help="Path to the checkpoint the weights are exported from",
    )
    return parser


def main():
    """Initializes the argument parser and presents the user with a CLI."""
    parser = _make_parser()
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
