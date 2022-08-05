"""
Exports the weights in a format that Rust can work with (.npz with everything stripped out
besides weights).
"""
import torch
import numpy as np

from .generate_models import models_from_spec

def get_export_keys(spec: str):
    """
    Returns the exact weight keys that need to be exported from the model.
    This removes unnecessary attributes such as training parameters.
    """
    model_defs = models_from_spec(spec)
    keys_to_export = []
    for model in model_defs:
        for layer in model.layers:
            for weight in layer.weights:
                if weight is not None:
                    keys_to_export.append(f"layers.{layer.name}.{weight.name}")
    return keys_to_export


def export(spec: str, checkpoint: str, out: str):
    """
    Loads the model from the given specification, loads the weights
    that are found in the checkpoint, and writes them to the file given by out.
    """
    print("Loading model...")
    model = torch.load(checkpoint)
    # model update populates some important variables,
    # this is why we have to call it here.
    state_dict = model.state_dict()

    exported_dict = {key: state_dict[key] for key in get_export_keys(spec)}
    # exported_dict["entropy_bottleneck._medians"] = state_dict["entropy_bottleneck.quantiles"][:, :, 1:2].squeeze()
    np.savez(out, **exported_dict)
    print(f"Successfully wrote weights to {out}")
