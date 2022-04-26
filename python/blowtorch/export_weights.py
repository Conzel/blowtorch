"""
Exports the weights in a format that Rust can work with (.npz with everything stripped out
besides weights).
"""
import torch
import numpy as np
from compressai.layers import GDN

from .generate_models import models_from_spec


def reparametrize_gdns(state_dict, model):
    """
    Modifies state_dict in place with reparametrized gamma and beta values of
    the GDN layers.
    """
    mods = dict(model.named_modules())
    with torch.no_grad():
        for key, mod in mods.items():
            module = mods[key]
            if isinstance(module, GDN):
                beta_rep = module.beta_reparam(module.beta)
                gamma_rep = module.gamma_reparam(module.gamma)
                state_dict[key + ".beta"] = beta_rep
                state_dict[key + ".gamma"] = gamma_rep


def get_export_keys(spec: str):
    model_defs = models_from_spec(spec)
    keys_to_export = []
    for model in model_defs:
        for layer in model.layers:
            for weight in layer.weights:
                if weight is not None:
                    keys_to_export.append(
                        f"layers.{layer.name}.{weight.name}")
    return keys_to_export


def export(spec: str, checkpoint: str, out: str):
    """TODO"""
    print("Loading model...")
    model = torch.load(checkpoint)
    # model update populates some important variables,
    # this is why we have to call it here.
    state_dict = model.state_dict()
    # Updates gamma and beta values of the state dict via the reparametrizers
    reparametrize_gdns(state_dict, model)

    exported_dict = {key: state_dict[key] for key in get_export_keys(spec)}
    #exported_dict["entropy_bottleneck._medians"] = state_dict["entropy_bottleneck.quantiles"][:, :, 1:2].squeeze()
    np.savez(out, **exported_dict)
    print(f"Successfully wrote weights to {out}")
