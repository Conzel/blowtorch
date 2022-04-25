"""
Exports the weights in a format that Rust can work with (.npz with everything stripped out
besides weights).
"""
import pathlib
import torch
import numpy as np
import argparse
from models import load_model
from compressai.layers import GDN


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


def main(args):
    print("Loading model...")
    model = load_model(str(args.checkpoint))

    # model update populates some important variables,
    # this is why we have to call it here.
    state_dict = model.state_dict()
    # Updates gamma and beta values of the state dict via the reparametrizers
    reparametrize_gdns(state_dict, model)

    # TODO: These keys will later be non-hardcoded
    keys_to_export = {
        'entropy_bottleneck.quantiles',
        'entropy_bottleneck._offset',
        'entropy_bottleneck._quantized_cdf',
        'entropy_bottleneck._cdf_length',
        'analysis_transform.conv0.weight',
        'analysis_transform.gdn0.beta',
        'analysis_transform.gdn0.gamma',
        'analysis_transform.gdn1.beta',
        'analysis_transform.gdn1.gamma',
        'analysis_transform.conv1.weight',
        'analysis_transform.gdn2.beta',
        'analysis_transform.gdn2.gamma',
        'analysis_transform.conv2.weight',
        'analysis_transform.conv3.weight',
        'synthesis_transform.conv_transpose0.weight',
        'synthesis_transform.igdn0.beta',
        'synthesis_transform.igdn0.gamma',
        'synthesis_transform.conv_transpose1.weight',
        'synthesis_transform.igdn1.beta',
        'synthesis_transform.igdn1.gamma',
        'synthesis_transform.conv_transpose2.weight',
        'synthesis_transform.igdn2.beta',
        'synthesis_transform.igdn2.gamma',
        'synthesis_transform.conv_transpose3.weight'
    }

    exported_dict = {key: state_dict[key] for key in keys_to_export}
    exported_dict["entropy_bottleneck._medians"] = state_dict["entropy_bottleneck.quantiles"][:, :, 1:2].squeeze()
    np.savez(args.out, **exported_dict)
    print(f"Successfully wrote weights to {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", metavar="CHECKPOINT",
                        type=pathlib.Path, help="Path to the checkpoint to export weights from, or pretrained model.")
    parser.add_argument("--out", metavar="OUT",
                        type=pathlib.Path, default="weights.npz", help="Name of the file the weights are saved to.")
    args = parser.parse_args()
    main(args)
