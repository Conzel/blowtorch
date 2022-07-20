# blowtorch
WIP framework for running Pytorch models in Rust for inference.

## Task list

- [x] Export and import trained weights
- [ ] Provide implementations for the follow layers:
    - [x] Conv
    - [x] ConvT
    - [ ] ReLU
    - [ ] GDN
    - [ ] iGDN
    - [ ] Flatten
    - [x] Linear
- [ ] Provide an example
- [ ] Provide possibilities to extend the framework
- [ ] Write Readme
- [x] Write documentation for Python
- [x] Write documentation for Rust

## How to implement new layers
### Rust part
1. Write an implementation of the layer in Rust. This can have any form you want it to have. Place the layer as a new file in `rust/src`, f.e. as `batch_norm.rs`. 
2. Implement the `Layer` interface for your Layer in `layer_implementations.rs`. You
might have to import your layer first with a `use` directive.
3. Export the layer in the library, by adding it to the `nn` module. This might
require you to first add the module via `mod batch_norm` at the top, then adding it 
as an export via adding under `pub mod nn` a line like `pub use batch_norm::BatchNorm`. 
4. Add a `use` directive for your module into the models template file. ! Import 
the layer from the public export of the library directly, e.g. with `use blowtorch::nn::BatchNorm`. 

### Python part
1. Create a new layer in `python/blowtorch/layers`, f.e. `_batch_norm.py`.
2. Implement the `Layer` interface in `_interfaces.py` according to the docstrings
given there.
3. Add your layer with a fitting name to the `LAYER_DISPATCH` dictionary in `_parsing.py`, f.e. `"BatchNorm": BatchNorm`. This might require you to import your layer first.
Do a relative import with a leading `.`, such as `from ._batch_norm import BatchNorm`. 

### Schema
1. Navigate to `python/blowtorch/schema` and add a fitting jsonschema for your 
python class: Which attributes are required, which form do they have etc.
You will have to add it to the defs in the bottom, see the `convolution` example.
2. Add the schema to the `anyOf` attribute in the layer array.

## How to install
Simply navigate to the python folder and run `pip install . -e`. This installs 
the package in edit mode, all changes you make will be reflected without
reinstalling.