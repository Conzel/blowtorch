
How to implement new layers
===========================
Make sure you have a developer installation of blowtorch as 
described in :ref:`installation`.

Rust part
---------

#. Write an implementation of the layer in Rust. This can have any form you want it to have. Place the layer as a new file in :file:`rust/src`, f.e. as :file:`batch_norm.rs`. 
#. Implement the :code:`Layer` interface for your Layer in :file:`layer_implementations.rs`. You might have to import your layer first with a :code:`use` directive.
#. Export the layer in the library, by adding it to the :code:`nn` module. This might require you to first add the module via `mod batch_norm` at the top, then adding it as an export via adding under `pub mod nn` a line like :code:`pub use batch_norm::BatchNorm`. 
#. Add a :code:`use` directive for your module into the models template file. ! Import the layer from the public export of the library directly, e.g. with :code:`use blowtorch::nn::BatchNorm`. 


Python part
-----------
#. Create a new layer in :file:`python/blowtorch/layers`, f.e. :file:`_batch_norm.py`.
#. Implement the :code:`Layer` interface in :file:`_interfaces.py` according to the docstrings given there.
#. Add your layer with a fitting name to the :code:`LAYER_DISPATCH` dictionary in :file:`_parsing.py`, f.e. `"BatchNorm": BatchNorm`. This might require you to import your layer first.  Do a relative import with a leading :code:`.`, such as :code:`from ._batch_norm import BatchNorm`. 

Schema
------
#. Navigate to :file:`python/blowtorch/schema` and add a fitting jsonschema for your python class: Which attributes are required, which form do they have etc.  You will have to add it to the defs in the bottom, see the :code:`convolution` example.
#. Add the schema to the :code:`anyOf` attribute in the layer array.