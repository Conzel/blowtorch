Welcome to blowtorch's documentation!
=====================================
Blowtorch is a Python package that allows you to train machine learning models
and run inference in pure Rust. This is done through a specifying the model
once in a JSON file. Blowtorch then exports your specification into Rust and Python 
models. You can train the Python model as you prefer, and Blowtorch can be run again
to share the weights to Rust. 

An example application built with a predecessor of Blowtorch is `ZipNet <https://conzel.github.io/zipnet/>`__, 
which is a neural-network based compression algorithm run entirely in the browser.
We built Blowtorch as we could not find any easily extensible machine learning frameworks
that could be compiled to WebAssembly.

Advantages over similar packages
- Inference is in pure Rust, meaning your model can run anywhere that Rust runs. You can for example compile it to WebAssembly.
- New layers can be implemented very easily, as one just has to write a forward pass in Rust
- Training is completely in Python, meaning you can use whatever training procedures you like
- Complex networks can be built by splitting the architecture into simpler modules, which are combined together by some glue code

This documentation has the following information:

* "Install" explains how to install Blowtorch (developer and user side)
* "Tutorial" showcases an MNIST example to get started with Blowtorch
* "Model Reference" shows how to create a JSON file that can be parsed by blowtorch to a correct model
* "CLI Reference" contains information about how to use the Blowtorch CLI

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   tutorial
   new_layers
   model_reference
   cli_reference
