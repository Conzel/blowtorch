CLI Reference
=============
The command line interface (CLI) is the main user-facing part of Blowtorch.
The CLI is used to generate model files from a jsonschema and to export the trained
weights to Rust.

.. argparse::
   :ref: blowtorch.cli._make_parser
   :prog: blowtorch