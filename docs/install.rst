.. _installation:

Installation 
============

User Installation
-----------------
You can install our pip package via::

    pip install blowtorch-py

Take care, as the package is blowtorch, but the PyPI name is blowtorch-py.
The Rust part can simply be added to your Cargo.toml by adding the line::

    blowtorch="x.y.z"

with x.y.z. being the current blowtorch version.

Developer Installation
----------------------
If you want to make changes to Blowtorch, you can use the following instructions:

First make sure to install or update the prerequisites:
- Poetry: https://python-poetry.org/docs/
- Rust: https://www.rust-lang.org/tools/install

Then, clone our repository::

    git clone https://github.com/Conzel/blowtorch

Navigate to the python folder::

    cd blowtorch/python

Activate a new poetry shell and install the dependencies (or use any other venv you like):

    poetry shell 
    poetry install

Alternatively, you can install the project with pip from the :file:`pyproject.toml` file. 
Run (in the :file:`blowtorch/python` folder)::

    pip install -e .
