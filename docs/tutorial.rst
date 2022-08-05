Tutorial: Building an MNIST classifier
======================================

Here we will write a brief tutorial on how to build and train an MNIST classifier by using BlowTorch.

First, clone the repository as follows:

.. code-block:: bash

            git clone https://github.com/Conzel/blowtorch.git


Next, navigate to the examples folder, i.e.

.. code-block:: bash

            cd examples/mnist/    

Model description
^^^^^^^^^^^^^^^^^^^
The first step is to define the architecture of the desired neural network that is trained using PyTorch. Create 
a json file, in our example :file:`mnist.json`, as Blowtorch uses JSON to describe the 
architecture. The required fields for each model are as follows:

.. code-block:: json

            [
                {
                    "module_name": " ", 
                    "layers": [
                        { }, 
                        { }
                    ]
                }
            ]

More specifically, each layer can be added as follows:

1. The type of layer must be described in the key named "type" and this is a required field. Ensure that
the "type" of the layer matches the naming convention of PyTorch accurately. For example, if you want to add 
a 2D convolutional layer, the PyTorch name equivalent is "Conv2d". Additionally, each layer requires a "name"
and this is mandatory for BlowTorch to correctly load weights to rust. Note that the names of each layer have to
be unique. Hence, 

.. code-block:: json

            [
                {
                    "module_name": "MnistClassifier", 
                    "layers": [
                        {
                            "type": "Conv2d", 
                            "name": "conv1"
                        },
                        {
                            "type": "Conv2d", 
                            "name": "conv2"
                        },
                    ]
                }
            ]

2. There are certain arguments that are necessary for each layer and are not considered by default. These "required" arguments
for each layer are chosen by following PyTorch, i.e. all the arguments for a layer that PyTorch does not consider by default are "required".
If we consider the example of "Conv2d", the required arguments are the following:

* in_channels
* out_channels
* kernel_size

The json dictionary can be constructed as follows:

.. code-block:: json

            [
                {
                    "module_name": "MnistClassifier",
                    "layers": [
                        {
                            "type": "Conv2d",
                            "name": "conv1",
                            "in_channels": 1,
                            "out_channels": 6,
                            "kernel_size": "(5,5)"
                        }
                    ]
                }
            ]

            
In a similar fashion, other layers can be added and a full example can be found in :file:`examples/mnist/mnist.json`. 

Generate network architecture (model) files for PyTorch and Rust
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Now, to simultaneously generate the network architecture according to the json description,
that conforms PyTorch format and compatible in Rust, we can do the following: 
(Assuming blowtorch is already installed via pip, please refer to installation instructions found in :file:`docs/install.rst`)

.. code-block:: bash

            blowtorch <path-to-json-file> generate
            blowtorch examples/mnist/mnist.json generate

This command will generate the model description in two files, :file:`model.py` contains the PyTorch model and :file:`model.rs` has the rust one. 

Train a classifier with PyTorch
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The training script for this example can be found in :file:`examples/mnist/train.py`. The file contains a training script and the model for training can be imported
from the step above where we generated the desired network architecture in PyTorch format. For this example, we train the model for 10 epochs.
The network in this example is trained as follows:

.. code-block:: bash

        python examples/mnist/train.py

It is important to save the network weights in the same folder as the models. Hence, ensure that the weights are saved in :file:`.pt` format.

Export model weights to Rust
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Next, we need to convert the PyTorch weights of each layer to a format that can be read by the rust code. We provide a conversion
script that saves the weights in numpy's :file:`.npz` format. 

.. code-block:: bash

        blowtorch <path-to-json-file> export <path-to-weights-file>
        blowtorch examples/mnist/mnist.json export examples/mnist/models/model.py

After this step, the code automatically saves a :file:`weights.npz` file in the same working directory. 

Inference with Rust!
^^^^^^^^^^^^^^^^^^^^
The training code additionally saves a random example image taken from the test dataset in :file:`.npy`
format in :file:`examples/mnist/examples/` folder. To generate multiple random examples, re-run the training script. 
Note that, the examples are saved as :file:`example_` , followed by the ground truth class of that particular example.
For instance, :file:`example_1.npy`  implies that the image belongs to class 1.

Note, please verify if the path of BlowTorch in :file:`Cargo.toml` is correct. 

To run inference on rust open, :file:`src/main.rs` and set the path of the example file generated
by the training code. Next, simply follow these steps:

.. code-block:: bash

        cd examples/mnist
        cargo build
        cargo run

The rust code will print the predicted class and you can verify it with the example file loaded!
