Model Reference
===============
The model must be specified via a .json file. This file is validated validated
via a jsonschema. 

The model definition must consist of a list of modules. Each module 
has an attribute "module_name", which can be any string, and an array of
layers. 

Each layer must have a type (Conv2d, Linear, etc.) and a **unique** name.
Depending on the type, the layer has certain required and optional parameters.
The possible layer types and their arguments are described in the following, while **required** parameters are bold.

Layer reference 
---------------

Conv2d
^^^^^^
Does a 2d-image convolution. Expects 3d-input and returns 3d-output.

Parameters:

+------------------+------------+----------------------------------------------------+
| Name             | Type       | Description                                        |
+==================+============+====================================================+
| **out_channels** | int        | #channels in the output image                      |
+------------------+------------+----------------------------------------------------+
| **in_channels**  | int        | #channels in the input image                       |
+------------------+------------+----------------------------------------------------+
| **kernel_size**  | string     | tuple (int,int) describing kernel height and width |
+------------------+------------+----------------------------------------------------+
| stride           | int        | stride of the convolution, default=1               |
+------------------+------------+----------------------------------------------------+
| padding          | string     | padding, either same or valid, default=valid       |
+------------------+------------+----------------------------------------------------+

Conv2dTranspose
^^^^^^^^^^^^^^^
Does a 2d-image transposed convolution (also known as deconvolution), used to increase
the image size in upsampling tasks.

Parameters: see Conv2d

Linear
^^^^^^
A layer of linear perceptrons.

Parameters:

+------------------+------------+----------------------------------------------------+
| Name             | Type       | Description                                        |
+==================+============+====================================================+
| **in_features**  | int        | # of input features                                |
+------------------+------------+----------------------------------------------------+
| **out_features** | int        | # of output features                               |
+------------------+------------+----------------------------------------------------+

Flatten
^^^^^^^
Flattens the input into 1d.

Parameters: none

ReLU
^^^^
ReLU activation layer. 

Parameters: none

