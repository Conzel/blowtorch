# Tests

This folder is for integration tests only. The folders for the rust and python tests are in the respective rust and python folders. The integration tests here may run for a bit of time, as an ML model has to be (partially) trained.

To run the tests, make sure to install the (dev) dependencies under
`blowtorch/python` via `poetry install`.

## Current tests
- mnist_test.sh: Trains an MNIST classifier and checks that the prediction matches the python one.
