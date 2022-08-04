#!/usr/bin/env bash
set -e

echo "Starting MNIST integration test. Make sure that your blowtorch installation is working and up to date."
cd mnist_test
rm -f model.pt
blowtorch mnist.json generate
python train.py
blowtorch mnist.json export model.pt
cargo run
exit 0
