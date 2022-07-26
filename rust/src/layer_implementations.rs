/// Contains the implementations of the layer trait
/// for all our layers. This is not terribly interesting,
/// implementations should only be 1 or 2 lines for every
/// layer.
use crate::{
    activation_functions::{GdnLayer, IgdnLayer, ReluLayer},
    flatten::Flatten,
    linear::LinearLayer,
    traits::{FloatLikePrimitive, Layer},
};
use convolutions_rs::{
    convolutions::ConvolutionLayer, transposed_convolutions::TransposedConvolutionLayer,
};
use ndarray::{Array1, Array2, Array3};

impl<F: FloatLikePrimitive> Layer<Array3<F>, Array3<F>> for ConvolutionLayer<F> {
    fn forward_pass(&self, input: &Array3<F>) -> Array3<F> {
        self.convolve(input)
    }
}

impl<F: FloatLikePrimitive> Layer<Array3<F>, Array3<F>> for TransposedConvolutionLayer<F> {
    fn forward_pass(&self, input: &Array3<F>) -> Array3<F> {
        self.transposed_convolve(input)
    }
}
impl<F: FloatLikePrimitive> Layer<Array2<F>, Array2<F>> for LinearLayer<F> {
    fn forward_pass(&self, input: &Array2<F>) -> Array2<F> {
        self.linear(input)
    }
}
impl<F: FloatLikePrimitive> Layer<Array3<F>, Array3<F>> for GdnLayer<F> {
    fn forward_pass(&self, input: &Array3<F>) -> Array3<F> {
        self.activate(input)
    }
}

impl<F: FloatLikePrimitive> Layer<Array3<F>, Array3<F>> for IgdnLayer<F> {
    fn forward_pass(&self, input: &Array3<F>) -> Array3<F> {
        self.activate(input)
    }
}

impl<F: FloatLikePrimitive> Layer<Array3<F>, Array3<F>> for ReluLayer {
    fn forward_pass(&self, input: &Array3<F>) -> Array3<F> {
        self.activate(input)
    }
}
impl<F: FloatLikePrimitive> Layer<Array3<F>, Array2<F>> for Flatten {
    fn forward_pass(&self, input: &Array3<F>) -> Array2<F> {
        self.activate(input)
    }
}
