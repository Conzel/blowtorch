//! Contains an implementation of the "Flatten" layer
//! See <https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html>
use ndarray::*;
use num_traits::Float;

/// Rust implementation of a linear layer.
pub struct Flatten {}

impl Flatten {
    pub fn new() -> Self {
        Self {}
    }

    pub fn activate<F: Float>(&self, x: &Array2<F>) -> Array1<F> {
        let in_channel = x.len_of(Axis(0));
        let out_channel = x.len_of(Axis(1));
        let x_array: ArrayView2<F> = x.into();
        let output = x_array.into_shape(in_channel * out_channel);
        output.unwrap().to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten1() {
        let test_array = array![[-1.0643, -0.8746, -0.5266,  0.6039], 
        [ 0.7219, -0.8092,  0.1590, -0.2309], 
        [ 0.6337, -1.4233,  0.7101, -0.9875]];
        let output = array![-1.0643, -0.8746, -0.5266,  0.6039, 0.7219, -0.8092,  0.1590, -0.2309, 0.6337, -1.4233,  0.7101, -0.9875];
        let flatten_layer = Flatten::new();
        let flatten_array = flatten_layer.activate(&test_array);
        assert_eq!(flatten_array, output);
    }

    #[test]
    fn test_flatten2() {
        let test_array = array![[-1.0643, -0.8746, -0.5266,  0.6039], 
        [ 0.7219, -0.8092,  0.1590, -0.2309]];
        let output = array![-1.0643, -0.8746, -0.5266,  0.6039, 0.7219, -0.8092,  0.1590, -0.2309];
        let flatten_layer = Flatten::new();
        let flatten_array = flatten_layer.activate(&test_array);
        assert_eq!(flatten_array, output);
    }
}