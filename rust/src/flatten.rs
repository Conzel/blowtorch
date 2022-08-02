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

    pub fn activate<F: Float>(&self, x: &Array3<F>) -> Array1<F> {
        // let out_channel = x.len_of(Axis(0));
        // let h_prime = x.len_of(Axis(1));
        // let w_prime = x.len_of(Axis(2));
        // let out_shape = out_channel * h_prime * w_prime;
        let x_array: ArrayView3<F> = x.into();
        let flatten_img: Array1<F> = Array::from_iter(x_array.map(|a| *a));
        flatten_img
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten1() {
        let test_array: Array3<f32> = array![[
            [-1.0643, -0.8746, -0.5266, 0.6039],
            [0.7219, -0.8092, 0.1590, -0.2309],
            [0.6337, -1.4233, 0.7101, -0.9875]
        ]];
        let output: Array1<f32> = array![
            -1.0643, -0.8746, -0.5266, 0.6039, 0.7219, -0.8092, 0.1590, -0.2309, 0.6337, -1.4233,
            0.7101, -0.9875
        ];
        let flatten_layer = Flatten::new();
        let flatten_array = flatten_layer.activate(&test_array);
        assert_eq!(flatten_array, output);
    }

    #[test]
    fn test_flatten2() {
        let test_array: Array3<f32> = array![[
            [-1.0643, -0.8746, -0.5266, 0.6039],
            [0.7219, -0.8092, 0.1590, -0.2309]
        ]];
        let output: Array1<f32> =
            array![-1.0643, -0.8746, -0.5266, 0.6039, 0.7219, -0.8092, 0.1590, -0.2309];
        let flatten_layer = Flatten::new();
        let flatten_array = flatten_layer.activate(&test_array);
        assert_eq!(flatten_array, output);
    }
}
