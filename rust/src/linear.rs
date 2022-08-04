//! Contains the implementation of a simple multi-layer perceptron,
//! also called Linear layer.
//! See <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html?highlight=linear#torch.nn.Linear>
use ndarray::*;
use num_traits::Float;

/// Rust implementation of a linear layer.

pub struct LinearLayer<F: Float> {
    /// Weight matrix of the kernel
    pub(in crate) weights: Array2<F>,
    pub(in crate) bias: Option<Array1<F>>,
}

impl<F: 'static + Float + std::ops::AddAssign> LinearLayer<F> {
    /// Creates new linear layer.
    /// The weights are given in Pytorch layout.

    pub fn new(weights_array: Array2<F>, bias_array: Option<Array1<F>>) -> LinearLayer<F> {
        LinearLayer {
            weights: weights_array,
            bias: bias_array,
        }
    }

    /// Analog to nn.Linear.
    pub fn linear(&self, input_array: &Array1<F>) -> Array1<F> {
        multiply(&self.weights, self.bias.as_ref(), input_array)
    }
}

/// Performs a linear on the given input_array data using this layers parameters.

/// Input:
/// -----------------------------------------------
/// - kernel_weights: weights of shape (F, C)
/// - im2d: Input data of shape (C, D)

/// Returns:
/// -----------------------------------------------
/// - out: Output data, of shape (F, D)
pub fn multiply<'a, T, V, F: 'static + Float + std::ops::AddAssign>(
    kernel_weights: T,
    bias: Option<&Array1<F>>,
    im2d: V,
) -> Array1<F>
where
    // This trait bound ensures that kernel and im2d can be passed as owned array or view.
    // AsArray just ensures that im2d can be converted to an array view via ".into()".
    // Read more here: https://docs.rs/ndarray/0.12.1/ndarray/trait.AsArray.html
    V: AsArray<'a, F, Ix1>,
    T: AsArray<'a, F, Ix2>,
{
    // Initialisations
    let im2d_1d: ArrayView1<F> = im2d.into();
    let out_shape = im2d_1d.len_of(Axis(0));
    let im2d_arr = im2d_1d.into_shape((1, out_shape)).unwrap();

    let kernel_weights_arr: ArrayView2<F> = kernel_weights.into();

    let mul = im2d_arr.dot(&kernel_weights_arr.t());
    let output_array = add_bias(&mul, bias);
    let flatten_output: Array1<F> = Array::from_iter(output_array.map(|a| *a));
    flatten_output
}

pub(in crate) fn add_bias<F>(x: &Array2<F>, bias: Option<&Array1<F>>) -> Array2<F>
where
    F: 'static + Float + std::ops::AddAssign,
{
    if let Some(bias_array) = bias {
        assert!(
            bias_array.shape()[0] == x.shape()[1],
            "Bias array has the wrong shape {:?} for vec of shape {:?}",
            bias_array.shape(),
            x.shape()
        );
        // Yes this is really necessary. Broadcasting with ndarray-rust
        // starts at the right side of the shape, so we have to add
        // the axes by hand (else it thinks that it should compare the
        // output width and the bias channels).
        (x + &bias_array
            .clone()
            .insert_axis(Axis(0))
            .broadcast(x.shape())
            .unwrap())
            .into_dimensionality()
            .unwrap()
    } else {
        x.clone()
    }
}

fn arr_allclose<D: Dimension>(current: &Array<f32, D>, target: &Array<f32, D>) -> bool {
    assert_eq!(
        current.shape(),
        target.shape(),
        "\ngiven array had shape {:?}, but target had shape {:?}",
        current.shape(),
        target.shape()
    );
    (current - target).map(|x| (*x as f32).abs()).sum() < 1e-3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear() {
        let test_img: Array1<f32> = Array::from_shape_vec(
            12,
            vec![
                -1.0643, -0.8746, -0.5266, 0.6039, 0.7219, -0.8092, 0.1590, -0.2309, 0.6337,
                -1.4233, 0.7101, -0.9875,
            ],
        )
        .unwrap();

        let kernel: Array2<f32> = Array::from_shape_vec(
            (4, 12),
            vec![
                0.0379, 0.1877, 0.2359, 0.0712, 0.0907, -0.0815, 0.1697, -0.0474, -0.0823, -0.1261,
                -0.1167, 0.0740, 0.2609, -0.0292, -0.2330, 0.1270, -0.0309, -0.2788, 0.1672,
                -0.1382, -0.2816, 0.2592, 0.0464, -0.2120, -0.0236, -0.1604, -0.1838, -0.1979,
                -0.1971, 0.0578, -0.0632, 0.1702, 0.2735, 0.1344, -0.1922, -0.0913, 0.0733, 0.0641,
                0.0564, -0.2869, -0.1568, 0.2572, -0.0046, -0.1427, 0.0275, -0.0283, -0.1056,
                0.2554,
            ],
        )
        .unwrap();
        let linear_layer = LinearLayer::new(kernel, None);
        let linear_output = linear_layer.linear(&test_img);
        let output: Array1<f32> =
            Array::from_shape_vec((4), vec![-0.1451, -0.0960, -0.1602, -0.8956]).unwrap();
        assert!(
            arr_allclose(&linear_output, &output),
            "{:?} was not equal to {:?}",
            linear_output,
            output
        );
    }

    #[test]
    fn test_linear1() {
        let test_img: Array1<f32> = Array::from_shape_vec(
            12,
            vec![
                -1.0643, -0.8746, -0.5266, 0.6039, 0.7219, -0.8092, 0.1590, -0.2309, 0.6337,
                -1.4233, 0.7101, -0.9875,
            ],
        )
        .unwrap();

        let kernel: Array2<f32> = Array::from_shape_vec(
            (4, 12),
            vec![
                0.0379, 0.1877, 0.2359, 0.0712, 0.0907, -0.0815, 0.1697, -0.0474, -0.0823, -0.1261,
                -0.1167, 0.0740, 0.2609, -0.0292, -0.2330, 0.1270, -0.0309, -0.2788, 0.1672,
                -0.1382, -0.2816, 0.2592, 0.0464, -0.2120, -0.0236, -0.1604, -0.1838, -0.1979,
                -0.1971, 0.0578, -0.0632, 0.1702, 0.2735, 0.1344, -0.1922, -0.0913, 0.0733, 0.0641,
                0.0564, -0.2869, -0.1568, 0.2572, -0.0046, -0.1427, 0.0275, -0.0283, -0.1056,
                0.2554,
            ],
        )
        .unwrap();

        let linear_layer = LinearLayer::new(
            kernel,
            Some(Array::from_shape_vec((4,), vec![0.0487, -0.1376, -0.2240, -0.1867]).unwrap()),
        );
        let linear_output = linear_layer.linear(&test_img);
        let output: Array1<f32> =
            Array::from_shape_vec((4), vec![-0.0964, -0.2336, -0.3842, -1.0823]).unwrap();
        assert!(
            arr_allclose(&linear_output, &output),
            "{:?} was not equal to {:?}",
            linear_output,
            output
        );
    }
}
